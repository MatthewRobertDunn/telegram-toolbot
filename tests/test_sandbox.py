import sys
import os
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
from sandbox import (
    DockerTimeoutError,
    Sandbox,
    SandboxManager,
    SandboxStartError,
)


class FakeDocker:
    """In-memory stand-in for the docker CLI.

    Tracks container states in a dict so lifecycle logic can be exercised
    without Docker. A container name maps to its state: 'true' (running),
    'false' (stopped), or absent (never created / removed). Commands whose
    last argument contains 'TIMEOUT_MARKER' raise DockerTimeoutError.
    """

    def __init__(self):
        self.calls: list[list[str]] = []
        self.containers: dict[str, str] = {}
        self.fail_run = False

    async def __call__(self, args: list[str], timeout=None, input=None):
        self.calls.append(list(args))
        verb = args[0]
        if verb == 'inspect':
            name = args[3]
            if name in self.containers:
                return 0, self.containers[name] + '\n', ''
            return 1, '', f'Error: No such object: {name}'
        if verb == 'run':
            name = args[args.index('--name') + 1]
            if self.fail_run:
                return 1, '', 'simulated docker run failure'
            self.containers[name] = 'true'
            return 0, f'{name}-id\n', ''
        if verb == 'rm':
            name = args[2]
            if name in self.containers:
                del self.containers[name]
                return 0, name, ''
            return 1, '', f'Error: No such container: {name}'
        if verb == 'exec':
            command = args[-1]
            if 'TIMEOUT_MARKER' in command:
                raise DockerTimeoutError('timed out')
            return 0, f'out({command})', f'err({command})'
        if verb == 'ps':
            return 0, ''.join(f'{name}\n' for name in self.containers), ''
        return 0, '', ''

    def run_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == 'run']

    def rm_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == 'rm']

    def exec_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == 'exec']


class TestSandboxManager(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.fake = FakeDocker()
        self.manager = SandboxManager(docker=self.fake)

    async def _ensure(self, user_id: int) -> Sandbox:
        return await self.manager.ensure_sandbox(user_id)

    async def test_container_name_per_user(self):
        self.assertEqual(SandboxManager.container_name_for(42),
                         'toolbot-sandbox-42')

    async def test_ensure_starts_container_once(self):
        sandbox = await self._ensure(1)
        again = await self._ensure(1)
        self.assertEqual(len(self.fake.run_calls()), 1)
        self.assertEqual(sandbox.container_name, 'toolbot-sandbox-1')
        self.assertIs(sandbox, again)

    async def test_one_sandbox_per_user_are_distinct(self):
        first = await self._ensure(1)
        second = await self._ensure(2)
        self.assertEqual(first.container_name, 'toolbot-sandbox-1')
        self.assertEqual(second.container_name, 'toolbot-sandbox-2')
        self.assertEqual(len(self.fake.run_calls()), 2)

    async def test_run_uses_label_and_limits(self):
        await self._ensure(1)
        run = self.fake.run_calls()[0]
        self.assertIn('toolbot.sandbox=1', run)
        self.assertIn(config.SANDBOX_MEMORY, run)
        self.assertIn(config.SANDBOX_IMAGE, run)
        self.assertEqual(run[-2:], ['sleep', 'infinity'])

    async def test_ensure_recreates_dead_container(self):
        await self._ensure(1)
        self.fake.containers['toolbot-sandbox-1'] = 'false'
        await self._ensure(1)
        self.assertEqual(len(self.fake.run_calls()), 2)
        self.assertEqual(self.fake.containers['toolbot-sandbox-1'], 'true')

    async def test_ensure_failure_raises(self):
        self.fake.fail_run = True
        with self.assertRaises(SandboxStartError):
            await self._ensure(1)
        self.assertNotIn(1, self.manager._sandboxes)

    async def test_exec_returns_output_and_updates_timer(self):
        sandbox = await self._ensure(1)
        sandbox.last_activity = time.time() - 1000
        result = await self.manager.exec_in_sandbox(1, 'echo hi')
        self.assertEqual(result['returncode'], 0)
        self.assertEqual(result['stdout'], 'out(echo hi)')
        self.assertEqual(result['stderr'], 'err(echo hi)')
        self.assertFalse(result['timeout'])
        self.assertGreater(sandbox.last_activity, time.time() - 60)
        self.assertEqual(len(self.fake.exec_calls()), 1)
        self.assertIn('echo hi', self.fake.exec_calls()[0])

    async def test_exec_starts_sandbox_if_needed(self):
        await self.manager.exec_in_sandbox(7, 'uptime')
        self.assertEqual(len(self.fake.run_calls()), 1)
        self.assertEqual(self.fake.exec_calls()[0][1], 'toolbot-sandbox-7')

    async def test_exec_timeout_reported(self):
        result = await self.manager.exec_in_sandbox(1, 'sleep TIMEOUT_MARKER')
        self.assertTrue(result['timeout'])
        self.assertIsNone(result['returncode'])
        self.assertIn('timed out', result['stderr'])

    async def test_ensure_does_not_refresh_timer(self):
        sandbox = await self._ensure(1)
        sandbox.last_activity = time.time() - 1000
        await self._ensure(1)
        self.assertLess(sandbox.last_activity, time.time() - 900)

    async def test_destroy_removes_container(self):
        await self._ensure(1)
        rm_before = len(self.fake.rm_calls())
        destroyed = await self.manager.destroy_sandbox(1)
        self.assertTrue(destroyed)
        self.assertNotIn('toolbot-sandbox-1', self.fake.containers)
        rm_calls = self.fake.rm_calls()
        self.assertEqual(len(rm_calls), rm_before + 1)
        self.assertEqual(rm_calls[-1], ['rm', '-f', 'toolbot-sandbox-1'])
        self.assertNotIn(1, self.manager._sandboxes)

    async def test_destroy_without_sandbox_returns_false(self):
        destroyed = await self.manager.destroy_sandbox(1)
        self.assertFalse(destroyed)

    async def test_reap_destroys_only_idle_sandboxes(self):
        await self._ensure(1)
        await self._ensure(2)
        assert isinstance(self.manager._sandboxes[1], Sandbox)
        self.manager._sandboxes[1].last_activity = (
            time.time() - config.SANDBOX_IDLE_TIMEOUT - 1)

        destroyed = await self.manager.reap_idle()

        self.assertEqual(destroyed, [1])
        self.assertNotIn('toolbot-sandbox-1', self.fake.containers)
        self.assertEqual(self.fake.containers['toolbot-sandbox-2'], 'true')

    async def test_reap_leaves_sandbox_just_under_timeout(self):
        await self._ensure(1)
        self.manager._sandboxes[1].last_activity = (
            time.time() - config.SANDBOX_IDLE_TIMEOUT + 1)
        self.assertEqual(await self.manager.reap_idle(), [])
        self.assertEqual(self.fake.containers['toolbot-sandbox-1'], 'true')

    async def test_cleanup_all_stale_removes_labeled_containers(self):
        self.fake.containers = {'abc123': 'true', 'def456': 'false'}
        count = await self.manager.cleanup_all_stale()
        self.assertEqual(count, 2)
        self.assertEqual([c[2] for c in self.fake.rm_calls()],
                         ['abc123', 'def456'])
        self.assertIn('--filter', self.fake.calls[0])


if __name__ == '__main__':
    unittest.main()
