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


def docker_time(epoch: float) -> str:
    """Format an epoch timestamp the way docker inspect reports times."""
    return time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(epoch)) + '.000000000Z'


class FakeDocker:
    """In-memory stand-in for the docker CLI.

    Tracks container states in a dict so lifecycle logic can be exercised
    without Docker. A container name maps to its state: 'true' (running),
    'false' (stopped), or absent (never created / removed). FinishedAt
    timestamps and the toolbot.uid label can be faked to simulate containers
    left over from a previous bot run. Commands whose last argument contains
    'TIMEOUT_MARKER' raise DockerTimeoutError.
    """

    def __init__(self):
        self.calls: list[list[str]] = []
        self.containers: dict[str, str] = {}
        self.finished_at: dict[str, str] = {}
        self.labels: dict[str, str] = {}
        self.fail_run = False
        self.fail_start = False

    async def __call__(self, args: list[str], timeout=None, input=None):
        self.calls.append(list(args))
        verb = args[0]
        if verb == 'inspect':
            name = args[3]
            if name not in self.containers:
                return 1, '', f'Error: No such object: {name}'
            if 'FinishedAt' in args[2]:
                finished = self.finished_at.get(name, '0001-01-01T00:00:00Z')
                uid = self.labels.get(name, '')
                return 0, f'{self.containers[name]} {finished} {uid}\n', ''
            return 0, self.containers[name] + '\n', ''
        if verb == 'run':
            name = args[args.index('--name') + 1]
            if self.fail_run:
                return 1, '', 'simulated docker run failure'
            self.containers[name] = 'true'
            return 0, f'{name}-id\n', ''
        if verb == 'start':
            name = args[1]
            if self.fail_start or name not in self.containers:
                return 1, '', f'Error: No such container: {name}'
            self.containers[name] = 'true'
            return 0, name + '\n', ''
        if verb == 'stop':
            name = args[1]
            if name not in self.containers:
                return 1, '', f'Error: No such container: {name}'
            self.containers[name] = 'false'
            return 0, name + '\n', ''
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

    def start_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == 'start']

    def stop_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == 'stop']

    def exec_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == 'exec']


class TestSandboxManager(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.fake = FakeDocker()
        self.manager = SandboxManager(docker=self.fake)

    async def _ensure(self, user_id: int) -> Sandbox:
        return await self.manager.ensure(user_id)

    def _add_leftover(self, name: str, uid: int | None, running: bool,
                      finished: float):
        self.fake.containers[name] = 'true' if running else 'false'
        self.fake.finished_at[name] = docker_time(finished)
        if uid is not None:
            self.fake.labels[name] = str(uid)

    async def test_ensure_starts_container_once(self):
        sandbox = await self._ensure(1)
        again = await self._ensure(1)
        self.assertEqual(len(self.fake.run_calls()), 1)
        self.assertEqual(sandbox.container_name, 'toolbot-sandbox-1')
        self.assertIs(sandbox, again)
        # The run carries the uid label (used to recover leftovers), the
        # configured resource limits, and the keep-alive command.
        run = self.fake.run_calls()[0]
        self.assertIn('toolbot.sandbox=1', run)
        self.assertIn(config.SANDBOX_MEMORY, run)
        self.assertIn(config.SANDBOX_IMAGE, run)
        self.assertEqual(run[-2:], ['sleep', 'infinity'])

    async def test_one_sandbox_per_user_are_distinct(self):
        first = await self._ensure(1)
        second = await self._ensure(2)
        self.assertEqual(first.container_name, 'toolbot-sandbox-1')
        self.assertEqual(second.container_name, 'toolbot-sandbox-2')
        self.assertEqual(len(self.fake.run_calls()), 2)

    async def test_ensure_restarts_stopped_container(self):
        sandbox = await self._ensure(1)
        sandbox.last_activity = (
            time.time() - config.SANDBOX_IDLE_STOP_TIMEOUT - 1)
        stopped, destroyed = await self.manager.reap_idle()
        self.assertEqual((stopped, destroyed), ([1], []))
        self.assertEqual(self.fake.containers['toolbot-sandbox-1'], 'false')

        restarted = await self._ensure(1)
        self.assertEqual(self.fake.start_calls()[-1],
                         ['start', 'toolbot-sandbox-1'])
        self.assertEqual(len(self.fake.run_calls()), 1)  # files preserved
        self.assertEqual(self.fake.containers['toolbot-sandbox-1'], 'true')
        self.assertGreater(restarted.last_activity, time.time() - 60)

    async def test_ensure_recreates_when_restart_fails(self):
        await self._ensure(1)
        self.fake.containers['toolbot-sandbox-1'] = 'false'
        self.fake.fail_start = True
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
        result = await self.manager.exec(1, 'echo hi')
        self.assertEqual(result['returncode'], 0)
        self.assertEqual(result['stdout'], 'out(echo hi)')
        self.assertEqual(result['stderr'], 'err(echo hi)')
        self.assertFalse(result['timeout'])
        self.assertGreater(sandbox.last_activity, time.time() - 60)
        self.assertEqual(len(self.fake.exec_calls()), 1)
        self.assertIn('echo hi', self.fake.exec_calls()[0])

    async def test_exec_timeout_reported(self):
        result = await self.manager.exec(1, 'sleep TIMEOUT_MARKER')
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
        destroyed = await self.manager.destroy(1)
        self.assertTrue(destroyed)
        self.assertNotIn('toolbot-sandbox-1', self.fake.containers)
        rm_calls = self.fake.rm_calls()
        self.assertEqual(len(rm_calls), rm_before + 1)
        self.assertEqual(rm_calls[-1], ['rm', '-f', 'toolbot-sandbox-1'])
        self.assertNotIn(1, self.manager._sandboxes)
        # Destroying a user with no sandbox at all is a no-op, not an error.
        self.assertFalse(await self.manager.destroy(1))

    async def test_reap_stops_only_idle_sandboxes(self):
        await self._ensure(1)
        await self._ensure(2)
        assert isinstance(self.manager._sandboxes[1], Sandbox)
        self.manager._sandboxes[1].last_activity = (
            time.time() - config.SANDBOX_IDLE_STOP_TIMEOUT - 1)

        stopped, destroyed = await self.manager.reap_idle()

        self.assertEqual((stopped, destroyed), ([1], []))
        self.assertEqual(self.fake.stop_calls()[-1], ['stop', 'toolbot-sandbox-1'])
        self.assertEqual(self.fake.containers['toolbot-sandbox-1'], 'false')
        self.assertEqual(self.fake.containers['toolbot-sandbox-2'], 'true')
        # Still tracked so it is destroyed once the destroy timeout passes.
        self.assertIn(1, self.manager._sandboxes)

    async def test_reap_leaves_sandbox_just_under_stop_timeout(self):
        # One reap pass, both thresholds: a sandbox a second under the
        # stop timeout is left running, while one past it is stopped but -
        # still under the destroy timeout - kept for a later pass.
        await self._ensure(1)
        await self._ensure(2)
        self.manager._sandboxes[1].last_activity = (
            time.time() - config.SANDBOX_IDLE_STOP_TIMEOUT + 1)
        self.manager._sandboxes[2].last_activity = (
            time.time() - config.SANDBOX_IDLE_DESTROY_TIMEOUT + 1)

        stopped, destroyed = await self.manager.reap_idle()

        self.assertEqual((stopped, destroyed), ([2], []))
        self.assertEqual(self.fake.containers['toolbot-sandbox-1'], 'true')
        self.assertEqual(self.fake.containers['toolbot-sandbox-2'], 'false')
        self.assertIn(1, self.manager._sandboxes)
        self.assertIn(2, self.manager._sandboxes)

    async def test_reap_destroys_sandboxes_past_destroy_timeout(self):
        await self._ensure(1)
        self.manager._sandboxes[1].last_activity = (
            time.time() - config.SANDBOX_IDLE_DESTROY_TIMEOUT - 1)

        stopped, destroyed = await self.manager.reap_idle()

        self.assertEqual((stopped, destroyed), ([], [1]))
        self.assertNotIn('toolbot-sandbox-1', self.fake.containers)
        self.assertNotIn(1, self.manager._sandboxes)

    async def test_reap_ignores_already_stopped_sandbox(self):
        await self._ensure(1)
        sandbox = self.manager._sandboxes[1]
        self.assertTrue(await self.manager._stop_sandbox(sandbox))
        sandbox.last_activity = (
            time.time() - config.SANDBOX_IDLE_STOP_TIMEOUT - 1)

        stop_before = len(self.fake.stop_calls())
        self.assertEqual(await self.manager.reap_idle(), ([], []))
        self.assertEqual(len(self.fake.stop_calls()), stop_before)
        self.assertIn(1, self.manager._sandboxes)

    async def test_recover_adopts_stopped_leftover(self):
        now = time.time()
        self._add_leftover('toolbot-sandbox-5', 5, running=False,
                           finished=now - 100)

        adopted, removed = await self.manager.recover_leftover_sandboxes()

        self.assertEqual((adopted, removed), (1, 0))
        sandbox = self.manager._sandboxes[5]
        self.assertEqual(sandbox.container_name, 'toolbot-sandbox-5')
        self.assertFalse(sandbox.running)
        self.assertAlmostEqual(
            sandbox.last_activity,
            now - 100 - config.SANDBOX_IDLE_STOP_TIMEOUT, delta=5)

    async def test_recover_adopts_running_leftover(self):
        now = time.time()
        self._add_leftover('toolbot-sandbox-6', 6, running=True,
                           finished=now - 90000)

        adopted, removed = await self.manager.recover_leftover_sandboxes()

        self.assertEqual((adopted, removed), (1, 0))
        sandbox = self.manager._sandboxes[6]
        self.assertTrue(sandbox.running)
        # A running container is assumed to be in use right now.
        self.assertGreater(sandbox.last_activity, now - 60)

    async def test_recover_removes_containers_past_destroy_timeout(self):
        now = time.time()
        self._add_leftover('toolbot-sandbox-5', 5, running=False,
                           finished=now - 100000)

        adopted, removed = await self.manager.recover_leftover_sandboxes()

        self.assertEqual((adopted, removed), (0, 1))
        self.assertEqual(self.fake.rm_calls()[-1],
                         ['rm', '-f', 'toolbot-sandbox-5'])
        self.assertNotIn(5, self.manager._sandboxes)

    async def test_recover_removes_unidentifiable_leftovers(self):
        now = time.time()
        self._add_leftover('toolbot-sandbox-5', None, running=True,
                           finished=now)

        adopted, removed = await self.manager.recover_leftover_sandboxes()

        self.assertEqual((adopted, removed), (0, 1))
        self.assertNotIn('toolbot-sandbox-5', self.fake.containers)


if __name__ == '__main__':
    unittest.main()
