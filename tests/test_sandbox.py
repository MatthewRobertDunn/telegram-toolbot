import asyncio
import shlex
import sys
import os
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
import sandbox
from sandbox import (
    _EXEC_GRACE_SECONDS,
    _STORE_FILE_TIMEOUT,
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
    left over from a previous bot run. exec() behaviours, all keyed on the
    (deadline-wrapped) command text:

    - 'TIMEOUT_MARKER' raises DockerTimeoutError: the client-side backstop
      for a docker CLI that hung past deadline plus grace.
    - 'SLOW_MARKER' returns exit code 124 with partial output: the
      container-side timeout(1) firing at the deadline.
    - exec_responses, when set, supply the first (substring, rc, stdout,
      stderr) tuple whose substring matches the command.
    - exec_delay sleeps before answering, so lock behaviour can be
      observed; exec_stderr, when set, is returned as a failure instead.
    """

    def __init__(self):
        self.calls: list[list[str]] = []
        self.timeouts: list[float | None] = []
        self.inputs: list[bytes | None] = []
        self.containers: dict[str, str] = {}
        self.finished_at: dict[str, str] = {}
        self.labels: dict[str, str] = {}
        self.fail_run = False
        self.fail_start = False
        self.fail_stop = False
        self.exec_delay = 0.0
        self.exec_stderr: str | None = None
        self.exec_responses: list[tuple[str, int, str, str]] = []

    async def __call__(self, args: list[str], timeout=None, input=None):
        self.calls.append(list(args))
        self.timeouts.append(timeout)
        self.inputs.append(input)
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
            if self.fail_stop:
                return 1, '', 'simulated docker stop failure'
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
            if self.exec_delay:
                await asyncio.sleep(self.exec_delay)
            if self.exec_stderr is not None:
                return 126, '', self.exec_stderr
            if 'SLOW_MARKER' in command:
                # The container-side timeout fired: exit 124 with whatever
                # partial output the workload had produced.
                return sandbox._TIMEOUT_EXIT_CODE, 'partial-out\n', 'partial-err\n'
            if 'KILL_MARKER' in command:
                # A TERM-ignoring command: timeout's -k escalation kills it.
                return sandbox._KILL_EXIT_CODE, '', ''
            for substring, rc, stdout, stderr in self.exec_responses:
                if substring in command:
                    return rc, stdout, stderr
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
        # configured resource limits, the reaping init process, and the
        # keep-alive command.
        run = self.fake.run_calls()[0]
        self.assertIn('toolbot.sandbox=1', run)
        self.assertIn('--init', run)
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
        with self.assertRaises(SandboxStartError) as caught:
            await self._ensure(1)
        self.assertNotIn(1, self.manager._sandboxes)
        # The raw docker stderr goes to the log; the message stays classified.
        self.assertNotIn('simulated docker run failure',
                         str(caught.exception))
        self.assertIn('Could not start the sandbox container',
                      str(caught.exception))

    async def test_exec_returns_output_and_updates_timer(self):
        sandbox = await self._ensure(1)
        sandbox.last_activity = time.time() - 1000
        result = await self.manager.exec(1, 'echo hi')
        # The fake echoes the deadline-wrapped command back; the marker
        # text proves the plain command reached the container.
        self.assertIn('echo hi', result['stdout'])
        self.assertEqual(result['returncode'], 0)
        self.assertFalse(result['timeout'])
        self.assertGreater(sandbox.last_activity, time.time() - 60)
        self.assertEqual(len(self.fake.exec_calls()), 1)
        self.assertIn('echo hi', self.fake.exec_calls()[0][-1])

    async def test_exec_deadline_runs_inside_the_container(self):
        await self.manager.exec(1, 'echo hi', timeout=45)
        wrapped = self.fake.exec_calls()[0][-1]
        # The container's timeout(1) owns the deadline and kills the whole
        # command's process group at it.
        self.assertIn('timeout -k 5 45 bash -c ', wrapped)
        self.assertIn("'echo hi'", wrapped)
        # The client-side wait is only a backstop past that deadline.
        self.assertEqual(self.fake.timeouts[-1],
                         45 + _EXEC_GRACE_SECONDS)

    async def test_exec_container_timeout_preserves_partial_output(self):
        # When the in-container timeout fires, the command exits 124 and
        # whatever it had produced survives.
        result = await self.manager.exec(1, 'sleep SLOW_MARKER', timeout=10)
        self.assertTrue(result['timeout'])
        self.assertEqual(result['returncode'], sandbox._TIMEOUT_EXIT_CODE)
        self.assertEqual(result['stdout'], 'partial-out\n')
        self.assertEqual(result['stderr'], 'partial-err\n')

    async def test_exec_kill_escalation_counts_as_timeout(self):
        # A command that ignores SIGTERM dies by the -k escalation; that
        # is still a deadline hit as far as the caller is concerned.
        result = await self.manager.exec(1, 'sleep KILL_MARKER', timeout=10)
        self.assertTrue(result['timeout'])
        self.assertEqual(result['returncode'], sandbox._KILL_EXIT_CODE)

    async def test_exec_backstop_reports_lost_output(self):
        # If the docker CLI itself hangs past deadline plus grace, the
        # client-side backstop kills it and the output is gone with it.
        result = await self.manager.exec(1, 'sleep TIMEOUT_MARKER', timeout=10)
        self.assertTrue(result['timeout'])
        self.assertIsNone(result['returncode'])
        self.assertEqual(result['stdout'], '')
        self.assertIn('timed out', result['stderr'])
        self.assertEqual(self.fake.timeouts[-1],
                         10 + _EXEC_GRACE_SECONDS)

    async def test_exec_translates_gone_container(self):
        # A container removed mid-command (/sandbox_reset, the reaper)
        # makes docker exec fail with daemon noise; the model sees the
        # plain message instead.
        await self._ensure(1)
        self.fake.exec_stderr = (
            'Error response from daemon: '
            'Container toolbot-sandbox-1 is not running')
        result = await self.manager.exec(1, 'echo hi')
        self.assertIsNone(result['returncode'])
        self.assertEqual(result['stdout'], '')
        self.assertIn('was reset or stopped', result['stderr'])
        self.assertFalse(result['timeout'])

    async def test_exec_passes_command_failures_through(self):
        # A failing command's own stderr is untouched: only daemon noise
        # about a vanished container is reclassified.
        self.fake.exec_responses = [('ls /nope', 1, '',
                                     'ls: /nope: No such file')]
        result = await self.manager.exec(1, 'ls /nope')
        self.assertEqual(result['returncode'], 1)
        self.assertEqual(result['stderr'], 'ls: /nope: No such file')
        self.assertFalse(result['timeout'])

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
        self.assertTrue(await self.manager._stop_sandbox(1, sandbox))
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

    async def test_stop_failure_keeps_sandbox_running_for_retry(self):
        # A stop that fails must not be reported as stopped: the sandbox is
        # still there, so running stays True and reap tries again next pass.
        await self._ensure(1)
        sandbox = self.manager._sandboxes[1]
        self.fake.fail_stop = True

        self.assertFalse(await self.manager._stop_sandbox(1, sandbox))
        self.assertTrue(sandbox.running)
        self.assertEqual(self.fake.containers['toolbot-sandbox-1'], 'true')

        sandbox.last_activity = (
            time.time() - config.SANDBOX_IDLE_STOP_TIMEOUT - 1)
        self.fake.fail_stop = False
        self.assertEqual(await self.manager.reap_idle(), ([1], []))

    async def test_reap_does_not_count_a_failed_stop(self):
        await self._ensure(1)
        sandbox = self.manager._sandboxes[1]
        sandbox.last_activity = (
            time.time() - config.SANDBOX_IDLE_STOP_TIMEOUT - 1)
        self.fake.fail_stop = True

        self.assertEqual(await self.manager.reap_idle(), ([], []))
        self.assertIn(1, self.manager._sandboxes)

    async def test_destroy_waits_for_a_running_exec(self):
        # The lock spans the whole operation, so a destroy issued while a
        # command runs cannot remove the container out from under it: the
        # destroy lands after the command's docker call, as the last call.
        self.fake.exec_delay = 0.05
        exec_task = asyncio.create_task(self.manager.exec(1, 'echo hi'))
        await asyncio.sleep(0.01)
        await self.manager.destroy(1)

        result = await exec_task
        self.assertEqual(result['returncode'], 0)
        self.assertEqual(self.fake.calls[-1],
                         ['rm', '-f', 'toolbot-sandbox-1'])
        self.assertNotIn(1, self.manager._sandboxes)

    async def test_read_marks_are_cleared_on_destroy(self):
        class Holder:
            def __init__(self):
                self.sandbox_read_files = {'/workspace/a.txt'}

        holder = Holder()
        other = Holder()
        self.manager.watch_read_marks(1, holder)
        self.manager.watch_read_marks(2, other)
        await self._ensure(1)
        # Registering the same object twice does not double-track it.
        self.manager.watch_read_marks(1, holder)
        self.assertEqual(len(self.manager._read_marks[1]), 1)

        self.assertTrue(await self.manager.destroy(1))

        self.assertEqual(holder.sandbox_read_files, set())
        self.assertEqual(other.sandbox_read_files, {'/workspace/a.txt'})
        self.assertNotIn(1, self.manager._read_marks)

    async def test_read_marks_holder_is_not_kept_alive(self):
        class Holder:
            def __init__(self):
                self.sandbox_read_files = set()

        self.manager.watch_read_marks(1, Holder())
        import gc
        gc.collect()
        # Nothing is resurrected: every tracked reference is dead, so the
        # registry held no strong reference to the throwaway object.
        refs = self.manager._read_marks[1]
        self.assertTrue(refs)
        self.assertTrue(all(ref() is None for ref in refs))

    async def test_store_file_writes_once_and_returns_path(self):
        self.fake.exec_responses = [
            ('cat >', 0, '/workspace/uploads/note.txt\n', '')]
        path, error = await self.manager.store_file(1, b'hello', 'note.txt')
        self.assertIsNone(error)
        self.assertEqual(path, '/workspace/uploads/note.txt')
        # One roundtrip: mkdir, name pick and write in a single exec, under
        # a bounded timeout (not the 300s command default).
        self.assertEqual(len(self.fake.exec_calls()), 1)
        self.assertEqual(self.fake.timeouts[-1],
                         _STORE_FILE_TIMEOUT + _EXEC_GRACE_SECONDS)
        self.assertEqual(self.fake.inputs[-1], b'hello')
        self.assertEqual(self.fake.exec_calls()[0][0], 'exec')
        self.assertIn('-i', self.fake.exec_calls()[0])

    async def test_store_file_script_is_atomic(self):
        # The script picks a free name itself, so two uploads racing cannot
        # both write to note.txt.
        await self.manager.store_file(1, b'hello', 'note.txt')
        script = self.fake.exec_calls()[-1][-1]
        self.assertIn('mkdir -p', script)
        self.assertIn('cat > "$p"', script)
        self.assertIn('/workspace/uploads', script)
        # The exact script the manager built is what got wrapped (the
        # printf's single quotes are shlex-escaped inside the wrapper).
        self.assertIn(shlex.quote(sandbox._store_script('note.txt')), script)

    async def test_store_file_reports_write_failure(self):
        self.fake.exec_responses = [
            ('cat >', 1, '', 'bash: cannot create file')]
        path, error = await self.manager.store_file(1, b'x', 'note.txt')
        self.assertIsNone(path)
        self.assertIn('bash: cannot create file', error or '')

    async def test_store_file_reports_timeout(self):
        # The backstop marker rides in on the file name: the wrapped script
        # contains it, so the fake raises DockerTimeoutError.
        path, error = await self.manager.store_file(
            1, b'x', 'TIMEOUT_MARKER.txt')
        self.assertIsNone(path)
        self.assertIn('timed out', error or '')

    async def test_store_file_reports_empty_path(self):
        self.fake.exec_responses = [('cat >', 0, '  \n', '')]
        path, error = await self.manager.store_file(1, b'x', 'note.txt')
        self.assertIsNone(path)
        self.assertIn('did not report the stored path', error or '')

    async def test_recover_removes_unidentifiable_leftovers(self):
        now = time.time()
        self._add_leftover('toolbot-sandbox-5', None, running=True,
                           finished=now)

        adopted, removed = await self.manager.recover_leftover_sandboxes()

        self.assertEqual((adopted, removed), (0, 1))
        self.assertNotIn('toolbot-sandbox-5', self.fake.containers)


if __name__ == '__main__':
    unittest.main()
