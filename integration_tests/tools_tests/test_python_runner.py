import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # noqa: F401
from bootstrap import next_uid, requires_sandbox_image
from mocks import create_mock_context

from sandbox import manager
from tools.python_runner import run_python_code


@requires_sandbox_image
class TestPythonRunner(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.uid = next_uid()
        await manager.destroy(self.uid)

    async def asyncTearDown(self):
        await manager.destroy(self.uid)

    async def _ctx(self):
        return create_mock_context(user_id=self.uid)

    async def test_runs_code_and_returns_stdout(self):
        result = await run_python_code(await self._ctx(), "print(2 + 2)")
        self.assertIn("Return code: 0", result)
        self.assertIn("stdout:", result)
        self.assertIn("4", result)

    async def test_nonzero_return_code(self):
        # An uncaught exception yields a nonzero code, and output written
        # to stderr comes back in the stderr section.
        result = await run_python_code(
            await self._ctx(), "raise ValueError('boom')")
        self.assertIn("Return code: 1", result)
        self.assertIn("boom", result)
        self.assertIn("stderr:", result)

    async def test_runs_in_workspace_and_persists_state(self):
        ctx = await self._ctx()
        result = await run_python_code(
            ctx, "open('note.txt', 'w').write('hello')")
        self.assertIn("Return code: 0", result)
        result = await run_python_code(
            ctx, "print(open('/workspace/note.txt').read())")
        self.assertIn("hello", result)
        result = await run_python_code(
            ctx, "import os; print(os.getcwd())")
        self.assertIn("/workspace", result)
        # Running code counts as sandbox activity, so it keeps the
        # container alive.
        self.assertGreater(
            manager._sandboxes[self.uid].last_activity, time.time() - 60)


if __name__ == "__main__":
    unittest.main()
