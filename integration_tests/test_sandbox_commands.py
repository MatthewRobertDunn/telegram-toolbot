import unittest

import bootstrap  # noqa: F401 — path setup, env, logging, DB reset
from bootstrap import next_uid, requires_docker
from mocks import create_mock_context

from handlers.sandbox_handlers import sandbox_reset_command
from sandbox import manager


@requires_docker
class TestSandboxResetCommand(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.uid = next_uid()
        await manager.destroy(self.uid)

    async def asyncTearDown(self):
        await manager.destroy(self.uid)

    async def test_sandbox_reset_destroys_the_sandbox(self):
        await manager.exec(self.uid, "echo doomed > /tmp/doomed.txt")
        ctx = create_mock_context(user_id=self.uid, command_name="sandbox_reset")

        await sandbox_reset_command(ctx)

        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Sandbox destroyed", ctx.sent[0]["text"])
        # The next use gets a fresh sandbox without the old files.
        result = await manager.exec(self.uid, "cat /tmp/doomed.txt")
        self.assertNotEqual(result["returncode"], 0)

    async def test_sandbox_reset_without_sandbox(self):
        ctx = create_mock_context(user_id=self.uid, command_name="sandbox_reset")

        await sandbox_reset_command(ctx)

        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Sandbox destroyed", ctx.sent[0]["text"])


if __name__ == "__main__":
    unittest.main()
