import unittest

import bootstrap  # noqa: F401 — path setup, env, logging, DB reset
from bootstrap import next_uid, requires_docker

from sandbox import SandboxManager


@requires_docker
class TestSandboxIntegration(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.manager = SandboxManager()
        self.uid = next_uid()
        await self.manager.destroy(self.uid)

    async def asyncTearDown(self):
        await self.manager.destroy(self.uid)

    async def test_exec_runs_and_state_persists_between_commands(self):
        result = await self.manager.exec(self.uid, "echo hello-sandbox")
        self.assertEqual(result["returncode"], 0)
        self.assertFalse(result["timeout"])
        self.assertIn("hello-sandbox", result["stdout"])

        await self.manager.exec(
            self.uid, "echo persistence-check > /tmp/persist.txt")
        result = await self.manager.exec(
            self.uid, "cat /tmp/persist.txt")
        self.assertIn("persistence-check", result["stdout"])

    async def test_state_survives_idle_stop_and_restart(self):
        await self.manager.exec(
            self.uid, "echo keep-me > /tmp/keep.txt")
        sandbox = self.manager._sandboxes[self.uid]
        self.assertTrue(await self.manager._stop_sandbox(self.uid, sandbox))
        result = await self.manager.exec(self.uid, "cat /tmp/keep.txt")
        self.assertEqual(result["returncode"], 0)
        self.assertIn("keep-me", result["stdout"])

    async def test_command_deadline_enforced_in_container(self):
        # The in-container timeout kills the command at the deadline; the
        # result reports the timeout without losing partial output.
        result = await self.manager.exec(
            self.uid, "echo partial; sleep 30", timeout=3)
        self.assertTrue(result["timeout"])
        self.assertIn("partial", result["stdout"])
        # The sandbox is still usable right after.
        follow = await self.manager.exec(self.uid, "echo alive")
        self.assertEqual(follow["returncode"], 0)
        self.assertIn("alive", follow["stdout"])

    async def test_software_can_be_installed(self):
        install = await self.manager.exec(
            self.uid,
            "apt-get update -qq && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends bc",
            timeout=240)
        self.assertEqual(install["returncode"], 0, install["stderr"])
        run = await self.manager.exec(self.uid, "echo '1+1' | bc")
        self.assertEqual(run["returncode"], 0)
        self.assertIn("2", run["stdout"])


if __name__ == "__main__":
    unittest.main()
