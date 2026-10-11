import unittest

import bootstrap  # noqa: F401 — path setup, env, logging, DB reset
from bootstrap import next_uid, requires_docker

from sandbox import SandboxManager


@requires_docker
class TestSandboxIntegration(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.manager = SandboxManager()
        self.uid = next_uid()
        await self.manager.destroy_sandbox(self.uid)

    async def asyncTearDown(self):
        await self.manager.destroy_sandbox(self.uid)

    async def test_exec_runs_command_and_output_returned(self):
        result = await self.manager.exec_in_sandbox(self.uid, "echo hello-sandbox")
        self.assertEqual(result["returncode"], 0)
        self.assertFalse(result["timeout"])
        self.assertIn("hello-sandbox", result["stdout"])

    async def test_state_persists_between_commands(self):
        await self.manager.exec_in_sandbox(
            self.uid, "echo persistence-check > /tmp/persist.txt")
        result = await self.manager.exec_in_sandbox(
            self.uid, "cat /tmp/persist.txt")
        self.assertIn("persistence-check", result["stdout"])

    async def test_software_can_be_installed(self):
        install = await self.manager.exec_in_sandbox(
            self.uid,
            "apt-get update -qq && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends bc",
            timeout=240)
        self.assertEqual(install["returncode"], 0, install["stderr"])
        run = await self.manager.exec_in_sandbox(self.uid, "echo '1+1' | bc")
        self.assertEqual(run["returncode"], 0)
        self.assertIn("2", run["stdout"])

    async def test_destroy_loses_changes(self):
        await self.manager.exec_in_sandbox(
            self.uid, "echo doomed > /tmp/doomed.txt")
        self.assertTrue(await self.manager.destroy_sandbox(self.uid))
        result = await self.manager.exec_in_sandbox(
            self.uid, "cat /tmp/doomed.txt")
        self.assertNotEqual(result["returncode"], 0)
        self.assertIn("No such file", result["stderr"])

if __name__ == "__main__":
    unittest.main()
