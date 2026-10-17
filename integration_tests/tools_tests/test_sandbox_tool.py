import base64
import os
import struct
import sys
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # noqa: F401 — path setup, env, logging, DB reset
from bootstrap import next_uid, requires_sandbox_image
from mocks import create_mock_context

from tools import sandbox_tool
from tool_result import ToolCallResult


def _make_png() -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        block = struct.pack(">I", len(data)) + tag + data
        return block + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw = b"\x00\xff\x00\x00"
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


@requires_sandbox_image
class TestSandboxToolsIntegration(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.uid = next_uid()
        import sandbox
        self.manager = sandbox.manager
        await self.manager.destroy(self.uid)

    async def asyncTearDown(self):
        await self.manager.destroy(self.uid)

    async def _ctx(self):
        return create_mock_context(user_id=self.uid)

    async def test_bash_runs_command(self):
        result = await sandbox_tool.bash(await self._ctx(), "echo hello-tools")
        self.assertIn("Return code: 0", result)
        self.assertIn("hello-tools", result)

    async def test_write_read_round_trip(self):
        ctx = await self._ctx()
        written = await sandbox_tool.write(ctx, "src/hello.txt", "alpha\nbeta\ngamma\n")
        self.assertEqual(written, "Wrote file successfully.")
        read = await sandbox_tool.read(ctx, "src/hello.txt")
        self.assertIn("1: alpha", read)
        self.assertIn("3: gamma", read)
        self.assertIn("End of file - total 3 lines", read)

    async def test_read_offset_and_limit(self):
        ctx = await self._ctx()
        await sandbox_tool.write(
            ctx, "nums.txt", "\n".join(str(i) for i in range(1, 21)) + "\n")
        read = await sandbox_tool.read(ctx, "nums.txt", offset=5, limit=3)
        self.assertIn("5: 5", read)
        self.assertIn("7: 7", read)
        self.assertIn("Showing lines 5-7 of 20", read)

    async def test_edit_exact_replacement(self):
        ctx = await self._ctx()
        await sandbox_tool.write(ctx, "app.py", "def greet():\n    print('Hi')\n")
        edited = await sandbox_tool.edit(
            ctx, "app.py", "print('Hi')", "print('Hello, world!')")
        self.assertEqual(edited, "Edit applied successfully.")
        read = await sandbox_tool.read(ctx, "app.py")
        self.assertIn("print('Hello, world!')", read)

    async def test_grep_finds_matches(self):
        ctx = await self._ctx()
        await sandbox_tool.write(ctx, "a.txt", "the needle is here\n")
        await sandbox_tool.write(ctx, "b.txt", "nothing relevant\n")
        await sandbox_tool.write(ctx, "code.py", "another needle goes here\n")
        result = await sandbox_tool.grep(ctx, "needle")
        self.assertIn("Found 2 matches", result)
        self.assertIn("the needle is here", result)
        # The include filter restricts the search to matching names.
        result = await sandbox_tool.grep(ctx, "needle", include="*.txt")
        self.assertIn("Found 1 matches", result)
        self.assertIn("a.txt", result)
        self.assertNotIn("code.py", result)

    async def test_glob_finds_files(self):
        ctx = await self._ctx()
        await sandbox_tool.write(ctx, "src/pkg/mod.py", "x = 1\n")
        await sandbox_tool.write(ctx, "src/pkg/util.py", "y = 2\n")
        await sandbox_tool.write(ctx, "src/readme.md", "# doc\n")
        result = await sandbox_tool.glob(ctx, "src/**/*.py")
        self.assertIn("/workspace/src/pkg/mod.py", result)
        self.assertIn("/workspace/src/pkg/util.py", result)
        self.assertNotIn("readme.md", result)

    async def test_patch_add_update_delete(self):
        ctx = await self._ctx()
        await sandbox_tool.write(ctx, "src/app.py", "def greet():\n    print('Hi')\n")
        await sandbox_tool.write(ctx, "obsolete.txt", "delete me\n")
        patch = (
            "*** Begin Patch\n"
            "*** Add File: hello.txt\n"
            "+Hello world\n"
            "*** Update File: src/app.py\n"
            "@@ def greet():\n"
            "-print('Hi')\n"
            "+print('Hello, world!')\n"
            "*** Delete File: obsolete.txt\n"
            "*** End Patch\n")
        result = await sandbox_tool.patch(ctx, patch)
        self.assertIn("Success. Updated the following files:", result)
        self.assertIn("A hello.txt", result)
        self.assertIn("M src/app.py", result)
        self.assertIn("D obsolete.txt", result)
        read = await sandbox_tool.read(ctx, "src/app.py")
        self.assertIn("print('Hello, world!')", read)
        hello = await sandbox_tool.read(ctx, "hello.txt")
        self.assertIn("Hello world", hello)
        gone = await sandbox_tool.read(ctx, "obsolete.txt")
        self.assertIn("File not found", gone)

    async def test_send_file_delivers_document(self):
        ctx = await self._ctx()
        await sandbox_tool.bash(ctx, "printf 'report body' > report.txt")
        result = await sandbox_tool.send_file(ctx, "report.txt")
        self.assertIn("Sent report.txt", result)
        sent = ctx.bot.sent[-1]
        self.assertEqual(sent["kind"], "document")
        self.assertEqual(sent["document"], b"report body")
        self.assertEqual(sent["filename"], "report.txt")

    async def test_read_image_returns_image_payload(self):
        ctx = await self._ctx()
        png_b64 = base64.b64encode(_make_png()).decode()
        await sandbox_tool.bash(
            ctx, f"echo {png_b64} | base64 -d > /workspace/tiny.png")
        result = await sandbox_tool.read_image(ctx, "tiny.png")
        assert isinstance(result, ToolCallResult)
        assert result.image is not None
        self.assertEqual(result.image.mime, "image/png")
        self.assertTrue(result.image.data.startswith(b"\x89PNG"))
        sent = ctx.bot.sent[-1]
        self.assertEqual(sent["kind"], "photo")


if __name__ == "__main__":
    unittest.main()
