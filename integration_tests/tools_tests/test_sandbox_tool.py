import base64
import os
import struct
import subprocess
import sys
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # noqa: F401 — path setup, env, logging, DB reset
from bootstrap import next_uid
from mocks import create_mock_context

import config
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


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, timeout=15)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _image_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", config.SANDBOX_IMAGE],
            capture_output=True, timeout=15)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


DOCKER_AVAILABLE = _docker_available()
IMAGE_AVAILABLE = _image_available()


@unittest.skipUnless(DOCKER_AVAILABLE, "Docker daemon not available")
@unittest.skipUnless(IMAGE_AVAILABLE,
                     f"sandbox image {config.SANDBOX_IMAGE} not built locally")
class TestSandboxToolsIntegration(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.uid = next_uid()
        import sandbox
        self.manager = sandbox.manager
        await self.manager.destroy_sandbox(self.uid)

    async def asyncTearDown(self):
        await self.manager.destroy_sandbox(self.uid)

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

    async def test_edit_requires_read_first(self):
        ctx = await self._ctx()
        await sandbox_tool.write(ctx, "their.txt", "data\n")
        # a fresh context simulates a new conversation with no read history,
        # but the same sandbox, where the file now exists
        fresh = create_mock_context(user_id=self.uid)
        result = await sandbox_tool.edit(fresh, "their.txt", "data", "x")
        self.assertIn("has not been read", result)

    async def test_edit_not_found_error(self):
        ctx = await self._ctx()
        await sandbox_tool.write(ctx, "app.py", "one\n")
        result = await sandbox_tool.edit(ctx, "app.py", "nonexistent", "x")
        self.assertIn("Could not find oldString", result)

    async def test_edit_multiple_matches_error(self):
        ctx = await self._ctx()
        await sandbox_tool.write(ctx, "multi.txt", "x = 1\ny = 2\nz = 3\nx = 4\n")
        result = await sandbox_tool.edit(ctx, "multi.txt", "x", "q")
        self.assertIn("Found multiple matches", result)

    async def test_grep_finds_matches(self):
        ctx = await self._ctx()
        await sandbox_tool.write(ctx, "a.txt", "the needle is here\n")
        await sandbox_tool.write(ctx, "b.txt", "nothing relevant\n")
        result = await sandbox_tool.grep(ctx, "needle")
        self.assertIn("Found 1 matches", result)
        self.assertIn("the needle is here", result)

    async def test_grep_include_filter(self):
        ctx = await self._ctx()
        await sandbox_tool.write(ctx, "code.py", "target = 1\n")
        await sandbox_tool.write(ctx, "notes.txt", "target = 2\n")
        result = await sandbox_tool.grep(ctx, "target", include="*.py")
        self.assertIn("Found 1 matches", result)
        self.assertIn("code.py", result)

    async def test_glob_finds_files(self):
        ctx = await self._ctx()
        await sandbox_tool.write(ctx, "src/pkg/mod.py", "x = 1\n")
        await sandbox_tool.write(ctx, "src/pkg/util.py", "y = 2\n")
        await sandbox_tool.write(ctx, "src/readme.md", "# doc\n")
        result = await sandbox_tool.glob(ctx, "src/**/*.py")
        self.assertIn("/workspace/src/pkg/mod.py", result)
        self.assertIn("/workspace/src/pkg/util.py", result)
        self.assertNotIn("readme.md", result)

    async def test_apply_patch_add_update_delete(self):
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
        result = await sandbox_tool.apply_patch(ctx, patch)
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

    async def test_apply_patch_rejects_bad_patch(self):
        result = await sandbox_tool.apply_patch(await self._ctx(), "not a patch")
        self.assertTrue(result.startswith("Error:"), result)

    async def test_send_file_delivers_document(self):
        ctx = await self._ctx()
        await sandbox_tool.bash(ctx, "printf 'report body' > report.txt")
        result = await sandbox_tool.send_file(ctx, "report.txt")
        self.assertIn("Sent report.txt", result)
        sent = ctx.bot.sent[-1]
        self.assertEqual(sent["kind"], "document")
        self.assertEqual(sent["document"], b"report body")
        self.assertEqual(sent["filename"], "report.txt")

    async def test_send_file_from_subdirectory(self):
        ctx = await self._ctx()
        await sandbox_tool.bash(ctx, "mkdir -p out && printf 'data' > out/blob.bin")
        result = await sandbox_tool.send_file(ctx, "out/blob.bin")
        self.assertIn("Sent blob.bin", result)
        sent = ctx.bot.sent[-1]
        self.assertEqual(sent["document"], b"data")

    async def test_send_file_missing(self):
        result = await sandbox_tool.send_file(await self._ctx(), "nope.txt")
        self.assertIn("File not found", result)

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

    async def test_read_image_missing(self):
        result = await sandbox_tool.read_image(await self._ctx(), "nope.png")
        assert isinstance(result, str)
        self.assertIn("File not found", result)

    async def test_read_image_unsupported_type(self):
        ctx = await self._ctx()
        await sandbox_tool.bash(ctx, "echo data > /workspace/file.txt")
        result = await sandbox_tool.read_image(ctx, "file.txt")
        assert isinstance(result, str)
        self.assertIn("not a supported image type", result)

    async def test_binary_read_rejected(self):
        ctx = await self._ctx()
        await sandbox_tool.bash(
            ctx, "printf 'PK\\x03\\x04' > binary.bin; head -c 100 /dev/zero >> binary.bin")
        result = await sandbox_tool.read(ctx, "binary.bin")
        self.assertIn("binary", result)


if __name__ == "__main__":
    unittest.main()
