import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # noqa: F401
from bootstrap import requires_sandbox_image
from mocks import create_mock_context

from tools.image_gen import generate_and_send_image
from tool_result import ToolCallResult


@requires_sandbox_image
class TestImageGenTool(unittest.IsolatedAsyncioTestCase):

    async def test_generates_and_sends_photo(self):
        ctx = create_mock_context()
        result = await generate_and_send_image(
            ctx, "a single red circle on a white background")
        assert isinstance(result, str)
        self.assertIn("Image sent to chat", result)
        photos = [m for m in ctx.bot.sent if m["kind"] == "photo"]
        self.assertTrue(photos, "no photo was sent")

    async def test_add_to_context(self):
        ctx = create_mock_context()
        result = await generate_and_send_image(
            ctx, "a single green square on a white background",
            add_to_context=True)
        assert isinstance(result, ToolCallResult)
        assert result.image is not None
        self.assertEqual(result.image.mime, "image/png")
        self.assertTrue(result.image.data)


if __name__ == "__main__":
    unittest.main()
