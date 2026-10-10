import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # noqa: F401
from mocks import create_mock_context

from tools.image_gen import generate_and_send_image


class TestImageGenTool(unittest.IsolatedAsyncioTestCase):

    async def test_generates_and_sends_photo(self):
        ctx = create_mock_context()
        result = await generate_and_send_image(
            ctx, "a single red circle on a white background")
        self.assertIn("Image sent to chat", result)
        photos = [m for m in ctx.bot.sent if m["kind"] == "photo"]
        self.assertTrue(photos, "no photo was sent")


if __name__ == "__main__":
    unittest.main()
