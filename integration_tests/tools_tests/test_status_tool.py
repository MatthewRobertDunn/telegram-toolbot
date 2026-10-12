import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # noqa: F401
from mocks import create_mock_context

from tools.status_tool import display_status


class TestStatusTool(unittest.IsolatedAsyncioTestCase):

    async def test_display_status_sends_message(self):
        ctx = create_mock_context()
        result = await display_status(ctx, "Working on it...")
        self.assertEqual(result, "Status displayed.")
        self.assertEqual(ctx.bot.sent[-1]["text"], "Working on it...")


if __name__ == "__main__":
    unittest.main()
