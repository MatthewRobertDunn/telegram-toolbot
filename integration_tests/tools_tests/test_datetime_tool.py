import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # noqa: F401

from tools.datetime_tool import get_current_datetime


class TestDatetimeTool(unittest.IsolatedAsyncioTestCase):

    async def test_returns_formatted_utc_datetime(self):
        result = await get_current_datetime()
        self.assertRegex(result,
                         r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC \(\w+\)")

    async def test_includes_current_year(self):
        from datetime import datetime, timezone
        year = datetime.now(timezone.utc).year
        result = await get_current_datetime()
        self.assertIn(str(year), result)


if __name__ == "__main__":
    unittest.main()
