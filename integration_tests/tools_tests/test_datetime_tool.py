import os
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # noqa: F401

from tools.datetime_tool import get_current_datetime


class TestDatetimeTool(unittest.IsolatedAsyncioTestCase):

    def _ctx_with_timezone(self, tz_name: str):
        from unittest.mock import MagicMock
        from user_settings import UserSettings
        settings = UserSettings.default()
        settings.timezone = tz_name
        ctx = MagicMock()
        ctx.user_data.settings = settings
        return ctx

    async def test_returns_iso_timestamp_in_user_timezone(self):
        result = await get_current_datetime(
            self._ctx_with_timezone("Australia/Sydney"))
        self.assertRegex(
            result,
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*[+-]\d{2}:\d{2} "
            r"\(\w+, Australia/Sydney\)")
        self.assertIn(str(datetime.now(timezone.utc).year), result)

    async def test_defaults_to_utc(self):
        result = await get_current_datetime(
            self._ctx_with_timezone("UTC"))
        self.assertIn("+00:00", result)
        self.assertIn("UTC", result)


if __name__ == "__main__":
    unittest.main()
