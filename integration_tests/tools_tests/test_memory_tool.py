import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # noqa: F401
from bootstrap import next_uid
from mocks import create_mock_context

from tools.memory_tool import (
    memory_delete_values,
    memory_get_keys,
    memory_get_values,
    memory_set_value,
)


class TestMemoryTool(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.ctx = create_mock_context(user_id=next_uid())

    async def test_set_get_round_trip(self):
        result = await memory_set_value(self.ctx, "color", "blue")
        self.assertEqual(result, "Remembered 'color'.")
        result = await memory_get_values(self.ctx, ["color"])
        self.assertEqual(json.loads(result), {"color": "blue"})

    async def test_get_missing_key_returns_null(self):
        result = await memory_get_values(self.ctx, ["missing"])
        self.assertEqual(json.loads(result), {"missing": None})

    async def test_list_keys(self):
        await memory_set_value(self.ctx, "a", "1")
        await memory_set_value(self.ctx, "b", "2")
        result = await memory_get_keys(self.ctx)
        self.assertEqual(sorted(json.loads(result)), ["a", "b"])

    async def test_delete_keys(self):
        await memory_set_value(self.ctx, "a", "1")
        result = await memory_delete_values(self.ctx, ["a", "gone"])
        self.assertIn("Forgot", result)
        result = await memory_get_keys(self.ctx)
        self.assertEqual(result, "I don't remember anything yet.")


if __name__ == "__main__":
    unittest.main()
