import unittest

import bootstrap  # noqa: F401 — path setup, env, logging, DB reset, next_uid
from bootstrap import next_uid as _uid
from mocks import create_mock_context
from handlers.memory_handlers import (
    memory_command,
    memory_list_command,
    memory_delete_command,
    memory_delete_all_command,
)


class TestMemoryCommands(unittest.IsolatedAsyncioTestCase):
    # Each test uses a distinct user_id so they don't collide on the
    # shared test database (wiped + recreated once per test run).

    async def test_memory_set_and_get(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="color blue", command_name="memory")
        await memory_command(ctx_set)
        self.assertIn("Remembered", ctx_set.sent[0]["text"])

        ctx_get = create_mock_context(
            user_id=uid, command_text="color", command_name="memory")
        await memory_command(ctx_get)
        self.assertEqual(len(ctx_get.sent), 1)
        self.assertEqual(ctx_get.sent[0]["text"], "<pre>blue</pre>")

    async def test_memory_get_nonexistent(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="nosuchkey", command_name="memory")
        await memory_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("remember", ctx.sent[0]["text"])

    async def test_memory_usage_without_args(self):
        for command, name in ((memory_command, "memory"),
                              (memory_delete_command, "memory_delete")):
            ctx = create_mock_context(user_id=_uid(), command_name=name)
            await command(ctx)
            self.assertEqual(len(ctx.sent), 1)
            self.assertIn("Usage", ctx.sent[0]["text"])

    async def test_memory_list_with_keys(self):
        uid = _uid()
        ctx_set1 = create_mock_context(
            user_id=uid, command_text="alpha 1", command_name="memory")
        await memory_command(ctx_set1)
        ctx_set2 = create_mock_context(
            user_id=uid, command_text="beta 2", command_name="memory")
        await memory_command(ctx_set2)
        ctx_list = create_mock_context(user_id=uid, command_name="memory_list")
        await memory_list_command(ctx_list)
        self.assertEqual(len(ctx_list.sent), 1)
        text = ctx_list.sent[0]["text"]
        self.assertIn("alpha", text)
        self.assertIn("beta", text)

    async def test_memory_delete(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="tmp val", command_name="memory")
        await memory_command(ctx_set)
        ctx_del = create_mock_context(
            user_id=uid, command_text="tmp", command_name="memory_delete")
        await memory_delete_command(ctx_del)
        self.assertEqual(len(ctx_del.sent), 1)
        self.assertIn("Forgot", ctx_del.sent[0]["text"])
        ctx_get = create_mock_context(
            user_id=uid, command_text="tmp", command_name="memory")
        await memory_command(ctx_get)
        self.assertIn("remember", ctx_get.sent[0]["text"])

        # Deleting a key that isn't there still reports success.
        ctx = create_mock_context(
            user_id=_uid(), command_text="ghost", command_name="memory_delete")
        await memory_delete_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Forgot", ctx.sent[0]["text"])

    async def test_memory_delete_all(self):
        uid = _uid()
        ctx_set1 = create_mock_context(
            user_id=uid, command_text="k1 v1", command_name="memory")
        await memory_command(ctx_set1)
        ctx_set2 = create_mock_context(
            user_id=uid, command_text="k2 v2", command_name="memory")
        await memory_command(ctx_set2)
        ctx_del = create_mock_context(
            user_id=uid, command_name="memory_delete_all")
        await memory_delete_all_command(ctx_del)
        self.assertEqual(len(ctx_del.sent), 1)
        self.assertIn("forgotten", ctx_del.sent[0]["text"])
        ctx_list = create_mock_context(
            user_id=uid, command_name="memory_list")
        await memory_list_command(ctx_list)
        self.assertIn("remember", ctx_list.sent[0]["text"])

        # An empty memory still reports success.
        ctx = create_mock_context(
            user_id=uid, command_name="memory_delete_all")
        await memory_delete_all_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("forgotten", ctx.sent[0]["text"])


if __name__ == "__main__":
    unittest.main()
