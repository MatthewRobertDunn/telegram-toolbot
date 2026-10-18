import unittest

import bootstrap  # noqa: F401 — path setup, env, logging, DB reset, next_uid
from bootstrap import next_uid as _uid
from mocks import create_mock_context
from handlers.store_handlers import (
    store_command,
    store_list_command,
    store_delete_command,
    store_delete_all_command,
)


class TestStoreCommands(unittest.IsolatedAsyncioTestCase):
    # Each test uses a distinct user_id so they don't collide on the
    # shared test database (wiped + recreated once per test run).

    # -- store_command: set + get ----------------------------------------

    async def test_store_set_and_get(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="color blue", command_name="store")
        await store_command(ctx_set)
        self.assertIn("Stored", ctx_set.sent[0]["text"])

        ctx_get = create_mock_context(
            user_id=uid, command_text="color", command_name="store")
        await store_command(ctx_get)
        self.assertEqual(len(ctx_get.sent), 1)
        self.assertEqual(ctx_get.sent[0]["text"], "<pre>blue</pre>")

    async def test_store_get_nonexistent(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="nosuchkey", command_name="store")
        await store_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("not found", ctx.sent[0]["text"])

    async def test_store_usage_without_args(self):
        for command, name in ((store_command, "store"),
                              (store_delete_command, "store_delete")):
            ctx = create_mock_context(user_id=_uid(), command_name=name)
            await command(ctx)
            self.assertEqual(len(ctx.sent), 1)
            self.assertIn("Usage", ctx.sent[0]["text"])

    # -- store_list_command ----------------------------------------------

    async def test_store_list_with_keys(self):
        uid = _uid()
        ctx_set1 = create_mock_context(
            user_id=uid, command_text="alpha 1", command_name="store")
        await store_command(ctx_set1)
        ctx_set2 = create_mock_context(
            user_id=uid, command_text="beta 2", command_name="store")
        await store_command(ctx_set2)
        ctx_list = create_mock_context(user_id=uid, command_name="store_list")
        await store_list_command(ctx_list)
        self.assertEqual(len(ctx_list.sent), 1)
        text = ctx_list.sent[0]["text"]
        self.assertIn("alpha", text)
        self.assertIn("beta", text)

    # -- store_delete_command --------------------------------------------

    async def test_store_delete(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="tmp val", command_name="store")
        await store_command(ctx_set)
        ctx_del = create_mock_context(
            user_id=uid, command_text="tmp", command_name="store_delete")
        await store_delete_command(ctx_del)
        self.assertEqual(len(ctx_del.sent), 1)
        self.assertIn("Deleted", ctx_del.sent[0]["text"])
        # Verify the key was actually deleted
        ctx_get = create_mock_context(
            user_id=uid, command_text="tmp", command_name="store")
        await store_command(ctx_get)
        self.assertIn("not found", ctx_get.sent[0]["text"])

    async def test_store_delete_nonexistent(self):
        # Deleting a key that doesn't exist still reports success
        ctx = create_mock_context(
            user_id=_uid(), command_text="ghost", command_name="store_delete")
        await store_delete_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Deleted", ctx.sent[0]["text"])

    # -- store_delete_all_command ----------------------------------------

    async def test_store_delete_all(self):
        uid = _uid()
        ctx_set1 = create_mock_context(
            user_id=uid, command_text="k1 v1", command_name="store")
        await store_command(ctx_set1)
        ctx_set2 = create_mock_context(
            user_id=uid, command_text="k2 v2", command_name="store")
        await store_command(ctx_set2)
        ctx_del = create_mock_context(
            user_id=uid, command_name="store_delete_all")
        await store_delete_all_command(ctx_del)
        self.assertEqual(len(ctx_del.sent), 1)
        self.assertIn("All keys deleted", ctx_del.sent[0]["text"])
        # Verify all keys were actually deleted
        ctx_list = create_mock_context(
            user_id=uid, command_name="store_list")
        await store_list_command(ctx_list)
        self.assertIn("No keys", ctx_list.sent[0]["text"])

        # An empty store still reports success.
        ctx = create_mock_context(
            user_id=uid, command_name="store_delete_all")
        await store_delete_all_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("All keys deleted", ctx.sent[0]["text"])


if __name__ == "__main__":
    unittest.main()
