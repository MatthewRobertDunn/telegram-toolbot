import os
import unittest

import bootstrap  # noqa: F401 — path setup, env, logging, DB reset
from mocks import create_mock_context
from handlers.command_handlers import (
    start_restart_command,
    systemprompt_command,
    systemprompt_reset_command,
    model_command,
    model_reset_command,
    strong_model_command,
    strong_model_reset_command,
    top_p_command,
    top_p_reset_command,
    temperature_command,
    temperature_reset_command,
    settings_reset_command,
    tool_output_command,
    user_profile_command,
    user_profile_reset_command,
)
from handlers.store_handlers import (
    store_command,
    store_list_command,
    store_delete_command,
    store_delete_all_command,
)

_next_uid = 3000


def _uid() -> int:
    global _next_uid
    _next_uid += 1
    return _next_uid


class TestCommands(unittest.IsolatedAsyncioTestCase):
    # Each test uses a distinct user_id so they don't collide on the
    # test database, which is wiped + recreated every test run.

    # -- start_restart_command -------------------------------------------

    async def test_start_restart(self):
        ctx = create_mock_context(user_id=_uid(), command_name="start")
        await start_restart_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Starting conversation",
                      ctx.sent[0]["text"])

    # -- model_command ---------------------------------------------------

    async def test_model_set(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="my-org/my-model", command_name="model")
        await model_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("my-org/my-model",
                      ctx.sent[0]["text"])
        self.assertEqual(ctx.settings.model, "my-org/my-model")

    async def test_model_query_without_arg(self):
        ctx = create_mock_context(user_id=_uid(), command_name="model")
        await model_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Current model",
                      ctx.sent[0]["text"])

    async def test_model_reset(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="my-org/my-model", command_name="model")
        await model_command(ctx_set)
        ctx_reset = create_mock_context(user_id=uid, command_name="model_reset")
        await model_reset_command(ctx_reset)
        self.assertEqual(len(ctx_reset.sent), 1)
        self.assertIn("reset",
                      ctx_reset.sent[0]["text"])
        self.assertEqual(ctx_reset.settings.model,
                         os.environ.get("LLM_MODEL", "minimaxai/minimax-m2.7"))

    # -- strong_model_command --------------------------------------------

    async def test_strong_model_set(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="strong-v1", command_name="strong_model")
        await strong_model_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("strong-v1",
                      ctx.sent[0]["text"])
        self.assertEqual(ctx.settings.strong_model, "strong-v1")

    async def test_strong_model_query_without_arg(self):
        ctx = create_mock_context(user_id=_uid(), command_name="strong_model")
        await strong_model_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Current strong model",
                      ctx.sent[0]["text"])

    async def test_strong_model_reset(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="custom-model", command_name="strong_model")
        await strong_model_command(ctx_set)
        ctx_reset = create_mock_context(user_id=uid, command_name="strong_model_reset")
        await strong_model_reset_command(ctx_reset)
        self.assertEqual(len(ctx_reset.sent), 1)
        self.assertIn("reset",
                      ctx_reset.sent[0]["text"])

    # -- temperature_command ---------------------------------------------

    async def test_temperature_set(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="0.7", command_name="temperature")
        await temperature_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("0.7",
                      ctx.sent[0]["text"])
        self.assertEqual(ctx.settings.temperature, 0.7)

    async def test_temperature_query_without_arg(self):
        ctx = create_mock_context(user_id=_uid(), command_name="temperature")
        await temperature_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Current temperature",
                      ctx.sent[0]["text"])

    async def test_temperature_invalid(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="abc", command_name="temperature")
        await temperature_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("convert",
                      ctx.sent[0]["text"])

    async def test_temperature_out_of_range(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="5.0", command_name="temperature")
        await temperature_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("must be between",
                      ctx.sent[0]["text"])

    async def test_temperature_reset(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="0.5", command_name="temperature")
        await temperature_command(ctx_set)
        ctx_reset = create_mock_context(user_id=uid, command_name="temperature_reset")
        await temperature_reset_command(ctx_reset)
        self.assertEqual(len(ctx_reset.sent), 1)
        self.assertIn("reset",
                      ctx_reset.sent[0]["text"])
        self.assertAlmostEqual(ctx_reset.settings.temperature, 1.0)

    # -- top_p_command ---------------------------------------------------

    async def test_top_p_set(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="0.5", command_name="top_p")
        await top_p_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("0.5",
                      ctx.sent[0]["text"])
        self.assertEqual(ctx.settings.top_p, 0.5)

    async def test_top_p_query_without_arg(self):
        ctx = create_mock_context(user_id=_uid(), command_name="top_p")
        await top_p_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Current top",
                      ctx.sent[0]["text"])

    async def test_top_p_invalid(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="abc", command_name="top_p")
        await top_p_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("convert",
                      ctx.sent[0]["text"])

    async def test_top_p_out_of_range(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="-0.1", command_name="top_p")
        await top_p_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("must be between",
                      ctx.sent[0]["text"])

    async def test_top_p_reset(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="0.3", command_name="top_p")
        await top_p_command(ctx_set)
        ctx_reset = create_mock_context(user_id=uid, command_name="top_p_reset")
        await top_p_reset_command(ctx_reset)
        self.assertEqual(len(ctx_reset.sent), 1)
        self.assertIn("reset",
                      ctx_reset.sent[0]["text"])
        self.assertAlmostEqual(ctx_reset.settings.top_p, 0.95)

    # -- systemprompt_command --------------------------------------------

    async def test_systemprompt_set(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="You are a test bot.", command_name="systemprompt")
        await systemprompt_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("updated",
                      ctx.sent[0]["text"])
        self.assertEqual(ctx.settings.system_prompt, "You are a test bot.")

    async def test_systemprompt_query(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="Custom prompt", command_name="systemprompt")
        await systemprompt_command(ctx_set)
        ctx_query = create_mock_context(user_id=uid, command_name="systemprompt")
        await systemprompt_command(ctx_query)
        # Sends two messages: usage text + the preformatted prompt.
        self.assertEqual(len(ctx_query.sent), 2)
        self.assertIn("Current system prompt",
                      ctx_query.sent[0]["text"])

    async def test_systemprompt_reset(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="Custom", command_name="systemprompt")
        await systemprompt_command(ctx_set)
        ctx_reset = create_mock_context(user_id=uid, command_name="systemprompt_reset")
        await systemprompt_reset_command(ctx_reset)
        self.assertEqual(len(ctx_reset.sent), 1)
        self.assertIn("reset",
                      ctx_reset.sent[0]["text"])

    # -- tool_output_command ---------------------------------------------

    async def test_tool_output_toggle_on(self):
        uid = _uid()
        # Default is True; first toggle turns it off
        ctx_off = create_mock_context(user_id=uid, command_name="tool_output")
        await tool_output_command(ctx_off)
        # Second toggle turns it back on
        ctx_on = create_mock_context(user_id=uid, command_name="tool_output")
        await tool_output_command(ctx_on)
        self.assertEqual(len(ctx_on.sent), 1)
        self.assertIn("on",
                      ctx_on.sent[0]["text"])
        self.assertTrue(ctx_on.settings.show_tool_output)

    async def test_tool_output_toggle_off(self):
        ctx = create_mock_context(user_id=_uid(), command_name="tool_output")
        await tool_output_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("off",
                      ctx.sent[0]["text"])
        self.assertFalse(ctx.settings.show_tool_output)

    # -- settings_reset_command ------------------------------------------

    async def test_settings_reset(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="0.3", command_name="temperature")
        await temperature_command(ctx_set)
        ctx_reset = create_mock_context(user_id=uid, command_name="settings_reset")
        await settings_reset_command(ctx_reset)
        self.assertEqual(len(ctx_reset.sent), 1)
        self.assertIn("reset",
                      ctx_reset.sent[0]["text"])
        self.assertAlmostEqual(ctx_reset.settings.temperature, 1.0)

    # -- user_profile_command --------------------------------------------

    async def test_user_profile_set(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="I like cats.", command_name="user_profile")
        await user_profile_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("updated",
                      ctx.sent[0]["text"])
        self.assertEqual(ctx.settings.user_profile, "I like cats.")

    async def test_user_profile_query_when_set(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="My profile", command_name="user_profile")
        await user_profile_command(ctx_set)
        ctx_query = create_mock_context(user_id=uid, command_name="user_profile")
        await user_profile_command(ctx_query)
        self.assertEqual(len(ctx_query.sent), 1)
        self.assertIn("current profile",
                      ctx_query.sent[0]["text"])

    async def test_user_profile_query_when_unset(self):
        ctx = create_mock_context(user_id=_uid(), command_name="user_profile")
        await user_profile_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("No profile set",
                      ctx.sent[0]["text"])

    async def test_user_profile_reset(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="Something", command_name="user_profile")
        await user_profile_command(ctx_set)
        ctx_reset = create_mock_context(user_id=uid, command_name="user_profile_reset")
        await user_profile_reset_command(ctx_reset)
        self.assertEqual(len(ctx_reset.sent), 1)
        self.assertIn("reset",
                      ctx_reset.sent[0]["text"])
        self.assertIsNone(ctx_reset.settings.user_profile)

    # -- store_command ---------------------------------------------------

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
        self.assertIn("not found",
                      ctx.sent[0]["text"])

    async def test_store_usage_without_args(self):
        ctx = create_mock_context(user_id=_uid(), command_name="store")
        await store_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Usage",
                      ctx.sent[0]["text"])

    # -- store_list_command ----------------------------------------------

    async def test_store_list_empty(self):
        ctx = create_mock_context(user_id=_uid(), command_name="store_list")
        await store_list_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("No keys",
                      ctx.sent[0]["text"])

    async def test_store_list_with_keys(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="a 1", command_name="store")
        await store_command(ctx_set)
        ctx_list = create_mock_context(user_id=uid, command_name="store_list")
        await store_list_command(ctx_list)
        self.assertEqual(len(ctx_list.sent), 1)
        self.assertIn("a", ctx_list.sent[0]["text"])

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
        self.assertIn("Deleted",
                      ctx_del.sent[0]["text"])

    async def test_store_delete_usage_without_args(self):
        ctx = create_mock_context(user_id=_uid(), command_name="store_delete")
        await store_delete_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Usage",
                      ctx.sent[0]["text"])

    # -- store_delete_all_command ----------------------------------------

    async def test_store_delete_all(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="k v", command_name="store")
        await store_command(ctx_set)
        ctx_del = create_mock_context(
            user_id=uid, command_name="store_delete_all")
        await store_delete_all_command(ctx_del)
        self.assertEqual(len(ctx_del.sent), 1)
        self.assertIn("All keys deleted",
                      ctx_del.sent[0]["text"])


if __name__ == "__main__":
    unittest.main()