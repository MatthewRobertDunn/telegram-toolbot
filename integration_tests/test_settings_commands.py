import os
import unittest

import bootstrap  # noqa: F401 — path setup, env, logging, DB reset, next_uid
from bootstrap import next_uid as _uid
from mocks import create_mock_context
from handlers.command_handlers import (
    start_restart_command,
    systemprompt_command,
    systemprompt_reset_command,
    model_command,
    model_reset_command,
    subagent_model_command,
    subagent_model_reset_command,
    top_p_command,
    top_p_reset_command,
    temperature_command,
    temperature_reset_command,
    settings_reset_command,
    tool_output_command,
    user_profile_command,
    user_profile_reset_command,
)


class TestSettingsCommands(unittest.IsolatedAsyncioTestCase):
    # Each test uses a distinct user_id so they don't collide on the
    # shared test database (wiped + recreated once per test run).

    # -- start_restart_command -------------------------------------------

    async def test_start_restart(self):
        ctx = create_mock_context(user_id=_uid(), command_name="start")
        await start_restart_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Starting conversation", ctx.sent[0]["text"])
        self.assertEqual(ctx.conversation, [])

    # -- model_command ---------------------------------------------------

    async def test_model_set(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="my-org/my-model", command_name="model")
        await model_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("my-org/my-model", ctx.sent[0]["text"])
        self.assertEqual(ctx.settings.model, "my-org/my-model")

    async def test_model_query_without_arg(self):
        ctx = create_mock_context(user_id=_uid(), command_name="model")
        await model_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Current model", ctx.sent[0]["text"])
        # The default model should appear in the query response
        self.assertIn(os.environ.get("LLM_MODEL", "minimaxai/minimax-m2.7"),
                      ctx.sent[0]["text"])

    async def test_model_reset(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="my-org/my-model", command_name="model")
        await model_command(ctx_set)
        ctx_reset = create_mock_context(user_id=uid, command_name="model_reset")
        await model_reset_command(ctx_reset)
        self.assertEqual(len(ctx_reset.sent), 1)
        self.assertIn("reset", ctx_reset.sent[0]["text"])
        self.assertEqual(ctx_reset.settings.model,
                         os.environ.get("LLM_MODEL", "minimaxai/minimax-m2.7"))

    # -- subagent_model_command ------------------------------------------

    async def test_subagent_model_set(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="sub-v1", command_name="subagent_model")
        await subagent_model_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("sub-v1", ctx.sent[0]["text"])
        self.assertEqual(ctx.settings.subagent_model, "sub-v1")

    async def test_subagent_model_query_without_arg(self):
        uid = _uid()
        ctx_model = create_mock_context(
            user_id=uid, command_text="my-org/my-model", command_name="model")
        await model_command(ctx_model)
        ctx = create_mock_context(user_id=uid, command_name="subagent_model")
        await subagent_model_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Current subagent model", ctx.sent[0]["text"])
        # Falls back to the model chosen with /model when no override is set.
        self.assertIn(os.environ.get("SUBAGENT_MODEL") or "my-org/my-model",
                      ctx.sent[0]["text"])

    async def test_subagent_model_reset(self):
        uid = _uid()
        ctx_model = create_mock_context(
            user_id=uid, command_text="my-org/my-model", command_name="model")
        await model_command(ctx_model)
        ctx_set = create_mock_context(
            user_id=uid, command_text="custom-model", command_name="subagent_model")
        await subagent_model_command(ctx_set)
        ctx_reset = create_mock_context(user_id=uid, command_name="subagent_model_reset")
        await subagent_model_reset_command(ctx_reset)
        self.assertEqual(len(ctx_reset.sent), 1)
        self.assertIn("reset", ctx_reset.sent[0]["text"])
        # After reset the subagent follows the /model choice again.
        self.assertEqual(
            ctx_reset.settings.subagent_model,
            os.environ.get("SUBAGENT_MODEL") or "my-org/my-model")

    # -- temperature_command ---------------------------------------------

    async def test_temperature_set(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="0.7", command_name="temperature")
        await temperature_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertEqual(ctx.settings.temperature, 0.7)

    async def test_temperature_query_without_arg(self):
        ctx = create_mock_context(user_id=_uid(), command_name="temperature")
        await temperature_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Current temperature", ctx.sent[0]["text"])

    async def test_temperature_invalid(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="abc", command_name="temperature")
        await temperature_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("convert", ctx.sent[0]["text"])
        # Setting should not have changed from default
        self.assertAlmostEqual(ctx.settings.temperature, 1.0)

    async def test_temperature_out_of_range(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="5.0", command_name="temperature")
        await temperature_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("must be between", ctx.sent[0]["text"])
        # Setting should not have changed from default
        self.assertAlmostEqual(ctx.settings.temperature, 1.0)

    async def test_temperature_reset(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="0.5", command_name="temperature")
        await temperature_command(ctx_set)
        ctx_reset = create_mock_context(user_id=uid, command_name="temperature_reset")
        await temperature_reset_command(ctx_reset)
        self.assertEqual(len(ctx_reset.sent), 1)
        self.assertIn("reset", ctx_reset.sent[0]["text"])
        self.assertAlmostEqual(ctx_reset.settings.temperature, 1.0)

    # -- top_p_command ---------------------------------------------------

    async def test_top_p_set(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="0.5", command_name="top_p")
        await top_p_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertEqual(ctx.settings.top_p, 0.5)

    async def test_top_p_query_without_arg(self):
        ctx = create_mock_context(user_id=_uid(), command_name="top_p")
        await top_p_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Current top", ctx.sent[0]["text"])

    async def test_top_p_invalid(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="abc", command_name="top_p")
        await top_p_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("convert", ctx.sent[0]["text"])
        # Setting should not have changed from default
        self.assertAlmostEqual(ctx.settings.top_p, 0.95)

    async def test_top_p_out_of_range(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="-0.1", command_name="top_p")
        await top_p_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("must be between", ctx.sent[0]["text"])
        # Setting should not have changed from default
        self.assertAlmostEqual(ctx.settings.top_p, 0.95)

    async def test_top_p_reset(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="0.3", command_name="top_p")
        await top_p_command(ctx_set)
        ctx_reset = create_mock_context(user_id=uid, command_name="top_p_reset")
        await top_p_reset_command(ctx_reset)
        self.assertEqual(len(ctx_reset.sent), 1)
        self.assertIn("reset", ctx_reset.sent[0]["text"])
        self.assertAlmostEqual(ctx_reset.settings.top_p, 0.95)

    # -- systemprompt_command --------------------------------------------

    async def test_systemprompt_set(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="You are a test bot.", command_name="systemprompt")
        await systemprompt_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("updated", ctx.sent[0]["text"])
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
        self.assertIn("Current system prompt", ctx_query.sent[0]["text"])
        # The second message is the preformatted prompt content
        self.assertEqual(ctx_query.sent[1]["text"], "<pre>Custom prompt</pre>")

    async def test_systemprompt_reset(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="Custom", command_name="systemprompt")
        await systemprompt_command(ctx_set)
        ctx_reset = create_mock_context(user_id=uid, command_name="systemprompt_reset")
        await systemprompt_reset_command(ctx_reset)
        self.assertEqual(len(ctx_reset.sent), 1)
        self.assertIn("reset", ctx_reset.sent[0]["text"])
        # After reset, system_prompt should fall back to the default
        from user_settings import DEFAULT_SYSTEM_PROMPT
        self.assertEqual(ctx_reset.settings.system_prompt, DEFAULT_SYSTEM_PROMPT)

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
        self.assertIn("on", ctx_on.sent[0]["text"])
        self.assertTrue(ctx_on.settings.show_tool_output)

    async def test_tool_output_toggle_off(self):
        ctx = create_mock_context(user_id=_uid(), command_name="tool_output")
        await tool_output_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("off", ctx.sent[0]["text"])
        self.assertFalse(ctx.settings.show_tool_output)

    # -- settings_reset_command ------------------------------------------

    async def test_settings_reset(self):
        uid = _uid()
        # Set multiple non-default values
        ctx_model = create_mock_context(
            user_id=uid, command_text="custom-model", command_name="model")
        await model_command(ctx_model)
        ctx_temp = create_mock_context(
            user_id=uid, command_text="0.3", command_name="temperature")
        await temperature_command(ctx_temp)
        ctx_top_p = create_mock_context(
            user_id=uid, command_text="0.4", command_name="top_p")
        await top_p_command(ctx_top_p)
        # Reset all settings
        ctx_reset = create_mock_context(user_id=uid, command_name="settings_reset")
        await settings_reset_command(ctx_reset)
        self.assertEqual(len(ctx_reset.sent), 1)
        self.assertIn("reset", ctx_reset.sent[0]["text"])
        # All values should be back to defaults
        self.assertEqual(ctx_reset.settings.model,
                         os.environ.get("LLM_MODEL", "minimaxai/minimax-m2.7"))
        self.assertAlmostEqual(ctx_reset.settings.temperature, 1.0)
        self.assertAlmostEqual(ctx_reset.settings.top_p, 0.95)

    # -- user_profile_command --------------------------------------------

    async def test_user_profile_set(self):
        ctx = create_mock_context(
            user_id=_uid(), command_text="I like cats.", command_name="user_profile")
        await user_profile_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("updated", ctx.sent[0]["text"])
        self.assertEqual(ctx.settings.user_profile, "I like cats.")

    async def test_user_profile_query_when_set(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="My profile text", command_name="user_profile")
        await user_profile_command(ctx_set)
        ctx_query = create_mock_context(user_id=uid, command_name="user_profile")
        await user_profile_command(ctx_query)
        self.assertEqual(len(ctx_query.sent), 1)
        self.assertIn("current profile", ctx_query.sent[0]["text"])
        # The actual profile text should appear in the message
        self.assertIn("My profile text", ctx_query.sent[0]["text"])

    async def test_user_profile_query_when_unset(self):
        ctx = create_mock_context(user_id=_uid(), command_name="user_profile")
        await user_profile_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("No profile set", ctx.sent[0]["text"])

    async def test_user_profile_reset(self):
        uid = _uid()
        ctx_set = create_mock_context(
            user_id=uid, command_text="Something", command_name="user_profile")
        await user_profile_command(ctx_set)
        ctx_reset = create_mock_context(user_id=uid, command_name="user_profile_reset")
        await user_profile_reset_command(ctx_reset)
        self.assertEqual(len(ctx_reset.sent), 1)
        self.assertIn("reset", ctx_reset.sent[0]["text"])
        self.assertIsNone(ctx_reset.settings.user_profile)


if __name__ == "__main__":
    unittest.main()
