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
from user_settings import UserSettings


class TestSettingsCommands(unittest.IsolatedAsyncioTestCase):
    # Each test uses a distinct user_id so they don't collide on the
    # shared test database (wiped + recreated once per test run).

    # The two numeric settings are driven by identical command code, so
    # they are exercised together rather than copy-pasted per setting:
    # (command, command name, default, value to set, out-of-range value,
    #  substring the bare-query reply starts with).
    NUMERIC_SETTINGS = (
        (temperature_command, "temperature", 1.0, "0.7", "5.0",
         "Current temperature"),
        (top_p_command, "top_p", 0.95, "0.5", "-0.1", "Current top"),
    )

    # -- start_restart_command -------------------------------------------

    async def test_start_restart(self):
        ctx = create_mock_context(user_id=_uid(), command_name="start")
        await start_restart_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Starting conversation", ctx.sent[0]["text"])
        self.assertEqual(ctx.conversation, [])

    # -- model_command ---------------------------------------------------

    async def test_model_set_and_query(self):
        uid = _uid()
        ctx = create_mock_context(
            user_id=uid, command_text="my-org/my-model", command_name="model")
        await model_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("my-org/my-model", ctx.sent[0]["text"])
        self.assertEqual(ctx.settings.model, "my-org/my-model")

        # Called bare it reports the stored value...
        ctx = create_mock_context(user_id=uid, command_name="model")
        await model_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Current model", ctx.sent[0]["text"])
        self.assertIn("my-org/my-model", ctx.sent[0]["text"])

        # ...and the default model for a user who never set one.
        ctx = create_mock_context(user_id=_uid(), command_name="model")
        await model_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Current model", ctx.sent[0]["text"])
        self.assertIn(os.environ.get("LLM_MODEL", "minimaxai/minimax-m2.7"),
                      ctx.sent[0]["text"])

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

    # -- temperature_command / top_p_command -----------------------------

    async def test_numeric_setting_set_and_query(self):
        for command, name, _, value, _, needle in self.NUMERIC_SETTINGS:
            with self.subTest(setting=name):
                ctx = create_mock_context(
                    user_id=_uid(), command_text=value, command_name=name)
                await command(ctx)
                self.assertEqual(len(ctx.sent), 1)
                self.assertAlmostEqual(
                    getattr(ctx.settings, name), float(value))

                ctx = create_mock_context(user_id=_uid(), command_name=name)
                await command(ctx)
                self.assertEqual(len(ctx.sent), 1)
                self.assertIn(needle, ctx.sent[0]["text"])

    async def test_numeric_setting_rejects_bad_values(self):
        for command, name, default, _, out_of_range, _ in self.NUMERIC_SETTINGS:
            with self.subTest(setting=name):
                for text, needle in (("abc", "convert"),
                                     (out_of_range, "must be between")):
                    ctx = create_mock_context(
                        user_id=_uid(), command_text=text, command_name=name)
                    await command(ctx)
                    self.assertEqual(len(ctx.sent), 1)
                    self.assertIn(needle, ctx.sent[0]["text"])
                    # The rejected value leaves the setting at its default.
                    self.assertAlmostEqual(
                        getattr(ctx.settings, name), default)

    # -- the per-setting reset commands ----------------------------------

    async def test_setting_resets_restore_the_defaults(self):
        defaults = UserSettings.default()
        # (set command, set name, value, echo, reset command, reset name,
        #  settings attribute)
        cases = (
            (model_command, "model", "my-org/my-model", "my-org/my-model",
             model_reset_command, "model_reset", "model"),
            (temperature_command, "temperature", "0.5", "0.5",
             temperature_reset_command, "temperature_reset", "temperature"),
            (top_p_command, "top_p", "0.3", "0.3",
             top_p_reset_command, "top_p_reset", "top_p"),
            (systemprompt_command, "systemprompt", "Custom", "updated",
             systemprompt_reset_command, "systemprompt_reset", "system_prompt"),
            (user_profile_command, "user_profile", "Something", "updated",
             user_profile_reset_command, "user_profile_reset", "user_profile"),
        )
        for setter, set_name, value, echo, reset, reset_name, attr in cases:
            with self.subTest(setting=attr):
                uid = _uid()
                ctx_set = create_mock_context(
                    user_id=uid, command_text=value, command_name=set_name)
                await setter(ctx_set)
                self.assertEqual(len(ctx_set.sent), 1)
                self.assertIn(echo, ctx_set.sent[0]["text"])
                self.assertNotEqual(getattr(ctx_set.settings, attr),
                                    getattr(defaults, attr))

                ctx_reset = create_mock_context(
                    user_id=uid, command_name=reset_name)
                await reset(ctx_reset)
                self.assertEqual(len(ctx_reset.sent), 1)
                self.assertIn("reset", ctx_reset.sent[0]["text"])
                self.assertEqual(getattr(ctx_reset.settings, attr),
                                 getattr(defaults, attr))

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

    # -- tool_output_command ---------------------------------------------

    async def test_tool_output_toggle(self):
        uid = _uid()
        # Default is True; the first toggle turns it off...
        ctx_off = create_mock_context(user_id=uid, command_name="tool_output")
        await tool_output_command(ctx_off)
        self.assertEqual(len(ctx_off.sent), 1)
        self.assertIn("off", ctx_off.sent[0]["text"])
        self.assertFalse(ctx_off.settings.show_tool_output)
        # ...and the second turns it back on.
        ctx_on = create_mock_context(user_id=uid, command_name="tool_output")
        await tool_output_command(ctx_on)
        self.assertEqual(len(ctx_on.sent), 1)
        self.assertIn("on", ctx_on.sent[0]["text"])
        self.assertTrue(ctx_on.settings.show_tool_output)

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

    async def test_user_profile_set_and_query(self):
        uid = _uid()
        ctx = create_mock_context(
            user_id=uid, command_text="My profile text", command_name="user_profile")
        await user_profile_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("updated", ctx.sent[0]["text"])
        self.assertEqual(ctx.settings.user_profile, "My profile text")

        # A bare query echoes the stored profile...
        ctx = create_mock_context(user_id=uid, command_name="user_profile")
        await user_profile_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("current profile", ctx.sent[0]["text"])
        self.assertIn("My profile text", ctx.sent[0]["text"])

        # ...and says so when none is set.
        ctx = create_mock_context(user_id=_uid(), command_name="user_profile")
        await user_profile_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("No profile set", ctx.sent[0]["text"])


if __name__ == "__main__":
    unittest.main()
