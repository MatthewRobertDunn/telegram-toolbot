import asyncio
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
import handlers.middleware as middleware
import user_settings
from handlers.command_handlers import timezone_command
from mock_context import mock_ctx
from user_settings import UserSettings


class _Bot:
    def __init__(self):
        self.sent: list = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append(text)
        return SimpleNamespace(message_id=1)

    async def _post(self, method, data):
        return {'message_id': 1}


def _run(ctx):
    with mock.patch.object(middleware, 'save_settings',
                           mock.AsyncMock()) as save:
        asyncio.run(timezone_command(ctx))
    return save


class TestTimezoneSetting(unittest.TestCase):

    def test_default_timezone_is_utc(self):
        self.assertEqual(UserSettings.default().timezone, 'UTC')

    def test_set_valid_timezone(self):
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot, conversation=[],
                       settings=UserSettings.default())
        ctx.command_text = 'Australia/Sydney'
        save = _run(ctx)
        self.assertEqual(ctx.settings.timezone, 'Australia/Sydney')
        self.assertEqual(bot.sent, ['Timezone set to `Australia/Sydney`.'])
        save.assert_awaited_once()

    def test_invalid_timezone_is_rejected(self):
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot, conversation=[],
                       settings=UserSettings.default())
        ctx.command_text = 'Not/AZone'
        _run(ctx)
        self.assertEqual(ctx.settings.timezone, 'UTC')
        self.assertTrue(bot.sent[0].startswith("Unknown timezone"))

    def test_no_arg_shows_current(self):
        bot = _Bot()
        settings = UserSettings.default()
        settings.timezone = 'America/New_York'
        ctx = mock_ctx(user_id=5, bot=bot, conversation=[],
                       settings=settings)
        ctx.command_text = ''
        _run(ctx)
        self.assertIn('America/New_York', bot.sent[0])
        self.assertEqual(ctx.settings.timezone, 'America/New_York')

    def test_timezone_survives_json_round_trip(self):
        settings = UserSettings.default()
        settings.timezone = 'Pacific/Auckland'
        restored = UserSettings.from_json(settings.to_json())
        self.assertEqual(restored.timezone, 'Pacific/Auckland')


class TestUserSettingsDefaults(unittest.TestCase):
    """What an untouched settings object resolves to, and what is rejected."""

    def test_defaults_come_from_the_module_and_config(self):
        settings = UserSettings.default()
        self.assertEqual(settings.model, config.LLM_MODEL)
        self.assertEqual(settings.temperature, config.temperature)
        self.assertEqual(settings.top_p, config.top_p)
        self.assertEqual(settings.system_prompt,
                         user_settings.DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(settings.timezone, user_settings.DEFAULT_TIMEZONE)
        self.assertTrue(settings.show_tool_output)
        self.assertFalse(settings.thinking_keep)
        self.assertIsNone(settings.user_profile)
        self.assertEqual(settings.subagent_model,
                         config.SUBAGENT_MODEL or config.LLM_MODEL)

    def test_default_prompt_is_read_at_call_time(self):
        # chatbot.py replaces the module default at startup; settings built
        # before that must still resolve to the replacement.
        original = user_settings.DEFAULT_SYSTEM_PROMPT
        self.addCleanup(setattr, user_settings,
                        'DEFAULT_SYSTEM_PROMPT', original)
        user_settings.DEFAULT_SYSTEM_PROMPT = 'replaced at startup'
        self.assertEqual(UserSettings.default().system_prompt,
                         'replaced at startup')

    def test_blank_model_is_rejected(self):
        settings = UserSettings.default()
        for blank in ('', '   '):
            with self.subTest(value=repr(blank)):
                with self.assertRaises(ValueError):
                    settings.model = blank
        # None is the reset, not a blank value.
        settings.model = None
        self.assertEqual(settings.model, config.LLM_MODEL)

    def test_blank_system_prompt_is_rejected(self):
        settings = UserSettings.default()
        with self.assertRaises(ValueError):
            settings.system_prompt = ''
        settings.system_prompt = None
        self.assertEqual(settings.system_prompt,
                         user_settings.DEFAULT_SYSTEM_PROMPT)

    def test_every_setting_survives_a_json_round_trip(self):
        settings = UserSettings.default()
        settings.model = 'some-model'
        settings.temperature = 0.3
        settings.top_p = 0.4
        settings.system_prompt = 'be brief'
        settings.user_profile = 'a tea person'
        settings.show_tool_output = False
        settings.thinking_keep = True
        settings.subagent_model = 'other-model'
        settings.timezone = 'Australia/Sydney'

        restored = UserSettings.from_json(settings.to_json())
        self.assertEqual(restored.model, 'some-model')
        self.assertAlmostEqual(restored.temperature, 0.3)
        self.assertAlmostEqual(restored.top_p, 0.4)
        self.assertEqual(restored.system_prompt, 'be brief')
        self.assertEqual(restored.user_profile, 'a tea person')
        self.assertFalse(restored.show_tool_output)
        self.assertTrue(restored.thinking_keep)
        self.assertEqual(restored.subagent_model, 'other-model')
        self.assertEqual(restored.timezone, 'Australia/Sydney')


if __name__ == '__main__':
    unittest.main()
