import asyncio
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import handlers.middleware as middleware
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


if __name__ == '__main__':
    unittest.main()
