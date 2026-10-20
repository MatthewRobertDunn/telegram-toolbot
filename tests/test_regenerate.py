import asyncio
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from conversation import CUSTOM, Conversation
from handlers import generation_queue as generation_queue_mod
from handlers.conversation_handlers import (
    regenerate_command,
    undo_command,
)
from mock_context import mock_ctx


def _roles(conversation):
    """Conversation roles, ignoring the leading system head if present."""
    roles = [m['role'] for m in conversation.messages]
    if roles and roles[0] == 'system':
        roles = roles[1:]
    return roles


class _Bot:
    """Captures sends, deletions, and rich posts."""

    def __init__(self):
        self.deleted: list = []
        self.sent: list = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append(text)
        return SimpleNamespace(message_id=1042)

    async def delete_message(self, chat_id, message_id):
        self.deleted.append((chat_id, message_id))
        return True

    async def _post(self, method, data):
        return {'message_id': 555}


async def _run_regenerate(ctx, canned='fresh reply'):
    """Run regenerate_command with a stubbed model replacing run_llm."""
    with mock.patch.object(generation_queue_mod, 'run_llm',
                           mock.AsyncMock(return_value=canned)):
        await regenerate_command(ctx)


class TestRegenerate(unittest.TestCase):

    def test_last_assistant_is_popped_deleted_and_regenerated(self):
        bot = _Bot()
        conversation = Conversation([
            {'role': 'user', 'content': 'hello'},
        ])
        conversation.add_assistant('old reply', telegram_message_id=77)
        ctx = mock_ctx(user_id=5, bot=bot, conversation=conversation)
        asyncio.run(_run_regenerate(ctx))
        # The stale assistant turn is gone; a fresh one was generated.
        messages = ctx.conversation.messages
        self.assertEqual(_roles(ctx.conversation), ['user', 'assistant'])
        self.assertEqual(messages[-1]['content'], 'fresh reply')
        # The old reply's telegram message was deleted...
        self.assertEqual(bot.deleted, [(5, 77)])
        # ...and the fresh reply was recorded with its new id.
        self.assertEqual(messages[-1][CUSTOM]['telegram_message_id'], 555)

    def test_last_user_turns_into_a_reply_without_appending(self):
        bot = _Bot()
        conversation = Conversation([
            {'role': 'user', 'content': 'hello'},
        ])
        ctx = mock_ctx(user_id=5, bot=bot, conversation=conversation)
        asyncio.run(_run_regenerate(ctx))
        # The user turn stays single; a reply was generated and recorded.
        self.assertEqual(_roles(ctx.conversation), ['user', 'assistant'])
        self.assertEqual(ctx.conversation.messages[-1]['content'],
                         'fresh reply')
        self.assertEqual(bot.deleted, [])

    def test_busy_generation_is_refused(self):
        bot = _Bot()
        conversation = Conversation([{'role': 'user', 'content': 'hi'}])
        ctx = mock_ctx(user_id=5, bot=bot, conversation=conversation)
        # A live generation task blocks regeneration. Plain stub: only
        # done() is consulted.
        ctx.generation_task = SimpleNamespace(  # type: ignore[assignment]
            done=lambda: False)
        asyncio.run(regenerate_command(ctx))
        self.assertEqual(len(_roles(ctx.conversation)), 1)
        self.assertEqual(bot.deleted, [])
        self.assertEqual(bot.sent, ["A response is currently being "
                                    "generated; use /stop first."])

    def test_empty_conversation_says_nothing_to_regenerate(self):
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot, conversation=[])
        asyncio.run(regenerate_command(ctx))
        self.assertEqual(_roles(ctx.conversation), [])
        self.assertEqual(bot.deleted, [])
        self.assertEqual(bot.sent, ['Nothing to regenerate yet.'])

    def test_trailing_tool_results_are_skipped_to_find_the_reply(self):
        bot = _Bot()
        conversation = Conversation([
            {'role': 'user', 'content': 'hello'},
            {'role': 'assistant', 'content': '',
             'tool_calls': [{'id': 'c1', 'type': 'function'}]},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': 'result'},
        ])
        ctx = mock_ctx(user_id=5, bot=bot, conversation=conversation)
        asyncio.run(_run_regenerate(ctx))
        messages = ctx.conversation.messages
        self.assertEqual(_roles(ctx.conversation), ['user', 'assistant'])
        self.assertNotIn('tool_calls', messages[-1])
        # The popped assistant turn had no telegram id, so the bot
        # deleted nothing.
        self.assertEqual(bot.deleted, [])

    def test_reply_without_stored_id_deletes_nothing(self):
        # An assistant turn recorded before message-id tracking, or with
        # no reply sent at all, carries no telegram id: pop without
        # attempting any deletion.
        conversation = Conversation()
        conversation.add_user('hello')
        conversation.add_assistant('no telegram id')
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot, conversation=conversation)
        asyncio.run(_run_regenerate(ctx))
        self.assertEqual(bot.deleted, [])
        self.assertEqual(ctx.conversation.messages[-1]['content'],
                         'fresh reply')


class TestUndo(unittest.TestCase):

    def test_undoes_last_message_and_deletes_it_from_telegram(self):
        bot = _Bot()
        conversation = Conversation()
        conversation.add_user('hello', telegram_message_id=42)
        ctx = mock_ctx(user_id=5, bot=bot, conversation=conversation)
        asyncio.run(undo_command(ctx))
        self.assertEqual(_roles(ctx.conversation), [])
        self.assertEqual(bot.deleted, [(5, 42)])
        self.assertEqual(bot.sent,
                         ["Removed message (User):\n```\nhello\n```"])

    def test_undo_preview_is_truncated_to_the_limit(self):
        bot = _Bot()
        conversation = Conversation(
            [{'role': 'user', 'content': 'a' * 120}])
        ctx = mock_ctx(user_id=5, bot=bot, conversation=conversation)
        asyncio.run(undo_command(ctx))
        self.assertEqual(
            bot.sent,
            ["Removed message (User):\n```\n" + 'a' * 100 + "\n```"])

    def test_undo_of_assistant_turn_names_the_role(self):
        bot = _Bot()
        conversation = Conversation()
        conversation.add_user('hello')
        conversation.add_assistant('sure thing')
        ctx = mock_ctx(user_id=5, bot=bot, conversation=conversation)
        asyncio.run(undo_command(ctx))
        self.assertEqual(bot.sent,
                         ["Removed message (Assistant):\n```\nsure thing\n```"])

    def test_undo_without_telegram_id_deletes_nothing(self):
        bot = _Bot()
        conversation = Conversation([{'role': 'user', 'content': 'hello'}])
        ctx = mock_ctx(user_id=5, bot=bot, conversation=conversation)
        asyncio.run(undo_command(ctx))
        self.assertEqual(_roles(ctx.conversation), [])
        self.assertEqual(bot.deleted, [])

    def test_busy_generation_is_refused(self):
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot,
                       conversation=[{'role': 'user', 'content': 'hi'}])
        ctx.generation_task = SimpleNamespace(  # type: ignore[assignment]
            done=lambda: False)
        asyncio.run(undo_command(ctx))
        self.assertEqual(len(_roles(ctx.conversation)), 1)
        self.assertEqual(bot.deleted, [])
        self.assertEqual(bot.sent, ["A response is currently being "
                                    "generated; use /stop first."])

    def test_only_system_head_says_nothing_to_undo(self):
        conversation = Conversation()
        conversation.set_system_head('prompt')
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot, conversation=conversation)
        asyncio.run(undo_command(ctx))
        self.assertEqual(_roles(ctx.conversation), [])
        self.assertEqual(bot.sent, ['Nothing to undo.'])
        self.assertEqual(bot.deleted, [])


if __name__ == '__main__':
    unittest.main()