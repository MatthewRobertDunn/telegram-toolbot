import asyncio
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from conversation import CUSTOM, Conversation, ConversationMessageType
from handlers import conversation_handlers
from handlers.conversation_handlers import (impersonate_command,
                                            regenerate_command)
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
        self.sent: list = []
        self.rich_sent: list = []
        self.deleted: list = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append(text)
        return SimpleNamespace(message_id=555)

    async def delete_message(self, chat_id, message_id):
        self.deleted.append((chat_id, message_id))
        return True

    async def _post(self, method, data):
        # Rich messages (parse_mode='LLM') go through here.
        self.rich_sent.append(
            data.get('rich_message', {}).get('markdown'))
        return {'message_id': 555}


async def _run_command(ctx, canned='i would say this'):
    with mock.patch.object(conversation_handlers, 'run_llm',
                           mock.AsyncMock(return_value=canned)):
        await impersonate_command(ctx)


class TestImpersonate(unittest.TestCase):

    def test_empty_conversation_says_nothing_to_impersonate(self):
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot,
                       conversation=Conversation([
                           {'role': 'system', 'content': 'sys'}]))
        asyncio.run(_run_command(ctx))
        self.assertEqual(bot.sent, ['Nothing to impersonate yet.'])
        self.assertEqual(_roles(ctx.conversation), [])

    def test_impersonation_is_added_as_a_user_turn(self):
        bot = _Bot()
        conversation = Conversation([
            {'role': 'user', 'content': 'hello'},
        ])
        conversation.add_assistant('hi!', telegram_message_ids=[77])
        ctx = mock_ctx(user_id=5, bot=bot, conversation=conversation)
        asyncio.run(_run_command(ctx))
        # The existing history is untouched; the fake turn was added.
        self.assertEqual(_roles(ctx.conversation),
                         ['user', 'assistant', 'user'])
        self.assertEqual(ctx.conversation.messages[-1]['content'],
                         'i would say this')
        # Telegram got the turn as rich markdown, and the posted id is
        # attached to the message.
        self.assertEqual(bot.sent, [])
        self.assertEqual(bot.rich_sent, ['i would say this'])
        self.assertEqual(
            ctx.conversation.messages[-1][CUSTOM]['telegram_message_ids'],
            [555])

    def test_impersonated_turn_is_tagged_and_scoped_locally(self):
        bot = _Bot()
        conversation = Conversation([
            {'role': 'user', 'content': 'hello'},
        ])
        ctx = mock_ctx(user_id=5, bot=bot, conversation=conversation)
        asyncio.run(_run_command(ctx))
        last = ctx.conversation.messages[-1]
        self.assertEqual(last.message_type,
                         ConversationMessageType.IMPERSONATION)
        # The tag lives in custom metadata, so the endpoint never sees it.
        self.assertNotIn(CUSTOM, ctx.conversation.to_openai()[-1])

    def test_clone_is_isolated_from_the_real_history(self):
        bot = _Bot()
        conversation = Conversation([
            {'role': 'user', 'content': 'hello'},
        ])
        conversation.add_assistant('hi!', telegram_message_ids=[77])
        ctx = mock_ctx(user_id=5, bot=bot, conversation=conversation)
        captured = {}

        async def fake_run_llm(context, **kwargs):
            # run_llm mutates the conversation it is handed during a tool
            # loop; the real history must not feel any of it.
            captured.update(kwargs)
            kwargs['conversation'].add_assistant('pollution')
            return 'as the user would'
        with mock.patch.object(conversation_handlers, 'run_llm',
                               fake_run_llm):
            asyncio.run(impersonate_command(ctx))
        clone = captured['conversation']
        self.assertIsNot(clone, ctx.conversation)
        # New identity: the clone is a separate session, like subagents.
        self.assertNotEqual(clone.id, ctx.conversation.id)
        self.assertIn('pollution', [m.get('content') for m in clone])
        self.assertNotIn('pollution',
                         [m.get('content') for m in ctx.conversation])

    def test_impersonation_prompt_added_as_final_user_turn(self):
        bot = _Bot()
        conversation = Conversation([
            {'role': 'user', 'content': 'hello'},
        ])
        ctx = mock_ctx(user_id=5, bot=bot, conversation=conversation)
        captured = {}

        async def fake_run_llm(context, **kwargs):
            captured.update(kwargs)
            return 'as the user would'
        with mock.patch.object(conversation_handlers, 'run_llm',
                               fake_run_llm):
            asyncio.run(impersonate_command(ctx))
        clone = captured['conversation']
        self.assertEqual(_roles(clone), ['user', 'user'])
        self.assertIn('impersonating the user', clone.messages[-1]['content'])

    def test_regenerate_of_impersonation_redoes_the_impersonation(self):
        bot = _Bot()
        conversation = Conversation([
            {'role': 'user', 'content': 'hello'},
        ])
        conversation.add_assistant('hi!', telegram_message_ids=[77])
        conversation.add_user(
            'as the user would', telegram_message_ids=[99],
            message_type=ConversationMessageType.IMPERSONATION)
        ctx = mock_ctx(user_id=5, bot=bot, conversation=conversation)
        with mock.patch.object(conversation_handlers, 'run_llm',
                               mock.AsyncMock(return_value='round two')):
            asyncio.run(regenerate_command(ctx))
        # The removed impersonation turn's message was deleted from
        # Telegram; the earlier assistant turn's message stays.
        self.assertEqual(bot.deleted, [(5, 99)])
        # ...and the impersonation flow ran again, replacing the turn.
        self.assertEqual(_roles(ctx.conversation),
                         ['user', 'assistant', 'user'])
        self.assertEqual(ctx.conversation.messages[-1]['content'],
                         'round two')
        self.assertEqual(ctx.conversation.messages[-1].message_type,
                         ConversationMessageType.IMPERSONATION)
        self.assertEqual(bot.rich_sent, ['round two'])

    def test_no_model_output_reports_failure(self):
        bot = _Bot()
        conversation = Conversation([
            {'role': 'user', 'content': 'hello'},
        ])
        ctx = mock_ctx(user_id=5, bot=bot, conversation=conversation)
        asyncio.run(_run_command(
            ctx, canned='Problem getting response from model.'))
        self.assertEqual(bot.sent,
                         ['Impersonation failed: the model produced '
                          'nothing usable.'])
        self.assertEqual(_roles(ctx.conversation), ['user'])

    def test_unbusy_check_reruns_nothing_when_busy(self):
        bot = _Bot()
        conversation = Conversation([
            {'role': 'user', 'content': 'hello'},
        ])
        ctx = mock_ctx(user_id=5, bot=bot, conversation=conversation)
        # A live generation task blocks the command. Plain stub: only
        # done() is consulted.
        ctx.generation_task = SimpleNamespace(  # type: ignore[assignment]
            done=lambda: False)
        asyncio.run(_run_command(ctx))
        self.assertEqual(bot.sent,
                         ['A response is currently being generated; use '
                          '/stop first.'])
        self.assertEqual(_roles(ctx.conversation), ['user'])


if __name__ == '__main__':
    unittest.main()
