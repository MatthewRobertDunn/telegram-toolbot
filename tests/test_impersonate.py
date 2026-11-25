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
from handlers import generation_queue as generation_queue_mod
from handlers.conversation_handlers import (impersonate_command,
                                            regenerate_command,
                                            undo_command)
from handlers.impersonate_prompt import IMPERSONATE_PROMPT
from mock_context import mock_ctx


def setUpModule():
    # The worker persists every queued job; keep unit tests hermetic by
    # stubbing the save (persistence itself is covered in
    # test_conversation_persistence.py).
    patcher = mock.patch.object(generation_queue_mod, 'save_conversation',
                                mock.AsyncMock())
    patcher.start()
    global _save_patcher
    _save_patcher = patcher


def tearDownModule():
    _save_patcher.stop()


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
        # Every queued item sends typing; keep it quiet.
        self.send_chat_action = mock.AsyncMock()

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

    def test_impersonation_is_added_as_a_tagged_user_turn(self):
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
        last = ctx.conversation.messages[-1]
        self.assertEqual(last['content'], 'i would say this')
        # Telegram got the turn as rich markdown, and the posted id is
        # attached to the message.
        self.assertEqual(bot.sent, [])
        self.assertEqual(bot.rich_sent, ['i would say this'])
        self.assertEqual(last[CUSTOM]['telegram_message_ids'], [555])
        # The turn is tagged as an impersonation...
        self.assertEqual(last.message_type,
                         ConversationMessageType.IMPERSONATION)
        # ...but the tag lives in custom metadata, so the endpoint
        # never sees it.
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
        # The appended turn is exactly the impersonation prompt.
        self.assertEqual(clone.messages[-1]['content'], IMPERSONATE_PROMPT)

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
        self.assertEqual(_roles(ctx.conversation),
                         ['user', 'assistant', 'user'])
        self.assertEqual(ctx.conversation.messages[-1]['content'],
                         'round two')
        self.assertEqual(ctx.conversation.messages[-1].message_type,
                         ConversationMessageType.IMPERSONATION)
        self.assertEqual(bot.rich_sent, ['round two'])

    def test_undo_after_impersonation_preserves_thinking_exchange(self):
        async def scenario(keep_thinking):
            bot = _Bot()
            conversation = Conversation()
            conversation.add_user('hello', telegram_message_ids=[11])
            if keep_thinking:
                conversation.add_thinking(telegram_message_ids=[22])
            conversation.add_assistant('hi!', telegram_message_ids=[33])
            previous = list(conversation.messages)
            ctx = mock_ctx(
                user_id=5, bot=bot, conversation=conversation,
                settings=SimpleNamespace(system_prompt='sys', user_profile=None,
                                         thinking_keep=keep_thinking))
            ctx.messaging.run_with_typing = lambda coro: coro

            await _run_command(ctx)
            await undo_command(ctx)

            # The impersonation is its own user turn. Undoing it must not
            # walk into the preceding answer or its kept thinking block.
            self.assertEqual(_roles(ctx.conversation),
                             ['user', 'assistant', 'assistant']
                             if keep_thinking else ['user', 'assistant'])
            self.assertEqual(ctx.conversation.messages[1:], previous)
            self.assertEqual(bot.deleted, [(5, 555)])

        for keep_thinking in (False, True):
            with self.subTest(thinking_keep=keep_thinking):
                asyncio.run(scenario(keep_thinking))

    def test_stop_during_post_commits_impersonation_for_undo(self):
        async def scenario():
            from handlers.message_handlers import stop_generation

            bot = _Bot()
            conversation = Conversation()
            conversation.add_user('hello', telegram_message_ids=[11])
            conversation.add_assistant('hi!', telegram_message_ids=[77])
            previous = list(conversation.messages)
            ctx = mock_ctx(user_id=5, bot=bot, conversation=conversation)
            ctx.messaging.run_with_typing = lambda coro: coro
            posting = asyncio.Event()
            release_post = asyncio.Event()

            async def delayed_send(text):
                # Telegram has accepted the post, but its response has not
                # reached the handler yet. /stop lands in this window.
                posting.set()
                await release_post.wait()
                return [555]

            ctx.messaging.send_rich_message = delayed_send
            command = asyncio.create_task(_run_command(ctx))
            await posting.wait()
            self.assertTrue(stop_generation(ctx))
            await asyncio.sleep(0)
            release_post.set()
            await asyncio.wait_for(command, 5)

            # Once the model output exists, finishing the post and history
            # commit is atomic with respect to /stop. Otherwise the visible
            # post has no row for a following /undo to delete.
            self.assertEqual(ctx.conversation.messages[-1]['content'],
                             'i would say this')
            self.assertEqual(
                ctx.conversation.messages[-1][CUSTOM]['telegram_message_ids'],
                [555])
            # atomic completes the commit, then lets /stop's cancellation
            # reach the queue supervisor.
            self.assertEqual(bot.sent, ['Impersonation stopped.'])

            await undo_command(ctx)
            self.assertEqual(ctx.conversation.messages[1:], previous)
            self.assertEqual(bot.deleted, [(5, 555)])

        asyncio.run(scenario())

    def test_impersonation_rechecks_content_when_its_queue_item_runs(self):
        async def scenario():
            bot = _Bot()
            ctx = mock_ctx(
                user_id=5, bot=bot,
                conversation=Conversation([
                    {'role': 'user', 'content': 'hello'}]))
            ctx.messaging.run_with_typing = lambda coro: coro
            clearing = asyncio.Event()
            release_clear = asyncio.Event()

            async def clear_history():
                clearing.set()
                await release_clear.wait()
                ctx.conversation.set_messages([])

            # The command's pre-queue check sees "hello", but the undo-like
            # job ahead of it removes that content before impersonation runs.
            clearer = asyncio.create_task(
                generation_queue_mod.submit_task(ctx, clear_history))
            await clearing.wait()
            with mock.patch.object(
                    conversation_handlers, 'run_llm',
                    mock.AsyncMock(return_value='must not be generated')) as llm:
                impersonation = asyncio.create_task(impersonate_command(ctx))
                while generation_queue_mod.generation_queue(ctx).qsize() < 1:
                    await asyncio.sleep(0)
                release_clear.set()
                await clearer
                await impersonation

            llm.assert_not_awaited()
            self.assertEqual(ctx.conversation.messages, [])
            self.assertEqual(bot.sent, ['Nothing to impersonate yet.'])

        asyncio.run(scenario())

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


if __name__ == '__main__':
    unittest.main()
