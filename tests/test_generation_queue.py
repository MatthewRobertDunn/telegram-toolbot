import asyncio
import os
import sys
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from conversation import CUSTOM, Conversation
from agent_inbox import SubagentInbox, SubagentMessage, SubagentStatus, \
    get_main_inbox
from handlers import generation_queue as gq
from messaging import Messaging, _split_at_linebreak
from mock_context import mock_ctx


def setUpModule():
    # The worker persists every queued job; keep unit tests hermetic by
    # stubbing the save (persistence itself is covered in
    # test_conversation_persistence.py).
    patcher = mock.patch.object(gq, 'save_conversation',
                                mock.AsyncMock())
    patcher.start()
    global _save_patcher
    _save_patcher = patcher


def tearDownModule():
    _save_patcher.stop()


class _Bot:
    def __init__(self):
        self.sent: list = []
        self.send_chat_action: Any = mock.AsyncMock()
        self.send_photo: Any = mock.AsyncMock()
        self.send_document: Any = mock.AsyncMock()

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append(text)
        return SimpleNamespace(message_id=7)

    async def _post(self, method, data):
        return {'message_id': 9}


class TestSubmitTask(unittest.TestCase):

    def test_jobs_run_one_at_a_time_in_order(self):
        ctx = mock_ctx(user_id=5, bot=_Bot())
        order: list = []

        async def job_a():
            order.append('a')
            await asyncio.sleep(0)

        async def job_b():
            order.append('b')

        asyncio.run(gq.submit_task(ctx, job_a))
        asyncio.run(gq.submit_task(ctx, job_b))
        self.assertEqual(order, ['a', 'b'])
        # The transient worker cleaned up after itself.
        self.assertIsNone(ctx.generation_worker)

    def test_submit_generation_queues_a_reply_unit(self):
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot, conversation=[])
        with mock.patch.object(gq, 'run_llm',
                               mock.AsyncMock(return_value='answer')):
            asyncio.run(gq.submit_generation(ctx, 'hello',
                                             telegram_message_id=42))
        roles = [m['role'] for m in ctx.conversation.messages
                 if m['role'] != 'system']
        self.assertEqual(roles, ['user', 'assistant'])
        self.assertEqual(ctx.conversation.messages[-1]['content'], 'answer')
        self.assertEqual(ctx.conversation.messages[-1][CUSTOM],
                         {'telegram_message_ids': [9]})
        self.assertIsNone(ctx.generation_task)

    def test_a_finished_generation_is_not_dangling_thinking(self):
        # The item-end cleanup only ever removes an aborted reply's
        # trailing thinking; a completed reply's thinking has its answer
        # after it and must survive.
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot,
                       conversation=[{'role': 'user', 'content': 'hi'}])

        async def thinking_then_reply(*args, **kwargs):
            ctx.conversation.add_thinking(telegram_message_ids=[55])
            return 'answer'

        deleted: list = []
        ctx.messaging.delete_message = mock.AsyncMock(
            side_effect=lambda message_id: deleted.append(message_id))
        with mock.patch.object(gq, 'run_llm', thinking_then_reply):
            asyncio.run(gq.submit_generation(ctx, None))
        self.assertEqual(deleted, [])
        # The thinking block (an assistant-role message) kept both its
        # history entry and its Telegram post, and the reply followed it.
        roles = [m['role'] for m in ctx.conversation.messages
                 if m['role'] != 'system']
        self.assertEqual(roles, ['user', 'assistant', 'assistant'])
        self.assertEqual(ctx.conversation.messages[-1]['content'], 'answer')


class TestGenerateResponse(unittest.TestCase):

    def test_no_content_appends_no_user_turn(self):
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot, conversation=[])
        with mock.patch.object(gq, 'run_llm',
                               mock.AsyncMock(return_value='only reply')):
            asyncio.run(gq.generate_response(ctx))
        roles = [m['role'] for m in ctx.conversation.messages
                 if m['role'] != 'system']
        self.assertEqual(roles, ['assistant'])

    def test_stopped_generation_records_no_reply(self):
        async def main():
            bot = _Bot()
            ctx = mock_ctx(user_id=5, bot=bot, conversation=[])
            started = asyncio.Event()

            async def hang(*args, **kwargs):
                started.set()
                await asyncio.sleep(60)

            with mock.patch.object(gq, 'run_llm', hang):
                driver = asyncio.ensure_future(
                    gq.submit_generation(ctx, 'hello'))
                await started.wait()
                tracked = ctx.generation_task
                assert tracked is not None
                tracked.cancel()
                await asyncio.wait_for(driver, 5)
            roles = [m['role'] for m in ctx.conversation.messages
                     if m['role'] != 'system']
            self.assertEqual(roles, ['user'])
            self.assertEqual(bot.sent, [])
            self.assertIsNone(ctx.generation_task)

        asyncio.run(main())

    def test_worker_survives_a_failing_job(self):
        ctx = mock_ctx(user_id=5, bot=_Bot())
        order: list = []

        async def boom():
            raise RuntimeError('boom')

        async def after():
            order.append('ran')

        # The worker keeps going; the submitter of the failed job sees
        # the error.
        with self.assertRaises(RuntimeError):
            asyncio.run(gq.submit_task(ctx, boom))
        asyncio.run(gq.submit_task(ctx, after))
        self.assertEqual(order, ['ran'])

    def test_abandon_releases_a_dropped_item_without_running_it(self):
        ctx = mock_ctx(user_id=5, bot=_Bot())
        ran: list = []

        def thunk():
            ran.append('ran')

            async def job():
                pass

            return job()

        item = gq.WorkItem(ctx=ctx, job=thunk)
        item.abandon()
        self.assertTrue(item.done.is_set())
        self.assertEqual(ran, [], 'abandon must not run the job')

    def test_worker_cancellation_releases_still_queued_items(self):
        # The second drop path: the worker is cancelled while one job
        # runs and another waits. The leftover must be released so its
        # submitter wakes, and its thunk must never run.
        async def main():
            ctx = mock_ctx(user_id=5, bot=_Bot())
            queue = gq.generation_queue(ctx)
            running = asyncio.Event()

            async def first_job():
                running.set()
                await asyncio.sleep(60)

            ran: list = []

            def dropped_thunk():
                ran.append('built')

                async def job():
                    pass

                return job()

            driver = asyncio.ensure_future(gq.submit_task(ctx, first_job))
            await running.wait()
            queued = gq.WorkItem(ctx=ctx, job=dropped_thunk)
            queue.put_nowait(queued)

            worker = ctx.generation_worker
            assert worker is not None
            worker.cancel()
            await asyncio.wait_for(driver, 5)

            self.assertTrue(queued.done.is_set())
            self.assertEqual(ran, [], 'a queued thunk must not run')
            self.assertTrue(queue.empty())
            self.assertIsNone(ctx.generation_worker)

        asyncio.run(main())

    def test_stop_deletes_orphan_thinking(self):
        async def main():
            ctx = mock_ctx(user_id=5, bot=_Bot(),
                           conversation=[{'role': 'user', 'content': 'hi'}])
            deleted: list = []
            ctx.messaging.delete_message = mock.AsyncMock(
                side_effect=lambda message_id: deleted.append(message_id))

            async def thinking_then_hang(*args, **kwargs):
                ctx.conversation.add_thinking(
                    telegram_message_ids=[55])
                await asyncio.sleep(60)

            with mock.patch.object(gq, 'run_llm', thinking_then_hang):
                driver = asyncio.ensure_future(
                    gq.submit_generation(ctx, None))
                for _ in range(1000):
                    if ctx.generation_task is not None:
                        break
                    await asyncio.sleep(0.01)
                self.assertIsNotNone(ctx.generation_task)
                inner = ctx.generation_task
                assert inner is not None
                inner.cancel()
                await asyncio.wait_for(driver, 5)
            # The queue item's cleanup removed the aborted reply's
            # thinking from history and Telegram.
            self.assertEqual(deleted, [55])
            roles = [m['role'] for m in ctx.conversation.messages
                     if m['role'] != 'system']
            self.assertEqual(roles, ['user'])

        asyncio.run(main())


class TestStopCommand(unittest.TestCase):

    def test_stop_drops_queued_items_and_reports(self):
        from handlers.command_handlers import stop_command
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot)
        items = [gq.WorkItem(ctx=ctx, job=lambda: asyncio.sleep(0))
                 for _ in range(2)]
        for item in items:
            gq.generation_queue(ctx).put_nowait(item)

        asyncio.run(stop_command(ctx))

        self.assertTrue(gq.generation_queue(ctx).empty())
        self.assertTrue(all(item.done.is_set() for item in items))
        self.assertIn('Dropped 2 queued request', bot.sent[-1])

    def test_stop_with_idle_bot_says_nothing_running(self):
        from handlers.command_handlers import stop_command
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot)
        asyncio.run(stop_command(ctx))
        self.assertIn('No response', bot.sent[-1])


class TestMessagingTyping(unittest.IsolatedAsyncioTestCase):
    # run_with_typing must reap its typing task without swallowing what
    # the wrapped coroutine did (an exception, or a /stop cancellation).

    def _ctx(self):
        bot = _Bot()
        return mock_ctx(user_id=5, bot=bot)

    async def test_returns_the_coroutine_result(self):
        ctx = self._ctx()

        async def ok():
            # Yield once so the typing task gets its first send in.
            await asyncio.sleep(0)
            return 'value'

        self.assertEqual(await ctx.messaging.run_with_typing(ok()), 'value')
        bot = ctx.bot
        assert isinstance(bot, _Bot)
        assert isinstance(bot.send_chat_action, mock.AsyncMock)
        bot.send_chat_action.assert_awaited()

    async def test_wrapped_exception_propagates(self):
        ctx = self._ctx()

        async def boom():
            raise RuntimeError('inner')

        with self.assertRaises(RuntimeError):
            await ctx.messaging.run_with_typing(boom())
        # The typing task was reaped: nothing but this test task remains.
        leftover = [t for t in asyncio.all_tasks()
                    if t is not asyncio.current_task() and not t.done()]
        self.assertEqual(leftover, [])

    async def test_cancellation_propagates(self):
        ctx = self._ctx()

        async def hang():
            await asyncio.sleep(60)

        task = asyncio.ensure_future(ctx.messaging.run_with_typing(hang()))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        leftover = [t for t in asyncio.all_tasks()
                    if t is not asyncio.current_task() and not t.done()]
        self.assertEqual(leftover, [])


class TestMessagingReturnShapes(unittest.IsolatedAsyncioTestCase):
    # The chat-posting helpers all return the posted message ids, so the
    # caller can link them to the conversation turn (/undo, /retry).

    def _bot(self):
        bot = _Bot()
        photo = bot.send_photo
        document = bot.send_document
        assert isinstance(photo, mock.AsyncMock)
        assert isinstance(document, mock.AsyncMock)
        photo.return_value = SimpleNamespace(message_id=11)
        document.return_value = SimpleNamespace(message_id=12)
        return bot

    async def test_posting_helpers_return_message_ids(self):
        cases: list[tuple[str, tuple[Any, ...], list[int]]] = [
            ('send_preformatted_message', ('hello',), [7]),
            ('send_image', (b'png',), [11]),
            ('send_document', (b'pdf', 'f.pdf'), [12]),
        ]
        for name, args, expected in cases:
            with self.subTest(helper=name):
                ctx = mock_ctx(user_id=5, bot=self._bot())
                ids = await getattr(ctx.messaging, name)(*args)
                self.assertEqual(list(ids), expected)


class TestSplitAtLinebreak(unittest.TestCase):

    def test_degenerate_limit_makes_progress(self):
        # A limit <= 0 cannot make progress, so the text returns whole.
        self.assertEqual(_split_at_linebreak('abc', 0), ['abc'])
        self.assertEqual(_split_at_linebreak('abc', -5), ['abc'])

    def test_leading_newline_produces_no_empty_chunk(self):
        self.assertEqual(_split_at_linebreak('\nabc', 3), ['abc'])
        self.assertEqual(_split_at_linebreak('a\n\nb', 1), ['a', 'b'])

    def test_splits_without_newlines(self):
        self.assertEqual(_split_at_linebreak('abcdef', 4), ['abcd', 'ef'])

    def test_prefers_line_breaks(self):
        self.assertEqual(_split_at_linebreak('ab\ncdef', 4), ['ab', 'cdef'])

    def test_short_and_empty_text(self):
        self.assertEqual(_split_at_linebreak('ab', 10), ['ab'])
        self.assertEqual(_split_at_linebreak('', 10), [])


SETTINGS = SimpleNamespace(
    show_tool_output=False, model='m', temperature=1.0, top_p=0.95,
    thinking_keep=False)


class TestStopLeavesInboxConsistent(unittest.IsolatedAsyncioTestCase):
    # Drain ordering: the turn is committed to the conversation and the
    # message consumed from the inbox in one synchronous step, so a stop
    # can neither lose the message nor duplicate it.

    async def test_stop_after_drain_keeps_the_turn_answerable(self):
        from handlers import run_llm as run_llm_mod
        ctx = mock_ctx(user_id=5, bot=_Bot(), conversation=[],
                       settings=SETTINGS)
        conversation = ctx.conversation
        inbox = get_main_inbox(ctx)
        inbox.put_nowait(SubagentMessage(
            id='abc', state=SubagentStatus.FINISHED, description='T',
            result='the report'))
        # The drain (what run_llm does at the top of every iteration)...
        await run_llm_mod._drain_inbox(conversation, inbox)
        self.assertTrue(inbox.empty())
        # ...then a /stop cancels the run: the committed turn survives.
        users = [m for m in conversation.messages if m['role'] == 'user']
        self.assertEqual(len(users), 1)
        self.assertIn('the report', users[0]['content'])
        # And the next request (/retry) answers it - the turn is in the
        # history the model sees.
        with mock.patch.object(gq, 'run_llm',
                               mock.AsyncMock(return_value='ack')):
            await gq.generate_response(ctx)
        self.assertEqual(conversation.messages[-1]['content'], 'ack')

    async def test_stop_before_drain_keeps_mail_for_the_next_run(self):
        from handlers import run_llm as run_llm_mod
        ctx = mock_ctx(user_id=5, bot=_Bot(), conversation=[],
                       settings=SETTINGS)
        inbox = get_main_inbox(ctx)
        inbox.put_nowait(SubagentMessage(
            id='abc', state=SubagentStatus.FINISHED, description='T',
            result='queued report'))

        # /stop lands before the run even starts: the task is cancelled
        # before its first step, so no drain happened at all.
        task = asyncio.ensure_future(run_llm_mod.run_llm(ctx, inbox=inbox))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        # Not lost: the message stays queued for the next turn.
        self.assertFalse(inbox.empty())
        self.assertEqual([m for m in ctx.conversation.messages
                          if m['role'] == 'user'], [])
        with mock.patch.object(
                run_llm_mod, 'client',
                SimpleNamespace(chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=_create_ok_stream)))):
            result = await run_llm_mod.run_llm(ctx, inbox=inbox)
        self.assertEqual(result, 'ok')
        self.assertTrue(inbox.empty())
        self.assertIn('queued report',
                      [m['content'] for m in ctx.conversation.messages
                       if m['role'] == 'user'][0])

    async def test_stop_mid_run_after_drain_keeps_the_turn(self):
        # Cancellation during the completion (after the top-of-loop
        # drain): the turn stays committed, the inbox is empty, and the
        # next request answers the turn.
        from handlers import run_llm as run_llm_mod
        ctx = mock_ctx(user_id=5, bot=_Bot(), conversation=[],
                       settings=SETTINGS)
        inbox = get_main_inbox(ctx)
        inbox.put_nowait(SubagentMessage(
            id='abc', state=SubagentStatus.FINISHED, description='T',
            result='queued report'))

        async def cancelled_create(**kwargs):
            raise asyncio.CancelledError

        with mock.patch.object(run_llm_mod, 'client', SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=cancelled_create)))):
            with self.assertRaises(asyncio.CancelledError):
                await run_llm_mod.run_llm(ctx, inbox=inbox)
        self.assertTrue(inbox.empty())
        users = [m for m in ctx.conversation.messages if m['role'] == 'user']
        self.assertEqual(len(users), 1)
        self.assertIn('queued report', users[0]['content'])


def _create_ok_stream(**kwargs):
    async def create(**_kwargs):
        async def stream():
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(
                    content='ok', reasoning_content=None, tool_calls=None),
                finish_reason=None)])
        return stream()
    return create(**kwargs)


if __name__ == '__main__':
    unittest.main()
