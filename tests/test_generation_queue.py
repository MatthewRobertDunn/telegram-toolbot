import asyncio
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from conversation import CUSTOM, Conversation
from handlers import generation_queue as gq
from mock_context import mock_ctx


class _Bot:
    def __init__(self):
        self.sent: list = []

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

        asyncio.run(gq.submit_task(ctx, job_a()))
        asyncio.run(gq.submit_task(ctx, job_b()))
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

    def test_stopped_generation_records_nothing(self):
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot, conversation=[])
        cancelled = mock.AsyncMock(return_value='never seen')
        cancelled.cancelled.return_value = True
        cancelled.close.return_value = None
        with mock.patch.object(gq, 'run_llm', cancelled):
            with mock.patch.object(gq, 'run_with_typing',
                                   _cancel_inner):
                asyncio.run(gq.generate_response(ctx, 'hello'))
        roles = [m['role'] for m in ctx.conversation.messages
                 if m['role'] != 'system']
        self.assertEqual(roles, ['user'])
        self.assertEqual(bot.sent, [])
        self.assertIsNone(ctx.generation_task)

    def test_worker_survives_a_failing_job(self):
        ctx = mock_ctx(user_id=5, bot=_Bot())
        order: list = []

        async def boom():
            raise RuntimeError('boom')

        async def after():
            order.append('ran')

        asyncio.run(gq.submit_task(ctx, boom()))
        asyncio.run(gq.submit_task(ctx, after()))
        self.assertEqual(order, ['ran'])


async def _cancel_inner(ctx, coro):
    """Stand-in for run_with_typing that leaves the tracked task cancelled."""
    task = asyncio.current_task()
    try:
        await coro
    except asyncio.CancelledError:
        pass
    inner = ctx.generation_task
    inner.cancelled = lambda: True  # type: ignore[method-assign]
    raise asyncio.CancelledError


class TestStopCommand(unittest.TestCase):

    def test_stop_drops_queued_items_and_reports(self):
        from handlers.command_handlers import stop_command
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot)
        items = [gq.WorkItem(ctx=ctx, job=asyncio.sleep(0)) for _ in range(2)]
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


if __name__ == '__main__':
    unittest.main()
