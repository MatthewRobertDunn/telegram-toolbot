"""Queue supervision: every submitted job gets the same /stop handling.

The semantics that must never drift between a reply, /undo, /regenerate,
/impersonate and /compact:
- /stop cancels the tracked (inner) task, posts the item's stopped
  message, and the submitter just resumes.
- When the worker itself is cancelled (bot shutdown) the tracked work is
  taken down too and nothing is posted.
- ctx.generation_task is cleared on every path.
"""

import asyncio
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from handlers import generation_queue as gq
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
    """Captures sends; send_chat_action is mocked so typing stays quiet."""

    def __init__(self):
        self.sent: list = []
        self.send_chat_action = mock.AsyncMock()

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append(text)
        return SimpleNamespace(message_id=1042)


async def _sleepy_job(started: asyncio.Event) -> str:
    started.set()
    await asyncio.sleep(60)
    return 'finished'


class TestQueueSupervision(unittest.IsolatedAsyncioTestCase):

    async def test_running_job_is_tracked_as_the_inner_task(self):
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot)
        started = asyncio.Event()
        driver = asyncio.ensure_future(
            gq.submit_task(ctx, lambda: _sleepy_job(started)))
        await started.wait()

        # /stop acts on the model-facing task, never on the queue worker
        # awaiting it.
        tracked = ctx.generation_task
        self.assertIsNotNone(tracked)
        assert tracked is not None
        self.assertIsNot(tracked, ctx.generation_worker)

        tracked.cancel()
        await asyncio.wait_for(driver, 5)
        self.assertIsNone(ctx.generation_task)
        self.assertEqual(bot.sent, [], 'no stopped message without one')

    async def test_stop_posts_the_stopped_message(self):
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot)
        started = asyncio.Event()
        driver = asyncio.ensure_future(gq.submit_task(
            ctx, lambda: _sleepy_job(started),
            stopped_message='Undo stopped.'))
        await started.wait()

        tracked = ctx.generation_task
        assert tracked is not None
        tracked.cancel()
        await asyncio.wait_for(driver, 5)

        self.assertEqual(bot.sent, ['Undo stopped.'])
        self.assertIsNone(ctx.generation_task)

    async def test_worker_cancel_takes_the_work_down_without_a_message(self):
        # Bot shutdown cancels the worker: the tracked work is taken down
        # too, the submitter wakes, and no stopped message goes out.
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot)
        started = asyncio.Event()
        driver = asyncio.ensure_future(gq.submit_task(
            ctx, lambda: _sleepy_job(started),
            stopped_message='Undo stopped.'))
        await started.wait()

        worker = ctx.generation_worker
        assert worker is not None
        worker.cancel()
        await asyncio.wait_for(driver, 5)

        self.assertEqual(bot.sent, [])
        self.assertIsNone(ctx.generation_task)

    async def test_job_failure_is_reraised_and_clears_tracking(self):
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot)

        async def boom() -> None:
            raise ValueError('work failed')

        with self.assertRaises(ValueError):
            await gq.submit_task(ctx, boom,
                                 stopped_message='Undo stopped.')
        self.assertIsNone(ctx.generation_task)
        self.assertEqual(bot.sent, [])

    async def test_a_fresh_job_wraps_the_thunk_exactly_once(self):
        # The job is a zero-arg callable: the coroutine it returns exists
        # only because the worker called it, exactly once - under the
        # typing wrapper like anywhere else.
        ctx = mock_ctx(user_id=5, bot=_Bot())
        calls: list = []

        def thunk():
            calls.append('built')

            async def job():
                pass

            return job()

        await gq.submit_task(ctx, thunk)
        self.assertEqual(calls, ['built'])


if __name__ == '__main__':
    unittest.main()
