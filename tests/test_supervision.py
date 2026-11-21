"""run_tracked: the supervision every command-driven background job shares.

The three semantics that must never drift between /undo, /regenerate,
/impersonate and /compact:
- /stop cancels the tracked (inner) task, posts "X stopped.", and the
  supervisor returns None.
- When the job itself ends cancelled (bot shutdown) the CancelledError
  re-raises with no message.
- ctx.generation_task is cleared on every path.
"""

import asyncio
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from handlers.supervision import run_tracked
from mock_context import mock_ctx


class _Bot:
    """Captures sends, like the stubs in the command tests."""

    def __init__(self):
        self.sent: list = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append(text)
        return SimpleNamespace(message_id=1042)


async def _sleepy_work(started: asyncio.Event) -> str:
    started.set()
    await asyncio.sleep(60)
    return 'finished'


async def _started_sleepy_factory(ctx, started: asyncio.Event):
    await run_tracked(ctx, lambda: _sleepy_work(started),
                      stopped_message='Undo stopped.')


class TestRunTracked(unittest.IsolatedAsyncioTestCase):

    async def test_returns_the_result_and_clears_tracking(self):
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot)

        async def work() -> str:
            return 'finished'

        self.assertEqual(await run_tracked(ctx, work), 'finished')
        self.assertIsNone(ctx.generation_task)
        self.assertEqual(bot.sent, [])

    async def test_tracked_task_is_the_inner_task_not_the_supervisor(self):
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot)
        started = asyncio.Event()
        supervisor = asyncio.create_task(_started_sleepy_factory(ctx, started))
        await started.wait()
        # /stop acts on the model-facing task, never on the supervisor the
        # queue worker is awaiting.
        self.assertIsNotNone(ctx.generation_task)
        inner = ctx.generation_task
        assert inner is not None
        self.assertIsNot(inner, supervisor)
        inner.cancel()
        self.assertIsNone(await supervisor)
        self.assertEqual(bot.sent, ['Undo stopped.'])

    async def test_stop_posts_the_stopped_message_and_returns_none(self):
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot)
        started = asyncio.Event()
        stopped_hook = StoppedHook()
        supervisor = asyncio.create_task(run_tracked(
            ctx, lambda: _sleepy_work(started),
            stopped_message='Undo stopped.',
            on_stopped=stopped_hook.record))
        await started.wait()
        inner = ctx.generation_task
        assert inner is not None

        inner.cancel()
        self.assertIsNone(await supervisor)

        self.assertEqual(bot.sent, ['Undo stopped.'])
        self.assertEqual(stopped_hook.calls, 1)
        self.assertIsNone(ctx.generation_task)
        self.assertTrue(inner.cancelled())

    async def test_outer_cancel_reraises_without_a_message(self):
        # The work completed, but the supervisor was cancelled in the
        # window before it resumed on the result (a /stop racing
        # completion, as at bot shutdown): the CancelledError must
        # re-raise and nothing may be posted.
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot)
        loop = asyncio.get_running_loop()

        async def work() -> str:
            # Cancel the supervisor after this coroutine is done but
            # before the supervisor wakes up on the result.
            loop.call_soon(supervisor.cancel)
            return 'finished'

        supervisor = asyncio.create_task(
            run_tracked(ctx, work, stopped_message='Undo stopped.'))
        with self.assertRaises(asyncio.CancelledError):
            await supervisor
        self.assertEqual(bot.sent, [])
        self.assertIsNone(ctx.generation_task)

    async def test_cancelling_the_supervisor_before_it_runs_posts_nothing(self):
        # Bot shutdown cancelling the queued job before it starts: the
        # CancelledError surfaces and no stopped message goes out.
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot)
        started = asyncio.Event()
        supervisor = asyncio.create_task(run_tracked(
            ctx, lambda: _sleepy_work(started),
            stopped_message='Undo stopped.'))
        supervisor.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await supervisor
        self.assertEqual(bot.sent, [])
        self.assertIsNone(ctx.generation_task)
        # The inner task was created before the cancel landed; take it
        # down so the loop shuts down cleanly.
        inner = ctx.generation_task
        if inner is not None:
            inner.cancel()
            await asyncio.gather(inner, return_exceptions=True)

    async def test_tracking_is_cleared_when_the_work_fails(self):
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot)

        async def boom() -> None:
            raise ValueError('work failed')

        with self.assertRaises(ValueError):
            await run_tracked(ctx, boom, stopped_message='Undo stopped.')
        self.assertIsNone(ctx.generation_task)
        self.assertEqual(bot.sent, [])


class StoppedHook:
    """A callable-async record of on_stopped invocations."""

    def __init__(self):
        self.calls = 0

    async def record(self) -> None:
        self.calls += 1


if __name__ == '__main__':
    unittest.main()
