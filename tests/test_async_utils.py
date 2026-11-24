"""atomic: cancellation cannot split a Telegram post from its commit."""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from async_utils import atomic, atomic_call


class TestAtomicDecorator(unittest.IsolatedAsyncioTestCase):

    async def test_decorator_runs_the_transaction_and_returns_its_result(self):
        calls: list[str] = []

        @atomic
        async def commit(value: str) -> str:
            calls.append('send')
            return f'recorded:{value}'

        self.assertEqual(await commit('id'), 'recorded:id')
        self.assertEqual(calls, ['send'])

    async def test_decorator_cancellation_finishes_the_transaction(self):
        started = asyncio.Event()
        release = asyncio.Event()
        committed: list[str] = []

        @atomic
        async def commit(value: str) -> None:
            started.set()
            await release.wait()
            committed.append(value)

        task = asyncio.create_task(commit('id'))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(committed, ['id'])

    async def test_decorator_transaction_failure_propagates(self):
        @atomic
        async def commit() -> None:
            raise ValueError('commit failed')

        with self.assertRaises(ValueError):
            await commit()


class TestAtomicCall(unittest.IsolatedAsyncioTestCase):

    async def test_runs_action_and_callback_in_order(self):
        calls: list[str] = []

        async def action():
            calls.append('action')
            return 'id'

        async def callback(result):
            calls.append(f'callback:{result}')

        await atomic_call(action, callback)
        self.assertEqual(calls, ['action', 'callback:id'])

    async def test_sync_callback_commits_the_result(self):
        committed: list[str] = []

        async def action():
            return 555

        await atomic_call(action, lambda message_id: committed.append(
            str(message_id)))
        self.assertEqual(committed, ['555'])

    async def test_cancellation_during_action_finishes_the_commit(self):
        # The action models Telegram accepting the post but not yet
        # returning its id; /stop lands in that window. The shielded
        # transaction must finish, then the cancellation re-raises.
        accepted = asyncio.Event()
        release = asyncio.Event()
        committed: list[int] = []

        async def action():
            accepted.set()
            await release.wait()
            return 555

        async def callback(message_id):
            committed.append(message_id)

        task = asyncio.create_task(atomic_call(action, callback))
        await accepted.wait()
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(committed, [555])

    async def test_cancellation_during_callback_finishes_the_commit(self):
        called = asyncio.Event()
        release = asyncio.Event()
        committed: list[int] = []

        async def action():
            return 555

        async def callback(message_id):
            called.set()
            await release.wait()
            committed.append(message_id)

        task = asyncio.create_task(atomic_call(action, callback))
        await called.wait()
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(committed, [555])

    async def test_commit_failure_is_not_hidden_by_cancellation(self):
        # If the callback itself fails while a cancellation is already
        # unwinding, the caller should see that concrete failure instead
        # of a bare CancelledError.
        release = asyncio.Event()

        async def action():
            return 555

        async def callback(message_id):
            await release.wait()
            raise ValueError('history commit failed')

        task = asyncio.create_task(atomic_call(action, callback))
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with self.assertRaises(ValueError):
            await task


if __name__ == '__main__':
    unittest.main()
