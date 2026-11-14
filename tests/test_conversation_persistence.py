import asyncio
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import settings_store
from settings_store import (CONVERSATION_KEY, load_conversation,
                            save_conversation)
from context_store import AsyncSQLiteStore, SETTINGS_SCHEMA
from conversation import Conversation
from mock_context import mock_ctx


def _temp_store(path):
    return AsyncSQLiteStore(path, 'settings', SETTINGS_SCHEMA)


class ConversationPersistenceTest(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        fd, self.path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        store = _temp_store(self.path)
        await store.create_schema()
        await store.close()
        patcher = mock.patch.object(settings_store, 'create_settings_store',
                                    lambda: _temp_store(self.path))
        patcher.start()
        self.addCleanup(patcher.stop)

    async def asyncTearDown(self):
        os.unlink(self.path)

    async def test_round_trip_preserves_id_messages_and_telegram_ids(self):
        conversation = Conversation([{'role': 'user', 'content': 'hello'}])
        conversation.add_assistant('hi', telegram_message_ids=[77])
        await save_conversation(5, conversation)

        loaded = await load_conversation(5)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.id, conversation.id)
        self.assertEqual(loaded, conversation.messages)
        self.assertEqual(
            loaded.messages[-1]['custom']['telegram_message_ids'], [77])

    async def test_load_missing_returns_none(self):
        self.assertIsNone(await load_conversation(999))

    async def test_load_corrupt_returns_none(self):
        store = _temp_store(self.path)
        await store.set(7, CONVERSATION_KEY, 'not json{{{')
        await store.close()
        self.assertIsNone(await load_conversation(7))

    async def test_overwrite_keeps_latest(self):
        first = Conversation([{'role': 'user', 'content': 'one'}])
        await save_conversation(5, first)
        second = Conversation([{'role': 'user', 'content': 'two'}])
        await save_conversation(5, second)
        loaded = await load_conversation(5)
        assert loaded is not None
        self.assertEqual(loaded.id, second.id)
        self.assertEqual(loaded.messages[-1]['content'], 'two')

    async def test_worker_persists_conversation_after_queued_job(self):
        from handlers import generation_queue as gq
        ctx = mock_ctx(user_id=5, conversation=[])

        async def job():
            ctx.conversation.add_user('hello')

        with mock.patch.object(gq, 'save_conversation',
                               wraps=save_conversation) as spy:
            await gq.submit_task(ctx, job())
            spy.assert_awaited_once()
            args, _ = spy.call_args
            self.assertEqual(args[0], 5)
            self.assertEqual(args[1].messages[-1]['content'], 'hello')

        loaded = await load_conversation(5)
        assert loaded is not None
        self.assertEqual(loaded.messages[-1]['content'], 'hello')

    async def test_init_conversation_loads_persisted_history(self):
        from handlers.middleware import init_conversation
        conversation = Conversation([{'role': 'user', 'content': 'hello'}])
        await save_conversation(5, conversation)

        settings = SimpleNamespace(system_prompt='sys', user_profile=None)
        ctx = mock_ctx(user_id=5, settings=settings)
        await init_conversation(ctx)

        # Persisted id survives; the system head is re-derived from the
        # current settings on top of the loaded history.
        self.assertEqual(ctx.conversation.id, conversation.id)
        self.assertEqual(ctx.conversation.messages[0]['role'], 'system')
        self.assertEqual(ctx.conversation.messages[0]['content'], 'sys')
        self.assertEqual(ctx.conversation.messages[-1]['content'], 'hello')


if __name__ == '__main__':
    unittest.main()
