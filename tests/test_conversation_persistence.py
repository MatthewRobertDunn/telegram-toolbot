import asyncio
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import persistence.settings_store as settings_store
from persistence.settings_store import (CONVERSATION_KEY, load_conversation,
                            save_conversation)
from persistence.settings_store import SettingsStore
from conversation import Conversation, MAX_IMAGES
from mock_context import mock_ctx


def _temp_store(path):
    return SettingsStore(path)


def _image_parts(messages):
    """Every inline image part in the message list, oldest first."""
    return [part
            for message in messages
            if isinstance(message.get('content'), list)
            for part in message['content']
            if isinstance(part, dict) and part.get('type') == 'image_url']


def _overcap_messages(count):
    """Raw user turns carrying `count` inline images, one per message."""
    return [{'role': 'user', 'content': [
        {'type': 'text', 'text': f'msg {i}'},
        {'type': 'image_url',
         'image_url': {'url': f'data:image/png;base64,{i}'}}]}
        for i in range(count)]


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
        await save_conversation(5, None, conversation)

        loaded = await load_conversation(5, None)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.id, conversation.id)
        self.assertEqual(loaded, conversation.messages)
        self.assertEqual(
            loaded.messages[-1]['custom']['telegram_message_ids'], [77])

    async def test_load_missing_or_corrupt_returns_none(self):
        for label, user_id, thread_id in [('missing row', 999, None),
                                          ('missing thread row', 999, 2)]:
            with self.subTest(case=label):
                self.assertIsNone(await load_conversation(user_id, thread_id))

        store = _temp_store(self.path)
        try:
            await store.set(7, CONVERSATION_KEY, 'not json{{{', None)
        finally:
            await store.close()
        with self.subTest(case='corrupt payload'):
            self.assertIsNone(await load_conversation(7, None))

    async def test_overwrite_keeps_latest(self):
        first = Conversation([{'role': 'user', 'content': 'one'}])
        await save_conversation(5, None, first)
        second = Conversation([{'role': 'user', 'content': 'two'}])
        await save_conversation(5, None, second)
        loaded = await load_conversation(5, None)
        assert loaded is not None
        self.assertEqual(loaded.id, second.id)
        self.assertEqual(loaded.messages[-1]['content'], 'two')

    async def test_worker_persists_conversation_after_queued_job(self):
        from handlers import generation_queue as gq
        ctx = mock_ctx(user_id=5, conversation=[],
                       bot=SimpleNamespace(
                           send_chat_action=mock.AsyncMock()))

        async def job():
            ctx.conversation.add_user('hello')

        with mock.patch.object(gq, 'save_conversation',
                               wraps=save_conversation) as spy:
            await gq.submit_task(ctx, job)
            spy.assert_awaited_once()
            args, _ = spy.call_args
            self.assertEqual(args[0], 5)
            self.assertIsNone(args[1])
            self.assertEqual(args[2].messages[-1]['content'], 'hello')

        loaded = await load_conversation(5, None)
        assert loaded is not None
        self.assertEqual(loaded.messages[-1]['content'], 'hello')

    async def test_init_conversation_loads_persisted_history(self):
        from handlers.middleware import init_conversation
        conversation = Conversation([{'role': 'user', 'content': 'hello'}])
        await save_conversation(5, None, conversation)

        settings = SimpleNamespace(system_prompt='sys', user_profile=None)
        ctx = mock_ctx(user_id=5, settings=settings)
        await init_conversation(ctx)

        # Persisted id survives; the system head is re-derived from the
        # current settings on top of the loaded history.
        self.assertEqual(ctx.conversation.id, conversation.id)
        self.assertEqual(ctx.conversation.messages[0]['role'], 'system')
        self.assertEqual(ctx.conversation.messages[0]['content'], 'sys')
        self.assertEqual(ctx.conversation.messages[-1]['content'], 'hello')

    async def test_threads_persist_independently(self):
        plain = Conversation([{'role': 'user', 'content': 'plain'}])
        topic = Conversation([{'role': 'user', 'content': 'topic'}])
        await save_conversation(5, None, plain)
        await save_conversation(5, 2, topic)

        loaded_plain = await load_conversation(5, None)
        loaded_topic = await load_conversation(5, 2)
        assert loaded_plain is not None and loaded_topic is not None
        self.assertEqual(loaded_plain.messages[-1]['content'], 'plain')
        self.assertEqual(loaded_topic.messages[-1]['content'], 'topic')

    async def test_save_prunes_old_inline_images(self):
        # Built from raw messages: add_image would itself prune to the
        # cap, and the point here is an over-cap history reaching save.
        count = MAX_IMAGES + 2
        conversation = Conversation(_overcap_messages(count))
        self.assertEqual(len(_image_parts(conversation.messages)), count)

        await save_conversation(5, None, conversation)

        # The save never touches the live conversation.
        self.assertEqual(len(_image_parts(conversation.messages)), count)

        loaded = await load_conversation(5, None)
        assert loaded is not None
        parts = _image_parts(loaded.messages)
        # Only the newest images survive the round trip; the row is
        # rewritten whole on every queued turn, so the payload stays
        # bounded.
        self.assertEqual(len(parts), MAX_IMAGES)
        newest = _image_parts(conversation.messages)[-1]
        self.assertEqual(parts[-1], newest)
        self.assertIn('[image omitted]', json.dumps(loaded.messages))

    async def test_load_prunes_unbounded_stored_images(self):
        # A stored blob carrying more images than the cap (written by an
        # older version, say) must not resurrect them on load.
        count = MAX_IMAGES + 4
        messages = _overcap_messages(count)
        store = _temp_store(self.path)
        try:
            await store.set(5, CONVERSATION_KEY,
                            json.dumps({'id': 'abc', 'messages': messages}),
                            None)
        finally:
            await store.close()

        loaded = await load_conversation(5, None)
        assert loaded is not None
        self.assertEqual(loaded.id, 'abc')
        self.assertEqual(len(loaded.messages), count)
        self.assertEqual(len(_image_parts(loaded.messages)), MAX_IMAGES)
        self.assertIn('[image omitted]', json.dumps(loaded.messages))
        # Message dict shapes are otherwise intact.
        self.assertEqual(loaded.messages[-1]['role'], 'user')
        self.assertEqual(loaded.messages[-1]['content'][0],
                         {'type': 'text', 'text': f'msg {count - 1}'})
        self.assertEqual(loaded.messages[0]['content'][0],
                         {'type': 'text', 'text': 'msg 0'})
        self.assertEqual(loaded.messages[0]['content'][1],
                         {'type': 'text', 'text': '[image omitted]'})


if __name__ == '__main__':
    unittest.main()
