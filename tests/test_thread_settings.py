"""Per-thread settings: isolation, template inheritance, dual save.

Settings changes (e.g. /model) affect only the current thread, while the
shared template is refreshed so newly created threads inherit them.
"""

import copy
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import settings_store
from settings_store import load_settings, save_settings
from context_store import AsyncSQLiteStore, SETTINGS_SCHEMA
from chat_update_context import create_chat_update_context
from handlers import middleware
from mock_context import mock_ctx
from user_settings import UserSettings


def _temp_store(path):
    return AsyncSQLiteStore(path, 'settings', SETTINGS_SCHEMA)


async def _settings_rows(path, user_id):
    """Raw (thread_id,) rows for the user's settings key, oldest first."""
    import sqlite3
    conn = sqlite3.connect(path)
    try:
        return conn.execute(
            "SELECT thread_id FROM settings "
            "WHERE user_id = ? AND key = 'user_settings' "
            "ORDER BY rowid",
            (user_id,)).fetchall()
    finally:
        conn.close()


def _ptb_pair(store: dict, thread_id):
    """Build a context through the real factory, as PTB would per update."""
    message = SimpleNamespace(text='hi', caption=None,
                              message_thread_id=thread_id)
    chat = SimpleNamespace(id=7, type='private')
    user = SimpleNamespace(id=1, first_name='T', last_name='U')
    update = SimpleNamespace(effective_chat=chat, effective_user=user,
                             message=message, effective_message=message)
    context = SimpleNamespace(user_data=store, bot=None, bot_data={})
    return create_chat_update_context(update, context)  # type: ignore[arg-type]


class _TempDbTest(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        import tempfile
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


class SettingsStoreThreadTest(_TempDbTest):

    async def test_missing_returns_none(self):
        self.assertIsNone(await load_settings(5, None))
        self.assertIsNone(await load_settings(5, 2))

    async def test_round_trip_per_thread(self):
        s = UserSettings.default()
        s.model = 'thread-model'
        await save_settings(5, 2, s)
        loaded = await load_settings(5, 2)
        assert loaded is not None
        self.assertEqual(loaded.model, 'thread-model')
        # No template row was created as a side effect.
        self.assertIsNone(await load_settings(5, None))

    async def test_thread_and_template_share_one_key(self):
        template = UserSettings.default()
        template.model = 'template-model'
        thread = UserSettings.default()
        thread.model = 'thread-model'
        await save_settings(5, None, template)
        await save_settings(5, 2, thread)
        loaded_template = await load_settings(5, None)
        loaded_thread = await load_settings(5, 2)
        assert loaded_template is not None and loaded_thread is not None
        self.assertEqual(loaded_template.model, 'template-model')
        self.assertEqual(loaded_thread.model, 'thread-model')
        # One key, told apart by the thread_id column.
        store = _temp_store(self.path)
        try:
            keys = await store.get_all_keys(5)
            self.assertEqual(keys, ['user_settings', 'user_settings'])
            conn_rows = await _settings_rows(self.path, 5)
            self.assertEqual({r[0] for r in conn_rows}, {2, None})
        finally:
            await store.close()

    async def test_repeated_template_save_does_not_duplicate(self):
        # NULL thread ids never conflict in a SQLite PK, so the save
        # must upsert: two saves leave exactly one template row.
        first = UserSettings.default()
        first.model = 'one'
        second = UserSettings.default()
        second.model = 'two'
        await save_settings(5, None, first)
        await save_settings(5, None, second)
        rows = await _settings_rows(self.path, 5)
        self.assertEqual(len(rows), 1)
        loaded = await load_settings(5, None)
        assert loaded is not None
        self.assertEqual(loaded.model, 'two')


class InitConversationThreadTest(_TempDbTest):

    async def test_new_thread_clones_template(self):
        template = UserSettings.default()
        template.model = 'template-model'
        await save_settings(1, None, template)
        ctx = _ptb_pair({}, 2)
        await middleware.init_conversation(ctx)
        self.assertEqual(ctx.settings.model, 'template-model')
        # The clone was persisted for the thread; mutating the thread
        # copy must not touch the template row.
        ctx.settings.model = 'thread-model'
        reloaded = await load_settings(1, 2)
        assert reloaded is not None
        self.assertEqual(reloaded.model, 'template-model')

    async def test_existing_thread_row_wins_over_template(self):
        template = UserSettings.default()
        template.model = 'template-model'
        thread = UserSettings.default()
        thread.model = 'thread-model'
        await save_settings(1, None, template)
        await save_settings(1, 2, thread)
        ctx = _ptb_pair({}, 2)
        await middleware.init_conversation(ctx)
        self.assertEqual(ctx.settings.model, 'thread-model')

    async def test_unthreaded_chat_shares_template(self):
        template = UserSettings.default()
        template.model = 'template-model'
        await save_settings(1, None, template)
        ctx = _ptb_pair({}, None)
        await middleware.init_conversation(ctx)
        self.assertIs(ctx.thread_data.settings, ctx.user_data.settings)
        self.assertEqual(ctx.settings.model, 'template-model')


class SettingsDecoratorTest(_TempDbTest):

    async def test_change_updates_thread_and_template_only(self):
        template = UserSettings.default()
        template.model = 'base'
        await save_settings(1, None, template)
        other = UserSettings.default()
        other.model = 'other'
        await save_settings(1, 4, other)

        ptb_store: dict = {}
        ctx = _ptb_pair(ptb_store, 2)
        await middleware.init_conversation(ctx)

        @middleware.settings
        async def _change(inner_ctx):
            inner_ctx.settings.model = 'new'

        await _change(ctx)

        thread_row = await load_settings(1, 2)
        template_row = await load_settings(1, None)
        other_row = await load_settings(1, 4)
        assert thread_row is not None
        assert template_row is not None
        assert other_row is not None
        self.assertEqual(thread_row.model, 'new')
        self.assertEqual(template_row.model, 'new')
        self.assertEqual(other_row.model, 'other')
        # A freshly created thread now inherits the change.
        late = _ptb_pair(ptb_store, 9)
        await middleware.init_conversation(late)
        self.assertEqual(late.settings.model, 'new')

    async def test_unthreaded_change_saves_once(self):
        template = UserSettings.default()
        template.model = 'base'
        await save_settings(1, None, template)
        ctx = _ptb_pair({}, None)
        await middleware.init_conversation(ctx)

        @middleware.settings
        async def _change(inner_ctx):
            inner_ctx.settings.model = 'new'

        with mock.patch.object(middleware, 'save_settings',
                               wraps=save_settings) as save:
            await _change(ctx)
        self.assertEqual(save.await_count, 1)
        self.assertEqual((await load_settings(1, None)).model, 'new')  # type: ignore[union-attr]


class SubagentSnapshotTest(unittest.TestCase):

    def test_subagent_gets_thread_settings_snapshot(self):
        from conversation import Conversation
        parent = mock_ctx(settings=UserSettings.default())
        parent.thread_data.settings = copy.deepcopy(
            parent.user_data.settings)
        parent.settings.model = 'thread-model'
        sub = parent.new_subagent_context(Conversation())
        self.assertEqual(sub.settings.model, 'thread-model')
        sub.settings.model = 'child-model'
        self.assertEqual(parent.settings.model, 'thread-model')


if __name__ == '__main__':
    unittest.main()
