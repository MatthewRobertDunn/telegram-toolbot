"""Per-thread settings: isolation and template inheritance.

Settings changes (e.g. /model) affect only the current thread, except
changes made in the unthreaded/'all' category (thread id None) which
also update the shared template that newly created threads clone.
"""

import copy
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import persistence.settings_store as settings_store
from persistence.settings_store import load_settings, save_settings
from persistence.settings_store import SettingsStore
from chat_update_context import create_chat_update_context
from handlers import middleware
from mock_context import mock_ctx
from user_settings import UserSettings


def _temp_store(path):
    return SettingsStore(path)


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


class SettingsDecoratorTest(_TempDbTest):

    async def test_change_scopes_to_the_current_category(self):
        """Threaded changes update the thread row only; unthreaded
        ('all' category) changes update the shared template instead,
        saving exactly once."""
        template = UserSettings.default()
        template.model = 'base'
        await save_settings(1, None, template)
        other = UserSettings.default()
        other.model = 'other'
        await save_settings(1, 4, other)

        for thread_id in (2, None):
            with self.subTest(thread_id=thread_id):
                ptb_store: dict = {}
                ctx = _ptb_pair(ptb_store, thread_id)
                await middleware.init_conversation(ctx)

                @middleware.settings
                async def _change(inner_ctx):
                    inner_ctx.settings.model = 'new'

                with mock.patch.object(middleware, 'save_settings',
                                       wraps=save_settings) as save:
                    await _change(ctx)

                template_row = await load_settings(1, None)
                other_row = await load_settings(1, 4)
                assert template_row is not None
                assert other_row is not None
                self.assertEqual(other_row.model, 'other')
                if thread_id is None:
                    # The unthreaded change saves once (upsert).
                    self.assertEqual(save.await_count, 1)
                    self.assertEqual(template_row.model, 'new')
                else:
                    thread_row = await load_settings(1, 2)
                    assert thread_row is not None
                    self.assertEqual(thread_row.model, 'new')
                    # Only the unthreaded/'all' category updates the template.
                    self.assertEqual(template_row.model, 'base')
                # A freshly created thread inherits the (un)changed template.
                # A per-category thread id: the threaded pass above already
                # cloned the template into thread 9's own row.
                late_id = 10 if thread_id is None else 9
                late = _ptb_pair(ptb_store, late_id)
                await middleware.init_conversation(late)
                self.assertEqual(late.settings.model,
                                 'new' if thread_id is None else 'base')


class SettingsAliasingFlagTest(_TempDbTest):
    """The two settings-aliasing decisions, pinned by the ThreadData flag.

    Unthreaded chats share the template object (writes go to both);
    threaded chats own a deep copy. ``settings_is_template`` names which
    one a thread is in, so the aliasing is visible state, not an
    undocumented identity accident.
    """

    async def test_aliasing_follows_the_thread_category(self):
        """Unthreaded chats share the template object; threaded chats
        own a deep copy. ``settings_is_template`` pins which one."""
        template = UserSettings.default()
        template.model = 'template-model'
        await save_settings(1, None, template)
        for thread_id, is_template in ((None, True), (2, False)):
            with self.subTest(thread_id=thread_id):
                ctx = _ptb_pair({}, thread_id)
                await middleware.init_conversation(ctx)
                self.assertIs(ctx.thread_data.settings_is_template,
                              is_template)
                if is_template:
                    self.assertIs(ctx.thread_data.settings,
                                  ctx.user_data.settings)
                else:
                    self.assertIsNot(ctx.thread_data.settings,
                                     ctx.user_data.settings)

    async def test_swap_preserves_the_aliasing_flag(self):
        """/settings_reset-style swap: the command replaces the object
        through ctx.settings (flag False mid-command), then the settings
        decorator re-applies the category's aliasing decision."""
        for thread_id, is_template in ((None, True), (2, False)):
            with self.subTest(thread_id=thread_id):
                ctx = _ptb_pair({}, thread_id)
                await middleware.init_conversation(ctx)

                @middleware.settings
                async def _swap(inner_ctx):
                    inner_ctx.settings = UserSettings.default()

                with mock.patch.object(middleware, 'save_settings',
                                       mock.AsyncMock()):
                    await _swap(ctx)
                self.assertIs(ctx.thread_data.settings_is_template,
                              is_template)
                if is_template:
                    self.assertIs(ctx.thread_data.settings,
                                  ctx.user_data.settings)
                else:
                    self.assertIsNot(ctx.thread_data.settings,
                                     ctx.user_data.settings)


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
