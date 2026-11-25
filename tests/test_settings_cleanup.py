"""Startup cleanup: stale thread-scoped settings rows are purged.

Rows in the settings table are grouped by (user_id, thread_id). When
no row in a threaded group has a ``last_updated`` newer than a week,
the whole group is deleted on startup. Unthreaded rows (thread_id
NULL) are never touched: they are the template new threads clone.
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import persistence.cleanup as cleanup
from persistence.settings_store import SettingsStore


def _temp_store(path):
    return SettingsStore(path)


def _insert_row(path, user_id, key, thread_id, last_updated, value='v'):
    """Insert one settings row with an explicit last_updated timestamp."""
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            'INSERT INTO settings (user_id, key, value, thread_id, '
            'last_updated) VALUES (?, ?, ?, ?, ?)',
            (user_id, key, value, thread_id, last_updated))
        conn.commit()
    finally:
        conn.close()


def _rows(path, user_id, key):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(
            'SELECT thread_id FROM settings '
            'WHERE user_id = ? AND key = ? ORDER BY rowid',
            (user_id, key)).fetchall()
    finally:
        conn.close()


# Far enough in the future that "more recent than a week" cannot
# depend on the test machine's clock.
_RECENT = '2999-01-01 00:00:00'
_OLD = '2020-01-01 00:00:00'


class PurgeStaleThreadsTest(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        fd, self.path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        self.addCleanup(os.unlink, self.path)
        store = _temp_store(self.path)
        await store.create_schema()
        await store.close()

    async def _purge(self, **kwargs):
        with mock.patch.object(cleanup, 'create_settings_store',
                               lambda: _temp_store(self.path)):
            return await cleanup.purge_stale_thread_rows(**kwargs)

    async def test_stale_thread_group_is_purged(self):
        # Both keys of one (user, thread) group go together: the purge
        # is per group, not per row.
        _insert_row(self.path, 5, 'user_settings', 2, _OLD)
        _insert_row(self.path, 5, 'conversation', 2, _OLD)
        deleted = await self._purge()
        self.assertEqual(deleted, 2)
        self.assertEqual(_rows(self.path, 5, 'user_settings'), [])
        self.assertEqual(_rows(self.path, 5, 'conversation'), [])

    async def test_group_with_one_recent_row_survives_whole(self):
        # A recent save anywhere in the group keeps the older rows of
        # the same group: the cleanup judges threads, not rows.
        _insert_row(self.path, 5, 'user_settings', 2, _OLD)
        _insert_row(self.path, 5, 'conversation', 2, _RECENT)
        # A group whose rows are all recent (thread 3) is left alone
        # entirely.
        _insert_row(self.path, 5, 'conversation', 3, _RECENT)
        self.assertEqual(await self._purge(), 0)
        self.assertEqual(
            _rows(self.path, 5, 'user_settings'), [(2,)])
        self.assertEqual(
            _rows(self.path, 5, 'conversation'), [(2,), (3,)])

    async def test_unthreaded_rows_are_never_touched(self):
        _insert_row(self.path, 5, 'user_settings', None, _OLD)
        _insert_row(self.path, 5, 'conversation', None, _OLD)
        self.assertEqual(await self._purge(), 0)
        self.assertEqual(
            _rows(self.path, 5, 'user_settings'), [(None,)])
        self.assertEqual(
            _rows(self.path, 5, 'conversation'), [(None,)])

    async def test_users_and_threads_are_judged_separately(self):
        # One stale user must not drag an active one sharing the
        # thread id down with it.
        _insert_row(self.path, 5, 'conversation', 2, _OLD)
        _insert_row(self.path, 6, 'conversation', 2, _RECENT)
        self.assertEqual(await self._purge(), 1)
        self.assertEqual(_rows(self.path, 5, 'conversation'), [])
        self.assertEqual(
            _rows(self.path, 6, 'conversation'), [(2,)])

    async def test_cutoff_days_widen_and_narrow_the_purge(self):
        # With a ten-day cutoff a three-day-old row survives; with a
        # two-day cutoff it goes. SQLite compares CURRENT_TIMESTAMP
        # strings, so a UTC timestamp in the same format behaves the
        # same way.
        three_days_ago = (datetime.now(timezone.utc)
                          - timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S')
        _insert_row(self.path, 5, 'conversation', 2, three_days_ago)
        self.assertEqual(await self._purge(max_age_days=10), 0)
        self.assertEqual(len(_rows(self.path, 5, 'conversation')), 1)
        self.assertEqual(await self._purge(max_age_days=2), 1)
        self.assertEqual(_rows(self.path, 5, 'conversation'), [])


if __name__ == '__main__':
    unittest.main()
