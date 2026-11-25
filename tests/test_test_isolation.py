"""Tests for the per-run isolated test database.

These pin the behaviour of ``tests/_bootstrap``:

* schema creation happening exactly once (no repeated DDL per connect),
* the connect guard blocking the real development databases however
  they are spelled (plain paths and ``file:`` URIs, sync and async),
* the isolated database and unrelated paths still opening normally.
"""

import asyncio
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import config
import tests._bootstrap as bootstrap


class TestOneIsolatedDatabasePerRun(unittest.TestCase):

    def test_schema_created_once_for_the_whole_run(self):
        # Every store operation opens a fresh aiosqlite connection, so if
        # each connect re-ran the DDL the suite would pay for it on every
        # operation. Force the one-time creation to run again and count
        # it. All three tables now share one code path, so the count is
        # one DDL script each: memory (kv), settings and journal.
        import aiosqlite

        import persistence.sqlite_store as sqlite_store

        real_executescript = aiosqlite.Connection.executescript
        real_create_schema = sqlite_store.AsyncSQLiteStore.create_schema
        script_calls: list[str] = []
        table_calls: list[str] = []

        async def counting_executescript(self, sql_script):
            script_calls.append(sql_script)
            return await real_executescript(self, sql_script)

        async def counting_create_schema(self):
            table_calls.append(self.table_name)
            return await real_create_schema(self)

        # Schema readiness is shared per (database path, table) across
        # the whole process (see ``sqlite_store._SCHEMA_READY``), so
        # whichever tests happened to run first decide which tables a
        # "re-run" still has to apply. Clear it to make the counts below
        # mean "once each" rather than "whatever was left over".
        already_ready = set(sqlite_store._SCHEMA_READY)
        self.addCleanup(sqlite_store._SCHEMA_READY.update, already_ready)
        sqlite_store._reset_schema_cache()

        self.addCleanup(setattr, bootstrap, '_schema_ready', True)
        bootstrap._schema_ready = False

        with mock.patch.object(aiosqlite.Connection, 'executescript',
                               counting_executescript), \
                mock.patch.object(sqlite_store.AsyncSQLiteStore,
                                  'create_schema', counting_create_schema):
            asyncio.run(bootstrap.ensure_schema_async())

        self.assertEqual(sorted(table_calls),
                         ['journal_entries', 'kv', 'settings'],
                         'every table in the shared store must be applied')
        self.assertEqual(len(script_calls), 3,
                         'a table DDL ran more than once')

        # Already created: the rest of the run reuses it instead of
        # re-running the DDL per connect.
        with mock.patch.object(aiosqlite.Connection, 'executescript',
                               counting_executescript), \
                mock.patch.object(sqlite_store.AsyncSQLiteStore,
                                  'create_schema', counting_create_schema):
            asyncio.run(bootstrap.ensure_schema_async())

        self.assertEqual(len(script_calls), 3,
                         'schema was re-created on a later connect')
        self.assertEqual(len(table_calls), 3)


class TestRealDatabaseGuard(unittest.TestCase):

    def test_opening_the_repo_db_raises_for_every_connect_and_spelling(self):
        # The guard must catch both connect callables (sqlite3 and
        # aiosqlite) however a test spells the path: plainly or as a
        # file: URI, with or without a query.
        import aiosqlite

        target = bootstrap.repo_path('store.db').replace('\\', '/')

        def sync_connect(database, **kwargs):
            sqlite3.connect(database, **kwargs)

        def async_connect(database, **kwargs):
            async def attempt():
                await aiosqlite.connect(database, **kwargs)

            asyncio.run(attempt())

        spellings = {
            'direct path': (bootstrap.repo_path('store.db'), {}),
            'file: URI': ('file:' + target, {'uri': True}),
            'file:/// URI with query':
                ('file:///' + target + '?mode=ro', {'uri': True}),
        }
        for connect in (sync_connect, async_connect):
            for name, (database, kwargs) in spellings.items():
                with self.subTest(connect=connect.__name__, spelling=name):
                    with self.assertRaises(RuntimeError) as ctx:
                        connect(database, **kwargs)
                    self.assertIn('real development database',
                                  str(ctx.exception))

    def test_relative_path_that_resolves_to_repo_db_raises(self):
        # A test hardcoding the default path with a relative spelling
        # must be caught too (realpath, not string comparison). Relative
        # paths resolve against the CWD, so pin it to the repo root for
        # this test — the suite must pass no matter where it launched.
        old_cwd = os.getcwd()
        os.chdir(bootstrap._ROOT)
        self.addCleanup(os.chdir, old_cwd)
        with self.assertRaises(RuntimeError):
            sqlite3.connect('store.db')

    def test_old_integration_test_store_db_raises(self):
        with self.assertRaises(RuntimeError):
            sqlite3.connect(bootstrap.repo_path('integration_tests',
                                                'test_store.db'))

    def test_isolated_db_and_unrelated_paths_still_open(self):
        # The guard only forbids the real databases; the isolated DB
        # keeps its schema and unrelated paths open normally.
        conn = sqlite3.connect(config.sql_db_path)
        try:
            names = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")
            }
        finally:
            conn.close()
        self.assertIn('settings', names)
        self.assertIn('kv', names)
        self.assertIn('journal_entries', names)

        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        conn = sqlite3.connect(path)
        conn.execute('CREATE TABLE t (x)')
        conn.close()


if __name__ == '__main__':
    unittest.main()
