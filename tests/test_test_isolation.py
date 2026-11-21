"""Tests for the per-run isolated test database.

These pin the behaviour of ``tests/_bootstrap``:

* exactly one temp directory and one SQLite file per test process,
* schema creation happening exactly once (no repeated DDL per connect),
* the temporary directory removed at interpreter exit,
* the connect guard blocking the real development databases.
"""

import asyncio
import os
import sqlite3
import unittest
from unittest import mock

import config
import tests._bootstrap as bootstrap


class TestOneIsolatedDatabasePerRun(unittest.TestCase):

    def test_schema_created_once_for_the_whole_run(self):
        # Every store operation opens a fresh aiosqlite connection, so if
        # each connect re-ran the DDL the suite would pay for it on every
        # operation. Force the one-time creation to run again and count
        # it: the journal DDL (executescript) once, the memory and
        # settings tables once each.
        import aiosqlite

        import sqlite_store

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

        self.addCleanup(setattr, bootstrap, '_schema_ready', True)
        bootstrap._schema_ready = False

        with mock.patch.object(aiosqlite.Connection, 'executescript',
                               counting_executescript), \
                mock.patch.object(sqlite_store.AsyncSQLiteStore,
                                  'create_schema', counting_create_schema):
            asyncio.run(bootstrap.ensure_schema_async())

        self.assertEqual(len(script_calls), 1,
                         'journal DDL ran more than once')
        self.assertEqual(len(table_calls), 2,
                         'memory/settings DDL ran more than once')

        # Already created: the rest of the run reuses it instead of
        # re-running the DDL per connect.
        with mock.patch.object(aiosqlite.Connection, 'executescript',
                               counting_executescript), \
                mock.patch.object(sqlite_store.AsyncSQLiteStore,
                                  'create_schema', counting_create_schema):
            asyncio.run(bootstrap.ensure_schema_async())

        self.assertEqual(len(script_calls), 1,
                         'schema was re-created on a later connect')
        self.assertEqual(len(table_calls), 2)

    def test_bootstrap_db_is_a_single_file_in_its_own_temp_dir(self):
        db = os.path.abspath(config.sql_db_path)
        self.assertTrue(os.path.isfile(db), 'isolated DB file not created')
        self.assertEqual(os.path.dirname(db), bootstrap.db_dir())
        self.assertTrue(os.path.basename(bootstrap.db_dir()).startswith(
            'toolbot-tests-'))
        # Only the one SQLite file (plus SQLite sidecars) lives there.
        extra = [f for f in os.listdir(bootstrap.db_dir())
                 if not f.startswith('store.db')]
        self.assertEqual(extra, [])

    def test_temp_dir_is_removed_at_exit(self):
        # setup() registered an atexit cleanup; simulate process end.
        with mock.patch('shutil.rmtree') as rmtree:
            bootstrap._cleanup()
        rmtree.assert_called_once_with(bootstrap.db_dir(), ignore_errors=True)


class TestRealDatabaseGuard(unittest.TestCase):

    def test_aiosqlite_open_of_repo_db_raises(self):
        import aiosqlite

        async def attempt():
            await aiosqlite.connect(bootstrap.repo_path('store.db'))

        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(attempt())
        self.assertIn('real development database', str(ctx.exception))

    def test_sqlite3_open_of_repo_db_raises(self):
        with self.assertRaises(RuntimeError):
            sqlite3.connect(bootstrap.repo_path('store.db'))

    def test_relative_path_that_resolves_to_repo_db_raises(self):
        # A test hardcoding the default path with a relative spelling
        # must be caught too (realpath, not string comparison).
        with self.assertRaises(RuntimeError):
            sqlite3.connect('store.db')

    def test_old_integration_test_store_db_raises(self):
        with self.assertRaises(RuntimeError):
            sqlite3.connect(bootstrap.repo_path('integration_tests',
                                                'test_store.db'))

    def test_test_db_itself_still_opens(self):
        # The guard only forbids the real databases; the isolated DB and
        # unrelated paths keep working.
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

    def test_unrelated_paths_still_open(self):
        import tempfile
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        conn = sqlite3.connect(path)
        conn.execute('CREATE TABLE t (x)')
        conn.close()


if __name__ == '__main__':
    unittest.main()
