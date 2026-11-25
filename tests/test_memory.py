import asyncio
import os
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import aiosqlite
import persistence.sqlite_store as sqlite_store
from persistence.memory_store import MemoryStore, create_memory_store


def _counting_executescript(counter: dict):
    """A replacement for aiosqlite.Connection.executescript that counts
    calls and passes them through unchanged. aiosqlite's execute methods
    return an object that is both awaitable and an async context
    manager; returning it untouched keeps both usages identical."""
    original = aiosqlite.Connection.executescript

    def counting(conn, sql, *args, **kwargs):
        counter['calls'] += 1
        return original(conn, sql, *args, **kwargs)

    return counting


class TestMemory(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.store = MemoryStore(':memory:')
        await self.store.create_schema()

    async def asyncTearDown(self):
        await self.store.close()

    async def test_set_and_get(self):
        await self.store.set(1, 'name', 'Alice')
        await self.store.set(1, 'age', '30')
        await self.store.set(2, 'name', 'Bob')

        result = await self.store.get_values(1, ['name', 'age', 'missing'])
        self.assertEqual(result, {'name': 'Alice', 'age': '30', 'missing': None})

        keys = await self.store.get_all_keys(1)
        self.assertEqual(sorted(keys), ['age', 'name'])

        keys = await self.store.get_all_keys(2)
        self.assertEqual(keys, ['name'])

        keys = await self.store.get_all_keys(3)
        self.assertEqual(keys, [])

        # Empty key lists short-circuit without touching the DB.
        self.assertEqual(await self.store.get_values(1, []), {})
        await self.store.delete_values(1, [])
        self.assertEqual(await self.store.get_all_keys(1),
                         ['age', 'name'])

    async def test_delete(self):
        await self.store.set(1, 'k1', 'v1')
        await self.store.set(1, 'k2', 'v2')
        await self.store.set(1, 'k3', 'v3')

        await self.store.delete_values(1, ['k1', 'k3', 'nonexistent'])

        keys = await self.store.get_all_keys(1)
        self.assertEqual(keys, ['k2'])

        result = await self.store.get_values(1, ['k1', 'k2'])
        self.assertEqual(result, {'k1': None, 'k2': 'v2'})

    async def test_overwrite(self):
        await self.store.set(1, 'key', 'first')
        await self.store.set(1, 'key', 'second')

        result = await self.store.get_values(1, ['key'])
        self.assertEqual(result, {'key': 'second'})

    async def test_schema_ensured_without_startup_call(self):
        # A store used ad hoc, without an explicit create_schema call,
        # applies the schema on its first operation and works from
        # there - the same contract as the journal store.
        store = MemoryStore(':memory:')
        try:
            await store.set(1, 'key', 'value')
            self.assertEqual(await store.get(1, 'key'), 'value')
            self.assertEqual(await store.get_all_keys(1), ['key'])
        finally:
            await store.close()

    async def test_db_path_resolved_at_call_time(self):
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        self.addCleanup(os.unlink, path)
        with mock.patch.object(sqlite_store.config, 'sql_db_path', path):
            # No explicit path: the factory must pick up the patched
            # config when the store connects, not at import time.
            store = create_memory_store()
            try:
                await store.set(1, 'key', 'value')
                self.assertEqual(await store.get(1, 'key'), 'value')
            finally:
                await store.close()
        self.assertGreater(os.path.getsize(path), 0)

    async def test_schema_runs_once_per_database_across_stores(self):
        # Stores are created per operation; the DDL script must run once
        # per (path, table) for the process, not once per store.
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        self.addCleanup(os.unlink, path)
        counter = {'calls': 0}
        with mock.patch.object(aiosqlite.Connection, 'executescript',
                               _counting_executescript(counter)):
            first = MemoryStore(path)
            try:
                await first.set(1, 'k1', 'v1')
                await first.set(1, 'k2', 'v2')
                self.assertEqual(await first.get(1, 'k1'), 'v1')
            finally:
                await first.close()
            # A second, unrelated store over the same file skips DDL.
            second = MemoryStore(path)
            try:
                self.assertEqual(await second.get(1, 'k2'), 'v2')
                self.assertEqual(await second.get_all_keys(1), ['k1', 'k2'])
            finally:
                await second.close()
        self.assertEqual(counter['calls'], 1)

        # Every connection to an in-memory database is a different,
        # empty database: readiness is never shared, so two separate
        # stores must each run the schema themselves.
        memory_calls = {'calls': 0}
        with mock.patch.object(aiosqlite.Connection, 'executescript',
                               _counting_executescript(memory_calls)):
            for _ in range(2):
                store = MemoryStore(':memory:')
                try:
                    await store.set(1, 'key', 'value')
                    self.assertEqual(await store.get(1, 'key'), 'value')
                finally:
                    await store.close()
        self.assertEqual(memory_calls['calls'], 2)

    async def test_concurrent_first_uses_run_schema_once(self):
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        self.addCleanup(os.unlink, path)
        counter = {'calls': 0}
        with mock.patch.object(aiosqlite.Connection, 'executescript',
                               _counting_executescript(counter)):
            stores = [MemoryStore(path) for _ in range(4)]

            async def use(store: MemoryStore) -> str | None:
                await store.set(1, 'key', 'value')
                return await store.get(1, 'key')

            try:
                # The readiness lock serialises the first uses: the
                # script still runs exactly once.
                results = await asyncio.gather(
                    *(use(store) for store in stores))
            finally:
                for store in stores:
                    await store.close()
        self.assertEqual(results, ['value'] * 4)
        self.assertEqual(counter['calls'], 1)

    async def test_reset_schema_cache_allows_recreated_file(self):
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        self.addCleanup(os.unlink, path)
        first = MemoryStore(path)
        try:
            await first.set(1, 'key', 'value')
        finally:
            await first.close()

        # Same path, recreated file: the stale memo skips the schema and
        # the explicit-creation path is a no-op too, so writes fail.
        os.unlink(path)
        stale = MemoryStore(path)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                await stale.set(1, 'key', 'value')
        finally:
            await stale.close()

        # After clearing the cache, the recreated file creates its own schema.
        sqlite_store._reset_schema_cache()
        recreated = MemoryStore(path)
        try:
            await recreated.set(1, 'key', 'value')
            self.assertEqual(await recreated.get(1, 'key'), 'value')
        finally:
            await recreated.close()


if __name__ == '__main__':
    unittest.main()