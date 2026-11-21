import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import sqlite_store
from memory_store import MemoryStore, create_memory_store


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

    async def test_empty_keys(self):
        result = await self.store.get_values(1, [])
        self.assertEqual(result, {})

        await self.store.delete_values(1, [])

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


if __name__ == '__main__':
    unittest.main()