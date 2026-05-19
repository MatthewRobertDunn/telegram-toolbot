import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from context_store import AsyncSQLiteStore, KV_SCHEMA


class TestKeyValueStore(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.store = AsyncSQLiteStore(':memory:', "kv", KV_SCHEMA)
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

    async def test_isolated_users(self):
        await self.store.set(1, 'key', 'v1')
        await self.store.set(2, 'key', 'v2')

        self.assertEqual(await self.store.get_values(1, ['key']), {'key': 'v1'})
        self.assertEqual(await self.store.get_values(2, ['key']), {'key': 'v2'})


if __name__ == '__main__':
    unittest.main()