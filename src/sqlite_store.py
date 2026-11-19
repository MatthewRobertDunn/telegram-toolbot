"""Shared SQLite plumbing for the table-specific stores.

``AsyncSQLiteStore`` is only connection and schema management: opening
the database, creating the schema, closing. Each table gets its own
subclass with a single ``get``/``set`` pair shaped for that table
(``memory_store.MemoryStore``, ``settings_store.SettingsStore``), so no
caller deals with columns its table does not have.
"""

import aiosqlite


class AsyncSQLiteStore:
    def __init__(self, db_path: str, table_name: str, schema_sql: str):
        self.db_path = db_path
        self.table_name = table_name
        self._schema_sql = schema_sql
        self._conn = None

    async def _get_conn(self):
        if self._conn is None:
            self._conn = await aiosqlite.connect(self.db_path)
        return self._conn

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def create_schema(self):
        db = await self._get_conn()
        await db.execute(self._schema_sql)
        await db.commit()

    async def close(self):
        if self._conn:
            await self._conn.close()
            self._conn = None
