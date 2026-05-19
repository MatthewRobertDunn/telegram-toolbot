import aiosqlite
from config import store_db_path


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

    async def get(self, user_id: int, key: str) -> str | None:
        db = await self._get_conn()
        cursor = await db.execute(
            f'SELECT value FROM {self.table_name} WHERE user_id = ? AND key = ?',
            (user_id, key))
        row = await cursor.fetchone()
        return row[0] if row else None

    async def set(self, user_id: int, key: str, value: str) -> None:
        db = await self._get_conn()
        await db.execute(
            f'INSERT OR REPLACE INTO {self.table_name} (user_id, key, value) VALUES (?, ?, ?)',
            (user_id, key, value))
        await db.commit()

    async def delete(self, user_id: int, key: str) -> None:
        db = await self._get_conn()
        await db.execute(
            f'DELETE FROM {self.table_name} WHERE user_id = ? AND key = ?',
            (user_id, key))
        await db.commit()

    async def get_all_keys(self, user_id: int) -> list[str]:
        db = await self._get_conn()
        cursor = await db.execute(
            f'SELECT key FROM {self.table_name} WHERE user_id = ?', (user_id,))
        rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def get_values(self, user_id: int, keys: list[str]) -> dict[str, str | None]:
        if not keys:
            return {}
        db = await self._get_conn()
        placeholders = ','.join(['?' for _ in keys])
        cursor = await db.execute(
            f'SELECT key, value FROM {self.table_name} WHERE user_id = ? AND key IN ({placeholders})',
            (user_id, *keys))
        rows = await cursor.fetchall()
        result: dict[str, str | None] = {k: None for k in keys}
        for key, value in rows:
            result[key] = value
        return result

    async def delete_values(self, user_id: int, keys: list[str]) -> None:
        if not keys:
            return
        db = await self._get_conn()
        placeholders = ','.join(['?' for _ in keys])
        await db.execute(
            f'DELETE FROM {self.table_name} WHERE user_id = ? AND key IN ({placeholders})',
            (user_id, *keys))
        await db.commit()

    async def delete_all(self, user_id: int) -> None:
        db = await self._get_conn()
        await db.execute(f'DELETE FROM {self.table_name} WHERE user_id = ?', (user_id,))
        await db.commit()


KV_SCHEMA = '''
    CREATE TABLE IF NOT EXISTS kv (
        user_id INTEGER NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        PRIMARY KEY (user_id, key)
    )
'''

SETTINGS_SCHEMA = '''
    CREATE TABLE IF NOT EXISTS settings (
        user_id INTEGER NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        PRIMARY KEY (user_id, key)
    )
'''


def create_store() -> AsyncSQLiteStore:
    return AsyncSQLiteStore(store_db_path, "kv", KV_SCHEMA)


def create_settings_store() -> AsyncSQLiteStore:
    return AsyncSQLiteStore(store_db_path, "settings", SETTINGS_SCHEMA)