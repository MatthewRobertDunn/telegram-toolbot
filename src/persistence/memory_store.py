"""Per-user key/value memory, backed by the ``kv`` table."""

from .sqlite_store import AsyncSQLiteStore

KV_SCHEMA = '''
    CREATE TABLE IF NOT EXISTS kv (
        user_id INTEGER NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        PRIMARY KEY (user_id, key)
    )
'''


class MemoryStore(AsyncSQLiteStore):
    def __init__(self, db_path: str | None = None):
        # Table name "kv" is kept for backward compatibility with
        # existing databases.
        super().__init__(db_path, "kv", KV_SCHEMA)

    async def get(self, user_id: int, key: str) -> str | None:
        db = await self._ready()
        cursor = await db.execute(
            'SELECT value FROM kv WHERE user_id = ? AND key = ?',
            (user_id, key))
        row = await cursor.fetchone()
        return row[0] if row else None

    async def set(self, user_id: int, key: str, value: str) -> None:
        db = await self._ready()
        await db.execute(
            'INSERT OR REPLACE INTO kv (user_id, key, value) VALUES (?, ?, ?)',
            (user_id, key, value))
        await db.commit()

    async def delete(self, user_id: int, key: str) -> None:
        db = await self._ready()
        await db.execute(
            'DELETE FROM kv WHERE user_id = ? AND key = ?', (user_id, key))
        await db.commit()

    # get_all_keys is inherited from AsyncSQLiteStore.

    async def get_values(self, user_id: int, keys: list[str]) -> dict[str, str | None]:
        if not keys:
            return {}
        db = await self._ready()
        placeholders = ','.join(['?' for _ in keys])
        cursor = await db.execute(
            f'SELECT key, value FROM kv WHERE user_id = ? AND key IN ({placeholders})',
            (user_id, *keys))
        rows = await cursor.fetchall()
        result: dict[str, str | None] = {k: None for k in keys}
        for key, value in rows:
            result[key] = value
        return result

    async def delete_values(self, user_id: int, keys: list[str]) -> None:
        if not keys:
            return
        db = await self._ready()
        placeholders = ','.join(['?' for _ in keys])
        await db.execute(
            f'DELETE FROM kv WHERE user_id = ? AND key IN ({placeholders})',
            (user_id, *keys))
        await db.commit()

    async def delete_all(self, user_id: int) -> None:
        db = await self._ready()
        await db.execute('DELETE FROM kv WHERE user_id = ?', (user_id,))
        await db.commit()


def create_memory_store() -> MemoryStore:
    # The db path is resolved when the store connects (config.sql_db_path),
    # so patching the config redirects stores created here.
    return MemoryStore()
