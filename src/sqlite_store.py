"""Shared SQLite plumbing for the table-specific stores.

``AsyncSQLiteStore`` owns the connection and schema lifecycle every
store shares: opening the database lazily, applying one PRAGMA policy,
creating the schema at most once per (database path, table) for the
process, and closing. Each table
gets its own subclass with methods shaped for that table
(``memory_store.MemoryStore``, ``settings_store.SettingsStore``,
``journal_store.JournalStore``), so no caller deals with columns its
table does not have.

The database path is resolved when the connection opens (see
``db_path``), not at import time: pointing every store at a scratch
database is just a matter of patching ``config.sql_db_path``.
"""

from __future__ import annotations

import asyncio
import re

import aiosqlite

import config

# Table names are interpolated into SQL (SQLite cannot bind identifiers
# as parameters), so only plain SQL identifiers are accepted; each
# subclass pins one fixed table.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# How long a write waits for a competing writer's lock before SQLite
# reports the database as busy (milliseconds).
DEFAULT_BUSY_TIMEOUT_MS = 5_000

# (db_path, table_name) pairs whose schema this process has already
# applied. Stores are created per operation (``async with
# create_x_store()``), so a per-instance flag alone would re-run the
# DDL script on every single operation; readiness has to be shared.
# _SCHEMA_LOCK guards the check-and-apply so two concurrent first uses
# cannot execute the script simultaneously. It is rebound when the
# running event loop changes (unit tests run one loop per test case);
# within one loop - the only place coroutines of this app interleave -
# mutual exclusion always holds.
_SCHEMA_READY: set[tuple[str, str]] = set()
_SCHEMA_LOCK: asyncio.Lock = asyncio.Lock()
_SCHEMA_LOCK_LOOP: asyncio.AbstractEventLoop | None = None


def _is_memory_path(db_path: str) -> bool:
    """Whether the path opens a private in-memory database.

    Covers ``:memory:`` and its URI forms (``file::memory:``,
    ``mode=memory``): every connection to one is a different, empty
    database, so schema readiness must never be shared or cached.
    """
    return ":memory:" in db_path or "mode=memory" in db_path


def _schema_lock() -> asyncio.Lock:
    """The schema lock for the running event loop (see _SCHEMA_LOCK)."""
    global _SCHEMA_LOCK, _SCHEMA_LOCK_LOOP
    loop = asyncio.get_running_loop()
    if _SCHEMA_LOCK_LOOP is not loop:
        _SCHEMA_LOCK = asyncio.Lock()
        _SCHEMA_LOCK_LOOP = loop
    return _SCHEMA_LOCK


class AsyncSQLiteStore:
    """Connection/schema lifecycle for one table in the shared database.

    Subclasses set ``table_name`` and ``schema_sql`` (a single statement
    or a multi-statement script) through ``__init__``. The schema is
    applied at most once per (database path, table) for the process -
    explicitly via ``create_schema`` (the startup path) or lazily on the
    first operation - so ad-hoc and test use needs no startup call and
    per-operation stores do not re-run the DDL script. In-memory
    databases are exempt (see ``create_schema``).
    """

    # Per-connection row factory hook: subclasses wanting typed rows
    # override this (``journal_store`` uses ``aiosqlite.Row``).
    row_factory: type | None = None
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS

    def __init__(self, db_path: str | None, table_name: str, schema_sql: str):
        if not _IDENTIFIER_RE.match(table_name):
            raise ValueError(f"Invalid table name: {table_name!r}")
        # ``None`` defers to config.sql_db_path when the connection opens.
        self._explicit_path = db_path
        self.table_name = table_name
        self._schema_sql = schema_sql
        self._conn: aiosqlite.Connection | None = None
        # Flipped by create_schema so the script - journal's runs to
        # eight statements across tables, indexes, FTS5 and triggers -
        # is not re-executed on every operation.
        self._schema_ready = False

    @property
    def db_path(self) -> str:
        """The database file this store opens.

        Resolved per call rather than at import, so a patched
        ``config.sql_db_path`` redirects stores created without an
        explicit path.
        """
        return self._explicit_path or config.sql_db_path

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            db = await aiosqlite.connect(self.db_path)
            await db.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
            if self.row_factory is not None:
                db.row_factory = self.row_factory
            self._conn = db
        return self._conn

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    def _schema_needed(self, key: tuple[str, str]) -> bool:
        """Whether this store still has to apply its schema script."""
        if self._schema_ready:
            return False
        return _is_memory_path(self.db_path) or key not in _SCHEMA_READY

    async def create_schema(self) -> None:
        """Create tables, indexes and triggers (idempotent).

        The schema may be a single statement or a multi-statement
        script. It is applied at most once per (database path, table)
        for the process: the explicit startup call populates the shared
        readiness just like the first operation does, and later
        per-operation stores skip the DDL. In-memory databases are
        exempt - every connection to one is a different, empty database,
        so they always run the script. Caveat: a database file deleted
        and recreated at the same path mid-process would be skipped as
        already ready; use a fresh path instead (the tests do).
        """
        key = (self.db_path, self.table_name)
        if not self._schema_needed(key):
            return
        async with _schema_lock():
            if not self._schema_needed(key):
                return
            db = await self._get_conn()
            await db.executescript(self._schema_sql)
            await db.commit()
            self._schema_ready = True
            if not _is_memory_path(self.db_path):
                _SCHEMA_READY.add(key)

    async def _ready(self) -> aiosqlite.Connection:
        """Return the connection with the schema in place.

        The first operation on a store applies the schema (unless this
        process already applied it for this database path and table);
        every later call is a plain connection fetch (no DDL re-run).
        """
        if not self._schema_ready:
            await self.create_schema()
        return await self._get_conn()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def get_all_keys(self, user_id: int) -> list[str]:
        """Every key ``user_id`` has a row for in this store's table."""
        db = await self._ready()
        cursor = await db.execute(
            f'SELECT key FROM {self.table_name} WHERE user_id = ?',
            (user_id,))
        rows = await cursor.fetchall()
        return [row[0] for row in rows]
