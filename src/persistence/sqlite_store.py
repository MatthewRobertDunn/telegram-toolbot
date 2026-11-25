"""Shared SQLite plumbing for the table-specific stores.

``AsyncSQLiteStore`` owns the connection and schema lifecycle every
store shares: opening the database lazily, applying one PRAGMA policy,
creating the schema at most once per (database path, table) for the
process, and closing.

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
_SCHEMA_READY: set[tuple[str, str]] = set()
# Guards the check-and-apply so two concurrent first uses cannot
# execute the script simultaneously. Created on demand for the running
# loop (asyncio primitives bind to the loop that first awaits them, so
# an import-time lock would be dead weight): mutual exclusion is per
# running loop by design, which is all this single-loop app needs;
# multi-loop callers fall back on SQLite's own file locking and
# busy_timeout.
_SCHEMA_LOCK: asyncio.Lock | None = None
_SCHEMA_LOCK_LOOP: asyncio.AbstractEventLoop | None = None


def _reset_schema_cache() -> None:
    """Forget the process-wide schema readiness (test/tool hook).

    The memo is only valid while the database file at each path is the
    one the schema was applied to: a file deleted and recreated at the
    same path mid-process would be skipped as already ready and fail
    with "no such table". Resetting is explicit on purpose - the app
    never deletes its database out from under itself, and automatic
    invalidation would race the writer.
    """
    _SCHEMA_READY.clear()


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
    if _SCHEMA_LOCK is None or _SCHEMA_LOCK_LOOP is not loop:
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
        # Flipped by create_schema so the DDL script is not
        # re-executed on every operation.
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

    def _schema_already_applied(self) -> bool:
        """Whether the schema is in place for this store's (path, table):
        applied by this store, or by any store in this process for the
        same database path (never true for in-memory databases)."""
        if self._schema_ready:
            return True
        return (not _is_memory_path(self.db_path)
                and (self.db_path, self.table_name) in _SCHEMA_READY)

    async def create_schema(self) -> None:
        """Create tables, indexes and triggers (idempotent).

        The schema may be a single statement or a multi-statement script.
        It is applied at most once per (database path, table) for the
        process; in-memory databases always run the script (see
        ``_is_memory_path``). A database file deleted and recreated at the
        same path mid-process needs an explicit ``_reset_schema_cache()``.
        """
        if self._schema_already_applied():
            # Memo hit: adopt the shared readiness so later operations
            # on this store skip re-entering this method entirely.
            self._schema_ready = True
            return
        async with _schema_lock():
            if self._schema_already_applied():
                self._schema_ready = True
                return
            db = await self._get_conn()
            await db.executescript(self._schema_sql)
            await db.commit()
            self._schema_ready = True
            if not _is_memory_path(self.db_path):
                _SCHEMA_READY.add((self.db_path, self.table_name))

    async def _ready(self) -> aiosqlite.Connection:
        """Return the connection with the schema in place.

        The first operation on a store applies the schema - or adopts
        the shared memo when this process already applied it for this
        database path and table, without re-entering create_schema
        (whose subclass overrides may do per-call work: journal checks
        FTS5 support there). Every later call is a plain connection
        fetch (no DDL re-run).
        """
        if not self._schema_ready:
            if self._schema_already_applied():
                self._schema_ready = True
            else:
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
