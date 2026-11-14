"""SQLite-backed persistent journal storage.

Strongly typed persistence layer: methods take and return dataclasses and
plain values, never pre-formatted strings. LLM-facing formatting lives in
tools/journal_tool.py.

Usage mirrors context_store.AsyncSQLiteStore: the schema is created once at
startup (chatbot.post_init) via ``async with create_journal_store() as
store: await store.create_schema()``. Operations additionally ensure the
schema themselves, so ad-hoc and test use works without a startup step.

Every operation is scoped to a single ``user_id`` supplied by the caller (the
authenticated application context). There is no way to operate on another
user's entries: each query, update and delete filters on ``user_id``, so an
entry belonging to someone else behaves as though it does not exist.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import aiosqlite

import config


@dataclass(frozen=True)
class JournalEntry:
    id: int
    user_id: int
    # Full ISO 8601 timestamp with UTC offset, in the user's timezone.
    created: str
    # The same instant as `created`, as a UTC Unix timestamp in milliseconds.
    # Used for all chronological sorting and date-range filtering.
    created_epoch_ms: int
    topic: str
    entry: str


class ReplaceOutcome(Enum):
    UPDATED = "updated"
    NOT_FOUND = "not_found"
    TEXT_NOT_FOUND = "text_not_found"
    MULTIPLE_MATCHES = "multiple_matches"


@dataclass(frozen=True)
class SearchPage:
    entries: list[JournalEntry]
    # Whether more matching entries exist beyond the returned page
    # in the requested sort direction.
    has_more: bool


MAX_TOPIC_LENGTH = 128
SEARCH_LIMIT = 50

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS journal_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    created TEXT NOT NULL,
    created_epoch_ms INTEGER NOT NULL,
    topic TEXT NOT NULL CHECK(length(topic) <= 128 AND length(topic) > 0),
    entry TEXT NOT NULL CHECK(length(entry) > 0)
);
CREATE INDEX IF NOT EXISTS idx_journal_entries_user_time
    ON journal_entries(user_id, created_epoch_ms);
CREATE INDEX IF NOT EXISTS idx_journal_entries_user_topic
    ON journal_entries(user_id, topic COLLATE NOCASE);
CREATE VIRTUAL TABLE IF NOT EXISTS journal_entries_fts
    USING fts5(entry, content='journal_entries', content_rowid='id',
               tokenize='unicode61');
CREATE TRIGGER IF NOT EXISTS journal_entries_ai
    AFTER INSERT ON journal_entries BEGIN
        INSERT INTO journal_entries_fts(rowid, entry)
        VALUES (new.id, new.entry);
    END;
CREATE TRIGGER IF NOT EXISTS journal_entries_ad
    AFTER DELETE ON journal_entries BEGIN
        INSERT INTO journal_entries_fts(journal_entries_fts, rowid, entry)
        VALUES ('delete', old.id, old.entry);
    END;
CREATE TRIGGER IF NOT EXISTS journal_entries_au
    AFTER UPDATE ON journal_entries BEGIN
        INSERT INTO journal_entries_fts(journal_entries_fts, rowid, entry)
        VALUES ('delete', old.id, old.entry);
        INSERT INTO journal_entries_fts(rowid, entry)
        VALUES (new.id, new.entry);
    END;
"""


class JournalStore:
    def __init__(self, db_path: str | None = None) -> None:
        self._explicit_path = db_path
        self._conn: aiosqlite.Connection | None = None

    @property
    def db_path(self) -> str:
        # Resolved at call time so tests can point at a scratch database by
        # patching config.sql_db_path.
        return self._explicit_path or config.sql_db_path

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            db = await aiosqlite.connect(self.db_path)
            db.row_factory = aiosqlite.Row
            self._conn = db
        return self._conn

    async def __aenter__(self) -> JournalStore:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def create_schema(self) -> None:
        """Create tables, indexes, FTS5 index and triggers (idempotent)."""
        db = await self._get_conn()
        await db.executescript(_SCHEMA_SQL)
        await db.commit()
        cursor = await db.execute("PRAGMA compile_options")
        options = {row[0] for row in await cursor.fetchall()}
        if "ENABLE_FTS5" not in options:
            raise RuntimeError("SQLite build does not support FTS5")

    async def _ready(self) -> aiosqlite.Connection:
        db = await self._get_conn()
        await db.executescript(_SCHEMA_SQL)
        await db.commit()
        return db

    @staticmethod
    def _row_to_entry(row: aiosqlite.Row) -> JournalEntry:
        return JournalEntry(
            id=row["id"],
            user_id=row["user_id"],
            created=row["created"],
            created_epoch_ms=row["created_epoch_ms"],
            topic=row["topic"],
            entry=row["entry"],
        )

    async def add_entry(self, *, user_id: int, topic: str, entry: str,
                        created: str, created_epoch_ms: int) -> JournalEntry:
        clean_topic = topic.strip()
        if not clean_topic:
            raise ValueError("Topic cannot be empty")
        if len(clean_topic) > MAX_TOPIC_LENGTH:
            raise ValueError(
                f"Topic must be at most {MAX_TOPIC_LENGTH} characters")
        if not entry.strip():
            raise ValueError("Entry cannot be empty")
        db = await self._ready()
        cursor = await db.execute(
            "INSERT INTO journal_entries "
            "(user_id, created, created_epoch_ms, topic, entry) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, created, created_epoch_ms, clean_topic, entry))
        await db.commit()
        entry_id = cursor.lastrowid
        assert entry_id is not None
        return JournalEntry(id=entry_id, user_id=user_id, created=created,
                            created_epoch_ms=created_epoch_ms,
                            topic=clean_topic, entry=entry)

    async def get_entry(self, *, user_id: int,
                        entry_id: int) -> JournalEntry | None:
        records = await self.get_entries(
            user_id=user_id, entry_ids=[entry_id])
        return records[0] if records else None

    async def get_entries(self, *, user_id: int,
                          entry_ids: list[int]) -> list[JournalEntry]:
        """Fetch multiple entries by ID with a single query.

        Returns the found entries in the order requested. Unknown IDs
        (including other users' entries) are silently skipped.
        """
        unique_ids = list(dict.fromkeys(entry_ids))
        if not unique_ids:
            return []
        placeholders = ",".join("?" for _ in unique_ids)
        db = await self._ready()
        cursor = await db.execute(
            "SELECT id, user_id, created, created_epoch_ms, topic, entry "
            f"FROM journal_entries WHERE user_id = ? AND id IN ({placeholders})",
            (user_id, *unique_ids))
        by_id = {row["id"]: self._row_to_entry(row)
                 for row in await cursor.fetchall()}
        return [by_id[entry_id] for entry_id in unique_ids
                if entry_id in by_id]

    async def delete_entry(self, *, user_id: int, entry_id: int) -> bool:
        db = await self._ready()
        cursor = await db.execute(
            "DELETE FROM journal_entries WHERE id = ? AND user_id = ?",
            (entry_id, user_id))
        await db.commit()
        return cursor.rowcount > 0

    async def replace_in_entry(self, *, user_id: int, entry_id: int,
                               old_string: str,
                               new_string: str,
                               replace_all: bool = False) -> ReplaceOutcome:
        if not old_string:
            raise ValueError("Existing text cannot be empty")
        if old_string == new_string:
            raise ValueError(
                "No changes to apply: oldString and newString are identical.")
        db = await self._ready()
        cursor = await db.execute(
            "SELECT entry FROM journal_entries "
            "WHERE id = ? AND user_id = ?",
            (entry_id, user_id))
        row = await cursor.fetchone()
        if row is None:
            return ReplaceOutcome.NOT_FOUND
        current: str = row["entry"]
        occurrences = current.count(old_string)
        if occurrences == 0:
            return ReplaceOutcome.TEXT_NOT_FOUND
        if occurrences > 1 and not replace_all:
            return ReplaceOutcome.MULTIPLE_MATCHES
        if replace_all:
            updated = current.replace(old_string, new_string)
        else:
            index = current.find(old_string)
            updated = (current[:index] + new_string
                       + current[index + len(old_string):])
        await db.execute(
            "UPDATE journal_entries SET entry = ? "
            "WHERE id = ? AND user_id = ?",
            (updated, entry_id, user_id))
        await db.commit()
        return ReplaceOutcome.UPDATED

    @staticmethod
    def _search_filter_clauses(*, user_id: int,
                               query: str | None,
                               topic: str | None
                               ) -> tuple[list[str], list[object]]:
        """Shared non-temporal WHERE clauses for journal search.

        Covers user scoping plus the optional topic and full-text filters.
        The paging cursor is added by the caller.
        """
        conditions = ["user_id = ?"]
        params: list[object] = [user_id]
        clean_topic = topic.strip() if isinstance(topic, str) else ""
        if clean_topic:
            conditions.append("topic = ? COLLATE NOCASE")
            params.append(clean_topic)
        if query is not None and query.strip():
            fts = build_fts_match(query)
            conditions.append(
                "id IN (SELECT rowid FROM journal_entries_fts "
                "WHERE journal_entries_fts MATCH ?)")
            params.append(fts)
        return conditions, params

    async def search_entries(self, *, user_id: int,
                             cursor_ms: int | None = None,
                             query: str | None = None,
                             topic: str | None = None,
                             limit: int = SEARCH_LIMIT,
                             sort_descending: bool = True,
                             ) -> list[JournalEntry]:
        """Return matching entries, newest-first by default.

        The cursor is exclusive and direction-aware: with newest-first
        ordering only entries created before ``cursor_ms`` match, with
        oldest-first ordering only entries created after it. At most
        ``limit`` entries are returned.
        """
        if limit < 1 or limit > SEARCH_LIMIT:
            raise ValueError(f"Limit must be between 1 and {SEARCH_LIMIT}")
        conditions, params = self._search_filter_clauses(
            user_id=user_id, query=query, topic=topic)
        if cursor_ms is not None:
            if sort_descending:
                conditions.append("created_epoch_ms < ?")
            else:
                conditions.append("created_epoch_ms > ?")
            params.append(cursor_ms)
        direction = "DESC" if sort_descending else "ASC"
        sql = ("SELECT id, user_id, created, created_epoch_ms, topic, entry "
               "FROM journal_entries WHERE " + " AND ".join(conditions)
               + f" ORDER BY created_epoch_ms {direction}, id {direction}"
               + " LIMIT ?")
        params.append(limit)
        db = await self._ready()
        cursor = await db.execute(sql, tuple(params))
        return [self._row_to_entry(row) for row in await cursor.fetchall()]

    async def search_page(self, *, user_id: int,
                          cursor_ms: int | None = None,
                          query: str | None = None,
                          topic: str | None = None,
                          limit: int = SEARCH_LIMIT,
                          sort_descending: bool = True,
                          ) -> SearchPage:
        """Return one page of matches plus a has_more flag.

        has_more is true when further matching entries exist beyond the
        returned page in the requested sort direction. Position
        comparisons use (created_epoch_ms, id) so entries sharing a
        millisecond are still detected.
        """
        entries = await self.search_entries(
            user_id=user_id, cursor_ms=cursor_ms,
            query=query, topic=topic, limit=limit,
            sort_descending=sort_descending)
        if not entries:
            return SearchPage(entries=[], has_more=False)
        conditions, params = self._search_filter_clauses(
            user_id=user_id, query=query, topic=topic)
        where = " AND ".join(conditions)
        db = await self._ready()
        if sort_descending:
            edge = min(entries, key=lambda r: (r.created_epoch_ms, r.id))
            cursor = await db.execute(
                "SELECT 1 FROM journal_entries WHERE " + where
                + " AND (created_epoch_ms < ? OR "
                "(created_epoch_ms = ? AND id < ?)) LIMIT 1",
                (*params, edge.created_epoch_ms,
                 edge.created_epoch_ms, edge.id))
        else:
            edge = max(entries, key=lambda r: (r.created_epoch_ms, r.id))
            cursor = await db.execute(
                "SELECT 1 FROM journal_entries WHERE " + where
                + " AND (created_epoch_ms > ? OR "
                "(created_epoch_ms = ? AND id > ?)) LIMIT 1",
                (*params, edge.created_epoch_ms,
                 edge.created_epoch_ms, edge.id))
        has_more = await cursor.fetchone() is not None
        return SearchPage(entries=entries, has_more=has_more)

    async def list_topics(self, *, user_id: int) -> list[str]:
        db = await self._ready()
        cursor = await db.execute(
            "SELECT DISTINCT topic FROM journal_entries "
            "WHERE user_id = ? ORDER BY topic COLLATE NOCASE",
            (user_id,))
        seen: set[str] = set()
        topics: list[str] = []
        for row in await cursor.fetchall():
            name: str = row["topic"]
            key = name.casefold()
            if key not in seen:
                seen.add(key)
                topics.append(name)
        return topics


_TOKEN_RE = re.compile(r'"([^"]*)"|(\S+)')


def build_fts_match(query: str) -> str:
    """Convert a free-text query into a safe FTS5 MATCH expression.

    Every term is double-quoted so FTS5 operators in user input cannot break
    the query, terms are ANDed so all must match, and bare words get a
    trailing ``*`` for prefix matching. Quoted phrases stay phrases.
    """
    parts: list[str] = []
    for match in _TOKEN_RE.finditer(query):
        phrase, word = match.group(1), match.group(2)
        if phrase is not None:
            text = phrase.strip()
            if text:
                parts.append('"' + text.replace('"', '""') + '"')
        elif word is not None:
            core = word.strip().rstrip("*").strip()
            if core:
                parts.append('"' + core.replace('"', '""') + '"*')
    if not parts:
        raise ValueError("Invalid search query")
    return " AND ".join(parts)


def create_journal_store(db_path: str | None = None) -> JournalStore:
    return JournalStore(db_path)
