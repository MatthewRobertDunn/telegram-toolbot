"""SQLite-backed persistent journal storage.

Strongly typed persistence layer: functions take and return dataclasses and
plain values, never pre-formatted strings. LLM-facing formatting lives in
tools/journal_tool.py.

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


def _db_path(db_path: str | None) -> str:
    # Read off config at call time so tests can point at a scratch database
    # by patching config.store_db_path.
    return db_path or config.store_db_path


async def _connect(db_path: str | None = None) -> aiosqlite.Connection:
    db = await aiosqlite.connect(_db_path(db_path))
    db.row_factory = aiosqlite.Row
    await db.executescript(_SCHEMA_SQL)
    await db.commit()
    cursor = await db.execute("PRAGMA compile_options")
    options = {row[0] for row in await cursor.fetchall()}
    if "ENABLE_FTS5" not in options:
        await db.close()
        raise RuntimeError("SQLite build does not support FTS5")
    return db


def _row_to_entry(row: aiosqlite.Row) -> JournalEntry:
    return JournalEntry(
        id=row["id"],
        user_id=row["user_id"],
        created=row["created"],
        created_epoch_ms=row["created_epoch_ms"],
        topic=row["topic"],
        entry=row["entry"],
    )


async def add_entry(*, user_id: int, topic: str, entry: str,
                    created: str, created_epoch_ms: int,
                    db_path: str | None = None) -> JournalEntry:
    clean_topic = topic.strip()
    if not clean_topic:
        raise ValueError("Topic cannot be empty")
    if len(clean_topic) > MAX_TOPIC_LENGTH:
        raise ValueError(
            f"Topic must be at most {MAX_TOPIC_LENGTH} characters")
    if not entry.strip():
        raise ValueError("Entry cannot be empty")
    db = await _connect(db_path)
    try:
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
    finally:
        await db.close()


async def get_entry(*, user_id: int, entry_id: int,
                    db_path: str | None = None) -> JournalEntry | None:
    db = await _connect(db_path)
    try:
        cursor = await db.execute(
            "SELECT id, user_id, created, created_epoch_ms, topic, entry "
            "FROM journal_entries WHERE id = ? AND user_id = ?",
            (entry_id, user_id))
        row = await cursor.fetchone()
        return _row_to_entry(row) if row is not None else None
    finally:
        await db.close()


async def delete_entry(*, user_id: int, entry_id: int,
                       db_path: str | None = None) -> bool:
    db = await _connect(db_path)
    try:
        cursor = await db.execute(
            "DELETE FROM journal_entries WHERE id = ? AND user_id = ?",
            (entry_id, user_id))
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


async def replace_in_entry(*, user_id: int, entry_id: int, existing: str,
                           replacement: str,
                           db_path: str | None = None) -> ReplaceOutcome:
    if not existing:
        raise ValueError("Existing text cannot be empty")
    db = await _connect(db_path)
    try:
        cursor = await db.execute(
            "SELECT entry FROM journal_entries "
            "WHERE id = ? AND user_id = ?",
            (entry_id, user_id))
        row = await cursor.fetchone()
        if row is None:
            return ReplaceOutcome.NOT_FOUND
        current: str = row["entry"]
        index = current.find(existing)
        if index < 0:
            return ReplaceOutcome.TEXT_NOT_FOUND
        updated = current[:index] + replacement + current[index + len(existing):]
        await db.execute(
            "UPDATE journal_entries SET entry = ? "
            "WHERE id = ? AND user_id = ?",
            (updated, entry_id, user_id))
        await db.commit()
        return ReplaceOutcome.UPDATED
    finally:
        await db.close()


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


async def search_entries(*, user_id: int,
                         start_ms: int | None = None,
                         end_ms: int | None = None,
                         query: str | None = None,
                         topic: str | None = None,
                         limit: int = SEARCH_LIMIT,
                         db_path: str | None = None) -> list[JournalEntry]:
    if start_ms is not None and end_ms is not None and start_ms >= end_ms:
        raise ValueError("Start timestamp must be earlier than end timestamp")
    conditions = ["user_id = ?"]
    params: list[object] = [user_id]
    if start_ms is not None:
        conditions.append("created_epoch_ms >= ?")
        params.append(start_ms)
    if end_ms is not None:
        conditions.append("created_epoch_ms < ?")
        params.append(end_ms)
    clean_topic = topic.strip() if isinstance(topic, str) else ""
    if clean_topic:
        conditions.append("topic = ? COLLATE NOCASE")
        params.append(clean_topic)
    fts: str | None = None
    if query is not None and query.strip():
        fts = build_fts_match(query)
    sql = ("SELECT id, user_id, created, created_epoch_ms, topic, entry "
           "FROM journal_entries WHERE " + " AND ".join(conditions))
    if fts is not None:
        sql += (" AND id IN (SELECT rowid FROM journal_entries_fts "
                "WHERE journal_entries_fts MATCH ?)")
        params.append(fts)
    sql += " ORDER BY created_epoch_ms DESC LIMIT ?"
    params.append(limit)
    db = await _connect(db_path)
    try:
        cursor = await db.execute(sql, tuple(params))
        return [_row_to_entry(row) for row in await cursor.fetchall()]
    finally:
        await db.close()


async def list_topics(*, user_id: int,
                      db_path: str | None = None) -> list[str]:
    db = await _connect(db_path)
    try:
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
    finally:
        await db.close()
