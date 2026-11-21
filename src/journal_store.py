"""SQLite-backed persistent journal storage.

Strongly typed persistence layer: methods take and return dataclasses and
plain values, never pre-formatted strings. LLM-facing formatting lives in
tools/journal_tool.py.

Extends sqlite_store.AsyncSQLiteStore (connection, PRAGMA and schema
lifecycle): the schema is created once at startup (chatbot.post_init) via
``async with create_journal_store() as store: await store.create_schema()``,
and operations ensure it themselves, so ad-hoc and test use works without
a startup step. Validation failures raise ``StoreValidationError``, a
ValueError subclass whose message is the user/model-facing text the tool
layer surfaces verbatim.

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

from sqlite_store import AsyncSQLiteStore


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


class StoreValidationError(ValueError):
    """A caller-supplied value failed store validation.

    Subclasses ValueError so ``except ValueError`` handling keeps
    working; the message is the user/model-facing text the tool layer
    surfaces verbatim, so validation exists in exactly one place.
    """


@dataclass(frozen=True)
class SearchPage:
    entries: list[JournalEntry]
    # Whether more matching entries exist beyond the returned page
    # in the requested sort direction.
    has_more: bool
    # Resume point for the next page when has_more is true (None
    # otherwise): the edge entry's id as a decimal string. The paging
    # comparison resolves that row's timestamp and uses the full
    # (created_epoch_ms, id) position, so the token is exact across
    # same-millisecond ties. Pass it back to search_page as cursor_id,
    # or to the journal_query tool as its cursor.
    next_cursor: str | None = None


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


def _validate_topic(topic: object, empty_message: str) -> str:
    """Stripped, lowercased topic, checked for emptiness and length.

    Raises StoreValidationError (not ValueError directly) so the tool
    layer can surface the message verbatim.
    """
    clean = topic.strip().lower() if isinstance(topic, str) else ""
    if not clean:
        raise StoreValidationError(empty_message)
    if len(clean) > MAX_TOPIC_LENGTH:
        raise StoreValidationError(
            f"Topic must be at most {MAX_TOPIC_LENGTH} characters")
    return clean


class JournalStore(AsyncSQLiteStore):
    # Typed rows for the dataclass mapping below.
    row_factory = aiosqlite.Row

    def __init__(self, db_path: str | None = None) -> None:
        super().__init__(db_path, "journal_entries", _SCHEMA_SQL)

    async def create_schema(self) -> None:
        """Create tables, indexes, FTS5 index and triggers (idempotent)."""
        db = await self._get_conn()
        cursor = await db.execute("PRAGMA compile_options")
        options = {row[0] for row in await cursor.fetchall()}
        if "ENABLE_FTS5" not in options:
            raise RuntimeError("SQLite build does not support FTS5")
        await super().create_schema()

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
        clean_topic = _validate_topic(topic, "Topic cannot be empty")
        if not isinstance(entry, str) or not entry.strip():
            raise StoreValidationError("Entry cannot be empty")
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
        if not isinstance(old_string, str) or not old_string:
            raise StoreValidationError("Existing text cannot be empty")
        if not isinstance(new_string, str):
            raise StoreValidationError("Invalid replacement text")
        if old_string == new_string:
            raise StoreValidationError(
                "No changes to apply: oldString and newString are identical.")
        db = await self._ready()
        # An explicit write transaction makes the read-modify-write
        # atomic: the row is locked from the read that counts occurrences
        # through the write that applies the replacement, so a competing
        # writer can neither slip a change in between nor be clobbered by
        # a stale rewrite.
        await db.execute("BEGIN IMMEDIATE")
        try:
            cursor = await db.execute(
                "SELECT entry FROM journal_entries "
                "WHERE id = ? AND user_id = ?",
                (entry_id, user_id))
            row = await cursor.fetchone()
            if row is None:
                await db.commit()
                return ReplaceOutcome.NOT_FOUND
            current: str = row["entry"]
            occurrences = current.count(old_string)
            if occurrences == 0:
                await db.commit()
                return ReplaceOutcome.TEXT_NOT_FOUND
            if occurrences > 1 and not replace_all:
                await db.commit()
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
        except BaseException:
            await db.rollback()
            raise

    async def update_entry_topic(self, *, user_id: int, entry_id: int,
                                 old_topic: str,
                                 new_topic: str) -> ReplaceOutcome:
        clean_old = _validate_topic(old_topic,
                                    "Existing topic cannot be empty")
        clean_new = _validate_topic(new_topic, "New topic cannot be empty")
        if clean_old == clean_new:
            raise StoreValidationError(
                "No changes to apply: oldTopic and newTopic are identical.")
        db = await self._ready()
        # One guarded UPDATE: the topic match rides in the WHERE clause,
        # so check-and-set is a single atomic statement instead of a
        # read-modify-write. Topics are stored lowercased and matched
        # case-insensitively, as before.
        cursor = await db.execute(
            "UPDATE journal_entries SET topic = ? "
            "WHERE id = ? AND user_id = ? AND topic = ? COLLATE NOCASE",
            (clean_new, entry_id, user_id, clean_old))
        await db.commit()
        if cursor.rowcount:
            return ReplaceOutcome.UPDATED
        # The guarded update did not fire: one read tells a missing
        # entry apart from a non-matching topic.
        cursor = await db.execute(
            "SELECT 1 FROM journal_entries WHERE id = ? AND user_id = ?",
            (entry_id, user_id))
        if await cursor.fetchone() is None:
            return ReplaceOutcome.NOT_FOUND
        return ReplaceOutcome.TEXT_NOT_FOUND

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

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if limit < 1 or limit > SEARCH_LIMIT:
            raise StoreValidationError(
                f"Limit must be between 1 and {SEARCH_LIMIT}")

    def _search_query(self, *, user_id: int, cursor_ms: int | None,
                      query: str | None, topic: str | None,
                      sort_descending: bool, limit: int,
                      cursor_id: int | None = None
                      ) -> tuple[str, tuple[object, ...]]:
        """The shared search SELECT with its bound parameters.

        The cursor is exclusive and direction-aware, comparing the same
        (created_epoch_ms, id) position the ORDER BY uses: with
        newest-first ordering only entries positioned before the cursor
        match, with oldest-first ordering only entries after it. The
        token cursor (``cursor_id``) is exact - its timestamp was
        resolved from the row it names - while a bare ``cursor_ms``
        (the ISO 8601 path) is millisecond-granular and skips entries
        sharing that millisecond.
        """
        conditions, params = self._search_filter_clauses(
            user_id=user_id, query=query, topic=topic)
        if cursor_ms is not None:
            if cursor_id is not None:
                comparison = "<" if sort_descending else ">"
                conditions.append(
                    f"(created_epoch_ms {comparison} ? "
                    f"OR (created_epoch_ms = ? AND id {comparison} ?))")
                params.extend([cursor_ms, cursor_ms, cursor_id])
            elif sort_descending:
                conditions.append("created_epoch_ms < ?")
                params.append(cursor_ms)
            else:
                conditions.append("created_epoch_ms > ?")
                params.append(cursor_ms)
        direction = "DESC" if sort_descending else "ASC"
        sql = ("SELECT id, user_id, created, created_epoch_ms, topic, entry "
               "FROM journal_entries WHERE " + " AND ".join(conditions)
               + f" ORDER BY created_epoch_ms {direction}, id {direction}"
               + " LIMIT ?")
        params.append(limit)
        return sql, tuple(params)

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
        self._validate_limit(limit)
        sql, params = self._search_query(
            user_id=user_id, cursor_ms=cursor_ms, query=query,
            topic=topic, sort_descending=sort_descending, limit=limit)
        db = await self._ready()
        cursor = await db.execute(sql, params)
        return [self._row_to_entry(row) for row in await cursor.fetchall()]

    async def _resolve_cursor_timestamp(self, *, user_id: int,
                                        cursor_id: int) -> int:
        """The timestamp half of a token cursor's (created_epoch_ms, id)
        position, read from the row the token names.

        Raises StoreValidationError when the row is gone (deleted) or
        not visible to this user: the token would be ambiguous, so
        paging must restart rather than guess."""
        db = await self._ready()
        cursor = await db.execute(
            "SELECT created_epoch_ms FROM journal_entries "
            "WHERE user_id = ? AND id = ?", (user_id, cursor_id))
        row = await cursor.fetchone()
        if row is None:
            raise StoreValidationError(
                "Cursor entry no longer exists; restart paging.")
        return row["created_epoch_ms"]

    async def search_page(self, *, user_id: int,
                          cursor_ms: int | None = None,
                          cursor_id: int | None = None,
                          query: str | None = None,
                          topic: str | None = None,
                          limit: int = SEARCH_LIMIT,
                          sort_descending: bool = True,
                          ) -> SearchPage:
        """Return one page of matches plus paging hints.

        has_more is true when further matching entries exist beyond the
        returned page in the requested sort direction, detected by
        fetching one row beyond ``limit`` rather than re-running the
        filtered query. ``next_cursor`` is the resume point for that
        next page - the edge entry's id as a decimal string, None when
        there is nothing to resume. The token is exact: on resume its
        timestamp is read back from the row it names and the comparison
        uses the full (created_epoch_ms, id) position the ORDER BY
        uses, so entries sharing a millisecond page without loss in
        either direction; a token whose row no longer exists (deleted,
        or not visible to this user) raises StoreValidationError. A
        bare ``cursor_ms`` (the ISO 8601 path) is millisecond-granular
        and skips entries sharing that millisecond. When both are
        given, the token wins.
        """
        self._validate_limit(limit)
        if cursor_id is not None:
            cursor_ms = await self._resolve_cursor_timestamp(
                user_id=user_id, cursor_id=cursor_id)
        # One extra row answers "is there another page?" cheaply.
        sql, params = self._search_query(
            user_id=user_id, cursor_ms=cursor_ms, cursor_id=cursor_id,
            query=query, topic=topic, sort_descending=sort_descending,
            limit=limit + 1)
        db = await self._ready()
        cursor = await db.execute(sql, params)
        rows = list(await cursor.fetchall())
        entries = [self._row_to_entry(row) for row in rows[:limit]]
        if len(rows) <= limit or not entries:
            return SearchPage(entries=entries, has_more=False,
                              next_cursor=None)
        if sort_descending:
            edge = min(entries, key=lambda r: (r.created_epoch_ms, r.id))
        else:
            edge = max(entries, key=lambda r: (r.created_epoch_ms, r.id))
        return SearchPage(entries=entries, has_more=True,
                          next_cursor=str(edge.id))

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
