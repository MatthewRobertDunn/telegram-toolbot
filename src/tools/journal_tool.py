"""Journal tools: persistent per-user journal for health, mood and notes.

All tools operate on the authenticated user's journal automatically.
Creation timestamps use the timezone
set with /timezone (UTC when never set).
"""
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from chat_update_context import AgentContext
from tool_registry import BOTH, agent_tool

from journal_store import (
    JournalEntry,
    ReplaceOutcome,
    StoreValidationError,
    SEARCH_LIMIT,
    create_journal_store,
)

logger = logging.getLogger(__name__)


def _user_timezone(ctx: AgentContext) -> str:
    try:
        return ctx.settings.timezone
    except Exception:
        return "UTC"


def _now_in_user_timezone(ctx: AgentContext) -> tuple[str, int]:
    name = _user_timezone(ctx)
    try:
        tz = ZoneInfo(name)
    except (ValueError, ZoneInfoNotFoundError):
        logger.warning("Unknown timezone '%s', falling back to UTC", name)
        tz = timezone.utc
    now = datetime.now(tz).replace(microsecond=0)
    return now.isoformat(), int(now.timestamp() * 1000)


def _parse_iso_timestamp(value: str, name: str) -> datetime:
    text = value.strip()
    if text[-1:] in ("Z", "z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(
            f"Error: Invalid {name} timestamp '{value}'. "
            "Use ISO 8601, e.g. '2026-11-01T00:00:00+10:30'.")
    if parsed.tzinfo is None:
        raise ValueError(
            f"Error: {name} timestamp must include an explicit timezone "
            "or UTC indicator, e.g. '2026-11-01T00:00:00+10:30'.")
    return parsed


def _coerce_int(value: object, error: str) -> int:
    """Accept ints (bool excluded), integral floats and integer strings;
    raise ValueError(error) for anything else. One coercer shared by the
    id and limit parameters."""
    if isinstance(value, bool):
        raise ValueError(error)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(error)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise ValueError(error)


def _coerce_id(value: object) -> int:
    return _coerce_int(value, "Error: Invalid journal entry id.")


def _coerce_cursor(value: object) -> int:
    """The entry id a paging cursor names - an exact resume point from
    the previous page's next_cursor.

    The declared type is an integer; a numeric string is still accepted
    defensively, since a model may quote the number. Anything else (a
    bool, a fractional float, arbitrary text) is a cursor the caller
    could not have got from a previous page."""
    return _coerce_int(
        value,
        "Error: Invalid cursor: expected the entry id from the previous "
        "page's next_cursor, e.g. 42.")


def _coerce_limit(value: object) -> int:
    error = f"Error: limit must be an integer between 1 and {SEARCH_LIMIT}."
    coerced = _coerce_int(value, error)
    if coerced < 1 or coerced > SEARCH_LIMIT:
        raise ValueError(error)
    return coerced


def _format_entry(record: JournalEntry) -> str:
    return (f"ID: {record.id}\n"
            f"Created: {record.created}\n"
            f"Topic: {record.topic}\n"
            f"Entry: {record.entry}")


@agent_tool(
    scope=BOTH,
    description=(
        "Create a journal entry for information worth retaining. Prefer an "
        "existing topic from journal_list_topics over new variations. "
        "Preserve the user's meaning; omit dates/times (recorded "
        "automatically)."),
    parameters={
        "topic": {
            "type": "string",
            "description": "Category or subject, max 128 characters.",
        },
        "entry": {
            "type": "string",
            "description": "Entry content, plain text.",
        },
    })
async def journal_add(ctx: AgentContext, topic: str, entry: str) -> str:
    created, created_epoch_ms = _now_in_user_timezone(ctx)
    try:
        async with create_journal_store() as store:
            record = await store.add_entry(
                user_id=ctx.user_id, topic=topic, entry=entry,
                created=created, created_epoch_ms=created_epoch_ms)
    except StoreValidationError as e:
        # The store validated the entry; its message is the user-facing
        # text by contract.
        return f"Error: {e}"
    except Exception as e:
        logger.warning("journal_add failed: %s", e)
        return f"Error: Could not save journal entry: {e}"
    return (f"Created journal entry id:{record.id} created:{record.created}\n"
            "Ensure the entry contains no data already captured by the timestamp; edit if needed.")

@agent_tool(
    scope=BOTH,
    description=("Get journal entries by ID. Use to inspect entries, verify "
                 "before editing, or get full contents after a search."),
    parameters={
        "ids": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Entry IDs to retrieve (max 50).",
        },
    })
async def journal_get(ctx: AgentContext,
                      ids: list[int | float | str]) -> str:
    if not isinstance(ids, (list, tuple)):
        return "Error: ids must be an array of entry IDs."
    if len(ids) == 0:
        return "Error: ids cannot be empty."
    if len(ids) > 50:
        return "Error: Too many ids (maximum 50)."
    try:
        entry_ids = [_coerce_id(value) for value in ids]
    except ValueError as e:
        return str(e)
    # Deduplicate while preserving request order.
    entry_ids = list(dict.fromkeys(entry_ids))
    try:
        async with create_journal_store() as store:
            records = await store.get_entries(
                user_id=ctx.user_id, entry_ids=entry_ids)
    except Exception as e:
        logger.warning("journal_get failed: %s", e)
        return f"Error: Could not retrieve journal entries: {e}"
    found_ids = {record.id for record in records}
    missing = [entry_id for entry_id in entry_ids
               if entry_id not in found_ids]
    if not records:
        if len(entry_ids) == 1:
            return f"Journal entry {entry_ids[0]} not found"
        return (f"Journal entries not found: "
                f"{', '.join(str(i) for i in missing)}")
    if len(records) == 1 and not missing:
        return _format_entry(records[0])
    noun = "entry" if len(records) == 1 else "entries"
    out = (f"Found {len(records)} journal {noun}:\n\n"
           + "\n\n".join(_format_entry(r) for r in records))
    if missing:
        out += (f"\n\nJournal entries not found: "
                f"{', '.join(str(i) for i in missing)}")
    return out


@agent_tool(
    scope=BOTH,
    description="Permanently delete a journal entry.",
    parameters={
        "id": {
            "type": "integer",
            "description": "ID of entry to delete.",
        },
    })
async def journal_delete(ctx: AgentContext, id: int | float | str) -> str:
    try:
        entry_id = _coerce_id(id)
    except ValueError as e:
        return str(e)
    try:
        async with create_journal_store() as store:
            deleted = await store.delete_entry(
                user_id=ctx.user_id, entry_id=entry_id)
    except Exception as e:
        logger.warning("journal_delete failed: %s", e)
        return f"Error: Could not delete journal entry: {e}"
    if not deleted:
        return f"Journal entry {entry_id} not found"
    return f"Journal entry {entry_id} deleted"


@agent_tool(
    scope=BOTH,
    description=(
        "Edit a journal entry via exact, case-sensitive string replacement. "
        "Get current contents with journal_get first. Fails if `oldString` "
        "is missing or matches multiple times: add context to disambiguate "
        "or use `replaceAll`. Creation timestamp is preserved. Empty "
        "`newString` deletes text."),
    parameters={
        "id": {
            "type": "integer",
            "description": "ID of entry to modify.",
        },
        "oldString": {
            "type": "string",
            "description": "Text to replace",
        },
        "newString": {
            "type": "string",
            "description": ("Replacement text (must differ from oldString)"),
        },
        "replaceAll": {
            "type": "boolean",
            "description": ("Replace all matches (default false)"),
        },
    })
async def journal_edit(ctx: AgentContext, id: int | float | str,
                       oldString: str, newString: str,
                       replaceAll: bool | None = None) -> str:
    try:
        entry_id = _coerce_id(id)
    except ValueError as e:
        return str(e)
    try:
        async with create_journal_store() as store:
            outcome = await store.replace_in_entry(
                user_id=ctx.user_id, entry_id=entry_id,
                old_string=oldString, new_string=newString,
                replace_all=bool(replaceAll))
    except StoreValidationError as e:
        # The store validated the edit; its message is the user-facing
        # text by contract.
        return f"Error: {e}"
    except Exception as e:
        logger.warning("journal_edit failed: %s", e)
        return f"Error: Could not update journal entry: {e}"
    if outcome is ReplaceOutcome.NOT_FOUND:
        return f"Journal entry {entry_id} not found"
    if outcome is ReplaceOutcome.TEXT_NOT_FOUND:
        return f"Text not found in journal entry {entry_id}"
    if outcome is ReplaceOutcome.MULTIPLE_MATCHES:
        return (f"Error: Found multiple matches for oldString in journal "
                f"entry {entry_id}. Provide more surrounding context to "
                f"make the match unique, or use replaceAll to replace all "
                f"occurrences.")
    return f"Journal entry {entry_id} updated"


@agent_tool(
    scope=BOTH,
    description=(
        "Change a journal entry's topic. Get current topic with journal_get "
        "first; oldTopic must match exactly. Prefer an existing topic from "
        "journal_list_topics."),
    parameters={
        "id": {
            "type": "integer",
            "description": "ID of entry to modify.",
        },
        "oldTopic": {
            "type": "string",
            "description": "Current topic (must match exactly)",
        },
        "newTopic": {
            "type": "string",
            "description": ("New topic (must differ from oldTopic, "
                            "max 128 characters)"),
        },
    })
async def journal_topic_edit(ctx: AgentContext, id: int | float | str,
                             oldTopic: str, newTopic: str) -> str:
    try:
        entry_id = _coerce_id(id)
    except ValueError as e:
        return str(e)
    try:
        async with create_journal_store() as store:
            outcome = await store.update_entry_topic(
                user_id=ctx.user_id, entry_id=entry_id,
                old_topic=oldTopic, new_topic=newTopic)
    except StoreValidationError as e:
        # The store validated the edit; its message is the user-facing
        # text by contract.
        return f"Error: {e}"
    except Exception as e:
        logger.warning("journal_topic_edit failed: %s", e)
        return f"Error: Could not update journal entry topic: {e}"
    if outcome is ReplaceOutcome.NOT_FOUND:
        return f"Journal entry {entry_id} not found"
    if outcome is ReplaceOutcome.TEXT_NOT_FOUND:
        return f"Topic does not match journal entry {entry_id}"
    return f"Journal entry {entry_id} topic updated"


@agent_tool(
    scope=BOTH,
    description=(
        "Search journal by text and topic; primary tool for recalling past "
        "experiences, mood, symptoms, sleep/wellbeing. All parameters "
        "optional; empty call returns recent entries. Newest-first by "
        "default (sortDescending=false for oldest-first). start is an ISO "
        "8601 timestamp with timezone that positions the FIRST page by "
        "time, inclusive: newest-first returns entries created at or "
        "before it, oldest-first at or after it. cursor is the entry id "
        "returned as next_cursor by the previous page - an exact resume "
        "point, even for entries sharing a timestamp - and is what every "
        "later page uses instead of start; supplying both is an error. "
        "Returns up to limit (default/max 50); if more exist, response "
        "ends with the next cursor (keep other params). Full-text matches "
        "words, not meanings: use concise terms or quoted phrases. "
        "Timestamps need explicit timezone/UTC."),
    parameters={
        "cursor": {
            "type": "integer",
            "description": ("Entry id from the previous page's "
                            "next_cursor. Exact resume point, even for "
                            "entries sharing a timestamp. Do not combine "
                            "with start."),
        },
        "start": {
            "type": "string",
            "description": ("ISO 8601 datetime with timezone positioning "
                            "the first page by time, inclusive: entries "
                            "at or before it when newest-first, at or "
                            "after it when oldest-first. First page only; "
                            "later pages use cursor."),
        },
        "query": {
            "type": "string",
            "description": "Full-text search query.",
        },
        "topic": {
            "type": "string",
            "description": "Exact topic filter (case-insensitive).",
        },
        "sortDescending": {
            "type": "boolean",
            "description": ("Newest-first when true (default), oldest-first "
                            "when false."),
        },
        "limit": {
            "type": "integer",
            "description": "Max entries to return, 1 to 50 (default 50).",
        },
    })
async def journal_query(ctx: AgentContext,
                        cursor: int | None = None,
                        start: str | None = None,
                        query: str | None = None,
                        topic: str | None = None,
                        sortDescending: bool | None = None,
                        limit: int | float | str | None = None) -> str:
    # cursor is the previous page's next_cursor - an entry id, resolved
    # inside the store to its (created_epoch_ms, id) position - while
    # start is a datetime positioning the first page inclusively. One
    # meaning each: a page resuming from a cursor needs no timestamp,
    # and a page positioned by start has no cursor yet.
    start_text = start.strip() if isinstance(start, str) else ""
    if cursor is not None and start_text:
        return ("Error: Provide either cursor or start, not both: cursor "
                "resumes paging from an entry id, start positions the "
                "first page by timestamp.")
    cursor_id: int | None = None
    if cursor is not None:
        try:
            cursor_id = _coerce_cursor(cursor)
        except ValueError as e:
            return str(e)
    start_ms: int | None = None
    if start_text:
        try:
            start_ms = int(
                _parse_iso_timestamp(start_text, "start").timestamp() * 1000)
        except ValueError as e:
            return str(e)
    if sortDescending is None:
        sort_descending = True
    elif isinstance(sortDescending, bool):
        sort_descending = sortDescending
    else:
        return "Error: sortDescending must be a boolean."
    if limit is None:
        max_entries = SEARCH_LIMIT
    else:
        try:
            max_entries = _coerce_limit(limit)
        except ValueError as e:
            return str(e)
    try:
        async with create_journal_store() as store:
            page = await store.search_page(
                user_id=ctx.user_id,
                cursor_id=cursor_id,
                start_ms=start_ms,
                query=query if isinstance(query, str) and query.strip() else None,
                topic=topic if isinstance(topic, str) and topic.strip() else None,
                limit=max_entries,
                sort_descending=sort_descending)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.warning("journal_query failed: %s", e)
        return f"Error: Could not search journal: {e}"
    records = page.entries
    if not records:
        return "No journal entries found."
    noun = "entry" if len(records) == 1 else "entries"
    out = (f"Found {len(records)} journal {noun}:\n\n"
           + "\n\n".join(_format_entry(r) for r in records))
    # next_cursor is the page edge the store already computed; no
    # recomputation here.
    if page.has_more and page.next_cursor is not None:
        out += (f"\n\nMore journal entries exist. To see more, call "
                f"journal_query with cursor={page.next_cursor} "
                f"(keep the same sortDescending, query and limit; do not "
                f"send start again - it positions the first page only).")
    return out


@agent_tool(
    scope=BOTH,
    description=(
        "List distinct journal topics, one per line, alphabetically. Use to "
        "discover categories before creating/searching; prefer existing "
        "topics."))
async def journal_list_topics(ctx: AgentContext) -> str:
    try:
        async with create_journal_store() as store:
            topics = await store.list_topics(user_id=ctx.user_id)
    except Exception as e:
        logger.warning("journal_list_topics failed: %s", e)
        return f"Error: Could not list journal topics: {e}"
    if not topics:
        return "No journal topics yet."
    return "\n".join(topics)
