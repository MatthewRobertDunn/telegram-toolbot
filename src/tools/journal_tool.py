"""Journal tools: persistent per-user journal for health, mood and notes."""
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from chat_update_context import AgentContext
from tool_registry import BOTH, agent_tool

from persistence.journal_store import (
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
async def journal_get(ctx: AgentContext, ids: list[int]) -> str:
    if not isinstance(ids, (list, tuple)):
        return "Error: ids must be an array of entry IDs."
    if len(ids) == 0:
        return "Error: ids cannot be empty."
    if len(ids) > 50:
        return "Error: Too many ids (maximum 50)."
    entry_ids = list(dict.fromkeys(ids))
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
async def journal_delete(ctx: AgentContext, id: int) -> str:
    try:
        async with create_journal_store() as store:
            deleted = await store.delete_entry(
                user_id=ctx.user_id, entry_id=id)
    except Exception as e:
        logger.warning("journal_delete failed: %s", e)
        return f"Error: Could not delete journal entry: {e}"
    if not deleted:
        return f"Journal entry {id} not found"
    return f"Journal entry {id} deleted"


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
async def journal_edit(ctx: AgentContext, id: int,
                       oldString: str, newString: str,
                       replaceAll: bool | None = None) -> str:
    try:
        async with create_journal_store() as store:
            outcome = await store.replace_in_entry(
                user_id=ctx.user_id, entry_id=id,
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
        return f"Journal entry {id} not found"
    if outcome is ReplaceOutcome.TEXT_NOT_FOUND:
        return f"Text not found in journal entry {id}"
    if outcome is ReplaceOutcome.MULTIPLE_MATCHES:
        return (f"Error: Found multiple matches for oldString in journal "
                f"entry {id}. Provide more surrounding context to "
                f"make the match unique, or use replaceAll to replace all "
                f"occurrences.")
    return f"Journal entry {id} updated"


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
async def journal_topic_edit(ctx: AgentContext, id: int,
                             oldTopic: str, newTopic: str) -> str:
    try:
        async with create_journal_store() as store:
            outcome = await store.update_entry_topic(
                user_id=ctx.user_id, entry_id=id,
                old_topic=oldTopic, new_topic=newTopic)
    except StoreValidationError as e:
        # The store validated the edit; its message is the user-facing
        # text by contract.
        return f"Error: {e}"
    except Exception as e:
        logger.warning("journal_topic_edit failed: %s", e)
        return f"Error: Could not update journal entry topic: {e}"
    if outcome is ReplaceOutcome.NOT_FOUND:
        return f"Journal entry {id} not found"
    if outcome is ReplaceOutcome.TEXT_NOT_FOUND:
        return f"Topic does not match journal entry {id}"
    return f"Journal entry {id} topic updated"


@agent_tool(
    scope=BOTH,
    description=(
        "Search journal entries by text and topic; the way to recall past "
        "notes. Newest first by default. All parameters are optional. "
        "Page with cursor (the next_cursor id from the previous page); "
        "to start from a date instead, give start (ISO 8601) on a first "
        "page. Returns up to limit entries, plus next_cursor when more "
        "exist."),
    parameters={
        "cursor": {
            "type": "integer",
            "description": ("Entry id from the previous page's "
                            "next_cursor. Cannot be combined with start."),
        },
        "start": {
            "type": "string",
            "description": ("Optional ISO 8601 datetime with timezone for "
                            "the first page; inclusive. Cannot be combined "
                            "with cursor."),
        },
        "query": {
            "type": "string",
            "description": ("Full-text query (matches words, not "
                            "meanings; quote phrases)."),
        },
        "topic": {
            "type": "string",
            "description": "Exact topic filter (case-insensitive).",
        },
        "sortDescending": {
            "type": "boolean",
            "description": "Newest first (default) or oldest first.",
        },
        "limit": {
            "type": "integer",
            "description": "Max entries, 1-50 (default 50).",
        },
    })
async def journal_query(ctx: AgentContext,
                        cursor: int | None = None,
                        start: str | None = None,
                        query: str | None = None,
                        topic: str | None = None,
                        sortDescending: bool | None = None,
                        limit: int | None = None) -> str:
    # cursor resumes paging from an entry id; start positions the first page by
    # timestamp; each page uses exactly one of them.
    if cursor is not None and start:
        return ("Error: Provide either cursor or start, not both: cursor "
                "resumes paging from an entry id, start positions the "
                "first page by timestamp.")
    start_ms: int | None = None
    if start:
        try:
            start_ms = int(
                _parse_iso_timestamp(start, "start").timestamp() * 1000)
        except ValueError as e:
            return str(e)
    sort_descending = True if sortDescending is None else sortDescending
    max_entries = SEARCH_LIMIT if limit is None else limit
    if not 1 <= max_entries <= SEARCH_LIMIT:
        return f"Error: limit must be an integer between 1 and {SEARCH_LIMIT}."
    try:
        async with create_journal_store() as store:
            page = await store.search_page(
                user_id=ctx.user_id,
                cursor_id=cursor,
                start_ms=start_ms,
                query=query,
                topic=topic,
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
    if page.has_more and page.next_cursor is not None:
        out += (f"\n\nMore journal entries exist. To see more, call "
                f"journal_query with cursor={page.next_cursor}.")
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
