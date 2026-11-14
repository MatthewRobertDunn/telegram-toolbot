"""Journal tools: persistent per-user journal for health, mood and notes.

All tools operate on the authenticated user's journal automatically.
Creation timestamps use the timezone
set with /set_timezone (UTC when never set).
"""
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from chat_update_context import ChatUpdateContext, context_tool

from journal_store import (
    JournalEntry,
    ReplaceOutcome,
    MAX_TOPIC_LENGTH,
    create_journal_store,
)

logger = logging.getLogger(__name__)


def _user_timezone(ctx: ChatUpdateContext) -> str:
    try:
        return ctx.settings.timezone
    except (KeyError, AttributeError):
        return "UTC"


def _now_in_user_timezone(ctx: ChatUpdateContext) -> tuple[str, int]:
    name = _user_timezone(ctx)
    try:
        tz = ZoneInfo(name)
    except (ValueError, ZoneInfoNotFoundError):
        logger.warning("Unknown timezone '%s', falling back to UTC", name)
        tz = timezone.utc
    now = datetime.now(tz)
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


def _coerce_id(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("Error: Invalid journal entry id.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError("Error: Invalid journal entry id.")
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise ValueError("Error: Invalid journal entry id.")


def _format_entry(record: JournalEntry) -> str:
    return (f"ID: {record.id}\n"
            f"Created: {record.created}\n"
            f"Topic: {record.topic}\n"
            f"Entry: {record.entry}")


@context_tool
async def journal_add(ctx: ChatUpdateContext, topic: str, entry: str) -> str:
    if not isinstance(topic, str) or not topic.strip():
        return "Error: Topic cannot be empty"
    if not isinstance(entry, str) or not entry.strip():
        return "Error: Entry cannot be empty"
    if len(topic.strip()) > MAX_TOPIC_LENGTH:
        return (f"Error: Topic must be at most "
                f"{MAX_TOPIC_LENGTH} characters")
    created, created_epoch_ms = _now_in_user_timezone(ctx)
    try:
        async with create_journal_store() as store:
            record = await store.add_entry(
                user_id=ctx.user_id, topic=topic, entry=entry,
                created=created, created_epoch_ms=created_epoch_ms)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.warning("journal_add failed: %s", e)
        return f"Error: Could not save journal entry: {e}"
    return f"Created journal entry id:{record.id} created:{record.created} topic:{record.topic}"


@context_tool
async def journal_get(ctx: ChatUpdateContext, id: int | float | str) -> str:
    try:
        entry_id = _coerce_id(id)
    except ValueError as e:
        return str(e)
    try:
        async with create_journal_store() as store:
            record = await store.get_entry(
                user_id=ctx.user_id, entry_id=entry_id)
    except Exception as e:
        logger.warning("journal_get failed: %s", e)
        return f"Error: Could not retrieve journal entry: {e}"
    if record is None:
        return f"Journal entry {entry_id} not found"
    return _format_entry(record)


@context_tool
async def journal_delete(ctx: ChatUpdateContext, id: int | float | str) -> str:
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


@context_tool
async def journal_edit(ctx: ChatUpdateContext, id: int | float | str,
                       oldString: str, newString: str,
                       replaceAll: bool | None = None) -> str:
    try:
        entry_id = _coerce_id(id)
    except ValueError as e:
        return str(e)
    if not isinstance(oldString, str) or not oldString:
        return "Error: oldString cannot be empty."
    if not isinstance(newString, str):
        return "Error: Invalid replacement text"
    if oldString == newString:
        return ("Error: No changes to apply: "
                "oldString and newString are identical.")
    try:
        async with create_journal_store() as store:
            outcome = await store.replace_in_entry(
                user_id=ctx.user_id, entry_id=entry_id,
                old_string=oldString, new_string=newString,
                replace_all=bool(replaceAll))
    except ValueError as e:
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


@context_tool
async def journal_search(ctx: ChatUpdateContext,
                         start: str | None = None,
                         end: str | None = None,
                         query: str | None = None,
                         topic: str | None = None) -> str:
    try:
        start_dt = (_parse_iso_timestamp(start, "start")
                    if isinstance(start, str) and start.strip() else None)
        end_dt = (_parse_iso_timestamp(end, "end")
                  if isinstance(end, str) and end.strip() else None)
    except ValueError as e:
        return str(e)
    if start_dt is not None and end_dt is not None and start_dt >= end_dt:
        return "Error: Start timestamp must be earlier than end timestamp"
    try:
        async with create_journal_store() as store:
            records = await store.search_entries(
                user_id=ctx.user_id,
                start_ms=(int(start_dt.timestamp() * 1000)
                          if start_dt is not None else None),
                end_ms=(int(end_dt.timestamp() * 1000)
                        if end_dt is not None else None),
                query=query if isinstance(query, str) and query.strip() else None,
                topic=topic if isinstance(topic, str) and topic.strip() else None)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.warning("journal_search failed: %s", e)
        return f"Error: Could not search journal: {e}"
    if not records:
        return "No journal entries found."
    noun = "entry" if len(records) == 1 else "entries"
    return (f"Found {len(records)} journal {noun}:\n\n"
            + "\n\n".join(_format_entry(r) for r in records))


@context_tool
async def journal_list_topics(ctx: ChatUpdateContext) -> str:
    try:
        async with create_journal_store() as store:
            topics = await store.list_topics(user_id=ctx.user_id)
    except Exception as e:
        logger.warning("journal_list_topics failed: %s", e)
        return f"Error: Could not list journal topics: {e}"
    if not topics:
        return "No journal topics yet."
    return "\n".join(topics)


JOURNAL_ADD = "journal_add"
JOURNAL_GET = "journal_get"
JOURNAL_DELETE = "journal_delete"
JOURNAL_EDIT = "journal_edit"
JOURNAL_SEARCH = "journal_search"
JOURNAL_LIST_TOPICS = "journal_list_topics"

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": JOURNAL_ADD,
            "strict": False,
            "description": (
                "Creates a new journal entry in the authenticated user's "
                "journal. Use this whenever the user shares information that "
                "should be retained, such as health, mood, symptoms, sleep, "
                "experiences or personal reflections. Prefer an existing "
                "topic from journal_list_topics; do not create unnecessary "
                "variations of existing topics. Preserve the user's meaning "
                "and relevant details; do not invent information. The "
                "creation timestamp is generated automatically in the user's "
                "timezone."),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Category or subject, max 128 characters."
                    },
                    "entry": {
                        "type": "string",
                        "description": "Journal entry content, plain text."
                    }
                },
                "required": ["topic", "entry"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": JOURNAL_GET,
            "strict": False,
            "description": (
                "Retrieves a complete journal entry by its ID. Use this to "
                "inspect an entry, verify its contents before editing it, or "
                "retrieve its full contents after a search. Only entries "
                "belonging to the authenticated user can be retrieved."),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "ID of the entry to retrieve."
                    }
                },
                "required": ["id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": JOURNAL_DELETE,
            "strict": False,
            "description": (
                "Permanently deletes a journal entry."),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "ID of the entry to delete."
                    }
                },
                "required": ["id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": JOURNAL_EDIT,
            "strict": False,
            "description": (
                "Performs exact string replacements in a journal entry. "
                "Retrieve the entry with journal_get first if you do not already have its "
                "current contents. Matching is exact and case-sensitive. "
                "The edit will FAIL if `oldString` is not found in the "
                "entry. The edit will FAIL if `oldString` is found multiple "
                "times in the entry: either provide a larger string with "
                "more surrounding context to make it unique or use "
                "`replaceAll` to change every instance. The original "
                "creation timestamp is preserved. Use an empty `newString` "
                "to delete a section of text."),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "ID of the entry to modify."
                    },
                    "oldString": {
                        "type": "string",
                        "description": "The text to replace"
                    },
                    "newString": {
                        "type": "string",
                        "description": "The text to replace it with (must be different from oldString)"
                    },
                    "replaceAll": {
                        "type": "boolean",
                        "description": "Replace all occurrences of oldString (default false)"
                    }
                },
                "required": ["id", "oldString", "newString"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": JOURNAL_SEARCH,
            "strict": False,
            "description": (
                "Searches the authenticated user's journal by date, text "
                "and topic. This is the primary tool for recalling recorded "
                "information: past experiences, mood over time, symptoms, "
                "sleep or wellbeing patterns. All parameters are optional; "
                "use the narrowest useful criteria rather than retrieving "
                "everything. Full-text search matches words, not meanings: "
                "use concise relevant terms, multiple terms when several "
                "concepts matter, quoted phrases for exact phrases, and "
                "topic or date filters when the category or period is known. "
                "If nothing matches, try alternative wording or broader "
                "terms. Absence of results does not mean an event never "
                "happened. Date ranges are inclusive of start and exclusive "
                "of end; newest entries come first; at most 50 are returned. "
                "Timestamps must be ISO 8601 with an explicit timezone or "
                "UTC indicator."),
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "string",
                        "description": "Optional start timestamp, ISO 8601 with timezone."
                    },
                    "end": {
                        "type": "string",
                        "description": "Optional end timestamp, ISO 8601 with timezone."
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional full-text search query."
                    },
                    "topic": {
                        "type": "string",
                        "description": "Optional exact topic filter (case-insensitive)."
                    }
                },
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": JOURNAL_LIST_TOPICS,
            "strict": False,
            "description": (
                "Returns the distinct topics in the authenticated user's "
                "journal, one per line, sorted alphabetically. Use this to "
                "discover existing categories before creating or searching "
                "entries, and prefer an existing topic over coining a new "
                "variation."),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        }
    },
]


def register(tools_list: list[dict], tool_map: dict):
    tools_list.extend(TOOL_DEFINITIONS)
    tool_map[JOURNAL_ADD] = journal_add
    tool_map[JOURNAL_GET] = journal_get
    tool_map[JOURNAL_DELETE] = journal_delete
    tool_map[JOURNAL_EDIT] = journal_edit
    tool_map[JOURNAL_SEARCH] = journal_search
    tool_map[JOURNAL_LIST_TOPICS] = journal_list_topics
