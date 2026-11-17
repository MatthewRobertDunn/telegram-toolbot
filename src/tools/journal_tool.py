"""Journal tools: persistent per-user journal for health, mood and notes.

All tools operate on the authenticated user's journal automatically.
Creation timestamps use the timezone
set with /timezone (UTC when never set).
"""
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from chat_update_context import AgentContext, context_tool
from tool_scope import subagent_tool

from journal_store import (
    JournalEntry,
    ReplaceOutcome,
    MAX_TOPIC_LENGTH,
    SEARCH_LIMIT,
    create_journal_store,
)

logger = logging.getLogger(__name__)


def _user_timezone(ctx: AgentContext) -> str:
    try:
        return ctx.settings.timezone
    except (KeyError, AttributeError):
        return "UTC"


def _now_in_user_timezone(ctx: AgentContext) -> tuple[str, int]:
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


def _coerce_limit(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"Error: limit must be an integer between 1 and {SEARCH_LIMIT}.")
    if isinstance(value, int):
        coerced = value
    elif isinstance(value, float):
        if value.is_integer():
            coerced = int(value)
        else:
            raise ValueError(
                f"Error: limit must be an integer between 1 and {SEARCH_LIMIT}.")
    elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
        coerced = int(value.strip())
    else:
        raise ValueError(
            f"Error: limit must be an integer between 1 and {SEARCH_LIMIT}.")
    if coerced < 1 or coerced > SEARCH_LIMIT:
        raise ValueError(
            f"Error: limit must be an integer between 1 and {SEARCH_LIMIT}.")
    return coerced


def _format_entry(record: JournalEntry) -> str:
    return (f"ID: {record.id}\n"
            f"Created: {record.created}\n"
            f"Topic: {record.topic}\n"
            f"Entry: {record.entry}")


@subagent_tool
@context_tool
async def journal_add(ctx: AgentContext, topic: str, entry: str) -> str:
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


@subagent_tool
@context_tool
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


@subagent_tool
@context_tool
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


@subagent_tool
@context_tool
async def journal_edit(ctx: AgentContext, id: int | float | str,
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


@subagent_tool
@context_tool
async def journal_topic_edit(ctx: AgentContext, id: int | float | str,
                              oldTopic: str, newTopic: str) -> str:
    try:
        entry_id = _coerce_id(id)
    except ValueError as e:
        return str(e)
    if not isinstance(oldTopic, str) or not oldTopic.strip():
        return "Error: oldTopic cannot be empty."
    if not isinstance(newTopic, str) or not newTopic.strip():
        return "Error: newTopic cannot be empty."
    if len(newTopic.strip()) > MAX_TOPIC_LENGTH:
        return (f"Error: Topic must be at most "
                f"{MAX_TOPIC_LENGTH} characters")
    if oldTopic.strip() == newTopic.strip():
        return ("Error: No changes to apply: "
                "oldTopic and newTopic are identical.")
    try:
        async with create_journal_store() as store:
            outcome = await store.update_entry_topic(
                user_id=ctx.user_id, entry_id=entry_id,
                old_topic=oldTopic, new_topic=newTopic)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.warning("journal_topic_edit failed: %s", e)
        return f"Error: Could not update journal entry topic: {e}"
    if outcome is ReplaceOutcome.NOT_FOUND:
        return f"Journal entry {entry_id} not found"
    if outcome is ReplaceOutcome.TEXT_NOT_FOUND:
        return f"Topic does not match journal entry {entry_id}"
    return f"Journal entry {entry_id} topic updated"


@subagent_tool
@context_tool
async def journal_query(ctx: AgentContext,
                         cursor: str | None = None,
                         query: str | None = None,
                         topic: str | None = None,
                         sortDescending: bool | None = None,
                         limit: int | float | str | None = None) -> str:
    try:
        cursor_dt = (_parse_iso_timestamp(cursor, "cursor")
                    if isinstance(cursor, str) and cursor.strip() else None)
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
                cursor_ms=(int(cursor_dt.timestamp() * 1000)
                           if cursor_dt is not None else None),
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
    if page.has_more:
        if sort_descending:
            edge = min(records, key=lambda r: (r.created_epoch_ms, r.id))
        else:
            edge = max(records, key=lambda r: (r.created_epoch_ms, r.id))
        out += (f"\n\nMore journal entries exist. To see more, call "
                f"journal_query with cursor=\"{edge.created}\" "
                f"(keep the same sortDescending, query, topic and limit).")
    return out


@subagent_tool
@context_tool
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


JOURNAL_ADD = "journal_add"
JOURNAL_GET = "journal_get"
JOURNAL_DELETE = "journal_delete"
JOURNAL_EDIT = "journal_edit"
JOURNAL_TOPIC_EDIT = "journal_topic_edit"
JOURNAL_QUERY = "journal_query"
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
                "Retrieves one or more journal entries by their IDs. Use this to "
                "inspect entries, verify their contents before editing them, or "
                "retrieve their full contents after a search. Only entries "
                "belonging to the authenticated user can be retrieved."),
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Array of entry IDs to retrieve (max 50)."
                    }
                },
                "required": ["ids"],
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
            "name": JOURNAL_TOPIC_EDIT,
            "strict": False,
            "description": (
                "Changes the topic of a journal entry. "
                "Retrieve the entry with journal_get first if you do not "
                "already know its current topic. oldTopic must match the "
                "entry's current topic exactly; the edit fails otherwise. "
                "Prefer an existing topic from journal_list_topics; do not "
                "create unnecessary variations of existing topics."),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "ID of the entry to modify."
                    },
                    "oldTopic": {
                        "type": "string",
                        "description": "The current topic (must match exactly)"
                    },
                    "newTopic": {
                        "type": "string",
                        "description": "The new topic (must be different from oldTopic, max 128 characters)"
                    }
                },
                "required": ["id", "oldTopic", "newTopic"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": JOURNAL_QUERY,
            "strict": False,
            "description": (
                "Searches the authenticated user's journal by text "
                "and topic. This is the primary tool for recalling recorded "
                "information: past experiences, mood over time, symptoms, "
                "sleep or wellbeing patterns. All parameters are optional; "
                "call with no parameters to retrieve recent entries. "
                "Results are newest-first by default; set sortDescending to "
                "false for oldest-first. cursor is an optional exclusive "
                "paging cursor, ISO 8601 with timezone: with newest-first "
                "it returns only entries created before cursor, with "
                "oldest-first only entries created after cursor. At most "
                "limit entries are returned (default 50, maximum 50). When "
                "more matching entries exist beyond the returned page, the "
                "response ends with a follow-up line telling you which date "
                "to pass as cursor next; keep the other parameters the same. "
                "Full-text search matches "
                "words, not meanings: "
                "use concise terms or quoted phrases for exact phrases. "
                "Timestamps must be ISO 8601 with an explicit timezone or "
                "UTC indicator."),
            "parameters": {
                "type": "object",
                "properties": {
                    "cursor": {
                        "type": "string",
                        "description": "Optional exclusive paging cursor, ISO 8601 with timezone."
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional full-text search query."
                    },
                    "topic": {
                        "type": "string",
                        "description": "Optional exact topic filter (case-insensitive)."
                    },
                    "sortDescending": {
                        "type": "boolean",
                        "description": "Newest-first when true (default), oldest-first when false."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max entries to return, 1 to 50 (default 50)."
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
    tool_map[JOURNAL_TOPIC_EDIT] = journal_topic_edit
    tool_map[JOURNAL_QUERY] = journal_query
    tool_map[JOURNAL_LIST_TOPICS] = journal_list_topics
