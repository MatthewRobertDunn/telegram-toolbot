import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from chat_update_context import ChatUpdateContext, context_tool

logger = logging.getLogger(__name__)


@context_tool
async def get_current_datetime(ctx: ChatUpdateContext) -> str:
    try:
        tz_name = ctx.settings.timezone
    except (KeyError, AttributeError):
        tz_name = "UTC"
    try:
        tz = ZoneInfo(tz_name)
    except (ValueError, ZoneInfoNotFoundError):
        logger.warning("Unknown timezone '%s', falling back to UTC", tz_name)
        tz_name = "UTC"
        tz = timezone.utc
    now = datetime.now(tz)
    return f"{now.isoformat()} ({now.strftime('%A')}, {tz_name})"


TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "strict": False,
            "description": "Get the current date, time, and day of the week in the user's timezone (set with /timezone, UTC when never set). Returns a full ISO 8601 timestamp with UTC offset, plus weekday and IANA timezone name.",
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
    tool_map["get_current_datetime"] = get_current_datetime
