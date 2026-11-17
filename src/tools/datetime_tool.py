import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from chat_update_context import AgentContext
from tool_registry import BOTH, agent_tool

logger = logging.getLogger(__name__)


@agent_tool(
    scope=BOTH,
    description=("Get the current date, time, and day of the week in the "
                 "user's timezone (set with /timezone, UTC when never set). "
                 "Returns a full ISO 8601 timestamp with UTC offset, plus "
                 "weekday and IANA timezone name."))
async def get_current_datetime(ctx: AgentContext) -> str:
    settings = ctx.user_data.settings
    if settings is None:
        tz_name = "UTC"
    else:
        try:
            tz_name = settings.timezone
        except AttributeError:
            tz_name = "UTC"
    try:
        tz = ZoneInfo(tz_name)
    except (ValueError, ZoneInfoNotFoundError):
        logger.warning("Unknown timezone '%s', falling back to UTC", tz_name)
        tz_name = "UTC"
        tz = timezone.utc
    now = datetime.now(tz)
    return f"{now.isoformat()} ({now.strftime('%A')}, {tz_name})"
