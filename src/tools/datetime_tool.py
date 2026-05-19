from datetime import datetime, timezone


async def get_current_datetime() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S UTC (%A)")


TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "strict": False,
            "description": "Get the current date, time, and day of the week in UTC. Use this when you need to know the current time, date, or day.",
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
