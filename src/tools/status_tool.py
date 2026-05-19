from chat_update_context import ChatUpdateContext, context_tool
from messaging import send_message

TOOL_NAME = "display_status"
TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "strict": False,
            "description": "Displays a message to the user. Use this during long running tool executions (web search, python, etc.) to indicate progress. Do not use it for normal messages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The text to display to the user."
                    }
                },
                "required": ["message"],
                "additionalProperties": False
            }
        }
    },
]


@context_tool
async def display_status(ctx: ChatUpdateContext, message: str) -> str:
    await send_message(ctx, message, parse_mode="LLM")
    return "Status displayed."


def register(tools_list: list[dict], tool_map: dict):
    tools_list.extend(TOOL_DEFINITIONS)
    tool_map[TOOL_NAME] = display_status