"""Progress messages for long tool calls, currently not registered.

Deliberately absent from tool_registration.TOOL_MODULES: a stray import cannot
add a tool, because @agent_tool only runs from declare(), and the
stray-declaration check would reject it anyway.

To enable: add 'tools.status_tool' to TOOL_MODULES and call
status_tool.declare() from load_tools().
"""

from chat_update_context import ChatUpdateContext

_DESCRIPTION = ("Show a progress message during long tool runs (web search, "
                "python, etc.). Not for normal messages.")
_PARAMETERS = {
    "message": {
        "type": "string",
        "description": "Text to display.",
    },
}


async def display_status(ctx: ChatUpdateContext, message: str) -> str:
    await ctx.messaging.send_message(message, parse_mode="LLM")
    return "Status displayed."


def declare() -> None:
    from tool_registry import agent_tool

    agent_tool(description=_DESCRIPTION, parameters=_PARAMETERS)(
        display_status)
