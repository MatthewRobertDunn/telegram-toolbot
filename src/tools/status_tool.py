"""Progress messages for long tool calls. PARKED.

Not currently offered: the parent agent narrates well enough without it and
the extra turns were not worth the token cost.

This module is deliberately absent from tool_registration.TOOL_MODULES, and
importing it declares nothing: the @agent_tool decorator only runs from
declare(), so a stray import cannot join a tool to the surface (and
tool_registration's stray-declaration check would reject it if it did).

To switch the tool back on: add 'tools.status_tool' to TOOL_MODULES and
call status_tool.declare() from load_tools().
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
    """Declare display_status as a parent tool; call this to re-enable."""
    from tool_registry import agent_tool

    agent_tool(description=_DESCRIPTION, parameters=_PARAMETERS)(
        display_status)
