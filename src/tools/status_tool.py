"""Progress messages for long tool calls.

Not currently offered: the parent agent narrates well enough without it and
the extra turns were not worth the token cost. Kept registered as a
declaration (not imported by tool_registration) so it can be switched back
on by adding this module to TOOL_MODULES.
"""

from chat_update_context import ChatUpdateContext
from tool_registry import agent_tool


@agent_tool(
    description=("Show a progress message during long tool runs (web search, "
                 "python, etc.). Not for normal messages."),
    parameters={
        "message": {
            "type": "string",
            "description": "Text to display.",
        },
    })
async def display_status(ctx: ChatUpdateContext, message: str) -> str:
    await ctx.messaging.send_message(message, parse_mode="LLM")
    return "Status displayed."
