"""Progress messages for long tool calls.

Not currently offered: the parent agent narrates well enough without it and
the extra turns were not worth the token cost. Kept registered as a
declaration (not imported by tool_registration) so it can be switched back
on by adding this module to TOOL_MODULES.
"""

from chat_update_context import ChatUpdateContext
from tool_registry import agent_tool


@agent_tool(
    description=("Displays a message to the user. Use this during long "
                 "running tool executions (web search, python, etc.) to "
                 "indicate progress. Do not use it for normal messages."),
    parameters={
        "message": {
            "type": "string",
            "description": "The text to display to the user.",
        },
    })
async def display_status(ctx: ChatUpdateContext, message: str) -> str:
    await ctx.messaging.send_message(message, parse_mode="LLM")
    return "Status displayed."
