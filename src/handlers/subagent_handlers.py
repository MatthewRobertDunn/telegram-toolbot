from chat_update_context import ChatUpdateContext
from handlers.middleware import handler
from tools.subagent import subagent_list


@handler
async def subagent_list_command(ctx: ChatUpdateContext):
    """User-facing /subagent_list: same info as the subagent_list tool."""
    await ctx.messaging.send_message(f"```{await subagent_list(ctx)}```")
