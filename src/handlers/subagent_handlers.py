from chat_update_context import ChatUpdateContext
from command_registry import command
from tools.subagent import subagent_list


@command('subagent_list', 'List running background subagents')
async def subagent_list_command(ctx: ChatUpdateContext):
    """User-facing /subagent_list: same info as the subagent_list tool."""
    await ctx.messaging.send_message(f"```{await subagent_list(ctx)}```")
