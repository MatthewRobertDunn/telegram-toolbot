from chat_update_context import ChatUpdateContext
from command_registry import command
from sandbox import manager


@command('sandbox_reset', 'Destroy your sandbox and all its files')
async def sandbox_reset_command(ctx: ChatUpdateContext):
    """Destroy the user's sandbox, losing all its files."""
    await manager.destroy(ctx.user_id)
    await ctx.messaging.send_message(
        "Sandbox destroyed; a fresh one will be created on next use.")
