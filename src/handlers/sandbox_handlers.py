from chat_update_context import ChatUpdateContext
from sandbox import manager

from .middleware import handler


@handler
async def sandbox_reset_command(ctx: ChatUpdateContext):
    """Destroy the user's sandbox, losing all its files."""
    await manager.destroy(ctx.user_id)
    await ctx.messaging.send_message(
        "Sandbox destroyed; a fresh one will be created on next use.")
