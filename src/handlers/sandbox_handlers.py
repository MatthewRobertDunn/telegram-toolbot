from chat_update_context import ChatUpdateContext
from messaging import send_message
from sandbox import manager

from .middleware import handler


@handler
async def sandbox_reset_command(ctx: ChatUpdateContext):
    """Destroy the user's sandbox, losing all its files."""
    await manager.destroy(ctx.user_id)
    await send_message(
        ctx, "Sandbox destroyed; a fresh one will be created on next use.")
