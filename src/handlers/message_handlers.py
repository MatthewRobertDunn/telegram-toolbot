from telegram.constants import ChatType

from chat_update_context import ChatUpdateContext
from handlers.middleware import handler
from handlers.generation_queue import submit_generation


def stop_generation(ctx: ChatUpdateContext) -> bool:
    task = ctx.generation_task
    if task is None or task.done():
        ctx.generation_task = None
        return False
    task.cancel()
    ctx.generation_task = None
    return True


async def group_text_message_handler(ctx: ChatUpdateContext):
    text = ctx.message.text or ctx.message.caption
    should_respond = f"@{ctx.bot.username}" in text # type: ignore
    text = text.replace(f"@{ctx.bot.username}", "").strip() # type: ignore

    sender = ctx.message.from_user  # type: ignore[union-attr]
    name = f"{sender.first_name or ''} {sender.last_name or ''}".strip()  # type: ignore[union-attr]

    if not should_respond:
        # Record the ambient chatter without answering it (the reply to a
        # later mention will see it in the history). No queue involved:
        # nothing is being submitted to the model.
        ctx.conversation.add_user(f"{name}: {text}")
        return

    await submit_generation(ctx, f"{name}: {text}")


@handler
async def text_message_handler(ctx: ChatUpdateContext):
    text = ctx.message.text or ctx.message.caption
    if not text:
        return

    if ctx.message.chat.type == ChatType.GROUP or ctx.message.chat.type == ChatType.SUPERGROUP:
        await group_text_message_handler(ctx)
        return

    await submit_generation(ctx, text)
