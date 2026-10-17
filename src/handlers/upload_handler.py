import logging
import os
import time
from typing import Any

from chat_update_context import ChatUpdateContext
from handlers.middleware import handler
from handlers.run_llm import run_llm
from messaging import run_with_typing, send_message
from sandbox import manager
from tools.sandbox_tool import IMAGE_MIME_TYPES

logger = logging.getLogger(__name__)


def _get_media(msg: Any) -> Any | None:
    if msg.document is not None:
        return msg.document
    if msg.photo:
        return msg.photo[-1]
    return (msg.audio or msg.video or msg.voice or msg.video_note
            or msg.animation or msg.sticker)


def _fallback_name(media: Any, file: Any) -> str:
    kind = type(media).__name__.lower()
    if kind == "photosize":
        kind = "photo"
    ext = os.path.splitext(file.file_path or "")[1]
    return f"{kind}_{int(time.time())}{ext}"


@handler
async def upload_handler(ctx: ChatUpdateContext):
    msg = ctx.message
    media = _get_media(msg)
    if media is None:
        return

    try:
        file = await ctx.bot.get_file(media.file_id)
        data = bytes(await file.download_as_bytearray())
    except Exception as e:
        logger.exception("error downloading upload: %s", e)
        await send_message(
            ctx,
            "Error processing file, encountered an error while downloading "
            f"the file: {e}")
        return

    name = getattr(media, "file_name", None) or _fallback_name(media, file)
    path, error = await manager.store_file(ctx.user_id, data, name)
    if error is not None:
        await send_message(ctx, error)
        return
    logger.info("Stored upload %s (%d bytes) for user %d",
                path, len(data), ctx.user_id)

    if os.path.splitext(name)[1].lower() in IMAGE_MIME_TYPES:
        hint = "the read_image tool"
    else:
        hint = "the read tool"
    notification = (
        "[The user uploaded a file to their sandbox: "
        f"{path} ({len(data):,} bytes). You can inspect it with "
        f"{hint} if it is relevant.]")
    caption = msg.caption
    content = f"{caption}\n\n{notification}" if caption else notification
    ctx.conversation.add_user(content)

    gpt_response = await run_with_typing(ctx, run_llm(ctx))

    await send_message(ctx, gpt_response, parse_mode='LLM')
    ctx.conversation.add_assistant(gpt_response)
