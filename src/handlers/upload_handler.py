import logging
import os
import shlex
import time
from typing import Any

from chat_update_context import ChatUpdateContext
from handlers.middleware import handler
from handlers.run_with_tools import run_with_tools
from messaging import run_with_typing, send_message
from sandbox import manager, UPLOADS_DIR
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


def _unique_name(name: str, existing: set[str]) -> str:
    if name not in existing:
        return name
    stem, ext = os.path.splitext(name)
    counter = 1
    while f"{stem}-{counter}{ext}" in existing:
        counter += 1
    return f"{stem}-{counter}{ext}"


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
    listing = await manager.exec_in_sandbox(
        ctx.user_id, f"mkdir -p {UPLOADS_DIR} && ls -1 {UPLOADS_DIR}")
    if listing["returncode"] != 0:
        await send_message(
            ctx,
            "Error preparing the sandbox uploads folder: "
            f"{listing['stderr'].strip()[:500]}")
        return
    existing = set(listing["stdout"].splitlines())
    path = f"{UPLOADS_DIR}/{_unique_name(name, existing)}"

    write = await manager.exec_in_sandbox(
        ctx.user_id, f"cat > {shlex.quote(path)}", stdin_data=data)
    if write["returncode"] != 0:
        await send_message(
            ctx,
            "Error storing the file in the sandbox: "
            f"{write['stderr'].strip()[:500]}")
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
    ctx.conversation.append({
        'role': 'user',
        'content': content
    })

    gpt_response = await run_with_typing(ctx, run_with_tools(ctx))

    await send_message(ctx, gpt_response, parse_mode='LLM')
    ctx.conversation.append({
        'role': 'assistant',
        'content': gpt_response
    })
