import asyncio
import logging
from telegram import constants
from telegram.error import BadRequest
from telegram.helpers import escape_markdown
from chat_update_context import ChatUpdateContext
import html

logger = logging.getLogger(__name__)
MAX_MSG_LEN = 4000
# Telegram's sendRichMessage limit is 32,768 UTF-8 chars / 500 blocks per
# message. Chunk well below to stay clear of the block-count cap.
MAX_RICH_MSG_LEN = 32000


async def send_rich_message(ctx: ChatUpdateContext, text: str):
    # PTB doesn't expose sendRichMessage yet (Bot API 10.1); we use the raw
    # request layer as recommended by the PTB maintainers until it ships.
    await ctx.bot._post(
        "sendRichMessage",
        data={
            "chat_id": ctx.chat_id,
            "rich_message": {"markdown": text},
        },
    )


async def _send_chunk_maybe_fallback(ctx: ChatUpdateContext, text: str, parse_mode, disable_web_page_preview):
    try:
        return await ctx.bot.send_message(chat_id=ctx.chat_id, text=text, parse_mode=parse_mode,
                                          disable_web_page_preview=disable_web_page_preview)
    except BadRequest as e:
        if parse_mode and 'parse entities' in str(e):
            escaped = escape_markdown(text, version=1)
            return await ctx.bot.send_message(chat_id=ctx.chat_id, text=escaped, parse_mode=parse_mode,
                                              disable_web_page_preview=disable_web_page_preview)
        raise


def _split_at_linebreak(text: str, limit: int) -> list[str]:
    chunks = []
    while len(text) > limit:
        break_at = text.rfind("\n", 0, limit)
        if break_at == -1:
            break_at = limit
        chunks.append(text[:break_at])
        text = text[break_at:]
        if text.startswith("\n"):
            text = text[1:]
    if text:
        chunks.append(text)
    return chunks


async def send_message(ctx: ChatUpdateContext, text: str, parse_mode='Markdown', disable_web_page_preview=True):
    if parse_mode == 'LLM':
        for chunk in _split_at_linebreak(text, MAX_RICH_MSG_LEN):
            await send_rich_message(ctx, chunk)
    else:
        for chunk in _split_at_linebreak(text, MAX_MSG_LEN):
            await _send_chunk_maybe_fallback(
                ctx,
                chunk,
                parse_mode,
                disable_web_page_preview,
            )


async def send_preformatted_message(ctx: ChatUpdateContext, text: str):
    OVERHEAD = len("<pre></pre>")
    chunk_len = MAX_MSG_LEN - OVERHEAD
    for chunk in _split_at_linebreak(text, chunk_len):
        await ctx.bot.send_message(
            chat_id=ctx.chat_id,
            text=f"<pre>{html.escape(chunk)}</pre>",
            parse_mode="HTML",
        )


async def send_image(ctx: ChatUpdateContext, photo: bytes, caption: str = ""):
    await ctx.bot.send_photo(
        chat_id=ctx.chat_id,
        photo=photo,
        caption=caption,
    )


async def _send_typing_periodically(ctx: ChatUpdateContext, stop_event: asyncio.Event):
    while not stop_event.is_set():
        try:
            await ctx.bot.send_chat_action(chat_id=ctx.chat_id, action=constants.ChatAction.TYPING)
        except Exception as e:
            logger.exception("Error occurred while sending typing action: %s", e)
            pass
        await asyncio.sleep(5)


async def run_with_typing(ctx: ChatUpdateContext, coro):
    typing_stop = asyncio.Event()
    typing_task = asyncio.create_task(
        _send_typing_periodically(ctx, typing_stop))
    try:
        return await coro
    finally:
        typing_stop.set()
        await typing_task
