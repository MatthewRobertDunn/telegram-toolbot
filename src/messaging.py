import asyncio
from telegram import constants
from telegram.error import BadRequest
from telegram.helpers import escape_markdown
from chat_update_context import ChatUpdateContext
from telegramify_markdown import ContentType, telegramify
from telegramify_markdown.config import get_runtime_config
import html
cfg = get_runtime_config()
cfg.markdown_symbol.heading_level_1 = "🔴"
cfg.markdown_symbol.heading_level_2 = "🟠"
cfg.markdown_symbol.heading_level_3 = "🟡"
cfg.markdown_symbol.heading_level_4 = "⚪"
MAX_MSG_LEN = 4000


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
        results = await telegramify(text, max_message_length=MAX_MSG_LEN, min_file_lines=100)
        for item in results:
            if item.content_type == ContentType.TEXT:
                await ctx.bot.send_message(
                    ctx.chat_id,
                    item.text,  # type: ignore
                    entities=[e.to_dict()
                              for e in item.entities],  # type: ignore
                )
            elif item.content_type == ContentType.PHOTO:
                await ctx.bot.send_photo(
                    ctx.chat_id,
                    filename=item.file_name,  # type: ignore
                    photo=item.file_data,  # type: ignore
                    caption=item.caption_text or None,  # type: ignore
                    caption_entities=[
                        e.to_dict() for e in item.caption_entities] or None,  # type: ignore
                )
            elif item.content_type == ContentType.FILE:
                await ctx.bot.send_document(
                    ctx.chat_id,
                    filename=item.file_name,  # type: ignore
                    document=item.file_data,  # type: ignore
                    caption=item.caption_text or None,  # type: ignore
                    caption_entities=[
                        e.to_dict() for e in item.caption_entities] or None,  # type: ignore
                )
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
            print(f"Error occurred while sending typing action: {e}")
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
