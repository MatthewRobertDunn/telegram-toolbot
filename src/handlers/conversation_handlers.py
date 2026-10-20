"""Commands operating on the conversation itself: regenerating the last
turn and compacting the history."""

import asyncio
import logging

from chat_update_context import ChatUpdateContext
from compaction import (DEFAULT_KEEP_TOKENS, CompactionError, compact_context)
from conversation import CUSTOM
from handlers.middleware import handler
from handlers.generation_queue import submit_generation, submit_task
from messaging import send_message, run_with_typing

logger = logging.getLogger(__name__)

BUSY_RESPONSE = ("A response is currently being generated; "
                 "use /stop first.")


async def refuse_while_generating(ctx: ChatUpdateContext) -> bool:
    """True if a response or compaction is in flight; the caller should
    stop and the user was told to /stop first."""
    if ctx.is_busy:
        await send_message(ctx, BUSY_RESPONSE)
        return True
    return False


async def delete_from_telegram(ctx: ChatUpdateContext, message_id) -> None:
    """Delete the Telegram message with that id, logging failures. Note
    that chunked replies only store their last chunk, so earlier chunks
    would be left behind."""
    if message_id is None:
        return
    try:
        await ctx.bot.delete_message(ctx.chat_id, message_id)
    except Exception as e:
        logger.warning("Could not delete telegram message %s: %s",
                       message_id, e)


def stored_message_id(message: dict):
    """The Telegram message id recorded in a message's custom metadata."""
    return (message.get(CUSTOM) or {}).get('telegram_message_id')


def message_preview(message: dict, limit: int = 100) -> str:
    """A short one-line identification of a message's contents, for
    feedback like /undo. Multimodal parts become placeholders and tool
    payloads are shown as raw json."""
    content = message.get('content')
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for part in content or []:
            if not isinstance(part, dict):
                continue
            if part.get('type') == 'text':
                parts.append(part.get('text') or '')
            elif part.get('type') == 'image_url':
                parts.append('[image]')
        text = ' '.join(parts)
    else:
        text = ''
    if not text:
        for tool_call in message.get('tool_calls') or []:
            function = (tool_call or {}).get('function') or {}
            text = (f"{function.get('name') or ''}"
                    f"({function.get('arguments') or ''})")
            if text != '()':
                break
        if not text and message.get('role') == 'tool':
            text = message.get('content') or ''
    text = ' '.join(str(text).split())
    if not text:
        return '(no content)'
    if len(text) <= limit:
        return text
    return text[:limit]


@handler
async def undo_command(ctx: ChatUpdateContext):
    """Remove the last conversation message, deleting it from Telegram
    when it has a known message id."""
    if await refuse_while_generating(ctx):
        return

    messages = ctx.conversation.messages
    if not messages or messages[-1].get('role') == 'system':
        await send_message(ctx, "Nothing to undo.")
        return

    last = messages.pop()
    await delete_from_telegram(ctx, stored_message_id(last))
    role = str(last.get('role') or 'message').capitalize()
    await send_message(
        ctx,
        f"Removed message ({role}):\n```\n{message_preview(last)}\n```")


@handler
async def regenerate_command(ctx: ChatUpdateContext):
    """Drop the last assistant turn (conversation and Telegram) and answer
    again, or - if the history ends on the user's turn - just answer it
    without re-adding it."""
    if await refuse_while_generating(ctx):
        return

    messages = ctx.conversation.messages
    # An unfinished assistant turn leaves its tool results trailing; skip
    # past them so the real last speaker is what decides the branch.
    while messages and messages[-1].get('role') == 'tool':
        messages.pop()
    if not messages or messages[-1].get('role') == 'system':
        await send_message(ctx, "Nothing to regenerate yet.")
        return

    last = messages[-1]
    if last.get('role') == 'assistant':
        messages.pop()
        await delete_from_telegram(ctx, stored_message_id(last))

    # content=None: the queued item runs the model without appending a
    # user turn - after the pop the history ends on the original prompt,
    # unchanged, and the model answers it again.
    await submit_generation(ctx, None)


@handler
async def compact_command(ctx: ChatUpdateContext):
    keep_tokens = DEFAULT_KEEP_TOKENS
    if ctx.command_text:
        try:
            keep_tokens = int(ctx.command_text)
            if keep_tokens < 1:
                raise ValueError
        except ValueError:
            await send_message(
                ctx,
                f'Usage: `/{ctx.command_name} [tokens of recent history to keep]`')
            return

    await submit_task(ctx, run_compaction(ctx, keep_tokens))


async def run_compaction(ctx: ChatUpdateContext, keep_tokens: int) -> None:
    """One unit of work: compact the conversation, reporting progress and
    the token savings. The task is tracked in ctx.generation_task, so
    /stop cancels it just like a reply generation."""
    ctx.generation_task = inner = asyncio.create_task(
        run_with_typing(
            ctx,
            compact_context(
                ctx,
                keep_tokens=keep_tokens,
                on_start=lambda: send_message(ctx, 'Starting Compaction...'))))
    try:
        result = await inner
    except asyncio.CancelledError:
        if not inner.cancelled():
            # This job itself was cancelled (bot shutdown).
            inner.cancel()
            raise
        await send_message(ctx, 'Compaction stopped.')
    except CompactionError as e:
        await send_message(ctx, str(e))
    else:
        await send_message(ctx, result.summary, parse_mode='LLM')
        await send_message(
            ctx,
            f'Compaction complete: {result.tokens_before:,} tokens '
            f'-> {result.tokens_after:,} tokens')
    finally:
        ctx.generation_task = None
