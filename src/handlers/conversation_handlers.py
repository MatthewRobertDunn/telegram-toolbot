"""Commands operating on the conversation itself: regenerating the last
turn, impersonating the user, and compacting the history."""

import asyncio
import copy
import logging

from chat_update_context import ChatUpdateContext
from compaction import (DEFAULT_KEEP_TOKENS, CompactionError, compact_context)
from conversation import CUSTOM, Conversation
from handlers.middleware import handler
from handlers.generation_queue import submit_generation, submit_task
from handlers.impersonate_prompt import IMPERSONATE_PROMPT
from handlers.run_llm import (NO_CONTENT_MESSAGE, NO_RESPONSE_MESSAGE,
                              run_llm)
from messaging import send_message, run_with_typing

logger = logging.getLogger(__name__)

BUSY_RESPONSE = ("A response is currently being generated; "
                 "use /stop first.")

# Seconds before a command's own message and its status replies are
# deleted, so the chat is left tidy without leaving debris behind.
COMMAND_CLEAR_DELAY = 10


async def _schedule_command_clear(ctx: ChatUpdateContext,
                                  message_ids: list) -> None:
    """Delete the command message and the status replies it sent
    after a short delay. Fire-and-forget: the handler returns at once and
    this task cleans up later, so a slow Telegram delete never blocks the
    next update."""
    await asyncio.sleep(COMMAND_CLEAR_DELAY)
    await delete_from_telegram(ctx, message_ids)


def _command_message_id(ctx: ChatUpdateContext) -> int | None:
    """The Telegram message id of the command itself, if known."""
    return getattr(ctx.message, 'message_id', None)


async def refuse_while_generating(ctx: ChatUpdateContext) -> bool:
    """True if a response or compaction is in flight; the caller should
    stop and the user was told to /stop first."""
    if ctx.is_busy:
        await send_message(ctx, BUSY_RESPONSE)
        return True
    return False


async def delete_from_telegram(ctx: ChatUpdateContext, message_ids) -> None:
    """Delete the Telegram messages with those ids, logging failures."""
    for message_id in message_ids or []:
        try:
            await ctx.bot.delete_message(ctx.chat_id, message_id)
        except Exception as e:
            logger.warning("Could not delete telegram message %s: %s",
                           message_id, e)


def stored_message_ids(message: dict) -> list:
    """The Telegram message ids recorded in a message's custom metadata."""
    return (message.get(CUSTOM) or {}).get('telegram_message_ids') or []


def _tool_call_name(call_id, context: list | None) -> str:
    """The function name a tool result's tool_call_id refers to, looked
    up among the tool calls of the other messages in the conversation."""
    for message in context or []:
        for call in message.get('tool_calls') or []:
            if (call or {}).get('id') == call_id:
                return ((call or {}).get('function') or {}).get('name') or ''
    return ''


def message_preview(message: dict, limit: int = 100,
                    context: list | None = None) -> str:
    """A short identification of a message's contents, for feedback like
    /undo: the text inside a code fence, trimmed to `limit` characters.
    Tool calls and results are prefixed with the tool's name and dump
    their raw json; multimodal parts become placeholders."""
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
        tool_calls = message.get('tool_calls') or []
        if tool_calls:
            function = (tool_calls[0] or {}).get('function') or {}
            text = (f"{function.get('name') or ''} "
                    f"{function.get('arguments') or ''}")
        elif message.get('role') == 'tool':
            name = _tool_call_name(message.get('tool_call_id'), context)
            text = f"{name} {message.get('content') or ''}"
    text = ' '.join(str(text).split())
    if not text:
        return '(no content)'
    if len(text) > limit:
        text = text[:limit]
    return f"```\n{text}\n```"


@handler
async def undo_command(ctx: ChatUpdateContext):
    """Remove the tail of the conversation back to and including the last
    user prompt, deleting the removed messages from Telegram when they
    have a known message id. When a removed message references tool call
    ids, every message sharing those ids is stripped too - a call left
    without its results (or vice versa) can't be replayed.

    The /undo command message and the status replies it sends are deleted
    a short while later, so the chat is left tidy."""
    if await refuse_while_generating(ctx):
        return

    messages = ctx.conversation.messages
    if not messages or messages[-1].get('role') == 'system':
        message_ids = await send_message(ctx, "Nothing to undo.")
        asyncio.create_task(_schedule_command_clear(
            ctx, [_command_message_id(ctx), *message_ids]))
        return

    # Walk the history back to and including the last user turn: a whole
    # exchange (prompt + reply + any tool rounds) is removed in one go.
    last_user_index = -1
    for i, message in enumerate(messages):
        if message.get('role') == 'user':
            last_user_index = i

    if last_user_index == -1:
        # No user turn exists (e.g. only an assistant greeting); undo just
        # the last message, preserving the old single-message behaviour.
        last = messages.pop()
        await delete_from_telegram(ctx, stored_message_ids(last))

        removed = ctx.conversation.remove_tool_calls(last)
        for message in removed:
            await delete_from_telegram(ctx, stored_message_ids(message))

        role = str(last.get('role') or 'message').capitalize()
        message_ids = await send_message(
            ctx,
            f"Removed message ({role}):\n"
            f"{message_preview(last, context=messages + removed)}")
        asyncio.create_task(_schedule_command_clear(
            ctx, [_command_message_id(ctx), *message_ids]))
        return

    removed_messages = []
    while len(messages) > last_user_index:
        last = messages.pop()
        removed_messages.append(last)
        await delete_from_telegram(ctx, stored_message_ids(last))

        removed = ctx.conversation.remove_tool_calls(last)
        for message in removed:
            await delete_from_telegram(ctx, stored_message_ids(message))
            if message not in removed_messages:
                removed_messages.append(message)

    if len(removed_messages) == 1:
        last = removed_messages[0]
        role = str(last.get('role') or 'message').capitalize()
        message_ids = await send_message(
            ctx,
            f"Removed message ({role}):\n"
            f"{message_preview(last, context=messages + removed_messages)}")
    else:
        message_ids = await send_message(
            ctx,
            f"Removed {len(removed_messages)} message(s).\n"
            + "\n".join(f"- {message_preview(m)}"
                         for m in removed_messages))

    asyncio.create_task(_schedule_command_clear(
        ctx, [_command_message_id(ctx), *message_ids]))


@handler
async def regenerate_command(ctx: ChatUpdateContext):
    """Drop the last assistant turn (conversation and Telegram) and answer
    again, or - if the history ends on the user's turn - just answer it
    without re-adding it.

    The /regenerate command message is deleted a short while later, so
    the chat is left tidy: the queued generation replies on its own and
    is not touched."""
    if await refuse_while_generating(ctx):
        return

    messages = ctx.conversation.messages
    # An unfinished assistant turn leaves its tool results trailing; skip
    # past them so the real last speaker is what decides the branch.
    while messages and messages[-1].get('role') == 'tool':
        messages.pop()
    if not messages or messages[-1].get('role') == 'system':
        message_ids = await send_message(ctx, "Nothing to regenerate yet.")
        asyncio.create_task(_schedule_command_clear(
            ctx, [_command_message_id(ctx), *message_ids]))
        return

    last = messages[-1]
    if last.get('role') == 'assistant':
        messages.pop()
        await delete_from_telegram(ctx, stored_message_ids(last))

    # content=None: the queued item runs the model without appending a
    # user turn - after the pop the history ends on the original prompt,
    # unchanged, and the model answers it again.
    await submit_generation(ctx, None)
    # The command message itself goes away shortly after; the queued
    # generation replies on its own and is left alone.
    asyncio.create_task(_schedule_command_clear(
        ctx, [_command_message_id(ctx)]))


@handler
async def retry_command(ctx: ChatUpdateContext):
    """Answer the conversation again exactly as it stands: nothing is
    popped or deleted, a fresh reply is appended after the history."""
    if await refuse_while_generating(ctx):
        return

    messages = ctx.conversation.messages
    if not any(message.get('role') != 'system' for message in messages):
        await send_message(ctx, "Nothing to retry yet.")
        return

    # content=None: the queued item runs the model without appending a
    # user turn - the history is left alone, and the model answers it
    # again wherever it currently ends.
    await submit_generation(ctx, None)


@handler
async def impersonate_command(ctx: ChatUpdateContext):
    """Fake a user turn: a clone of the whole conversation (system head
    with the user profile, full history) is shown to the model, which
    impersonates the user to write their next message. The response is
    added to the real conversation as a user message and posted as a
    user-labelled reply.

    Runs as a queued unit of work like /compact, with the task tracked
    in ctx.generation_task, so /stop stops it like any other work."""
    if await refuse_while_generating(ctx):
        return

    messages = ctx.conversation.messages
    if not any(message.get('role') != 'system' for message in messages):
        message_ids = await send_message(ctx, "Nothing to impersonate yet.")
        asyncio.create_task(_schedule_command_clear(
            ctx, [_command_message_id(ctx), *message_ids]))
        return

    await submit_task(ctx, run_impersonation(ctx))
    # The command message itself goes away shortly after; the generated
    # user turn is real content and stays.
    asyncio.create_task(_schedule_command_clear(
        ctx, [_command_message_id(ctx)]))


async def run_impersonation(ctx: ChatUpdateContext) -> None:
    """One unit of work: clone the conversation, impersonate the user
    against the clone, then add the response to the real conversation as
    a user message and post it. The task is tracked in
    ctx.generation_task, so /stop cancels it just like a reply
    generation."""
    ctx.generation_task = inner = asyncio.create_task(
        run_with_typing(ctx, impersonate_turn(ctx)))
    try:
        await inner
    except asyncio.CancelledError:
        if not inner.cancelled():
            # This job itself was cancelled (bot shutdown).
            inner.cancel()
            raise
        await send_message(ctx, 'Impersonation stopped.')
    finally:
        ctx.generation_task = None


async def impersonate_turn(ctx: ChatUpdateContext) -> None:
    """The impersonation itself, runs as its own task."""
    # Deep-copy the whole conversation, custom Telegram ids included:
    # the clone must not share message dicts with the real history (and
    # to_openai() strips the custom keys on the way out anyway).
    clone = Conversation([
        copy.deepcopy(message) for message in ctx.conversation.messages])
    clone.add_user(IMPERSONATE_PROMPT)

    response = (await run_llm(ctx, conversation=clone, allow_tools=False)
                or '').strip()
    if not response or response in (NO_RESPONSE_MESSAGE, NO_CONTENT_MESSAGE):
        await send_message(
            ctx, 'Impersonation failed: the model produced nothing usable.')
        return

    # The real conversation gets the bare text as a user turn - the
    # "## User" heading is only presentation for Telegram. The posted
    # message ids are attached to it, so /undo and /regenerate can
    # delete it from the chat too.
    message_ids = await send_message(ctx, f'## User\n{response}',
                                     parse_mode='LLM')
    ctx.conversation.add_user(response, telegram_message_ids=message_ids)


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
    async def on_start() -> None:
        await send_message(ctx, 'Starting Compaction...')

    ctx.generation_task = inner = asyncio.create_task(
        run_with_typing(
            ctx,
            compact_context(ctx,
                            keep_tokens=keep_tokens,
                            on_start=on_start)))
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
