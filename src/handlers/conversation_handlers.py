"""Commands operating on the conversation itself: regenerating the last
turn, impersonating the user, and compacting the history."""

import asyncio
import logging

from chat_update_context import ChatUpdateContext
from compaction import DEFAULT_KEEP_TOKENS, CompactionError, compact_context
from conversation import (ConversationMessageType, RegenerationAction,
                          is_thinking_message)
from handlers.generation_queue import (
    generate_response,
    submit_generation,
    submit_task,
)
from handlers.impersonate_prompt import IMPERSONATE_PROMPT
from handlers.middleware import handler
from handlers.run_llm import (NO_CONTENT_MESSAGE, NO_RESPONSE_MESSAGE,
                              run_llm)
from handlers.supervision import run_tracked
from messages import iter_parts, telegram_ids_of
# Re-exports: the token accounting lives in tokens.py, but other branches
# and tests import these names from here.
from tokens import estimate_conversation_tokens, token_count

logger = logging.getLogger(__name__)

# Seconds before a command's own message and its status replies are
# deleted, so the chat is left tidy without leaving debris behind.
COMMAND_CLEAR_DELAY = 10

# Strong references to the fire-and-forget cleanup tasks: asyncio keeps
# only weak references to running tasks, so an unreferenced one could be
# garbage-collected mid-sleep.
_PENDING_CLEAR_TASKS: set[asyncio.Task] = set()


async def _schedule_command_clear(ctx: ChatUpdateContext,
                                  message_ids: list) -> None:
    """Delete the command message and the status replies it sent
    after a short delay. Fire-and-forget: the handler returns at once and
    this task cleans up later, so a slow Telegram delete never blocks the
    next update."""
    await asyncio.sleep(COMMAND_CLEAR_DELAY)
    await delete_from_telegram(ctx, message_ids)


def _clear_command_later(ctx: ChatUpdateContext, message_ids) -> None:
    """Schedule deletion of the command message plus the status replies
    `message_ids` a short while later. Fire-and-forget, with a strong
    reference kept to the task (asyncio does not keep one)."""
    task = asyncio.create_task(_schedule_command_clear(
        ctx, [_command_message_id(ctx), *message_ids]))
    _PENDING_CLEAR_TASKS.add(task)
    task.add_done_callback(_PENDING_CLEAR_TASKS.discard)


def _command_message_id(ctx: ChatUpdateContext) -> int | None:
    """The Telegram message id of the command itself, if known."""
    return getattr(ctx.message, 'message_id', None)


async def delete_from_telegram(ctx: ChatUpdateContext, message_ids) -> None:
    """Delete the Telegram messages with those ids, logging failures."""
    for message_id in message_ids or []:
        try:
            await ctx.messaging.delete_message(message_id)
        except Exception as e:
            logger.warning("Could not delete telegram message %s: %s",
                           message_id, e)


def stored_message_ids(message: dict) -> list:
    """The Telegram message ids recorded in a message's custom metadata."""
    return telegram_ids_of(message)


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
        for part in iter_parts(message):
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


def _has_content(messages: list) -> bool:
    """Whether the history holds anything beyond the system head and
    kept thinking blocks - anything the "Nothing to X yet." commands
    (/retry, /impersonate) could act on."""
    return any(message.get('role') != 'system'
               and not is_thinking_message(message)
               for message in messages)


@handler
async def undo_command(ctx: ChatUpdateContext):
    """Remove the tail of the conversation back to and including the last
    user prompt, deleting the removed messages from Telegram when they
    have a known message id. When a removed message references tool call
    ids, every message sharing those ids is stripped too - a call left
    without its results (or vice versa) can't be replayed.

    Runs as a queued unit of work like /compact, with the task tracked
    in ctx.generation_task, so /stop stops it like any other work.

    The /undo command message and the status replies it sends are deleted
    a short while later, so the chat is left tidy."""
    await submit_task(ctx, run_undo(ctx))


async def run_undo(ctx: ChatUpdateContext) -> None:
    """One unit of work: remove the last exchange. The task is tracked in
    ctx.generation_task, so /stop cancels it just like a reply
    generation."""
    await run_tracked(ctx, lambda: undo_turn(ctx),
                      stopped_message='Undo stopped.')


async def undo_turn(ctx: ChatUpdateContext) -> None:
    """The undo itself, runs as its own task.

    High-level orchestration only: the history trim lives in
    Conversation.undo_last_exchange, this just deletes the Telegram posts
    and reports what went."""
    removed_messages = ctx.conversation.undo_last_exchange()
    if not removed_messages:
        message_ids = await ctx.messaging.send_message("Nothing to undo.")
        _clear_command_later(ctx, message_ids)
        return

    for message in removed_messages:
        await delete_from_telegram(ctx, stored_message_ids(message))

    if len(removed_messages) == 1:
        last = removed_messages[0]
        if is_thinking_message(last):
            role = 'Thinking'
        else:
            role = str(last.get('role') or 'message').capitalize()
        context = ctx.conversation.messages + removed_messages
        message_ids = await ctx.messaging.send_message(
            f"Removed message ({role}):\n"
            f"{message_preview(last, context=context)}")
    else:
        message_ids = await ctx.messaging.send_message(
            f"Removed {len(removed_messages)} message(s).\n"
            + "\n".join(f"- {message_preview(m)}"
                        for m in removed_messages))

    _clear_command_later(ctx, message_ids)


@handler
async def regenerate_command(ctx: ChatUpdateContext):
    """Drop the last assistant turn (conversation and Telegram) and answer
    again, or - if the history ends on the user's turn - just answer it
    without re-adding it.

    When the last user turn is an impersonation, that turn is removed and
    the impersonation flow runs again instead.

    Runs as a queued unit of work like /compact, with the task tracked
    in ctx.generation_task, so /stop stops it like any other work.

    The /regenerate command message is deleted a short while later, so
    the chat is left tidy: the queued generation replies on its own and
    is not touched."""
    await submit_task(ctx, run_regeneration(ctx))


async def run_regeneration(ctx: ChatUpdateContext) -> None:
    """One unit of work: pop the last turn and generate again. The task
    is tracked in ctx.generation_task, so /stop cancels it just like a
    reply generation. The generation phase itself is tracked by
    generate_response / impersonate_turn, which overwrite the tracking;
    the finally in run_tracked is then idempotent."""
    await run_tracked(ctx, lambda: regeneration_turn(ctx),
                      stopped_message='Regeneration stopped.')


async def regeneration_turn(ctx: ChatUpdateContext) -> None:
    """The regeneration itself, runs as its own task.

    High-level orchestration only: the history trim lives in
    Conversation.prepare_regeneration, this just deletes the Telegram
    posts and runs the next flow (answer again or redo impersonation)."""
    removed, action = ctx.conversation.prepare_regeneration()
    for message in removed:
        await delete_from_telegram(ctx, stored_message_ids(message))

    if action is RegenerationAction.NOTHING:
        message_ids = await ctx.messaging.send_message("Nothing to regenerate yet.")
        _clear_command_later(ctx, message_ids)
        return

    if action is RegenerationAction.IMPERSONATE:
        # The last turn was an impersonation: it is already removed, so
        # run the impersonation flow again instead of answering as usual.
        # Already inside the worker: run the flow directly instead of
        # submitting a second queued item (which would deadlock waiting
        # for itself).
        await run_impersonation(ctx)
        _clear_command_later(ctx, [])
        return

    # content=None: run the model without appending a user turn - after
    # the trim the history ends on the original prompt, unchanged, and the
    # model answers it again. Already inside the worker: call directly
    # instead of submitting a second queued item.
    await generate_response(ctx, None)
    # The command message itself goes away shortly after; the queued
    # generation replies on its own and is left alone.
    _clear_command_later(ctx, [])


@handler
async def retry_command(ctx: ChatUpdateContext):
    """Answer the conversation again exactly as it stands: nothing is
    popped or deleted, a fresh reply is appended after the history.

    Runs as a queued unit of work like /compact, so it waits for any
    in-flight work instead of refusing."""
    if not _has_content(ctx.conversation.messages):
        await ctx.messaging.send_message("Nothing to retry yet.")
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
    if not _has_content(ctx.conversation.messages):
        message_ids = await ctx.messaging.send_message("Nothing to impersonate yet.")
        _clear_command_later(ctx, message_ids)
        return

    await submit_task(ctx, run_impersonation(ctx))
    #We don't clear the impersonate command because it helps the user know the text below is the impersonated text and not the original user text

async def run_impersonation(ctx: ChatUpdateContext) -> None:
    """One unit of work: clone the conversation, impersonate the user
    against the clone, then add the response to the real conversation as
    a user message and post it. The task is tracked in
    ctx.generation_task, so /stop cancels it just like a reply
    generation."""
    await run_tracked(
        ctx,
        lambda: ctx.messaging.run_with_typing(impersonate_turn(ctx)),
        stopped_message='Impersonation stopped.')


async def impersonate_turn(ctx: ChatUpdateContext) -> None:
    """The impersonation itself, runs as its own task.

    High-level orchestration only: the detached clone lives in
    Conversation.clone_for_impersonation, this just runs the model and
    records the fake turn."""
    clone = ctx.conversation.clone_for_impersonation(IMPERSONATE_PROMPT)

    response = (await run_llm(ctx, conversation=clone)
                or '').strip()
    if not response or response in (NO_RESPONSE_MESSAGE, NO_CONTENT_MESSAGE):
        await ctx.messaging.send_message('Impersonation failed: the model produced nothing usable.')
        return

    # The real conversation gets the bare text as a user turn, tagged as
    # an impersonation so /regenerate can redo this flow. The posted
    # message ids are attached, so /undo and /regenerate can delete it
    # from the chat too.
    message_ids = await ctx.messaging.send_message(response, parse_mode='LLM')
    ctx.conversation.add_user(
        response, telegram_message_ids=message_ids,
        message_type=ConversationMessageType.IMPERSONATION)


@handler
async def compact_command(ctx: ChatUpdateContext):
    keep_tokens = DEFAULT_KEEP_TOKENS
    if ctx.command_text:
        try:
            keep_tokens = int(ctx.command_text)
            if keep_tokens < 1:
                raise ValueError
        except ValueError:
            await ctx.messaging.send_message(f'Usage: `/{ctx.command_name} [tokens of recent history to keep]`')
            return

    await submit_task(ctx, run_compaction(ctx, keep_tokens))


async def run_compaction(ctx: ChatUpdateContext, keep_tokens: int) -> None:
    """One unit of work: compact the conversation, reporting progress and
    the token savings. The task is tracked in ctx.generation_task, so
    /stop cancels it just like a reply generation."""
    async def on_start() -> None:
        await ctx.messaging.send_message('Starting Compaction...')

    try:
        result = await run_tracked(
            ctx,
            lambda: ctx.messaging.run_with_typing(
                compact_context(ctx,
                                keep_tokens=keep_tokens,
                                on_start=on_start)),
            stopped_message='Compaction stopped.')
    except CompactionError as e:
        await ctx.messaging.send_message(str(e))
        return
    if result is None:
        # /stop: run_tracked already reported it.
        return
    await ctx.messaging.send_message(result.summary, parse_mode='LLM')
    await ctx.messaging.send_message(
        f'Compaction complete: {result.tokens_before:,} tokens '
        f'-> {result.tokens_after:,} tokens')
