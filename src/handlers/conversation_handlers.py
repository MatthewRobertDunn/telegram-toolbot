"""Commands operating on the conversation itself: regenerating the last
turn, impersonating the user, and compacting the history."""

import asyncio
import logging

from async_utils import atomic_call
from chat_update_context import ChatUpdateContext
from command_spec import usage_line
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
from messages import iter_parts, telegram_ids_of
# Re-exports: the token accounting lives in tokens.py, but other branches
# and tests import these names from here.
from tokens import estimate_conversation_tokens, token_count

logger = logging.getLogger(__name__)

# Seconds before a command's own message and its status replies are deleted.
COMMAND_CLEAR_DELAY = 10

# /compact's argument syntax: the handler renders it and the command
# table row declares the same constant, so the two cannot drift.
COMPACT_USAGE = "[tokens of recent history to keep]"

# Strong references to the fire-and-forget cleanup tasks: asyncio keeps
# only weak references to running tasks, so an unreferenced one could be
# garbage-collected mid-sleep.
_PENDING_CLEAR_TASKS: set[asyncio.Task] = set()


async def _schedule_command_clear(ctx: ChatUpdateContext,
                                  message_ids: list) -> None:
    """Delete the command message and the status replies after
    COMMAND_CLEAR_DELAY seconds."""
    await asyncio.sleep(COMMAND_CLEAR_DELAY)
    await delete_from_telegram(ctx, message_ids)


def _clear_command_later(ctx: ChatUpdateContext, message_ids) -> None:
    """Delete the command message plus ``message_ids`` a short while later.

    Fire-and-forget: the handler returns at once and this task cleans up
    later, so a slow Telegram delete never blocks the next update.
    """
    task = asyncio.create_task(_schedule_command_clear(
        ctx, [_command_message_id(ctx), *message_ids]))
    _PENDING_CLEAR_TASKS.add(task)
    task.add_done_callback(_PENDING_CLEAR_TASKS.discard)


def _command_message_id(ctx: ChatUpdateContext) -> int | None:
    return getattr(ctx.message, 'message_id', None)


async def delete_from_telegram(ctx: ChatUpdateContext, message_ids) -> None:
    for message_id in message_ids or []:
        try:
            await ctx.messaging.delete_message(message_id)
        except Exception as e:
            logger.warning("Could not delete telegram message %s: %s",
                           message_id, e)


def stored_message_ids(message: dict) -> list:
    return telegram_ids_of(message)


def _tool_call_name(call_id, context: list | None) -> str:
    """The function name a tool result's tool_call_id refers to, looked up
    among the conversation's other tool calls."""
    for message in context or []:
        for call in message.get('tool_calls') or []:
            if (call or {}).get('id') == call_id:
                return ((call or {}).get('function') or {}).get('name') or ''
    return ''


def message_preview(message: dict, limit: int = 100,
                    context: list | None = None) -> str:
    """A short identification of a message's contents for feedback such as
    /undo: fenced text trimmed to ``limit`` characters. Tool calls and
    results are prefixed with the tool's name; multimodal parts become
    placeholders."""
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
    """Whether history holds anything beyond the system head and kept
    thinking blocks, i.e. something /retry and /impersonate could act on."""
    return any(message.get('role') != 'system'
               and not is_thinking_message(message)
               for message in messages)


@handler
async def undo_command(ctx: ChatUpdateContext):
    """Remove the tail back to and including the last user prompt, deleting
    the removed messages from Telegram where their ids are known. Tool-call
    ids are followed too: a call left without its results (or vice versa)
    can't be replayed.

    Runs as a queued unit of work like /compact, so /stop stops it; the
    command message and its status replies are deleted shortly after.
    """
    await submit_task(ctx, lambda: undo_turn(ctx),
                      stopped_message='Undo stopped.')


async def undo_turn(ctx: ChatUpdateContext) -> None:
    """The undo itself, inside the queued unit of work.

    Only the Telegram deletes and the report: the trim lives in
    Conversation.undo_last_exchange.
    """
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
    again; when history ends on the user's turn, answer it without
    re-adding it. An impersonated last user turn is removed and the
    impersonation flow reruns instead.

    Runs as a queued unit of work like /compact; only the command message
    is deleted afterwards, never the queued reply.
    """
    await submit_task(ctx, lambda: regeneration_turn(ctx),
                      stopped_message='Regeneration stopped.')


async def regeneration_turn(ctx: ChatUpdateContext) -> None:
    """The regeneration itself, inside the queued unit of work.

    Only deletes the Telegram posts and runs the next flow: the trim lives
    in Conversation.prepare_regeneration.
    """
    removed, action = ctx.conversation.prepare_regeneration()
    for message in removed:
        await delete_from_telegram(ctx, stored_message_ids(message))

    if action is RegenerationAction.NOTHING:
        message_ids = await ctx.messaging.send_message("Nothing to regenerate yet.")
        _clear_command_later(ctx, message_ids)
        return

    if action is RegenerationAction.IMPERSONATE:
        # Already inside the worker: run the flow directly instead of
        # submitting a second queued item, which would deadlock waiting
        # for itself.
        await run_impersonation(ctx)
        _clear_command_later(ctx, [])
        return

    # content=None: run the model without appending a user turn - after
    # the trim the history ends on the original prompt, unchanged, and the
    # model answers it again. Already inside the worker: call directly
    # instead of submitting a second queued item.
    await generate_response(ctx, None)
    _clear_command_later(ctx, [])


@handler
async def retry_command(ctx: ChatUpdateContext):
    """Answer the conversation again exactly as it stands: nothing is popped
    or deleted, a fresh reply is appended.

    Runs as a queued unit of work like /compact, so it waits for any
    in-flight work instead of refusing.
    """
    if not _has_content(ctx.conversation.messages):
        await ctx.messaging.send_message("Nothing to retry yet.")
        return

    # content=None: the queued item runs the model without appending a
    # user turn - the history is left alone, and the model answers it
    # again wherever it currently ends.
    await submit_generation(ctx, None)


@handler
async def impersonate_command(ctx: ChatUpdateContext):
    """Fake a user turn: the model sees a clone of the whole conversation
    (system head with the user profile, full history) and writes the user's
    next message, which is added to the real conversation and posted as a
    user-labelled reply.

    Runs as a queued unit of work like /compact, so /stop stops it.
    """
    # Not cleared: its text tells the user the reply below is impersonated,
    # not their own.
    await submit_task(ctx, lambda: run_impersonation(ctx),
                      stopped_message='Impersonation stopped.')


async def run_impersonation(ctx: ChatUpdateContext) -> None:
    """The queued unit of work behind /impersonate, reused by /regenerate."""
    # Check here, not in the command handler: earlier queued work may
    # empty the history after the command arrives but before it runs.
    if not _has_content(ctx.conversation.messages):
        message_ids = await ctx.messaging.send_message("Nothing to impersonate yet.")
        _clear_command_later(ctx, message_ids)
        return

    await impersonate_turn(ctx)


async def impersonate_turn(ctx: ChatUpdateContext) -> None:
    """The impersonation itself, inside the queued unit of work.

    Only runs the model and records the fake turn: the clone lives in
    Conversation.clone_for_impersonation.
    """
    clone = ctx.conversation.clone_for_impersonation(IMPERSONATE_PROMPT)

    response = (await run_llm(ctx, conversation=clone)
                or '').strip()
    if not response or response in (NO_RESPONSE_MESSAGE, NO_CONTENT_MESSAGE):
        await ctx.messaging.send_message('Impersonation failed: the model produced nothing usable.')
        return

    # The real conversation gets the bare text as a tagged user turn so
    # /regenerate can redo this flow, with the posted ids attached so
    # /undo can delete it from the chat too. Atomic keeps /stop from
    # tearing the post away from its history row.
    await atomic_call(
        lambda: ctx.messaging.send_message(response, parse_mode='LLM'),
        lambda message_ids: ctx.conversation.add_user(
            response, telegram_message_ids=message_ids,
            message_type=ConversationMessageType.IMPERSONATION))


@handler
async def compact_command(ctx: ChatUpdateContext):
    keep_tokens = DEFAULT_KEEP_TOKENS
    if ctx.command_text:
        try:
            keep_tokens = int(ctx.command_text)
            if keep_tokens < 1:
                raise ValueError
        except ValueError:
            await ctx.messaging.send_message(
                usage_line(ctx.command_name, COMPACT_USAGE))
            return

    await submit_task(ctx, lambda: run_compaction(ctx, keep_tokens),
                      stopped_message='Compaction stopped.')


async def run_compaction(ctx: ChatUpdateContext, keep_tokens: int) -> None:
    """One unit of work: compact the conversation, reporting progress and
    the token savings."""
    async def on_start() -> None:
        await ctx.messaging.send_message('Starting Compaction...')

    try:
        result = await compact_context(ctx, keep_tokens=keep_tokens,
                                       on_start=on_start)
    except CompactionError as e:
        await ctx.messaging.send_message(str(e))
        return
    await ctx.messaging.send_message(result.summary, parse_mode='LLM')
    await ctx.messaging.send_message(
        f'Compaction complete: {result.tokens_before:,} tokens '
        f'-> {result.tokens_after:,} tokens')
