"""The single submission point for serialized per-thread work.

Handlers never run generation themselves. They hand a job (usually a user
turn handed to submit_generation) to the per-thread queue, where one
dedicated worker task drains it. This keeps exactly one piece of work
running per chat/thread - a message typed while a previous reply is still
being generated queues up behind it instead of racing with it - and jobs
can be any callable, so units of work beyond replies (compaction today,
others tomorrow) can join the same serialization. All of that state (the
queue, the worker claim, the tracked task) lives on the thread's
``ThreadData``, so two threads of one user never share a serialization.

Queue items carry a job thunk and a done event: submitting waits for the
item's turn, runs to completion, and the handler resumes after its work
has gone out. A job is a zero-arg callable returning the awaitable, so
the coroutine is created only when the worker dequeues the item - a
dropped item (/stop) leaves nothing un-awaited behind.

The worker is per-thread and transient: it exits whenever the queue is
empty and a new one is started on the next submission, so users who are
never coming back don't leave a task parked forever.

Inbox wakeups are just another kind of job here. There is no
"parent loop is active" flag any more: every delivery schedules one
wakeup job (see ``tools.subagent``), the first drains the whole inbox and
the rest no-op, so nothing can strand mail behind a flag stuck True.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agent_inbox import get_main_inbox
from chat_update_context import ChatUpdateContext
from settings_store import save_conversation
from tools.subagent import get_parent_tool_defs
from .run_llm import run_llm

logger = logging.getLogger(__name__)


@dataclass
class WorkItem:
    """One queued unit of work.

    ``job`` is a zero-arg callable returning the awaitable (a thunk): the
    coroutine comes into existence only when the worker dequeues the item,
    so an item dropped without running (a /stop drain) leaves nothing
    un-awaited.

    ``done`` is owned by the queue machinery and set once the item is
    processed or released; ``error`` carries a failed job's exception out
    to the submitter (submit_task re-raises it).
    """

    ctx: ChatUpdateContext
    job: Callable[[], Awaitable[Any]]
    done: asyncio.Event = field(default_factory=asyncio.Event)
    error: BaseException | None = None

    def abandon(self) -> None:
        """Release this item without running it and wake the submitter.

        A queued item never built its coroutine (``job`` is a thunk), so
        there is nothing to close - releasing it is simply waking whoever
        is waiting on ``done``.
        """
        self.done.set()


def generation_queue(ctx: ChatUpdateContext) -> asyncio.Queue:
    queue = ctx.thread_data.generation_queue
    if queue is None:
        queue = asyncio.Queue()
        ctx.thread_data.generation_queue = queue
    return queue


async def submit_task(ctx: ChatUpdateContext,
                      job: Callable[[], Awaitable[Any]],
                      ) -> None:
    """Enqueue a unit of work and wait for it to finish.

    Returns once the job ran, re-raising a failure the job ended with so
    callers see it instead of the worker swallowing it silently (a
    dropped or cancelled item simply returns).

    ``job`` must be a zero-arg callable returning the awaitable: a
    coroutine built eagerly at the call site would exist even for an item
    that is later dropped without ever running.
    """
    item = WorkItem(ctx=ctx, job=job)
    generation_queue(ctx).put_nowait(item)
    if ctx.generation_worker is None:
        # Queue idle: nothing is draining it. Claims run without an await
        # between the put and the spawn, and the worker releases and
        # re-takes ownership under the same no-await condition, so at most
        # one worker exists per thread and no enqueued item can outlive
        # every worker.
        ctx.generation_worker = asyncio.create_task(_generation_worker(ctx))
    await item.done.wait()
    if item.error is not None:
        raise item.error


def submit_generation(ctx, content: str | None = None,
                      telegram_message_id: int | None = None):
    """Enqueue a reply for a user turn (a variant of submit_task), and
    wait for the reply to be sent.

    `content` is the user turn, already formatted for the conversation
    (mention-stripped, sender-prefixed in groups); None submits a bare
    generation with no new user turn (e.g. /regenerate). Added to the
    conversation when the item's turn to run comes up: a queued message
    for another chat doesn't end up in the wrong history at the wrong
    time.
    """
    # Thunk: the reply's coroutine is built by the worker on dequeue, so
    # a /stop that drops this item leaves nothing un-awaited.
    return submit_task(
        ctx, lambda: generate_response(
            ctx, content=content,
            telegram_message_id=telegram_message_id))


async def generate_response(ctx: ChatUpdateContext, content: str | None = None,
                            telegram_message_id: int | None = None) -> None:
    """One unit of work: append the user turn (if given) and carry a full
    generation+reply cycle. The generation task is tracked in
    ctx.generation_task, so /stop always acts on whatever is running.

    One run_llm call per turn: it drains the inbox at the top of every
    iteration, so mail present when the reply starts is answered by this
    turn. Mail that arrives after that last drain has its own queued
    wakeup (always-enqueue), which is why no further looping is needed
    here.
    """
    if content is not None:
        ctx.conversation.add_user(
            content, telegram_message_ids=telegram_message_id)

    try:
        ctx.generation_task = inner = asyncio.create_task(
            ctx.messaging.run_with_typing(
                run_llm(ctx,
                        tools=get_parent_tool_defs(),
                        inbox=get_main_inbox(ctx))))
        try:
            gpt_response = await inner
        except asyncio.CancelledError:
            if not inner.cancelled():
                # This job itself was cancelled (bot shutdown): take the
                # generation down too and re-raise.
                inner.cancel()
                raise
            # /stop cancelled just the generation task. Thinking already
            # appended for the aborted reply has no answer to belong to,
            # so drop it now (history and Telegram): leaving it would
            # strand a dangling block a later /regenerate would miss.
            for thinking in ctx.conversation.pop_trailing_thinking():
                for message_id in thinking.telegram_message_ids:
                    try:
                        await ctx.messaging.delete_message(message_id)
                    except Exception:
                        logger.warning(
                            'Could not delete thinking message %s',
                            message_id, exc_info=True)
            return
        finally:
            ctx.generation_task = None

        message_ids = await ctx.messaging.send_message(gpt_response,
                                                       parse_mode='LLM')
        ctx.conversation.add_assistant(gpt_response,
                                       telegram_message_ids=message_ids)
    finally:
        ctx.generation_task = None


async def _run_item(item: WorkItem) -> None:
    """Run one dequeued item: the job, its failure record, persistence.

    A failing job is recorded on the item (submit_task re-raises it to
    the submitter) and logged, but never kills the worker: the next item
    still runs. A cancelled job skips persistence (nothing completed to
    save) but still sets ``done``, so its submitter wakes either way.
    """
    try:
        try:
            await item.job()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            item.error = e
            logger.error('Queued work failed', exc_info=True)
        # Single persistence point: every dequeued job is persisted once,
        # right here - replies, /undo, /regenerate, /retry, /impersonate,
        # /compact, and parent wakeups alike. A wakeup that finds an
        # empty inbox still triggers this save (a harmless rewrite of an
        # unchanged conversation): the pre-enqueue check makes that race
        # rare, and a per-job persist opt-out would special-case one job
        # type for no correctness gain. Only when the thread actually has
        # a conversation: a bare context (none loaded) has nothing to
        # persist and reading ctx.conversation would trip its assert.
        # Never let a DB failure kill the worker loop.
        conversation = item.ctx.thread_data.conversation
        if conversation is not None:
            try:
                await save_conversation(item.ctx.user_id,
                                        item.ctx.message_thread_id,
                                        conversation)
            except Exception:
                logger.warning('Could not persist conversation for user %d',
                               item.ctx.user_id, exc_info=True)
    finally:
        item.done.set()


async def _generation_worker(submitter: ChatUpdateContext) -> None:
    """Drain queued units of work, one at a time.

    Transient: exits as soon as the queue runs dry, so an idle thread
    doesn't keep a task parked forever; the next submit respawns one.
    Every exit - clean drain, worker cancellation - drops the claim and
    releases anything still queued, so a submitter never waits on a dead
    worker and no dropped job leaks an un-awaited coroutine. Dropping is
    safe for inbox mail too: an undrained message stays in the inbox and
    the next turn (user message or wakeup) drains it.
    """
    queue = generation_queue(submitter)
    try:
        while True:
            item = await queue.get()
            await _run_item(item)
            # Queue ran dry: drop the claim and exit. Releasing the claim
            # *before* the final empty check keeps the no-worker race
            # impossible - this whole sequence is synchronous, so a submit
            # that sees a live claim knows this worker is still looping,
            # and one that sees no claim spawns fresh for its
            # already-queued item.
            submitter.generation_worker = None
            if queue.empty():
                return
            submitter.generation_worker = asyncio.current_task()
    finally:
        me = asyncio.current_task()
        if me is not None and submitter.generation_worker is me:
            submitter.generation_worker = None
        # Whatever killed the loop, nothing else will drain the queue
        # now: release the leftovers so their submitters wake instead of
        # hanging, and so no legacy coroutine dies un-awaited.
        while True:
            try:
                leftover = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            leftover.abandon()
