"""The single submission point for serialized per-thread work.

Handlers never run generation themselves: they hand a job to the
per-thread queue, where one dedicated worker task drains it. Exactly one
piece of work runs per chat/thread - a message typed while a reply is
still being generated queues behind it - and jobs are any callable, so
compaction and inbox wakeups share the same serialization. Queue state
lives on the thread's ``ThreadData``, so two threads never share it.

Every queued job runs under the typing indicator, is tracked in
``ctx.generation_task`` so /stop cancels the work rather than the worker
waiting on it, and reports the stop through ``stopped_message``. When the
job ends - successfully, with a failure, or stopped - dangling trailing
thinking is dropped and the conversation is persisted once.

Queue items carry a job thunk and a done event: submitting waits for the
item's turn, and the coroutine is created only when the worker dequeues
the item, so a dropped item leaves nothing un-awaited.

The worker is transient: it exits when the queue is empty and a new one
starts on the next submission.

Inbox wakeups are just another job: every delivery schedules one, the
first drains the whole inbox and the rest no-op.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agent_inbox import get_main_inbox
from async_utils import atomic_call
from chat_update_context import ChatUpdateContext
from persistence.settings_store import save_conversation
from tool_registry import PARENT, surface_for
from .run_llm import run_llm

logger = logging.getLogger(__name__)


@dataclass
class WorkItem:
    """One queued unit of work.

    ``stopped_message`` is what the item tells the user when /stop
    cancelled it mid-run; None for jobs whose stop is visible in the
    missing work itself (a reply generation). ``done`` is set once the
    item is processed or released; ``error`` carries a failed job's
    exception to the submitter.
    """

    ctx: ChatUpdateContext
    job: Callable[[], Awaitable[Any]]
    stopped_message: str | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    error: BaseException | None = None

    def abandon(self) -> None:
        """Release this item without running it and wake the submitter."""
        self.done.set()


def generation_queue(ctx: ChatUpdateContext) -> asyncio.Queue:
    queue = ctx.thread_data.generation_queue
    if queue is None:
        queue = asyncio.Queue()
        ctx.thread_data.generation_queue = queue
    return queue


async def submit_task(ctx: ChatUpdateContext,
                      job: Callable[[], Awaitable[Any]],
                      *,
                      stopped_message: str | None = None,
                      ) -> None:
    """Enqueue a unit of work and wait for it to finish.

    Returns once the job ran, re-raising a failure the job ended with so
    callers see it instead of the worker swallowing it silently.

    ``stopped_message`` is posted to the chat when /stop cancels the job
    mid-run. Everything else a queued job needs (typing indicator, /stop
    supervision, cleanup, persistence) is the queue's job.
    """
    item = WorkItem(ctx=ctx, job=job, stopped_message=stopped_message)
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
    return submit_task(
        ctx, lambda: generate_response(
            ctx, content=content,
            telegram_message_id=telegram_message_id))


async def generate_response(ctx: ChatUpdateContext, content: str | None = None,
                            telegram_message_id: int | None = None) -> None:
    """One unit of work: append the user turn (if given) and carry a full
    generation+reply cycle. The supervision around it belongs to the queue
    item that runs it.

    One run_llm call per turn: it drains the inbox at the top of every
    iteration, so mail present when the reply starts is answered here.
    Later mail has its own queued wakeup, which is why no further looping
    is needed.
    """
    if content is not None:
        ctx.conversation.add_user(
            content, telegram_message_ids=telegram_message_id)

    gpt_response = await run_llm(ctx,
                                 tools=surface_for(PARENT),
                                 inbox=get_main_inbox(ctx))

    # The reply is visible once Telegram accepts it, so the history
    # row must not be lost to cancellation in that window.
    await atomic_call(
        lambda: ctx.messaging.send_message(gpt_response,
                                           parse_mode='LLM'),
        lambda message_ids: ctx.conversation.add_assistant(
            gpt_response, telegram_message_ids=message_ids))


async def _supervise(item: WorkItem) -> None:
    """Run the item's job under the typing indicator, tracked for /stop.

    The tracked task is always the freshly created inner task, never the
    queue worker awaiting it, so /stop cancels the model-facing work
    directly. Cancellation semantics:
    - /stop (the inner task cancelled): posts the item's
      ``stopped_message`` and returns.
    - Bot shutdown (the worker itself cancelled): cancels the inner task
      and re-raises, posting nothing.
    ``ctx.generation_task`` is cleared on every path.
    """
    ctx = item.ctx
    ctx.generation_task = inner = asyncio.create_task(
        ctx.messaging.run_with_typing(item.job()))
    try:
        await inner
    except asyncio.CancelledError:
        # This worker was cancelled (bot shutdown). The cancellation rode
        # down the await into the inner task; make sure it is down, take
        # the worker with it, and post nothing. current_task() is always
        # the worker here, but the type checker cannot prove it.
        worker = asyncio.current_task()
        assert worker is not None
        if worker.cancelling():
            inner.cancel()
            raise
        if item.stopped_message is not None:
            await ctx.messaging.send_message(item.stopped_message)
    finally:
        ctx.generation_task = None


async def _drop_dangling_thinking(ctx: ChatUpdateContext) -> None:
    """Delete thinking left behind by an aborted reply.

    Thinking already appended for a reply that never arrived has no answer
    to belong to, so the history entry and its Telegram post go. A finished
    reply has its answer after the thinking, so this is a no-op there.
    """
    if ctx.thread_data.conversation is None:
        return
    for thinking in ctx.conversation.pop_trailing_thinking():
        for message_id in thinking.telegram_message_ids:
            try:
                await ctx.messaging.delete_message(message_id)
            except Exception:
                logger.warning('Could not delete thinking message %s',
                               message_id, exc_info=True)


async def _run_item(item: WorkItem) -> None:
    """Run one dequeued item: supervise the job, clean up, persist.

    A failing job is recorded on the item (submit_task re-raises it to
    the submitter) and logged, but never kills the worker: the next item
    still runs. A cancelled job skips the cleanup and the persistence
    (nothing completed to save) but still sets ``done``, so its submitter
    wakes either way.
    """
    try:
        try:
            await _supervise(item)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            item.error = e
            logger.error('Queued work failed', exc_info=True)
        await _drop_dangling_thinking(item.ctx)
        # Single persistence point: every dequeued job is persisted here -
        # replies, /undo, /regenerate, /retry, /impersonate, /compact and
        # parent wakeups alike. A wakeup with an empty inbox still rewrites
        # the unchanged conversation, which is harmless. Only when the
        # thread has a conversation: a bare context would trip
        # ctx.conversation's assert. Never let a DB failure kill the loop.
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
    Every exit drops the claim and releases anything still queued, so a
    submitter never waits on a dead worker. Dropping is safe for inbox
    mail too: an undrained message stays in the inbox and the next turn
    (user message or wakeup) drains it.
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
        # Nothing else will drain the queue now: release the leftovers so
        # their submitters wake instead of hanging.
        while True:
            try:
                leftover = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            leftover.abandon()
