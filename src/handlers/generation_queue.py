"""The single submission point for serialized per-user work.

Handlers never run generation themselves. They hand a job (usually a user
turn handed to submit_generation) to the per-user queue, where one
dedicated worker task drains it. This keeps exactly one piece of work
running per user - a message typed while a previous reply is still being
generated queues up behind it instead of racing with it - and jobs can be
any callable, so units of work beyond replies (compaction today, others
tomorrow) can join the same serialization.

Queue items carry the job and a done event: submitting waits for the
item's turn, runs to completion, and the handler resumes after its work
has gone out.

The worker is per-user and transient: it exits whenever the queue is
empty and a new one is started on the next submission, so users who are
never coming back don't leave a task parked forever.
"""

import asyncio
import logging
from collections.abc import Awaitable
from dataclasses import dataclass, field

from chat_update_context import ChatUpdateContext
from .run_llm import run_llm
from messaging import run_with_typing, send_message

logger = logging.getLogger(__name__)


@dataclass
class WorkItem:
    """One queued unit of work: an awaitable job (usually a bound call to
    generate_response or another queued coroutine). `done` is owned by
    the queue machinery and set once processed."""
    ctx: ChatUpdateContext
    job: Awaitable
    done: asyncio.Event = field(default_factory=asyncio.Event)


def generation_queue(ctx: ChatUpdateContext) -> asyncio.Queue:
    return ctx.user_data.setdefault('generation_queue', asyncio.Queue())


async def submit_task(ctx: ChatUpdateContext, job: Awaitable) -> None:
    """Enqueue a unit of work and wait for it to finish.

    Returns once the job ran (or failed, or was stopped), so callers keep
    the old run-and-reply sequencing.
    """
    item = WorkItem(ctx=ctx, job=job)
    generation_queue(ctx).put_nowait(item)
    if ctx.generation_worker is None:
        # Queue idle: nothing is draining it. Claims run without an await
        # between the put and the spawn, and the worker releases and
        # re-takes ownership under the same no-await condition, so at most
        # one worker exists per user and no enqueued item can outlive
        # every worker.
        ctx.generation_worker = asyncio.create_task(_generation_worker(ctx))
    await item.done.wait()


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
    return submit_task(ctx, generate_response(
        ctx, content=content, telegram_message_id=telegram_message_id))


async def generate_response(ctx: ChatUpdateContext, content: str | None = None,
                            telegram_message_id: int | None = None) -> None:
    """One unit of work: append the user turn (if given) and carry a full
    generation+reply cycle. The generation task is tracked in
    ctx.generation_task, so /stop always acts on whatever is running."""
    if content is not None:
        ctx.conversation.add_user(
            content, telegram_message_id=telegram_message_id)

    ctx.generation_task = inner = asyncio.create_task(
        run_with_typing(ctx, run_llm(ctx)))
    try:
        gpt_response = await inner
    except asyncio.CancelledError:
        if not inner.cancelled():
            # This job itself was cancelled (bot shutdown): take the
            # generation down too and re-raise.
            inner.cancel()
            raise
        # /stop cancelled just the generation task; leave nothing behind
        # (the user can /regenerate or simply talk again).
    else:
        message_id = await send_message(ctx, gpt_response, parse_mode='LLM')
        ctx.conversation.add_assistant(
            gpt_response, telegram_message_id=message_id)
    finally:
        ctx.generation_task = None


async def _generation_worker(submitter: ChatUpdateContext) -> None:
    """Drain queued units of work, one at a time.

    Exits when the queue runs dry, so an idle user doesn't keep a task
    around forever; the next submit respawns one. If it ever dies with
    requests pending (a bug), the next submit spawns a replacement.
    """
    queue = generation_queue(submitter)
    while True:
        item = await queue.get()
        try:
            await item.job
        except Exception:
            logger.exception('Queued work failed')
        finally:
            item.done.set()

        # Queue ran dry: drop the claim and exit. Releasing the claim
        # *before* the final empty check keeps the no-worker race
        # impossible - this whole sequence is synchronous, so a submit
        # that sees a live claim knows this worker is still looping, and
        # one that sees no claim spawns fresh for its already-queued item.
        submitter.generation_worker = None
        if queue.empty():
            return
        submitter.generation_worker = asyncio.current_task()
