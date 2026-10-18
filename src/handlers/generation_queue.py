"""The single submission point for reply generation work.

Handlers never call run_llm(ctx) themselves. They hand their user turn to
submit_generation, which parks the handler on a per-user queue while one
dedicated worker task drains it and runs the model. This keeps exactly one
generation running per user - a message typed while a previous reply is
still being generated queues up behind it instead of racing with it - and
the worker is also the only place that creates the task tracked in
ctx.generation_task, so /stop always acts on whatever is actually running.

Queue items carry the reply's context and the already-rendered user turn
text, so the worker adds it to the conversation only when its turn to run
comes up (a queued message addressed to another chat doesn't end up in the
wrong history at the wrong time).

Compaction and the subagent tool call run_llm directly: they work on their
own private conversations, so they have nothing to queue behind.

The worker is per-user and transient: it exits whenever the queue is
empty and a new one is started on the next submission, so users who are
never coming back don't leave a task parked forever.
"""

import asyncio
import logging

from chat_update_context import ChatUpdateContext
from .run_llm import run_llm
from messaging import run_with_typing, send_message

logger = logging.getLogger(__name__)

STOPPED_BY_USER = '(response was stopped by the user)'


def generation_queue(ctx: ChatUpdateContext) -> "asyncio.Queue":
    return ctx.user_data.setdefault('generation_queue', asyncio.Queue())


async def submit_generation(ctx: ChatUpdateContext, content: str) -> None:
    """Queue a generation for a user turn, and wait for its reply to be sent.

    `content` is the user turn, already formatted for the conversation
    (mention-stripped, sender-prefixed in groups). The worker appends it
    when the item is dequeued, just before the model sees it.

    Returns once the reply went out (or the generation was stopped), so
    callers keep the old run-and-reply sequencing.
    """
    item: asyncio.Event = asyncio.Event()
    generation_queue(ctx).put_nowait((item, ctx, content))
    if ctx.generation_worker is None:
        # Queue idle: nothing is draining it. Claims run without an await
        # between the put and the spawn, and the worker releases and
        # re-takes ownership under the same no-await condition, so at most
        # one worker exists per user and no enqueued item can outlive
        # every worker.
        ctx.generation_worker = asyncio.create_task(_generation_worker(ctx))
    await item.wait()


async def _generation_worker(submitter: ChatUpdateContext) -> None:
    """Pump queued items through run_llm, one generation at a time.

    Exits when the queue runs dry, so an idle user doesn't keep a task
    around forever; the next submit respawns one. If it ever dies with
    requests pending (a bug), the next submit_generation spawns a
    replacement.
    """
    queue = generation_queue(submitter)
    while True:
        event, ctx, content = await queue.get()
        try:
            if content is not None:
                ctx.conversation.add_user(content)
            # A concurrent compaction rewrites the same conversation; wait
            # it out so the generation starts against the compacted state.
            compaction = ctx.compaction_task
            if compaction is not None and not compaction.done():
                await asyncio.wait([compaction])

            ctx.generation_task = inner = asyncio.create_task(
                run_with_typing(ctx, run_llm(ctx)))
            try:
                gpt_response = await inner
            except asyncio.CancelledError:
                if not inner.cancelled():
                    # The worker itself was cancelled (bot shutdown):
                    # take the generation down too and re-raise.
                    inner.cancel()
                    raise
                # /stop cancelled just the generation task.
                ctx.conversation.add_assistant(STOPPED_BY_USER)
            else:
                await send_message(ctx, gpt_response, parse_mode='LLM')
                ctx.conversation.add_assistant(gpt_response)
        except Exception:
            logger.exception('Generation failed')
        finally:
            ctx.generation_task = None
            event.set()

        # Queue ran dry: drop the claim and exit. Releasing the claim
        # *before* the final empty check keeps the no-worker race
        # impossible - this whole sequence is synchronous, so a submit
        # that sees a live claim knows this worker is still looping, and
        # one that sees no claim spawns fresh for its already-queued item.
        submitter.generation_worker = None
        if queue.empty():
            return
        submitter.generation_worker = asyncio.current_task()
