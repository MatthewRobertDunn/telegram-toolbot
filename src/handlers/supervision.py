"""Shared supervision for command-driven background work.

The /undo, /regenerate, /impersonate and /compact flows all run the same
shape: the queued unit of work creates a task for the model-facing
operation, tracks it in ctx.generation_task so /stop acts on the model
work (never the queue worker waiting on it), tells the user when the work
was stopped, and always clears the tracking. This module is that shape,
written once instead of pasted four times.
"""

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar

from chat_update_context import ChatUpdateContext

_T = TypeVar('_T')


async def run_tracked(ctx: ChatUpdateContext,
                      factory: Callable[[], Coroutine[Any, Any, _T]], *,
                      stopped_message: str | None = None,
                      on_stopped: Callable[[], Awaitable[None]] | None = None
                      ) -> _T | None:
    """Run `factory()` as a task tracked in ctx.generation_task.

    The tracked task is always the freshly created inner task, never the
    queued worker awaiting it, so /stop cancels model work directly.
    Cancellation semantics:
    - /stop (the inner task cancelled): posts `stopped_message` ("X
      stopped.") and, when given, awaits `on_stopped`, then returns None.
    - Bot shutdown (this job itself cancelled): cancels the inner task
      and re-raises, posting nothing.
    `ctx.generation_task` is cleared on every path. Returns the inner
    task's result, or None on the stopped path.
    """
    ctx.generation_task = inner = asyncio.create_task(factory())
    try:
        return await inner
    except asyncio.CancelledError:
        if not inner.cancelled():
            # This job itself was cancelled (bot shutdown): take the
            # tracked work down too and re-raise.
            inner.cancel()
            raise
        if stopped_message is not None:
            await ctx.messaging.send_message(stopped_message)
        if on_stopped is not None:
            await on_stopped()
        return None
    finally:
        ctx.generation_task = None
