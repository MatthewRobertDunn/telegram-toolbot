"""Small asyncio primitives shared across the bot.

The first resident is the atomic transaction helper: a post-then-record
sequence that must not be torn in half by cancellation. Telegram accepts
a message and returns its id only after a network round trip, so a bare
``send_message`` followed by a synchronous history commit has a window in
which the post is visible but its id (and the history row) was never
recorded. Wrapping both steps makes cancellation wait for the commit to
finish, then propagates the cancellation as usual.
"""

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Coroutine
from functools import wraps
from typing import Any, ParamSpec, TypeVar

_T = TypeVar('_T')
_P = ParamSpec('_P')
_R = TypeVar('_R')


async def _run_transaction(
        transaction: Callable[[], Coroutine[Any, Any, _R]]) -> _R:
    """Run `transaction` shielded; a cancel waits for it, then re-raises."""
    commit = asyncio.create_task(transaction())
    try:
        return await asyncio.shield(commit)
    except asyncio.CancelledError:
        await commit
        raise


async def atomic_call(action: Callable[[], Awaitable[_T]],
                      callback: Callable[[_T], object]) -> None:
    """Run ``action()`` and commit its result with ``callback(result)``
    as one cancellation unit.

    The callback may be synchronous (the normal, instant history commit)
    or async. If cancellation arrives while ``action`` is awaiting, the
    shielded transaction still runs to completion - the Telegram post
    already happened or is about to, so the caller must finish recording
    it - and the cancellation is then re-raised. If cancellation arrives
    before the transaction starts or while its callback is committing,
    the same rule applies: the caller observes the cancellation only
    after the visible side effect and its bookkeeping agree.
    """
    async def transaction() -> None:
        result = await action()
        outcome = callback(result)
        if inspect.isawaitable(outcome):
            await outcome

    await _run_transaction(transaction)


def atomic(func: Callable[_P, Coroutine[Any, Any, _R]]
           ) -> Callable[_P, Coroutine[Any, Any, _R]]:
    """Decorator form of :func:`atomic_call` for a self-contained helper.

    The decorated coroutine's visible work and local commit run as one
    cancellation unit. This reads best when the send and the bookkeeping
    belong together:

        @atomic
        async def _post_thinking(...):
            ids = await chat.messaging.send_collapsible(...)
            conversation.add_thinking(telegram_message_ids=ids)
    """
    @wraps(func)
    async def atomic_wrapper(*args, **kwargs) -> _R:
        return await _run_transaction(lambda: func(*args, **kwargs))

    return atomic_wrapper
