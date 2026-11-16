"""Per-user agent inbox state: a leaf module with no project imports.

The main assistant owns one ``asyncio.Queue`` inbox plus a small set of
coordination flags in ``ctx.user_data``. Both ``handlers.generation_queue``
(the main loop owner) and ``tools.subagent`` (background producers) share
this state, so it lives here instead of in either of them.
"""

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chat_update_context import ChatUpdateContext


def get_main_inbox(ctx: "ChatUpdateContext") -> asyncio.Queue:
    inbox = ctx.user_data.get('main_inbox')
    if inbox is None:
        inbox = asyncio.Queue()
        ctx.user_data['main_inbox'] = inbox
    return inbox


def clear_main_inbox(ctx: "ChatUpdateContext") -> None:
    """Drop the main inbox (e.g. on /start: stale reports must not leak
    into the fresh conversation)."""
    ctx.user_data.pop('main_inbox', None)


def main_loop_active(ctx: "ChatUpdateContext") -> bool:
    """Whether a main-loop drainer exists that will pick up inbox mail."""
    return bool(ctx.user_data.get('main_loop_active'))


def set_main_loop_active(ctx: "ChatUpdateContext", active: bool) -> None:
    ctx.user_data['main_loop_active'] = active


def conversation_epoch(ctx: "ChatUpdateContext") -> int:
    return ctx.user_data.get('conversation_epoch', 0)


def bump_conversation_epoch(ctx: "ChatUpdateContext") -> int:
    epoch = conversation_epoch(ctx) + 1
    ctx.user_data['conversation_epoch'] = epoch
    return epoch


def take_wakeup_task(ctx: "ChatUpdateContext") -> asyncio.Task | None:
    return ctx.user_data.pop('wakeup_task', None)


def set_wakeup_task(ctx: "ChatUpdateContext",
                    task: asyncio.Task | None) -> None:
    if task is None:
        ctx.user_data.pop('wakeup_task', None)
    else:
        ctx.user_data['wakeup_task'] = task
