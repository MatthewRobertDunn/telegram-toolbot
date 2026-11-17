"""Per-thread agent inbox state: a leaf module with no project imports.

The main assistant owns one ``asyncio.Queue`` inbox plus a small set of
coordination flags on ``ctx.thread_data``. Both ``handlers.generation_queue``
(the main loop owner) and ``tools.subagent`` (background producers) share
this state, so it lives here instead of in either of them.

Cross-agent traffic is strongly typed: every inbox carries
``SubagentMessage`` (never raw strings) via ``SubagentInbox``, whose
``put_nowait``/``get_nowait`` take and return the message shape. Display
text is built from the fields at drain time (see ``SubagentMessage.format``)
instead of being threaded through as pre-rendered strings.
"""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chat_update_context import ChatUpdateContext

#: Sender id used for parent-originated inbox mail (steers into a
#: subagent's inbox, parent notes back into its own).
PARENT_ID = "parent"

#: Inbox states after which the sender will send no more mail. Receipts
#: name these in the summary; anything else (interim progress) gets no
#: suffix — the sender is still alive.
TERMINAL_STATES = frozenset({"finished", "failed", "timed out", "stopped"})


@dataclass
class SubagentMessage:
    """One structured inbox delivery.

    id: sender — subagent id, or "parent" for parent-originated mail.
    state: running (still alive) | finished | timed out | failed |
        stopped | steer.
    description: subagent description ("" when unknown/inapplicable).
    result: report body, error detail, progress text, or steer text.
    """

    id: str
    state: str
    description: str = ""
    result: str = ""

    def format(self) -> str:
        """Full conversation text for the model."""
        label = f"[{self.id}"
        if self.description:
            label += f" ({self.description})"
        label += f" {self.state}]"
        if self.result:
            label += f":\n{self.result}"
        return label


class SubagentInbox(asyncio.Queue[SubagentMessage]):
    """An inbox strongly typed to ``SubagentMessage``.

    ``put_nowait`` rejects anything else so unstructured strings can never
    sneak back in; ``get_nowait`` therefore always returns the message
    shape ``_drain_inbox`` formats.
    """

    def put_nowait(self, item: SubagentMessage) -> None:
        if not isinstance(item, SubagentMessage):
            raise TypeError(
                "SubagentInbox only accepts SubagentMessage, got "
                f"{type(item).__name__}")
        super().put_nowait(item)

    def get_nowait(self) -> SubagentMessage:
        return super().get_nowait()


def get_main_inbox(ctx: "ChatUpdateContext") -> SubagentInbox:
    inbox = ctx.thread_data.main_inbox
    if inbox is None:
        inbox = SubagentInbox()
        ctx.thread_data.main_inbox = inbox
    return inbox


def clear_main_inbox(ctx: "ChatUpdateContext") -> None:
    """Drop the main inbox (e.g. on /start: stale reports must not leak
    into the fresh conversation)."""
    ctx.thread_data.main_inbox = None


def main_loop_active(ctx: "ChatUpdateContext") -> bool:
    """Whether a main-loop drainer exists that will pick up inbox mail."""
    return ctx.thread_data.main_loop_active


def set_main_loop_active(ctx: "ChatUpdateContext", active: bool) -> None:
    ctx.thread_data.main_loop_active = active


def conversation_epoch(ctx: "ChatUpdateContext") -> int:
    return ctx.thread_data.conversation_epoch


def bump_conversation_epoch(ctx: "ChatUpdateContext") -> int:
    ctx.thread_data.conversation_epoch += 1
    return ctx.thread_data.conversation_epoch


def take_wakeup_task(ctx: "ChatUpdateContext") -> asyncio.Task | None:
    task = ctx.thread_data.wakeup_task
    ctx.thread_data.wakeup_task = None
    return task


def set_wakeup_task(ctx: "ChatUpdateContext",
                    task: asyncio.Task | None) -> None:
    ctx.thread_data.wakeup_task = task
