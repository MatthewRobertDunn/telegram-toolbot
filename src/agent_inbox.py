"""Per-thread agent inbox state: a leaf module with no project imports.

The main assistant owns one ``asyncio.Queue`` inbox plus the conversation
epoch on ``ctx.thread_data``. Both ``handlers.generation_queue`` (the
worker that drains the parent loop) and ``tools.subagent`` (background
producers) share this state, so it lives here instead of in either of
them.

Wakeup protocol (always-enqueue): every notification puts its mail on the
inbox and schedules a wakeup job no matter what the parent is doing.
There is deliberately no "parent loop is active" flag to consult or
release - the flag used to leak True on any wakeup path that exited
early, silently silencing every later notification until a human typed.
Redundant wakeups are acceptable instead: N notifications may queue N
jobs, the first drains the whole inbox and the rest find it empty and
no-op (no send, no persistence). Losing mail now requires dropping a
wakeup job *and* never draining the inbox again; a /stop that drops
queued items can always be followed by the next user turn, which drains
whatever is still queued.

Cross-agent traffic is strongly typed: every inbox carries
``SubagentMessage`` (never raw strings) via ``SubagentInbox``, whose
``put_nowait``/``get_nowait`` take and return the message shape. Display
text is built from the fields at drain time (see ``SubagentMessage.format``)
instead of being threaded through as pre-rendered strings.
"""

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chat_update_context import ChatUpdateContext

#: Sender id used for parent-originated inbox mail (steers into a
#: subagent's inbox, parent notes back into its own).
PARENT_ID = "parent"


class SubagentStatus(StrEnum):
    """The one status vocabulary for subagent records and inbox mail.

    Both ``SubagentRecord.status`` and ``SubagentMessage.state`` use it,
    so a record can be funneled into a receipt without a state rename
    (``"done"`` and ``"finished"`` used to disagree, and ``"done"`` was
    not terminal). Being a ``StrEnum`` the members format and compare as
    their lowercase values, so stored text and model-facing rendering
    stay unchanged.
    """

    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    STOPPED = "stopped"
    # Parent-originated steering mail: not a subagent state at all.
    STEER = "steer"


#: Inbox states after which the sender will send no more mail. Receipts
#: name these in the summary; anything else (interim progress) gets no
#: suffix — the sender is still alive.
TERMINAL_STATES = frozenset({SubagentStatus.FINISHED, SubagentStatus.FAILED,
                             SubagentStatus.STOPPED})


@dataclass
class SubagentMessage:
    """One structured inbox delivery.

    id: sender — subagent id, or "parent" for parent-originated mail.
    state: a SubagentStatus - running (still alive) | finished | failed |
        stopped | steer.
    description: subagent description ("" when unknown/inapplicable).
    result: report body, error detail, progress text, or steer text.
    """

    id: str
    state: SubagentStatus
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


def conversation_epoch(ctx: "ChatUpdateContext") -> int:
    return ctx.thread_data.conversation_epoch


def bump_conversation_epoch(ctx: "ChatUpdateContext") -> int:
    ctx.thread_data.conversation_epoch += 1
    return ctx.thread_data.conversation_epoch
