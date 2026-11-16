"""Per-user agent inbox state: a leaf module with no project imports.

The main assistant owns one ``asyncio.Queue`` inbox plus a small set of
coordination flags in ``ctx.user_data``. Both ``handlers.generation_queue``
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

#: Telegram receipt preview cap, mirroring the tool-call announcement
#: truncation (``args[:1024]``).
RECEIPT_PREVIEW_LIMIT = 1024


@dataclass
class SubagentMessage:
    """One structured inbox delivery.

    id: subagent id the notice is about ("" for parent-originated mail).
    state: finished | timed out | failed | stopped | status | steer.
    description: subagent description ("" when unknown/inapplicable).
    result: report body, error detail, status text, or steer text.
    """

    id: str = ""
    state: str = "status"
    description: str = ""
    result: str = ""

    def format(self, result_limit: int | None = None) -> str:
        """Full conversation text (result_limit=None) or a truncated
        Telegram preview (result_limit=1024)."""
        body = self.result
        if result_limit is not None and len(body) > result_limit:
            body = body[:result_limit] + "\n…(truncated)"
        if self.state == "steer":
            return f"[parent message]:\n{body}"
        if self.state == "status" and not self.id:
            return f"[parent note]:\n{body}"
        label = f"[subagent {self.id}"
        if self.description:
            label += f" ({self.description})"
        if self.state == "finished":
            return f"{label} finished]:\n{body}"
        if self.state == "status":
            return f"{label} status]:\n{body}"
        if self.state == "failed":
            return f"{label} failed: {body}]"
        if self.state == "stopped":
            return f"{label} stopped by parent]"
        return f"{label} {self.state}]"

    def format_preview(self,
                       limit: int = RECEIPT_PREVIEW_LIMIT) -> str:
        """Truncated Telegram receipt text."""
        return self.format(result_limit=limit)


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
    inbox = ctx.user_data.get('main_inbox')
    if inbox is None:
        inbox = SubagentInbox()
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
