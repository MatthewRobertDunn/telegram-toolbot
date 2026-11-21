"""Background subagents with message queues.

The main assistant is agent ``0`` (the root): it owns ``ctx.conversation``
and its lifecycle belongs to the generation queue. Every spawned subagent
gets its own fresh ``Conversation``, its own silent
``AgentContext`` (a copy of the mutable state, with no way to
reach the chat), an ``asyncio.Queue`` inbox, and an ``asyncio.Task``
running ``run_llm`` to completion in the background.

All cross-agent delivery is queue-mediated and drained only at the top of
an agent loop iteration, so an injected message can never split an
assistant tool_calls message from its tool results.
"""

import asyncio
import contextvars
import logging
import time
import uuid
from dataclasses import dataclass, field

from agent_inbox import (PARENT_ID, SubagentStatus, clear_main_inbox,
                         conversation_epoch, get_main_inbox, SubagentInbox,
                         SubagentMessage)
from chat_update_context import AgentContext, ChatUpdateContext
from conversation import Conversation
from handlers.run_llm import run_llm
from tool_registry import BOTH, PARENT, SUBAGENT, agent_tool, definitions_for

logger = logging.getLogger(__name__)

SUBAGENT_SYSTEM_PROMPT = (
    "You are a subagent working on behalf of a larger assistant, completing a "
    "single task in isolation.\n"
    "Work through the task independently, using the available tools as needed.\n"
    "You cannot talk to the user directly: your only audience is the parent "
    "assistant. Anything the user should see (files, images, results) has to "
    "be described or pointed at in your report.\n"
    "When you are done, end with a clear, self-contained final report: the "
    "parent assistant only reads your last message, so it must contain "
    "everything worth keeping - results, findings, and any facts the task "
    "asked for."
)

MAX_SUBAGENTS = 8
# Finished records kept for subagent_list history; older ones are pruned.
MAX_RECORDS = 32

# The subagent whose tool loop is currently running (for
# subagent_send_to_parent). The record also carries the parent context, so
# a subagent's mail and wakeups always go through the interactive parent,
# never through its own silent context.
_current_subagent: contextvars.ContextVar["SubagentRecord | None"] = (
    contextvars.ContextVar('current_subagent', default=None))


@dataclass
class SubagentRecord:
    id: str
    description: str
    conversation: Conversation
    # The interactive context that spawned this subagent. Notifications and
    # wakeups run against it; the subagent's own silent context deliberately
    # has no access to the parent's inbox or generation queue.
    parent: ChatUpdateContext
    inbox: SubagentInbox = field(default_factory=SubagentInbox)
    task: asyncio.Task | None = None
    status: SubagentStatus = SubagentStatus.RUNNING
    result: str | None = None
    notified: bool = False
    started_at: float = field(default_factory=time.monotonic)
    ended_at: float | None = None

    def finish(self, status: SubagentStatus) -> None:
        """Mark the record terminal, stamping the end time once."""
        self.status = status
        if self.ended_at is None:
            self.ended_at = time.monotonic()

    def mark_notified(self) -> bool:
        """Claim this record's single parent notification.

        Several paths can race to report one record (the stop tool, the
        cancelled-task fallback, the finished/failed tail, a /start
        reset). The first caller wins and notifies; later callers lose
        and stay silent, instead of every path relying on call order to
        be first. Returns whether the caller won.
        """
        if self.notified:
            return False
        self.notified = True
        return True

    @property
    def elapsed(self) -> float:
        """Seconds from start until end (or now, while running)."""
        end = self.ended_at if self.ended_at is not None else time.monotonic()
        return max(0.0, end - self.started_at)


def format_duration(seconds: float) -> str:
    """Compact human duration: 45s, 1m23s, 2h04m."""
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def get_subagents(ctx: ChatUpdateContext) -> dict[str, SubagentRecord]:
    return ctx.thread_data.subagents


def cancel_all_subagents(ctx: ChatUpdateContext) -> None:
    """Cancel every tracked subagent (e.g. on /start reset).

    Suppresses their stop notifications: the conversation is being wiped,
    so there is no parent left to notify. Stale inbox mail is dropped, so
    any wakeup already queued no-ops on the empty inbox (and /start bumps
    the epoch, so even a wakeup scheduled before the reset skips).
    """
    for record in ctx.thread_data.subagents.values():
        record.mark_notified()
        if record.task is not None and not record.task.done():
            record.task.cancel()
        record.finish(SubagentStatus.STOPPED)
    ctx.thread_data.subagents = {}
    clear_main_inbox(ctx)


def _running_count(ctx: ChatUpdateContext) -> int:
    return sum(1 for r in get_subagents(ctx).values()
               if r.status is SubagentStatus.RUNNING)


def get_parent_tool_defs() -> list[dict]:
    """Every tool the parent agent is offered."""
    return definitions_for(PARENT)


def get_subagent_tool_defs() -> list[dict]:
    """The tools a subagent may use: the shared toolbox plus its own
    variants of the tools that would otherwise post to the chat.

    A subagent never sees a chat-writing tool: those are declared
    parent-only, and the shared ones are declared once per audience so the
    model-facing name stays the same for both agents.
    """
    return definitions_for(SUBAGENT)


def ensure_parent_wakeup(ctx: ChatUpdateContext) -> asyncio.Task:
    """Queue a wakeup job for a delivery, whatever the parent is doing.

    Deliberately unconditional: there is no "is a parent loop running?"
    flag to consult, because such a flag leaks True on every path that
    exits without draining (stale epoch, a racing finisher, a dropped
    job) and then silences all later notifications. Redundant wakeups are
    cheap instead - the first drains the whole inbox, the rest find it
    empty and return without sending or touching the conversation.

    Returns the scheduled task; it is not tracked, because nothing needs
    to cancel it (see cancel_all_subagents) and the inbox/epoch checks
    make a stale one harmless.
    """
    return asyncio.create_task(
        _wakeup_parent(ctx, conversation_epoch(ctx)))


async def _notify_parent(ctx: ChatUpdateContext,
                         message: SubagentMessage) -> None:
    """Queue a message for the main agent and schedule its wakeup.

    The put and the wakeup scheduling are done without an await between
    them, so the wakeup cannot miss its own mail.
    """
    get_main_inbox(ctx).put_nowait(message)
    ensure_parent_wakeup(ctx)


async def _wakeup_parent(ctx: ChatUpdateContext, epoch: int) -> None:
    # Imported late: generation_queue needs this module's tool getters at
    # its own import time, so a top-level import here would be circular.
    from handlers.generation_queue import generate_response, submit_task

    def mail_pending() -> bool:
        # An epoch mismatch means /start reset the conversation meanwhile:
        # the mail this wakeup was scheduled for is gone with it.
        return (conversation_epoch(ctx) == epoch
                and not get_main_inbox(ctx).empty())

    # Cheap no-op first: with nothing to deliver there is nothing to
    # enqueue, so a redundant wakeup costs no job, no generation and no
    # persistence. Checked again inside the job below, because the mail
    # can be drained by whoever runs between here and that job's turn.
    if not mail_pending():
        return

    async def wakeup_job() -> None:
        if not mail_pending():
            return
        await generate_response(ctx, content=None)

    try:
        await submit_task(ctx, wakeup_job)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.error('parent wakeup failed', exc_info=True)


async def _run_subagent(parent: ChatUpdateContext, record: SubagentRecord,
                        model: str | None) -> None:
    token = _current_subagent.set(record)
    try:
        result = await run_llm(
            parent.new_subagent_context(record.conversation),
            model=model,
            conversation=record.conversation,
            tools=get_subagent_tool_defs(),
            inbox=record.inbox,
            stream_drafts=False,
            announce_tools=False,
        )
    except asyncio.CancelledError:
        record.finish(SubagentStatus.STOPPED)
        # A task cancelled before its first run never enters the body, so
        # subagent_stop (which runs in the parent loop) owns the stop
        # notice; this is only a fallback for a mid-run cancel. The inbox
        # receipt is the single user-visible path (see _drain_inbox).
        if record.mark_notified():
            try:
                await _notify_parent(
                    parent,
                    SubagentMessage(id=record.id, state=SubagentStatus.STOPPED,
                                    description=record.description))
            except Exception:
                logger.warning('subagent stop notification failed',
                               exc_info=True)
        raise
    except Exception as e:
        record.finish(SubagentStatus.FAILED)
        record.result = f"Subagent error: {e}"
        if record.mark_notified():
            logger.warning('subagent %s failed: %s', record.id, e,
                           exc_info=True)
            try:
                await _notify_parent(
                    parent,
                    SubagentMessage(id=record.id, state=SubagentStatus.FAILED,
                                    description=record.description,
                                    result=str(e)))
            except Exception:
                logger.warning('subagent failure notification failed',
                               exc_info=True)
        return
    finally:
        _current_subagent.reset(token)

    record.finish(SubagentStatus.FINISHED)
    record.result = result
    if record.mark_notified():
        try:
            await _notify_parent(
                parent,
                SubagentMessage(id=record.id, state=SubagentStatus.FINISHED,
                                description=record.description,
                                result=result))
        except Exception:
            logger.warning('subagent parent notification failed',
                           exc_info=True)


@agent_tool(
    description=(
        "Start a background subagent and return its id immediately (no wait). "
        "Subagent has fresh context, no history: include all context in "
        "prompt. Shares your tools (except subagent controls and chat-posting "
        "tools) and sandbox; files persist. It can message you via "
        "subagent_send_to_parent. You get its final report on completion; "
        "no polling or tool call loops needed."),
    parameters={
        "description": {
            "type": "string",
            "description": ("Short 3-5 word task label, shown to user"),
        },
        "prompt": {
            "type": "string",
            "description": ("Subagent task. Include context, constraints, and "
                            "required final report contents."),
        },
    })
async def subagent_start(ctx: ChatUpdateContext, description: str,
                         prompt: str) -> str:
    """Start a background subagent and return its id immediately."""
    description = description.strip()
    if not description:
        return "Error: description must be non-empty."
    if _running_count(ctx) >= MAX_SUBAGENTS:
        return (f"Error: too many running subagents "
                f"(max {MAX_SUBAGENTS}). Stop one first.")
    sid = uuid.uuid4().hex[:6]
    subs = get_subagents(ctx)
    while sid in subs:
        sid = uuid.uuid4().hex[:6]
    # Prune oldest finished records so long sessions don't accumulate
    # full subagent histories (dicts preserve insertion order).
    finished = [sid for sid, old in subs.items()
                if old.status is not SubagentStatus.RUNNING]
    while len(subs) >= MAX_RECORDS and finished:
        del subs[finished.pop(0)]
    record = SubagentRecord(
        id=sid,
        description=description,
        conversation=Conversation([
            {"role": "system",
             "content": f"{SUBAGENT_SYSTEM_PROMPT}\n\nTask: {description}"},
            {"role": "user", "content": prompt},
        ]),
        parent=ctx,
    )
    subs[sid] = record
    record.task = asyncio.create_task(
        _run_subagent(ctx, record, model=ctx.settings.subagent_model))
    return (f"Started subagent {sid} (`{description}`). It runs in the "
            f"background; you will be notified when it finishes. "
            f"Use subagent_steer to message it, subagent_list to check it.")


@agent_tool(
    description=("List subagents with ids, statuses, and runtimes."))
async def subagent_list(ctx: ChatUpdateContext) -> str:
    subs = get_subagents(ctx)
    if not subs:
        return "No subagents."
    lines = []
    for sid, rec in subs.items():
        if rec.status is SubagentStatus.RUNNING:
            age = f"running {format_duration(rec.elapsed)}"
        elif rec.ended_at is not None:
            age = f"ran {format_duration(rec.elapsed)}"
        else:
            age = str(rec.status)
        lines.append(f"- {sid}: `{rec.description}` [{rec.status}, {age}]")
    return "Subagents:\n" + "\n".join(lines)


@agent_tool(
    description="Stop a running subagent by id.",
    parameters={
        "id": {
            "type": "string",
            "description": "Subagent id from subagent_start",
        },
    })
async def subagent_stop(ctx: ChatUpdateContext, id: str) -> str:
    subs = get_subagents(ctx)
    record = subs.get(id)
    if record is None:
        return f"Error: unknown subagent {id}."
    if record.status is not SubagentStatus.RUNNING:
        return f"Subagent {id} already {record.status}."
    # Own the stop notice here: a task cancelled before its first run
    # never executes its own cleanup, so relying on it would lose the
    # Telegram notification. No parent-inbox notice: the tool result below
    # already informs the parent loop that called us. mark_notified makes
    # the claim explicit - the cancelled-task fallback must not double up.
    record.finish(SubagentStatus.STOPPED)
    record.mark_notified()
    if record.task is not None and not record.task.done():
        record.task.cancel()
    try:
        await ctx.messaging.send_message(
            f"Subagent `{record.description}` ({id}) stopped.")
    except Exception:
        logger.warning('subagent stop notification failed', exc_info=True)
    return f"Stopped subagent {id} (`{record.description}`)."


@agent_tool(
    description=(
        "Message a running subagent mid-run (delivered as parent message). "
        "Use to redirect, answer questions, or assign next task when idle."),
    parameters={
        "id": {
            "type": "string",
            "description": "Subagent id from subagent_start",
        },
        "message": {
            "type": "string",
            "description": "Message for subagent",
        },
    })
async def subagent_steer(ctx: ChatUpdateContext, id: str,
                         message: str) -> str:
    subs = get_subagents(ctx)
    record = subs.get(id)
    if record is None:
        return f"Error: unknown subagent {id}."
    if record.status is not SubagentStatus.RUNNING:
        return f"Error: subagent {id} is {record.status}, cannot message it."
    record.inbox.put_nowait(SubagentMessage(id=PARENT_ID,
                                            state=SubagentStatus.STEER,
                                            result=message))
    return f"Message queued for subagent {id}."


@agent_tool(
    scope=SUBAGENT,
    description=(
        "Send progress update to parent assistant (you keep running). "
        "Use for progress worth surfacing before final report."),
    parameters={
        "message": {
            "type": "string",
            "description": "Progress update for parent",
        },
    })
async def subagent_send_to_parent(ctx: AgentContext,
                                  message: str) -> str:
    """Send a progress update back to the parent assistant.

    Subagents use it for progress worth surfacing before the final
    report; the state is RUNNING - the most up-to-date fact at send
    time - so the parent can tell interim mail apart from terminal
    reports (finished/failed/stopped). Subagent-only: the record comes
    from the running subagent context, never from the caller.

    The parent context comes from the record, never from the sender's
    silent context: that copy owns no inbox, wakeup flag or generation
    queue, and running the parent loop on it would write to the wrong
    conversation.
    """
    record = _current_subagent.get()
    if record is None:
        return "Error: no parent assistant is available for this context."
    parent = record.parent
    description = record.description
    sender_id = record.id
    await _notify_parent(
        parent, SubagentMessage(id=sender_id, state=SubagentStatus.RUNNING,
                                description=description, result=message))
    return "Status sent to parent."
