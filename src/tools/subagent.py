"""Background subagents with message queues.

The main assistant is agent ``0`` (the root): it owns ``ctx.conversation``
and its lifecycle belongs to the generation queue. Every spawned subagent
gets its own fresh ``Conversation``, an ``asyncio.Queue`` inbox, and an
``asyncio.Task`` running ``run_llm`` to completion in the background.

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

from agent_inbox import (clear_main_inbox, conversation_epoch,
                         get_main_inbox, main_loop_active,
                         set_main_loop_active, set_wakeup_task,
                         take_wakeup_task, SubagentInbox, SubagentMessage)
from chat_update_context import ChatUpdateContext, context_tool
from conversation import Conversation
from handlers.run_llm import run_llm
from messaging import send_message

logger = logging.getLogger(__name__)

SUBAGENT_SYSTEM_PROMPT = (
    "You are a subagent working on behalf of a larger assistant, completing a "
    "single task in isolation.\n"
    "Work through the task independently, using the available tools as needed.\n"
    "When you are done, end with a clear, self-contained final report: the "
    "parent assistant only reads your last message, so it must contain "
    "everything worth keeping - results, findings, and any facts the task "
    "asked for."
)

MAX_SUBAGENTS = 8
# Finished records kept for subagent_list history; older ones are pruned.
MAX_RECORDS = 32
# A hung subagent is failed after this long so it cannot hold a slot and
# stay silent forever.
SUBAGENT_TIMEOUT = 30 * 60

# Id of the subagent whose tool loop is currently running (for
# subagent_send_to_parent labelling). The main agent never sets it.
_current_subagent: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar('current_subagent', default=None))


@dataclass
class SubagentRecord:
    id: str
    description: str
    conversation: Conversation
    inbox: SubagentInbox = field(default_factory=SubagentInbox)
    task: asyncio.Task | None = None
    status: str = "running"
    result: str | None = None
    notified: bool = False
    started_at: float = field(default_factory=time.monotonic)
    ended_at: float | None = None

    def finish(self, status: str) -> None:
        """Mark the record terminal, stamping the end time once."""
        self.status = status
        if self.ended_at is None:
            self.ended_at = time.monotonic()

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
    return ctx.user_data.setdefault('subagents', {})


def cancel_all_subagents(ctx: ChatUpdateContext) -> None:
    """Cancel every tracked subagent (e.g. on /start reset).

    Suppresses their stop notifications: the conversation is being wiped,
    so there is no parent left to notify. Stale inbox mail is dropped and
    any scheduled wakeup is cancelled (epoch-guarded anyway).
    """
    subs = ctx.user_data.get('subagents') or {}
    for record in subs.values():
        record.notified = True
        if record.task is not None and not record.task.done():
            record.task.cancel()
        record.finish("stopped")
    ctx.user_data['subagents'] = {}
    clear_main_inbox(ctx)
    wakeup = take_wakeup_task(ctx)
    if wakeup is not None and not wakeup.done():
        wakeup.cancel()
    set_main_loop_active(ctx, False)


def _running_count(ctx: ChatUpdateContext) -> int:
    return sum(1 for r in get_subagents(ctx).values() if r.status == "running")


def get_parent_tool_defs() -> list[dict]:
    """All registered tools except subagent-only ones.

    The steering tool is shared, so the parent offers it too: a parent
    that thinks it is a subagent gets a working tool instead of an
    'Unknown tool' error.
    """
    from tool_definitions import TOOL_DEFINITIONS
    return [t for t in TOOL_DEFINITIONS
            if t.get("function", {}).get("name") not in SUBAGENT_ONLY_NAMES]


def get_subagent_tool_defs() -> list[dict]:
    """All registered tools except the subagent controls (parent-only)."""
    from tool_definitions import TOOL_DEFINITIONS
    return [t for t in TOOL_DEFINITIONS
            if t.get("function", {}).get("name") not in PARENT_ONLY_NAMES]


async def _notify_parent(ctx: ChatUpdateContext,
                         message: SubagentMessage) -> None:
    """Queue a message for the main agent, waking it if idle.

    When the parent loop is still running it drains the inbox on its next
    iteration (and takes another turn for mid-round mail). When idle, a
    serialized worker job runs the parent loop so the message is actually
    processed. This body has no awaits: the put/check/claim sequence is
    atomic, so the wakeup decision cannot be lost.
    """
    inbox = get_main_inbox(ctx)
    inbox.put_nowait(message)
    if main_loop_active(ctx):
        return
    set_main_loop_active(ctx, True)
    set_wakeup_task(ctx, asyncio.create_task(
        _wakeup_parent(ctx, conversation_epoch(ctx))))


async def _wakeup_parent(ctx: ChatUpdateContext, epoch: int) -> None:
    # Imported late: generation_queue needs this module's tool getters at
    # its own import time, so a top-level import here would be circular.
    from handlers.generation_queue import generate_response, submit_task

    async def wakeup_job() -> None:
        # Serialized in the worker: a second finisher racing us sees an
        # empty inbox here and skips instead of causing a spurious call.
        # An epoch mismatch means /start reset the conversation meanwhile.
        if conversation_epoch(ctx) != epoch:
            return
        if get_main_inbox(ctx).empty():
            return
        await generate_response(ctx, content=None)

    try:
        await submit_task(ctx, wakeup_job())
    except Exception:
        logger.error('parent wakeup failed', exc_info=True)
    finally:
        set_wakeup_task(ctx, None)


async def _run_subagent(ctx: ChatUpdateContext, record: SubagentRecord,
                        model: str | None) -> None:
    token = _current_subagent.set(record.id)
    try:
        result = await asyncio.wait_for(
            run_llm(
                ctx,
                model=model,
                conversation=record.conversation,
                tools=get_subagent_tool_defs(),
                inbox=record.inbox,
                stream_drafts=False,
                announce_tools=False,
            ),
            timeout=SUBAGENT_TIMEOUT,
        )
    except asyncio.CancelledError:
        record.finish("stopped")
        # A task cancelled before its first run never enters the body, so
        # subagent_stop (which runs in the parent loop) owns the stop
        # notice; this is only a fallback for a mid-run cancel. The inbox
        # receipt is the single user-visible path (see _drain_inbox).
        if not record.notified:
            record.notified = True
            try:
                await _notify_parent(
                    ctx, SubagentMessage(id=record.id, state="stopped",
                                        description=record.description))
            except Exception:
                logger.warning('subagent stop notification failed', exc_info=True)
        raise
    except (asyncio.TimeoutError, TimeoutError):
        record.finish("timed out")
        record.result = f"Subagent timed out after {SUBAGENT_TIMEOUT}s."
        record.notified = True
        logger.warning('subagent %s timed out', record.id)
        try:
            await _notify_parent(
                ctx, SubagentMessage(id=record.id, state="timed out",
                                     description=record.description))
        except Exception:
            logger.warning('subagent timeout notification failed',
                           exc_info=True)
        return
    except Exception as e:
        record.finish("failed")
        record.result = f"Subagent error: {e}"
        record.notified = True
        logger.warning('subagent %s failed: %s', record.id, e, exc_info=True)
        try:
            await _notify_parent(
                ctx, SubagentMessage(id=record.id, state="failed",
                                     description=record.description,
                                     result=str(e)))
        except Exception:
            logger.warning('subagent failure notification failed',
                           exc_info=True)
        return
    finally:
        _current_subagent.reset(token)

    record.finish("done")
    record.result = result
    record.notified = True
    try:
        await _notify_parent(
            ctx, SubagentMessage(id=record.id, state="finished",
                                 description=record.description,
                                 result=result))
    except Exception:
        logger.warning('subagent parent notification failed', exc_info=True)


@context_tool
async def subagent_start(ctx: ChatUpdateContext, description: str,
                         prompt: str) -> str:
    """Start a background subagent and return its id immediately."""
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
                if old.status != "running"]
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
    )
    subs[sid] = record
    record.task = asyncio.create_task(
        _run_subagent(ctx, record, model=ctx.settings.subagent_model))
    return (f"Started subagent {sid} (`{description}`). It runs in the "
            f"background; you will be notified when it finishes. "
            f"Use subagent_steer to message it, subagent_list to check it.")


@context_tool
async def subagent_list(ctx: ChatUpdateContext) -> str:
    subs = get_subagents(ctx)
    if not subs:
        return "No subagents."
    lines = []
    for sid, rec in subs.items():
        if rec.status == "running":
            age = f"running {format_duration(rec.elapsed)}"
        elif rec.ended_at is not None:
            age = f"ran {format_duration(rec.elapsed)}"
        else:
            age = rec.status
        lines.append(f"- {sid}: `{rec.description}` [{rec.status}, {age}]")
    return "Subagents:\n" + "\n".join(lines)


@context_tool
async def subagent_stop(ctx: ChatUpdateContext, id: str) -> str:
    subs = get_subagents(ctx)
    record = subs.get(id)
    if record is None:
        return f"Error: unknown subagent {id}."
    if record.status != "running":
        return f"Subagent {id} already {record.status}."
    # Own the stop notice here: a task cancelled before its first run
    # never executes its own cleanup, so relying on it would lose the
    # Telegram notification. No parent-inbox notice: the tool result below
    # already informs the parent loop that called us.
    record.finish("stopped")
    record.notified = True
    if record.task is not None and not record.task.done():
        record.task.cancel()
    try:
        await send_message(
            ctx, f"Subagent `{record.description}` ({id}) stopped.")
    except Exception:
        logger.warning('subagent stop notification failed', exc_info=True)
    return f"Stopped subagent {id} (`{record.description}`)."


@context_tool
async def subagent_steer(ctx: ChatUpdateContext, id: str,
                       message: str) -> str:
    subs = get_subagents(ctx)
    record = subs.get(id)
    if record is None:
        return f"Error: unknown subagent {id}."
    if record.status != "running":
        return f"Error: subagent {id} is {record.status}, cannot message it."
    record.inbox.put_nowait(SubagentMessage(state="steer", result=message))
    return f"Message queued for subagent {id}."


@context_tool
async def subagent_send_to_parent(ctx: ChatUpdateContext,
                                  message: str) -> str:
    """Send a status message back to the parent assistant.

    Subagents use it for progress worth surfacing before the final
    report. The parent is also offered it (as a safety net for role
    confusion): with no subagent context it just notes the message back
    into the parent's own inbox.
    """
    sid = _current_subagent.get()
    description = ""
    if sid:
        record = get_subagents(ctx).get(sid)
        if record is not None:
            description = record.description
    await _notify_parent(
        ctx, SubagentMessage(id=sid or "", state="status",
                             description=description, result=message))
    return "Status sent to parent."


_SEND_TO_PARENT_DEF: dict = {
    "type": "function",
    "function": {
        "name": "subagent_send_to_parent",
        "description": (
            "Send a status message back to the parent assistant. "
            "Use it for progress worth surfacing before your final "
            "report."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Status update for the parent assistant",
                },
            },
            "required": ["message"],
            "additionalProperties": False,
        },
    },
}


PARENT_TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "subagent_start",
            "description": (
                "Start a background subagent on a task and return its id "
                "immediately (does not wait). The subagent has fresh context "
                "and none of this conversation's history, so include all "
                "relevant context in the prompt. It shares your tools "
                "(except these subagent controls) and the sandbox; file "
                "changes persist. The subagent can communicate with you via "
                "its subagent_send_to_parent tool. You will be notified with its final "
                "report when it finishes; use subagent_steer to steer it "
                "meanwhile."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "A short 3-5 word label for the task, displayed to the user",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The task for the subagent. Include context, constraints, and what its final report should contain.",
                    },
                },
                "required": ["description", "prompt"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subagent_list",
            "description": ("List all subagents with their ids, statuses, "
                            "and how long each has been running."),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subagent_stop",
            "description": "Stop a running background subagent by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The subagent id returned by subagent_start",
                    },
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subagent_steer",
            "description": (
                "Steer a running background subagent mid-run: queue a "
                "message delivered as a parent message in its conversation. "
                "Use it to redirect, answer its questions, or give it its "
                "next task once idle."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The subagent id returned by subagent_start",
                    },
                    "message": {
                        "type": "string",
                        "description": "Message for the subagent",
                    },
                },
                "required": ["id", "message"],
                "additionalProperties": False,
            },
        },
    },
    _SEND_TO_PARENT_DEF,
]

SUBAGENT_TOOL_DEFINITIONS: list[dict] = [
    _SEND_TO_PARENT_DEF,
]

CONTROL_TOOL_MAP: dict = {
    "subagent_start": subagent_start,
    "subagent_list": subagent_list,
    "subagent_stop": subagent_stop,
    "subagent_steer": subagent_steer,
    "subagent_send_to_parent": subagent_send_to_parent,
}

# Role subsets derived from the definition lists. A tool in both lists is
# shared, so each ONLY set excludes names present in the other role; that
# way renaming a tool in one place cannot silently expose it to the wrong
# role, and sharing one needs only a list entry.
_PARENT_NAMES = {t["function"]["name"] for t in PARENT_TOOL_DEFINITIONS}
_SUBAGENT_NAMES = {t["function"]["name"]
                   for t in SUBAGENT_TOOL_DEFINITIONS}
PARENT_ONLY_NAMES = _PARENT_NAMES - _SUBAGENT_NAMES
SUBAGENT_ONLY_NAMES = _SUBAGENT_NAMES - _PARENT_NAMES


def register(tools_list: list[dict], tool_map: dict):
    """Register all five control tools globally.

    Visibility is controlled per-agent by which definitions are passed to
    run_llm (get_parent_tool_defs / get_subagent_tool_defs); run_llm only
    executes tool names present in the offered definitions. Shared tools
    appear in both definition lists but are registered once (duplicates
    would offer the model the same function twice).
    """
    seen: set[str] = set()
    for t in list(PARENT_TOOL_DEFINITIONS) + list(SUBAGENT_TOOL_DEFINITIONS):
        name = t.get("function", {}).get("name")
        if name not in seen:
            tools_list.append(t)
            seen.add(name)
    tool_map.update(CONTROL_TOOL_MAP)
