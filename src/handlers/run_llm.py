import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import config
from agent_inbox import (PARENT_ID, TERMINAL_STATES, SubagentInbox,
                         SubagentMessage)
from tool_registry import functions_for, role_for
from chat_update_context import AgentContext, ChatUpdateContext
from conversation import Conversation, ConversationMessageType
from user_settings import UserSettings
from .draft_streamer import DraftSink, DraftStreamer, NullDraftSink
from .consume_stream import ConsumeStreamResult, consume_stream
from tool_result import ToolCallResult
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=config.OPENAI_KEY,
                     base_url=config.OPENAI_BASE_URL)

# Returned in place of a reply when the model could not be reached or said
# nothing usable. Callers that need to detect failure (e.g. compaction)
# compare against these instead of the literal strings.
NO_RESPONSE_MESSAGE = "Problem getting response from model."
NO_CONTENT_MESSAGE = "Model response had no content."

# Safety valve for the agent loop: the most completions one run_llm call
# may spend on tool rounds (plus inbox-driven turns, which count the same
# way). A pathological model/inbox loop stops here instead of burning the
# serialized worker forever; it is meant to be unreachable in practice.
MAX_TOOL_ROUNDS = 100

# Retry backoff keeps the long-standing 1.2x growth, but one sleep is
# capped so a single backoff cannot swallow hours of the retry budget.
RETRY_BACKOFF_GROWTH = 1.2
# Upper bound on a single retry sleep (seconds). config.API_RETRY_MAX_DELAY
# (with its env knob) is being added by the config branch; the getattr
# keeps this branch working standalone until it lands.
RETRY_MAX_DELAY_DEFAULT = 300.0

# Completion retry budget. DO NOT "fix" this into a short fail-fast
# policy: requests normally go to an LLM router that reroutes a failing
# request to another model, so nearly every failure is transient, and the
# budget is intentionally huge (config.api_retry_time, 10 h by default).
# Subagents in particular retry indefinitely by default (config
# .SUBAGENT_API_RETRY_TIME, default 0 = no deadline): a subagent that
# gives up loses all of its background work, so a proxy restart or outage
# must not destroy it. The parent has a finite budget because the user
# can simply /stop it.


def _loop_time() -> float:
    """Monotonic clock for the retry budget. Module-level so tests can
    fake it without patching asyncio globally."""
    return asyncio.get_running_loop().time()


# Module alias of asyncio.sleep, for the same reason.
_sleep = asyncio.sleep


def _retry_max_delay() -> float:
    return float(getattr(config, 'API_RETRY_MAX_DELAY',
                         RETRY_MAX_DELAY_DEFAULT))


def _retry_budget(ctx: AgentContext, override: float | None) -> float:
    """Wall-clock retry budget in seconds; <= 0 means retry indefinitely.

    Interactive contexts keep the long-standing config.api_retry_time.
    Subagent contexts default to unlimited (see the rationale above);
    config.SUBAGENT_API_RETRY_TIME lands with the config branch and the
    getattr keeps this branch working standalone until then. ``override``
    is the per-call hook (RunConfig.retry_budget) for callers that need
    their own bound.
    """
    if override is not None:
        return float(override)
    if isinstance(ctx, ChatUpdateContext):
        return float(config.api_retry_time)
    return float(getattr(config, 'SUBAGENT_API_RETRY_TIME', 0))


def _non_retryable(error: Exception) -> bool:
    """Whether waiting cannot fix this error: give up immediately.

    Authentication/permission failures (401/403) fail for every model the
    router could reroute to, and a request-construction TypeError/
    ValueError (bad argument, bad kwarg - raised while building the
    request, not by the server) reproduces identically forever; retrying
    either only burns the serialized worker's slot. Everything else is
    presumed transient per the rationale above: timeouts, connection
    errors, 429, 5xx, and 400-class responses the router may route to
    another model all keep retrying.
    """
    status = getattr(error, 'status_code', None)
    if isinstance(status, int) and status in (401, 403):
        return True
    return isinstance(error, (TypeError, ValueError))


async def execute_tool_call(ctx: AgentContext, tool_call,
                            tool_map: dict) -> str | ToolCallResult:
    name = tool_call.function.name
    func = tool_map.get(name)
    if func is None:
        return f"Unknown tool: {name}"
    try:
        # Model-provided: malformed JSON is a model mistake, not a crash.
        # It is answered as a tool result so the loop can tell the model
        # (which can retry with fixed arguments) instead of killing the
        # whole queued job with an unhandled exception.
        arguments = json.loads(tool_call.function.arguments)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        return (f"Error: arguments for '{name}' are not valid JSON "
                f"({e}); call the tool again with a JSON object.")
    try:
        if getattr(func, '_needs_ctx', False):
            result = await func(ctx, **arguments)
        else:
            result = await func(**arguments)
        if isinstance(result, ToolCallResult):
            return result
        return str(result)
    except Exception as e:
        return f"Tool '{name}' execution error: {e}"


def _receipt_summary(message: SubagentMessage) -> str:
    """Telegram receipt summary for one drained inbox message.

    Names the sender and, for terminal states, how it ended. Interim
    progress ("running") needs no suffix: the summary alone says the
    sender is still alive. Parent-originated notes (role-confusion
    fallback) get a "Parent note" summary instead of a subagent one.
    The body is the full report, passed separately to send_collapsible.
    """
    if message.id == PARENT_ID:
        return "Parent note"
    # Non-parent mail always carries a description: subagent_start
    # rejects empty ones, so every record-backed message has one.
    summary = f"Subagent **{message.id}** *{message.description}*"
    if message.state in TERMINAL_STATES:
        summary += f" — {message.state}"
    return summary


async def _drain_inbox(conversation: Conversation,
                       inbox: SubagentInbox | None,
                       chat: ChatUpdateContext | None = None) -> int:
    """Move queued agent messages into the conversation as user turns.

    Called at the top of every agent loop iteration - the only injection
    point, so an injected message can never split an assistant tool_calls
    message from its tool results. Returns the number of messages drained.

    Display text is built here from the structured fields (never threaded
    through as pre-rendered strings). When ``chat`` is given (the parent
    loop only) and show_tool_output is on, each message is also posted to
    Telegram as a collapsible rich-text receipt (see _receipt_summary),
    with its message ids linked to the injected turn - so /undo deletes
    the post along with the turn. Subagent loops pass no chat and stay
    silent: their inbox traffic is not user-chat material.
    """
    if inbox is None:
        return 0
    drained = 0
    while True:
        try:
            message = inbox.get_nowait()
        except asyncio.QueueEmpty:
            break
        # No await between the queue pop and the conversation append:
        # asyncio can only cancel at an await, so the delivery is either
        # fully committed with the message consumed, or not consumed at
        # all. Do not insert an await here (a DB write, an API call)
        # without reordering - that would open a window where /stop can
        # drop the message on the floor.
        user_message = conversation.add_user(
            message.format(), message_type=ConversationMessageType.AGENT_MESSAGE)
        if chat is not None and chat.settings.show_tool_output:
            for message_id in await chat.messaging.send_collapsible(
                    _receipt_summary(message),
                    message.result or f"({message.state})"):
                user_message.add_telegram_message_id(message_id)
        drained += 1
    return drained


async def _maybe_post_thinking(conversation: Conversation,
                               chat: ChatUpdateContext,
                               reasoning: str | None) -> None:
    """Post one thinking block as a collapsible and record its Telegram
    ids in history. No-op when there is no reasoning to keep. Only the
    ids are kept - the reasoning text itself is display-only and never
    stored or sent back."""
    if not reasoning:
        return
    message_ids = await chat.messaging.send_collapsible("Thinking", reasoning)
    conversation.add_thinking(telegram_message_ids=message_ids)


def _tool_map_for(tool_defs: list[dict], role: str) -> dict:
    """The executable map matching the offered definitions.

    Derived from the role's registry view on demand, so a model that
    hallucinates a non-offered tool name cannot execute it: it simply is
    not in the map. The role matters because one name can have an
    implementation per audience (``read_image``).
    """
    offered = {t.get("function", {}).get("name") for t in tool_defs}
    return {name: func for name, func in functions_for(role).items()
            if name in offered}


@dataclass(frozen=True)
class RunConfig:
    """Everything one run_llm call can vary, with today's defaults.

    run_llm's public keyword arguments map onto this one-for-one (they
    stay the call signature for compatibility); the phases below take
    the config object instead of an eight-parameter breadcrumb trail.

    ``retry_budget`` is the per-call override for the completion retry
    budget (see _retry_budget); None defers to the context kind. It is
    not part of the public kwargs yet.
    """

    model: str | None = None
    reasoning_effort: str | None = None
    extra_body: dict | None = None
    conversation: Conversation | None = None
    tools: list[dict] | None = None
    inbox: SubagentInbox | None = None
    stream_drafts: bool = True
    announce_tools: bool = True
    retry_budget: float | None = None


@dataclass(frozen=True)
class _Capabilities:
    """What one run may do to the user-visible chat, computed once.

    Only an interactive (ChatUpdateContext) context can touch the chat;
    everything here degrades to a no-op for a subagent's silent context,
    replacing the isinstance/identity checks previously scattered through
    the loop:

    streamer - the live draft sink (NullDraftSink when silent or drafts
        are disabled for this call).
    announce_chat - where tool-call announcements and inbox receipts go
        (parent only, and only when announce_tools is on).
    think_chat - where kept thinking blocks go: parent-main-loop-only,
        because nested calls with their own history (compaction,
        impersonation) reuse the interactive ctx but must not spam it,
        and subagent loops have no chat at all. Respects /thinking_keep.
    """

    streamer: DraftSink
    announce_chat: ChatUpdateContext | None
    think_chat: ChatUpdateContext | None


def _capabilities(ctx: AgentContext, cfg: RunConfig,
                  conversation: Conversation) -> _Capabilities:
    """The one computed view of this run's chat-facing capabilities."""
    # Only an interactive context can stream drafts or announce tool calls.
    chat = ctx if isinstance(ctx, ChatUpdateContext) else None
    streamer: DraftSink = (DraftStreamer(chat)
                           if chat is not None and cfg.stream_drafts
                           else NullDraftSink())
    announce_chat = chat if cfg.announce_tools else None
    think_chat = (chat if chat is not None
                  and conversation is ctx.conversation
                  and ctx.settings.thinking_keep else None)
    return _Capabilities(streamer=streamer, announce_chat=announce_chat,
                         think_chat=think_chat)


def _request_kwargs(cfg: RunConfig, conversation: Conversation,
                    settings: UserSettings, effective_model: str,
                    offer_tools: bool) -> dict:
    """The chat.completions.create kwargs for one round. Built per
    attempt inside the guarded call: a construction TypeError/ValueError
    is a fail-fast retry outcome, not a crash of the queued job."""
    create_kwargs: dict = dict(
        model=effective_model,
        messages=conversation.to_openai(),
        n=1,
        temperature=settings.temperature,
        top_p=settings.top_p,
        stream=True,
        extra_headers={
            "x-opencode-session": conversation.id,
        },
    )
    if offer_tools:
        create_kwargs["tools"] = cfg.tools  # type: ignore
        create_kwargs["tool_choice"] = "auto"
    if cfg.reasoning_effort is not None:
        create_kwargs["reasoning_effort"] = cfg.reasoning_effort
    if cfg.extra_body is not None:
        create_kwargs["extra_body"] = cfg.extra_body
    return create_kwargs


async def _one_completion(cfg: RunConfig, conversation: Conversation,
                          settings: UserSettings, effective_model: str,
                          offer_tools: bool, streamer: DraftSink,
                          strip_thinking: bool,
                          budget: float) -> ConsumeStreamResult | None:
    """One completion request under the module retry policy.

    Retries within the wall-clock ``budget`` (<= 0 means no deadline:
    subagents retry indefinitely so a proxy restart or outage does not
    destroy their background work - see the rationale at the top of this
    module), except for errors waiting cannot fix (_non_retryable), which
    return None right away. Backoff grows 1.2x per attempt, clamped to a
    per-sleep maximum so one sleep cannot swallow hours of the budget.
    Returns the consumed stream, or None when the budget is spent or a
    non-retryable error came back (the caller turns that into
    NO_RESPONSE_MESSAGE). Cancellation propagates immediately, so
    /stop and subagent_stop interrupt the retry sleep.

    The completion itself is guarded end to end - request construction
    included - so a TypeError/ValueError from building a malformed
    request surfaces as NO_RESPONSE_MESSAGE instead of killing the
    queued job.
    """
    give_up_at = _loop_time() + budget if budget > 0 else None
    time_to_wait = 1.0
    attempt = 0
    while True:
        if give_up_at is not None and _loop_time() >= give_up_at:
            break
        attempt += 1
        try:
            streamer.new_round()
            create_kwargs = _request_kwargs(cfg, conversation, settings,
                                            effective_model, offer_tools)
            response: Any = await client.chat.completions.create(
                **create_kwargs)
            message = await consume_stream(
                response, streamer, strip_thinking=strip_thinking)
            if (not message.content and not message.reasoning_content
                    and (not message.tool_calls or not offer_tools)):
                raise RuntimeError('empty response from model')
            return message
        except Exception as e:
            if _non_retryable(e):
                logger.warning(
                    'completion failed with a non-retryable error '
                    '(%s, status %s): %s; not retrying',
                    type(e).__name__, getattr(e, 'status_code', '-'), e)
                return None
            if give_up_at is None:
                logger.warning(
                    'completion attempt %d failed (%s: %s); retrying '
                    'with no deadline', attempt, type(e).__name__, e)
            else:
                logger.warning(
                    'completion attempt %d failed (%s: %s); %.0fs of '
                    'retry budget left', attempt, type(e).__name__, e,
                    max(0.0, give_up_at - _loop_time()))
            await _sleep(time_to_wait)
            time_to_wait = min(time_to_wait * RETRY_BACKOFF_GROWTH,
                               _retry_max_delay())
    logger.warning('completion retry budget of %.0fs spent after %d '
                   'attempt(s); giving up', budget, attempt)
    return None


async def _dispatch_tool_calls(ctx: AgentContext,
                               conversation: Conversation,
                               message: ConsumeStreamResult,
                               tool_map: dict,
                               announce_chat: ChatUpdateContext | None,
                               ) -> None:
    """Record one tool-call turn in history and execute every call.

    Appends the assistant message (with the reasoning passback thinking
    models require), announces each call to Telegram when announcements
    are on, executes it, and appends its result - images attached by
    tools ride along as a follow-up user message. Leaves the history
    ready for the next completion round.
    """
    assistant_message = conversation.add_assistant(message.content or '')
    if message.reasoning_content:
        # DeepSeek/Kimi thinking mode require the reasoning_content of a
        # tool-call turn to be passed back on every later request; dropping
        # it makes the API reject the turn and poisons the whole session.
        assistant_message['reasoning_content'] = message.reasoning_content
    assistant_message['tool_calls'] = [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
        }
        for tc in message.tool_calls or []
    ]

    images = []
    for tc in message.tool_calls or []:
        name = tc.function.name
        args = tc.function.arguments
        logger.info("tool call: %s(%s)", name, args)
        if announce_chat is not None and ctx.settings.show_tool_output:
            for message_id in await announce_chat.messaging.send_json(
                    f"Tool call: `{name}`", args):
                assistant_message.add_telegram_message_id(message_id)
        result = await execute_tool_call(ctx, tc, tool_map)
        if isinstance(result, ToolCallResult):
            logger.info("tool result (%s): image attached", name)
            conversation.add_tool(result.text, tc.id)
            if result.image is not None:
                images.append(result.image)
            continue
        logger.info("tool result (%s): %s...", name, result[:200])
        conversation.add_tool(result, tc.id)

    for image in images:
        conversation.add_image(image.data, image.mime,
                               "(a tool attached this image)")


async def run_llm(ctx: AgentContext,
                  model: str | None = None,
                  reasoning_effort: str | None = None,
                  extra_body: dict | None = None,
                  conversation: Conversation | None = None,
                  tools: list[dict] | None = None,
                  inbox: SubagentInbox | None = None,
                  stream_drafts: bool = True,
                  announce_tools: bool = True) -> str:
    """Run a model conversation to completion, executing tool calls as needed.

    ``ctx`` may be either kind of context: the interactive parent context or
    a subagent's silent one. Chat interaction (drafts, tool announcements,
    inbox receipts) is type-checked here and simply does not happen without
    an interactive context; everything else (history, tools, inbox) is
    identical for both agents.

    tools=None means no tools are offered to the model (used for nested
    calls such as context compaction). Pass an explicit tool list otherwise;
    there is intentionally no "all tools" default. Only the passed
    definitions are offered to the model, and only those names are
    executable.

    inbox, when given, is drained into the conversation as user turns at
    the top of every loop iteration - the only injection point, so queued
    messages never split a tool-call batch.

    conversation defaults to ctx.conversation. Nested calls with their own
    history (subagent, compaction) construct their own Conversation, whose
    fresh id becomes that request's x-opencode-session.
    """
    cfg = RunConfig(model=model, reasoning_effort=reasoning_effort,
                    extra_body=extra_body, conversation=conversation,
                    tools=tools, inbox=inbox, stream_drafts=stream_drafts,
                    announce_tools=announce_tools)
    if cfg.conversation is not None:
        conversation = cfg.conversation
    else:
        conversation = ctx.conversation
    settings = ctx.settings
    effective_model = cfg.model if cfg.model is not None else settings.model

    offer_tools = cfg.tools is not None and len(cfg.tools) > 0
    tool_map = _tool_map_for(cfg.tools, role_for(ctx)) if cfg.tools else {}
    caps = _capabilities(ctx, cfg, conversation)
    await caps.streamer.start()

    rounds = 0
    while True:
        await _drain_inbox(conversation, cfg.inbox, caps.announce_chat)
        if rounds >= MAX_TOOL_ROUNDS:
            # Bail between rounds, while the history is still consistent
            # (every dispatched call already has its result), with a
            # string the caller sends as the reply.
            logger.warning('run_llm: tool round cap (%d) reached',
                           MAX_TOOL_ROUNDS)
            return (f"(Stopped after {MAX_TOOL_ROUNDS} tool rounds; "
                    "send a message to continue.)")
        rounds += 1
        message = await _one_completion(
            cfg, conversation, settings, effective_model, offer_tools,
            caps.streamer,
            strip_thinking=effective_model in config.THINK_SPLIT_MODELS,
            budget=_retry_budget(ctx, cfg.retry_budget))
        if message is None:
            caps.streamer.new_round()
            return NO_RESPONSE_MESSAGE

        # Cutoff is request time: mail arriving mid-flight waits for the
        # next wakeup (a queued one was scheduled when it was delivered;
        # a finished subagent has no use for it).
        if caps.think_chat is not None:
            await _maybe_post_thinking(conversation, caps.think_chat,
                                       message.reasoning_content)
        if not message.tool_calls or not offer_tools:
            if message.tool_calls:
                logger.warning('model returned tool calls while tools are '
                               'disabled; ignoring them')
            if message.content:
                return message.content
            if message.reasoning_content:
                return message.reasoning_content
            return NO_CONTENT_MESSAGE
        await _dispatch_tool_calls(ctx, conversation, message, tool_map,
                                   caps.announce_chat)
