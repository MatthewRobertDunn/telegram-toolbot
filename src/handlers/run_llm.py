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

# Retry backoff keeps the long-standing 1.2x growth, but one sleep is
# capped (config.api_retry_max_delay / API_RETRY_MAX_DELAY) so a single
# backoff cannot swallow hours of the retry budget.
RETRY_BACKOFF_GROWTH = 1.2

# Completion retry budget. DO NOT "fix" this into a short fail-fast
# policy: requests normally go to an LLM router that reroutes a failing
# request to another model, so nearly every failure is transient, and the
# budget is intentionally huge (config.api_retry_time, 10 h by default).
# Every context kind gets the same budget: a per-subagent "retry forever"
# rule is a known deferred design - it only makes sense together with
# persisted/resumable subagent state, which does not exist yet, so do not
# add a special case here until that lands.


def _loop_time() -> float:
    """Monotonic clock for the retry budget. Module-level so tests can
    fake it without patching asyncio globally."""
    return asyncio.get_running_loop().time()


# Module alias of asyncio.sleep, for the same reason.
_sleep = asyncio.sleep


def _retry_max_delay() -> float:
    """Upper bound, in seconds, on one retry sleep."""
    return float(config.api_retry_max_delay)


def _retry_budget() -> float:
    """Wall-clock retry budget in seconds for every context kind.

    One budget, deliberately long (config.api_retry_time): see the
    router-failover rationale at the top of this module. There is
    deliberately no per-context-kind or per-call override.
    """
    return float(config.api_retry_time)


def _non_retryable(error: Exception) -> bool:
    """Whether waiting cannot fix this error: give up immediately.

    Only authentication/permission failures (401/403): they fail for
    every model the router could reroute to, so retrying just burns the
    serialized worker's slot. Everything else is presumed transient per
    the rationale above - timeouts, connection errors, 429, 5xx, 400-class
    responses the router may route to another model, and even TypeError/
    ValueError arriving from the SDK or the stream handling (a mid-stream
    parsing hiccup is transient; failing it with zero retries would turn
    a router blip into a dropped reply).
    """
    status = getattr(error, 'status_code', None)
    return isinstance(status, int) and status in (401, 403)


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
    ``conversation`` is resolved before construction: it is the history
    this run operates on - the caller's argument, or ctx.conversation -
    so the phases never see a None to second-guess.
    """

    # No default: the resolved history is not optional, and every
    # constructor passes it by keyword.
    conversation: Conversation
    model: str | None = None
    reasoning_effort: str | None = None
    extra_body: dict | None = None
    tools: list[dict] | None = None
    inbox: SubagentInbox | None = None
    stream_drafts: bool = True
    announce_tools: bool = True

    @property
    def offer_tools(self) -> bool:
        """Whether tools were offered this run."""
        return self.tools is not None and len(self.tools) > 0


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


def _capabilities(ctx: AgentContext, cfg: RunConfig) -> _Capabilities:
    """The one computed view of this run's chat-facing capabilities."""
    # Only an interactive context can stream drafts or announce tool calls.
    chat = ctx if isinstance(ctx, ChatUpdateContext) else None
    streamer: DraftSink = (DraftStreamer(chat)
                           if chat is not None and cfg.stream_drafts
                           else NullDraftSink())
    announce_chat = chat if cfg.announce_tools else None
    # Thinking is chat-facing only for the main loop: a run on its own
    # scratch history (compaction's summary conversation, the
    # impersonation clone) reuses the interactive ctx, and its thinking
    # must not be posted to the chat or recorded on a history that is
    # about to be discarded.
    think_chat = (chat if chat is not None
                  and cfg.conversation is ctx.conversation
                  and ctx.settings.thinking_keep else None)
    return _Capabilities(streamer=streamer, announce_chat=announce_chat,
                         think_chat=think_chat)


def _request_kwargs(cfg: RunConfig, settings: UserSettings) -> dict:
    """The chat.completions.create kwargs for one round. Pure assembly
    from cfg and the history: an exception here is a bug, not a retry
    condition, so the retry loop does not catch it."""
    create_kwargs: dict = dict(
        model=cfg.model if cfg.model is not None else settings.model,
        messages=cfg.conversation.to_openai(),
        n=1,
        temperature=settings.temperature,
        top_p=settings.top_p,
        stream=True,
        extra_headers={
            "x-opencode-session": cfg.conversation.id,
        },
    )
    if cfg.offer_tools:
        create_kwargs["tools"] = cfg.tools  # type: ignore
        create_kwargs["tool_choice"] = "auto"
    if cfg.reasoning_effort is not None:
        create_kwargs["reasoning_effort"] = cfg.reasoning_effort
    if cfg.extra_body is not None:
        create_kwargs["extra_body"] = cfg.extra_body
    return create_kwargs


async def _one_completion(cfg: RunConfig,
                          settings: UserSettings,
                          streamer: DraftSink,
                          strip_thinking: bool,
                          budget: float) -> ConsumeStreamResult | None:
    """One completion request under the module retry policy.

    Retries within the wall-clock ``budget`` (config.api_retry_time for
    every context kind; see the rationale at the top of this module);
    a budget of zero or less means a single attempt and no retries.
    Backoff grows 1.2x per attempt, clamped to a per-sleep maximum so
    one sleep cannot swallow hours of the budget. Returns the consumed
    stream, or None when the budget is spent or a non-retryable error
    came back (the caller turns that into NO_RESPONSE_MESSAGE).
    Cancellation propagates immediately, so /stop and subagent_stop
    interrupt the retry sleep.

    Fail-fast is 401/403 only. Exceptions from building the request are
    not caught: _request_kwargs is pure assembly, so an exception there
    is a bug (or broken history) and propagates to the caller.
    """
    # A non-positive budget means "no retries", never "unbounded": the
    # first attempt always runs, and the first failure gives up.
    budget = max(0.0, float(budget))
    give_up_at = _loop_time() + budget
    time_to_wait = 1.0
    attempt = 0
    while True:
        # The first attempt always runs; after that, no attempt starts
        # once the budget is spent.
        if attempt and _loop_time() >= give_up_at:
            break
        attempt += 1
        # Outside the retry try: request assembly is deterministic, so an
        # exception here is a bug or broken history, not a transient
        # condition to wait out.
        create_kwargs = _request_kwargs(cfg, settings)
        try:
            streamer.new_round()
            response: Any = await client.chat.completions.create(
                **create_kwargs)
            message = await consume_stream(
                response, streamer, strip_thinking=strip_thinking)
            if (not message.content and not message.reasoning_content
                    and (not message.tool_calls or not cfg.offer_tools)):
                raise RuntimeError('empty response from model')
            return message
        except Exception as e:
            if _non_retryable(e):
                logger.warning(
                    'completion failed with a non-retryable error '
                    '(%s, status %s): %s; not retrying',
                    type(e).__name__, getattr(e, 'status_code', '-'), e)
                return None
            logger.warning(
                'completion attempt %d failed (%s: %s); %.0fs of '
                'retry budget left', attempt, type(e).__name__, e,
                max(0.0, give_up_at - _loop_time()))
            if _loop_time() >= give_up_at:
                # Budget already spent (e.g. a zero budget): no point
                # sleeping before a next attempt that will not run.
                break
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
    conversation = (conversation if conversation is not None
                    else ctx.conversation)
    cfg = RunConfig(model=model, reasoning_effort=reasoning_effort,
                    extra_body=extra_body, conversation=conversation,
                    tools=tools, inbox=inbox, stream_drafts=stream_drafts,
                    announce_tools=announce_tools)
    settings = ctx.settings
    effective_model = cfg.model if cfg.model is not None else settings.model

    tool_map = _tool_map_for(cfg.tools, role_for(ctx)) if cfg.tools else {}
    caps = _capabilities(ctx, cfg)
    await caps.streamer.start()

    while True:
        await _drain_inbox(conversation, cfg.inbox, caps.announce_chat)
        message = await _one_completion(
            cfg, settings, caps.streamer,
            strip_thinking=effective_model in config.THINK_SPLIT_MODELS,
            budget=_retry_budget())
        if message is None:
            caps.streamer.new_round()
            return NO_RESPONSE_MESSAGE

        # Cutoff is request time: mail arriving mid-flight waits for the
        # next wakeup (a queued one was scheduled when it was delivered;
        # a finished subagent has no use for it).
        if caps.think_chat is not None:
            await _maybe_post_thinking(conversation, caps.think_chat,
                                       message.reasoning_content)
        if not message.tool_calls or not cfg.offer_tools:
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
