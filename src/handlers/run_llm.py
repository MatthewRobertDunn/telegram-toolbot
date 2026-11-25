import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import config
from agent_inbox import (PARENT_ID, TERMINAL_STATES, SubagentInbox,
                         SubagentMessage)
from async_utils import atomic, atomic_call
from tool_registry import ToolSurface
from chat_update_context import AgentContext, ChatUpdateContext
from conversation import (Conversation, ConversationMessage,
                          ConversationMessageType)
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

# Backoff grows 1.2x per attempt; one sleep is capped
# (config.api_retry_max_delay) so a single backoff cannot swallow hours of
# the retry budget.
RETRY_BACKOFF_GROWTH = 1.2

# Completion retry budget. Do not turn this into a short fail-fast policy:
# requests normally go to an LLM router that reroutes a failing request to
# another model, so nearly every failure is transient and the budget is
# intentionally huge (config.api_retry_time). Every context kind gets the
# same budget; no per-context or per-call override.


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
    """Wall-clock retry budget in seconds, shared by every context kind."""
    return float(config.api_retry_time)


def _non_retryable(error: Exception) -> bool:
    """Whether waiting cannot fix this error: only 401/403, which fail for
    every model the router could reroute to. Everything else (timeouts,
    connection errors, 429, 5xx, router-remapped 400s, even TypeError or
    ValueError from the SDK or stream handling) is presumed transient."""
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

    Parent-originated notes (role-confusion fallback) get a "Parent note"
    summary. For subagents, interim progress ("running") needs no suffix:
    the summary alone says the sender is still alive.
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
    message from its tool results. Returns the number drained. Display text
    is built here from the structured fields. With ``chat`` (the parent
    loop only) and show_tool_output on, each message is also posted as a
    collapsible receipt whose ids are linked to the injected turn, so
    /undo deletes the post with the turn. Subagent loops pass no chat and
    stay silent.
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
            await atomic_call(
                lambda: chat.messaging.send_collapsible(
                    _receipt_summary(message),
                    message.result or f"({message.state})"),
                user_message.add_telegram_message_id,
            )
        drained += 1
    return drained


@atomic
async def _maybe_post_thinking(conversation: Conversation,
                               chat: ChatUpdateContext,
                               reasoning: str | None) -> None:
    """Post one thinking block as a collapsible and record its Telegram ids
    in history. Only the ids are kept: the reasoning text itself is
    display-only and never stored or sent back."""
    if not reasoning:
        return
    message_ids = await chat.messaging.send_collapsible("Thinking", reasoning)
    conversation.add_thinking(telegram_message_ids=message_ids)


@dataclass(frozen=True)
class RunConfig:
    """Everything one run_llm call can vary.

    run_llm's keyword arguments map onto this one-for-one; the phases take
    the config object instead. ``conversation`` is resolved before
    construction (the caller's argument, or ctx.conversation), so the
    phases never see a None.
    """

    # No default: the resolved history is not optional, and every
    # constructor passes it by keyword.
    conversation: Conversation
    model: str | None = None
    tools: ToolSurface | None = None
    inbox: SubagentInbox | None = None
    stream_drafts: bool = True
    announce_tools: bool = True

    @property
    def offer_tools(self) -> bool:
        return self.tools is not None


@dataclass(frozen=True)
class _Capabilities:
    """What one run may do to the user-visible chat, computed once.

    Only an interactive (ChatUpdateContext) context can touch the chat;
    for a subagent's silent context everything here degrades to a no-op:

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
    """The chat.completions.create kwargs for one round, assembled from
    cfg and the history."""
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
    tools = cfg.tools
    if tools is not None:
        create_kwargs["tools"] = tools.definitions
        create_kwargs["tool_choice"] = "auto"
    return create_kwargs


async def _one_completion(cfg: RunConfig,
                          settings: UserSettings,
                          streamer: DraftSink,
                          strip_thinking: bool,
                          budget: float) -> ConsumeStreamResult | None:
    """One completion request under the module retry policy.

    Retries within the wall-clock ``budget``; a budget of zero or less
    means a single attempt. Returns the consumed stream, or None when the
    budget is spent or a non-retryable error came back (the caller turns
    that into NO_RESPONSE_MESSAGE). Cancellation propagates immediately,
    so /stop and subagent_stop interrupt the retry sleep. Fail-fast is
    401/403 only.
    """
    # A non-positive budget means "no retries", never "unbounded": the
    # first attempt always runs, and the first failure gives up.
    budget = max(0.0, float(budget))
    give_up_at = _loop_time() + budget
    time_to_wait = 1.0
    attempt = 0
    while True:
        if attempt and _loop_time() >= give_up_at:
            break
        attempt += 1
        # Outside the retry try: assembly is deterministic, so a failure
        # here is a bug (or broken history), not a transient condition.
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
                break
            await _sleep(time_to_wait)
            time_to_wait = min(time_to_wait * RETRY_BACKOFF_GROWTH,
                               _retry_max_delay())
    logger.warning('completion retry budget of %.0fs spent after %d '
                   'attempt(s); giving up', budget, attempt)
    return None


@atomic
async def _announce_tool_call(assistant_message: ConversationMessage,
                              chat: ChatUpdateContext, name: str,
                              args: str) -> None:
    message_ids = await chat.messaging.send_json(
        f"Tool call: `{name}`", args)
    assistant_message.add_telegram_message_id(message_ids)


async def _dispatch_tool_calls(ctx: AgentContext,
                               conversation: Conversation,
                               message: ConsumeStreamResult,
                               tool_map: dict,
                               announce_chat: ChatUpdateContext | None,
                               ) -> None:
    """Record one tool-call turn in history and execute every call.

    Announces each call when announcements are on, executes it, appends
    its result, and attaches tool-supplied images as a follow-up user
    message, leaving history ready for the next completion round.
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
            await _announce_tool_call(assistant_message, announce_chat,
                                      name, args)
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
                  conversation: Conversation | None = None,
                  tools: ToolSurface | None = None,
                  inbox: SubagentInbox | None = None,
                  stream_drafts: bool = True,
                  announce_tools: bool = True) -> str:
    """Run a model conversation to completion, executing tool calls as needed.

    ``ctx`` may be the interactive parent context or a subagent's silent
    one; chat interaction (drafts, announcements, receipts) simply does not
    happen without an interactive context. tools is a ToolSurface (see
    tool_registry.surface_for); None means no tools. A hallucinated name
    outside the surface gets 'Unknown tool'. inbox, when given, is drained
    at the top of every iteration - the only injection point. conversation
    defaults to ctx.conversation; nested calls construct their own, whose
    fresh id becomes that request's x-opencode-session.
    """
    conversation = (conversation if conversation is not None
                    else ctx.conversation)
    cfg = RunConfig(model=model, conversation=conversation,
                    tools=tools, inbox=inbox, stream_drafts=stream_drafts,
                    announce_tools=announce_tools)
    settings = ctx.settings
    effective_model = cfg.model if cfg.model is not None else settings.model

    tool_map = cfg.tools.functions if cfg.tools else {}
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
