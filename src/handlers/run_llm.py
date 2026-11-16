import asyncio
import json
import logging
from typing import Any

import config
from agent_inbox import (PARENT_ID, TERMINAL_STATES, SubagentInbox,
                         SubagentMessage)
from tool_definitions import TOOL_MAP
from chat_update_context import ChatUpdateContext
from conversation import Conversation, ConversationMessageType
from .draft_streamer import DraftStreamer
from .consume_stream import ConsumeStreamResult, consume_stream
from messaging import send_collapsible, send_json
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


async def execute_tool_call(ctx: ChatUpdateContext, tool_call,
                           tool_map: dict) -> str | ToolCallResult:
    name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    func = tool_map.get(name)
    if func is None:
        return f"Unknown tool: {name}"
    try:
        if getattr(func, '_requires_context', False):
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


async def _drain_inbox(ctx: ChatUpdateContext,
                       conversation: Conversation,
                       inbox: SubagentInbox | None,
                       announce: bool = False) -> int:
    """Move queued agent messages into the conversation as user turns.

    Only called at the top of an agent loop iteration and just before a
    final reply is returned (never mid-tool-batch), so an injected message
    can never split an assistant tool_calls message from its tool results.
    Returns the number of messages drained.

    Display text is built here from the structured fields (never threaded
    through as pre-rendered strings). When announce is set (and
    show_tool_output is on), each message is also posted to Telegram as a
    collapsible rich-text receipt (see _receipt_summary), with its
    message ids linked to the injected turn - so /undo deletes the post
    along with the turn. Subagent loops pass announce=False to stay
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
        user_message = conversation.add_user(
            message.format(), message_type=ConversationMessageType.AGENT_MESSAGE)
        if announce and ctx.settings.show_tool_output:
            for message_id in await send_collapsible(
                    ctx, _receipt_summary(message),
                    message.result or f"({message.state})"):
                user_message.add_telegram_message_id(message_id)
        drained += 1
    return drained


def _tool_map_for(tool_defs: list[dict]) -> dict:
    """The executable map matching the offered definitions.

    Derived from the global TOOL_MAP on demand, so a model that
    hallucinates a non-offered tool name cannot execute it: it simply
    is not in the map.
    """
    offered = {t.get("function", {}).get("name") for t in tool_defs}
    return {name: func for name, func in TOOL_MAP.items()
            if name in offered}


async def run_llm(ctx: ChatUpdateContext,
                  model: str | None = None,
                  reasoning_effort: str | None = None,
                  extra_body: dict | None = None,
                  conversation: Conversation | None = None,
                  tools: list[dict] | None = None,
                  inbox: SubagentInbox | None = None,
                  stream_drafts: bool = True,
                  announce_tools: bool = True) -> str:
    """Run a model conversation to completion, executing tool calls as needed.

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
    if conversation is None:
        conversation = ctx.conversation
    settings = ctx.settings
    effective_model = model if model is not None else settings.model

    offer_tools = tools is not None and len(tools) > 0
    active_map = _tool_map_for(tools) if tools else {}

    streamer = DraftStreamer(ctx, enabled=stream_drafts)
    await streamer.start()

    while True:
        await _drain_inbox(ctx, conversation, inbox, announce_tools)
        time_to_wait = 1
        t0 = asyncio.get_event_loop().time()
        message: ConsumeStreamResult | None = None
        while asyncio.get_event_loop().time() - t0 < config.api_retry_time:
            try:
                streamer.new_round()
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
                    create_kwargs["tools"] = tools  # type: ignore
                    create_kwargs["tool_choice"] = "auto"
                if reasoning_effort is not None:
                    create_kwargs["reasoning_effort"] = reasoning_effort
                if extra_body is not None:
                    create_kwargs["extra_body"] = extra_body
                # type: ignore
                response: Any = await client.chat.completions.create(**create_kwargs)
                message = await consume_stream(
                    response, streamer,
                    strip_thinking=effective_model
                    in config.THINK_SPLIT_MODELS)
                break
            except Exception as e:
                logger.warning('%s: %s, trying again...', type(e).__name__, e)
                await asyncio.sleep(time_to_wait)

        if message is None:
            streamer.new_round()
            return NO_RESPONSE_MESSAGE

        if not message.tool_calls or not offer_tools:
            if message.tool_calls:
                logger.warning('model returned tool calls while tools are '
                               'disabled; ignoring them')
            if message.content:
                # Anything that arrived while this round was in flight gets
                # another turn instead of a stale reply (see _drain_inbox).
                if await _drain_inbox(ctx, conversation, inbox,
                                       announce_tools):
                    continue
                return message.content
            elif message.reasoning_content:
                if await _drain_inbox(ctx, conversation, inbox,
                                      announce_tools):
                    continue
                return message.reasoning_content
            else:
                if await _drain_inbox(ctx, conversation, inbox,
                                      announce_tools):
                    continue
                return NO_CONTENT_MESSAGE
            
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
            for tc in message.tool_calls
        ]

        images = []
        for tc in message.tool_calls:
            name = tc.function.name
            args = tc.function.arguments
            logger.info("tool call: %s(%s)", name, args)
            if announce_tools and ctx.settings.show_tool_output:
                for message_id in await send_json(
                        ctx, f"Tool call: `{name}`", args):
                    assistant_message.add_telegram_message_id(message_id)
            result = await execute_tool_call(ctx, tc, active_map)
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

