import asyncio
import base64
import json
import logging
from types import SimpleNamespace
from typing import Any

import config
from tool_definitions import TOOL_DEFINITIONS, TOOL_MAP
from chat_update_context import ChatUpdateContext
from .draft_streamer import DraftStreamer
from .consume_stream import consume_stream
from messaging import send_message
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


async def execute_tool_call(ctx: ChatUpdateContext, tool_call) -> str | ToolCallResult:
    name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    func = TOOL_MAP.get(name)
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


async def run_llm(ctx: ChatUpdateContext,
                  model: str | None = None,
                  reasoning_effort: str | None = None,
                  extra_body: dict | None = None,
                  conversation: list | None = None,
                  excluded_tools: list[str] | None = None,
                  allow_tools: bool = True) -> str:
    """Run a model conversation to completion, executing tool calls as needed.

    With allow_tools=False no tools are offered to the model, so the reply is
    always plain text (used for nested calls such as context compaction).
    """
    if conversation is None:
        conversation = ctx.conversation
    settings = ctx.settings
    effective_model = model if model is not None else settings.model

    tools = TOOL_DEFINITIONS
    if excluded_tools:
        tools = [t for t in TOOL_DEFINITIONS
                 if t.get("function", {}).get("name") not in excluded_tools]

    streamer = DraftStreamer(ctx)
    await streamer.start()

    while True:
        time_to_wait = 1
        t0 = asyncio.get_event_loop().time()
        message: SimpleNamespace | Any = None
        while asyncio.get_event_loop().time() - t0 < config.api_retry_time:
            try:
                streamer.new_round()
                create_kwargs: dict = dict(
                    model=effective_model,
                    messages=conversation,
                    n=1,
                    temperature=settings.temperature,
                    top_p=settings.top_p,
                    stream=streamer.enabled,
                    extra_headers={
                        "x-opencode-session": str(ctx.chat_id),
                    },
                )
                if allow_tools:
                    create_kwargs["tools"] = tools  # type: ignore
                    create_kwargs["tool_choice"] = "auto"
                if reasoning_effort is not None:
                    create_kwargs["reasoning_effort"] = reasoning_effort
                if extra_body is not None:
                    create_kwargs["extra_body"] = extra_body
                # type: ignore
                response: Any = await client.chat.completions.create(**create_kwargs)
                if streamer.enabled:
                    message = await consume_stream(
                        response, streamer,
                        strip_thinking=effective_model
                        in config.THINK_SPLIT_MODELS)
                else:
                    message = response.choices[0].message
                break
            except Exception as e:
                logger.warning('%s: %s, trying again...', type(e).__name__, e)
                await asyncio.sleep(time_to_wait)
                time_to_wait = time_to_wait * 2

        if message is None:
            streamer.new_round()
            return NO_RESPONSE_MESSAGE

        if not message.tool_calls or not allow_tools:
            if message.tool_calls:
                logger.warning('model returned tool calls while tools are '
                               'disabled; ignoring them')
            if message.content:
                return message.content
            elif message.reasoning_content:
                return message.reasoning_content
            else:
                return NO_CONTENT_MESSAGE
            
        conversation.append({
            "role": "assistant",
            "content": message.content or '',
            "tool_calls": [
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
        })

        images = []
        for tc in message.tool_calls:
            name = tc.function.name
            args = tc.function.arguments
            logger.info("tool call: %s(%s)", name, args)
            if ctx.settings.show_tool_output:
                await send_message(ctx, f"Tool call: `{name}`\n```json\n{args[:1024]}\n```")
            result = await execute_tool_call(ctx, tc)
            if isinstance(result, ToolCallResult):
                logger.info("tool result (%s): image attached", name)
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result.text,
                })
                if result.image is not None:
                    images.append(result.image)
                continue
            logger.info("tool result (%s): %s...", name, result[:200])
            conversation.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        for image in images:
            b64 = base64.b64encode(image.data).decode()
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "(a tool attached this image)"},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{image.mime};base64,{b64}"}},
                ],
            })

