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
from messaging import send_message
from tool_result import ToolCallResult
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=config.OPENAI_KEY,
                     base_url=config.OPENAI_BASE_URL)


async def _consume_stream(stream, streamer: DraftStreamer) -> SimpleNamespace:
    """Read a chat completion stream, forwarding content to live drafts.

    Returns an attribute-compatible stand-in for the non-streaming
    ChatCompletionMessage (.content/.tool_calls/.reasoning_content).
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_acc: dict[int, dict] = {}

    async for chunk in stream:
        choices = getattr(chunk, 'choices', None)
        if not choices:
            continue
        delta = choices[0].delta
        if delta is None:
            continue
        text = getattr(delta, 'content', None)
        if text:
            content_parts.append(text)
            await streamer.update(text)
        reasoning = getattr(delta, 'reasoning_content', None)
        if reasoning:
            reasoning_parts.append(reasoning)
        for part in getattr(delta, 'tool_calls', None) or []:
            idx = getattr(part, 'index', None)
            if not isinstance(idx, int):
                idx = max(tool_acc, default=-1) + 1
            entry = tool_acc.setdefault(
                idx, {'id': '', 'name': '', 'arguments': ''})
            if getattr(part, 'id', None):
                entry['id'] = part.id
            fn = getattr(part, 'function', None)
            if fn is not None:
                if getattr(fn, 'name', None):
                    entry['name'] += fn.name
                entry['arguments'] += getattr(fn, 'arguments', None) or ''

    await streamer.flush(force=True)

    tool_calls = []
    for idx in sorted(tool_acc):
        entry = tool_acc[idx]
        if entry['id'] or entry['name']:
            tool_calls.append(SimpleNamespace(
                id=entry['id'], type='function',
                function=SimpleNamespace(name=entry['name'],
                                         arguments=entry['arguments'])))
    return SimpleNamespace(
        content=''.join(content_parts) or None,
        tool_calls=tool_calls or None,
        reasoning_content=''.join(reasoning_parts) or None,
    )


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


async def run_with_tools(ctx: ChatUpdateContext,
                         model: str | None = None,
                         reasoning_effort: str | None = None,
                         extra_body: dict | None = None,
                         conversation: list | None = None,
                         excluded_tools: list[str] | None = None) -> str:
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
        create_kwargs: dict = {}
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
                    tools=tools,  # type: ignore
                    tool_choice="auto",
                    extra_headers={
                        "x-opencode-session": str(ctx.chat_id),
                    },
                )
                if reasoning_effort is not None:
                    create_kwargs["reasoning_effort"] = reasoning_effort
                if extra_body is not None:
                    create_kwargs["extra_body"] = extra_body
                # type: ignore
                response: Any = await client.chat.completions.create(**create_kwargs)
                if streamer.enabled:
                    message = await _consume_stream(response, streamer)
                else:
                    message = response.choices[0].message
                break
            except Exception as e:
                if create_kwargs.get("stream") and 'stream' in str(e).lower():
                    logger.info('backend rejected streaming (%s); '
                                'disabling draft streaming', e)
                    streamer.enabled = False
                    time_to_wait = 1
                    continue
                logger.warning('%s: %s, trying again...', type(e).__name__, e)
                await asyncio.sleep(time_to_wait)
                time_to_wait = time_to_wait * 2

        if message is None:
            streamer.new_round()
            return "Problem getting response from model."

        if not message.tool_calls:
            if message.content:
                return message.content
            elif message.reasoning_content:
                return message.reasoning_content
            else:
                return "Model response had no content."
            
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

