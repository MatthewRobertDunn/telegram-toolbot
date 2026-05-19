import asyncio
import json
import config
from tool_definitions import TOOL_DEFINITIONS, TOOL_MAP
from chat_update_context import ChatUpdateContext
from messaging import send_message
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=config.OPENAI_KEY,
                     base_url=config.OPENAI_BASE_URL)


async def execute_tool_call(ctx: ChatUpdateContext, tool_call) -> str:
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

    for _ in range(config.MAX_TOOL_ROUNDS):
        time_to_wait = 1
        t0 = asyncio.get_event_loop().time()
        response = None
        while asyncio.get_event_loop().time() - t0 < config.api_retry_time:
            try:
                create_kwargs: dict = dict(
                    model=effective_model,
                    messages=conversation,
                    n=1,
                    temperature=settings.temperature,
                    top_p=settings.top_p,
                    stream=False,
                    tools=tools,  # type: ignore
                    tool_choice="auto",
                )
                if reasoning_effort is not None:
                    create_kwargs["reasoning_effort"] = reasoning_effort
                if extra_body is not None:
                    create_kwargs["extra_body"] = extra_body
                # type: ignore
                response = await client.chat.completions.create(**create_kwargs)
                break
            except Exception as e:
                print(f'{type(e).__name__}: {e}, trying again...')
                await asyncio.sleep(time_to_wait)
                time_to_wait = time_to_wait * 2

        if response is None:
            return "Problem getting response from model."

        message = response.choices[0].message

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

        for tc in message.tool_calls:
            name = tc.function.name
            args = tc.function.arguments
            print(f"tool call: {name}({args})")
            if ctx.settings.show_tool_output:
                await send_message(ctx, f"Tool call: `{name}`\n```json\n{args[:1024]}\n```")
            result = await execute_tool_call(ctx, tc)
            print(f"tool result ({name}): {result[:200]}...")
            conversation.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return 'Maximum tool-call rounds reached without a final response.'
