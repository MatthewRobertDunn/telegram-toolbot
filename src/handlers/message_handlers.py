import asyncio
from telegram.constants import ChatType

from chat_update_context import ChatUpdateContext
from handlers.middleware import handler
from handlers.run_llm import run_llm
from messaging import run_with_typing, send_message


def stop_generation(ctx: ChatUpdateContext) -> bool:
    task = ctx.generation_task
    if task is None or task.done():
        ctx.generation_task = None
        return False
    task.cancel()
    ctx.generation_task = None
    return True


async def _run_generation(ctx: ChatUpdateContext):
    task = asyncio.create_task(run_with_typing(ctx, run_llm(ctx)))
    ctx.generation_task = task
    try:
        gpt_response = await task
    except asyncio.CancelledError:
        ctx.conversation.append({
            'role': 'assistant',
            'content': '(response was stopped by the user)',
        })
        return
    finally:
        ctx.generation_task = None

    await send_message(ctx, gpt_response, parse_mode='LLM')
    ctx.conversation.append({
        'role': 'assistant',
        'content': gpt_response
    })


async def group_text_message_handler(ctx: ChatUpdateContext):
    text = ctx.message.text or ctx.message.caption
    should_respond = f"@{ctx.bot.username}" in text # type: ignore
    text = text.replace(f"@{ctx.bot.username}", "").strip() # type: ignore

    ctx.conversation.append({
        'role': 'user',
        'content': f"{ctx.message.from_user.first_name or ''} {ctx.message.from_user.last_name or ''}: {text}" # type: ignore
    })

    if not should_respond:
        return
    
    text = text.replace(f"@{ctx.bot.username}", "").strip() # type: ignore

    await _run_generation(ctx)


@handler
async def text_message_handler(ctx: ChatUpdateContext):
    text = ctx.message.text or ctx.message.caption
    if not text:
        return
    
    if ctx.message.chat.type == ChatType.GROUP or ctx.message.chat.type == ChatType.SUPERGROUP:
        await group_text_message_handler(ctx)
        return

    ctx.conversation.append({
        'role': 'user',
        'content': text
    })

    await _run_generation(ctx)