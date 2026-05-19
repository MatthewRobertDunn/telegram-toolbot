from chat_update_context import ChatUpdateContext
from handlers.middleware import handler
from handlers.run_with_tools import run_with_tools
from messaging import run_with_typing, send_message


@handler
async def text_message_handler(ctx: ChatUpdateContext):
    text = ctx.message.text or ctx.message.caption
    if not text:
        return

    ctx.conversation.append({
        'role': 'user',
        'content': text
    })

    gpt_response = await run_with_typing(ctx,
        run_with_tools(ctx))

    await send_message(ctx, gpt_response, parse_mode='LLM')
    ctx.conversation.append({
        'role': 'assistant',
        'content': gpt_response
    })