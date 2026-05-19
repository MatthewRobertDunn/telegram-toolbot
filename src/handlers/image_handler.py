from chat_update_context import ChatUpdateContext
from handlers.middleware import handler
from handlers.run_with_tools import run_with_tools
from messaging import send_message, run_with_typing


@handler
async def image_file_handler(ctx: ChatUpdateContext):
    msg = ctx.message
    conversation = ctx.conversation

    photo = msg.photo

    fid = photo[-1].file_id
    image_file = await ctx.bot.get_file(fid)
    image_url = str(image_file.file_path)

    s = msg.caption
    if s is None:
        s = ''

    conversation.append({
        "role": "user",
        "content": [
            {"type": "text", "text": s},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            },
        ],
    })

    gpt_response = await run_with_typing(ctx,
        run_with_tools(ctx))

    await send_message(ctx, gpt_response, parse_mode='LLM')
    conversation.append({
        'role': 'assistant',
        'content': gpt_response
    })