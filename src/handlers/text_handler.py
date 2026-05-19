import tempfile
from chat_update_context import ChatUpdateContext
from handlers.middleware import handler
from handlers.run_with_tools import run_with_tools
from messaging import send_message, run_with_typing


@handler
async def text_file_handler(ctx: ChatUpdateContext):
    msg = ctx.message
    user_id = ctx.user_id
    assert msg.document is not None

    tmp = tempfile.NamedTemporaryFile(mode='w+', suffix='_chatbot_text', delete=False)
    try:
        fid = msg.document.file_id
        doc_file = await ctx.bot.get_file(fid)
        await doc_file.download_to_drive(custom_path=tmp.name)

        s = tmp.read()
    except Exception as e:
        print('error processing file: ', e)
        await send_message(ctx,
            f'Error processing file, encountered an error while downloading the file: {e}')
        return
    finally:
        tmp.close()

    caption = msg.caption
    if caption:
        s = caption + '\n' + s

    ctx.conversation.append({
        'role': 'user',
        'content': s
    })

    gpt_response = await run_with_typing(ctx,
        run_with_tools(ctx))

    await send_message(ctx, gpt_response, parse_mode='LLM')
    ctx.conversation.append({
        'role': 'assistant',
        'content': gpt_response
    })