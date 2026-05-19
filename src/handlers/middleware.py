import functools

from chat_update_context import ChatUpdateContext
from settings_store import load_settings, save_settings


def handler(func):
    @functools.wraps(func)
    async def wrapper(ctx: ChatUpdateContext):
        user_id = ctx.user_id
        allowed = ctx.context.bot_data.get('allowed_ids', [])
        if allowed and user_id not in allowed:
            await ctx.bot.send_message(
                chat_id=ctx.chat_id,
                text=f'You are not authorized to use this chatbot. Your user ID is {user_id}.')
            return
        await init_conversation(ctx)
        return await func(ctx)
    return wrapper


def settings(func):
    @functools.wraps(func)
    async def wrapper(ctx: ChatUpdateContext):
        result = await func(ctx)
        await save_settings(ctx.user_id, ctx.settings)
        return result
    return wrapper


async def init_conversation(ctx: ChatUpdateContext):
    if not ctx.settings_loaded:
        ctx.settings = await load_settings(ctx.user_id)
    if 'conversation' not in ctx.user_data:
        print(f'Initializing conversation for user {ctx.user_id}')
        ctx.conversation = []
    if len(ctx.conversation) == 0:
        ctx.conversation.append({
            'role': 'system',
            'content': '',
        })
    ctx.conversation[0] = {'role': 'system', 'content': ctx.system_prompt}