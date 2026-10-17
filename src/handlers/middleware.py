import functools
import logging

from chat_update_context import ChatUpdateContext
from settings_store import load_settings, save_settings
from conversation import Conversation

logger = logging.getLogger(__name__)


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
        logger.info('Initializing conversation for user %d', ctx.user_id)
        ctx.conversation = Conversation()
    ctx.conversation.set_system_head(ctx.system_prompt)