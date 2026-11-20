import copy
import functools
import logging

from chat_update_context import ChatUpdateContext
from settings_store import load_conversation, load_settings, save_settings
from conversation import Conversation
from user_settings import UserSettings

logger = logging.getLogger(__name__)


def handler(func):
    @functools.wraps(func)
    async def wrapper(ctx: ChatUpdateContext):
        user_id = ctx.user_id
        allowed = ctx.context.bot_data.get('allowed_ids', [])
        if allowed and user_id not in allowed:
            await ctx.messaging.send_message(
                f'You are not authorized to use this chatbot. Your user ID is {user_id}.')
            return
        await init_conversation(ctx)
        return await func(ctx)
    return wrapper


def settings(func):
    @functools.wraps(func)
    async def wrapper(ctx: ChatUpdateContext):
        result = await func(ctx)
        if ctx.message_thread_id is None:
            # Unthreaded chat (the 'all' category): the thread shares the
            # template object, so one write persists both.
            ctx.user_data.settings = ctx.thread_data.settings
            await save_settings(ctx.user_id, None, ctx.settings)
        else:
            # Threaded chat: persist only this thread's copy. Only
            # changes made in the unthreaded/'all' category (thread id
            # None) are saved to the template, so other threads and
            # future threads keep their own values.
            await save_settings(
                ctx.user_id, ctx.message_thread_id, ctx.settings)
        return result
    return wrapper


async def init_conversation(ctx: ChatUpdateContext):
    if not ctx.settings_loaded:
        ctx.user_data.settings = (
            await load_settings(ctx.user_id, None)
            or UserSettings.default()
        )
    if ctx.thread_data.settings is None:
        if ctx.message_thread_id is None:
            # Unthreaded chat: share the template object.
            ctx.thread_data.settings = ctx.user_data.settings
        else:
            per_thread = await load_settings(
                ctx.user_id, ctx.message_thread_id)
            if per_thread is None:
                assert ctx.user_data.settings is not None
                per_thread = copy.deepcopy(ctx.user_data.settings)
                await save_settings(
                    ctx.user_id, ctx.message_thread_id, per_thread)
            ctx.thread_data.settings = per_thread
    if ctx.thread_data.conversation is None:
        logger.info('Initializing conversation for user %d', ctx.user_id)
        ctx.conversation = await load_conversation(
            ctx.user_id, ctx.message_thread_id) or Conversation()
    ctx.conversation.set_system_head(ctx.system_prompt)
