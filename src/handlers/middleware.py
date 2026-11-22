import asyncio
import copy
import functools
import logging

from chat_update_context import ChatUpdateContext
from persistence.settings_store import load_conversation, load_settings, save_settings
from conversation import Conversation
from user_settings import UserSettings

logger = logging.getLogger(__name__)


def handler(func):
    """Authorize, then run a command behind an error boundary.

    The allowlist check and the state initialisation (``init_conversation``)
    keep their order and stay outside the boundary, exactly as before; the
    boundary wraps only the command itself, so a crash inside a handler is
    logged and answered with a terse message instead of vanishing into
    PTB's update loop (where the user would simply see nothing happen).
    """

    @functools.wraps(func)
    async def wrapper(ctx: ChatUpdateContext):
        user_id = ctx.user_id
        allowed = ctx.context.bot_data.get('allowed_ids', [])
        if allowed and user_id not in allowed:
            await ctx.messaging.send_message(
                f'You are not authorized to use this chatbot. Your user ID is {user_id}.')
            return
        await init_conversation(ctx)
        try:
            return await func(ctx)
        except asyncio.CancelledError:
            # /stop and shutdown cancel real work: that is a control
            # signal, not a failure, so it reaches the caller untouched.
            raise
        except Exception:
            label = ctx.command_name or 'update'
            logger.error('Command /%s failed for user %d', label, user_id,
                         exc_info=True)
            try:
                await ctx.messaging.send_message(
                    f'Something went wrong running /{label}. '
                    'The error has been logged.')
            except Exception:
                # The user still deserves the log even if delivery of the
                # failure notice itself blew up.
                logger.error('Could not report the failure to user %d',
                             user_id, exc_info=True)
            return None
    return wrapper


def settings(func):
    """Persist the user's settings after the command changed them.

    Where the write lands follows the thread/template aliasing (see
    ``ThreadData.settings_is_template``): an unthreaded chat shares the
    template object with its thread, so one save - through the shared
    object - updates both; a threaded chat owns a deep copy, so only its
    own row is written and other threads keep their values.
    """

    @functools.wraps(func)
    async def wrapper(ctx: ChatUpdateContext):
        result = await func(ctx)
        if ctx.message_thread_id is None:
            # Unthreaded chat (the 'all' category): the thread shares the
            # template object, so one write persists both.
            ctx.user_data.settings = ctx.thread_data.settings
            ctx.thread_data.settings_is_template = True
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
            # Unthreaded chat: the thread aliases the user's template
            # object in place (ThreadData.settings_is_template), so a
            # write through the thread is a write to the template - that
            # is what lets a setting changed without a thread be inherited
            # by every thread created afterwards.
            ctx.thread_data.settings = ctx.user_data.settings
            ctx.thread_data.settings_is_template = True
        else:
            # Threaded chat: the thread gets its own deep copy of the
            # template (settings_is_template stays False), so changes
            # here never leak into the template or other threads.
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
