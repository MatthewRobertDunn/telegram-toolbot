import asyncio
import logging

import openai
from chat_update_context import ChatUpdateContext
from command_spec import usage_line
from conversation import Conversation
from command_registry import command
from handlers.middleware import settings
from handlers.generation_queue import generation_queue
from handlers.message_handlers import stop_generation
from persistence.settings_store import save_conversation
from config import OPENAI_KEY, OPENAI_BASE_URL

logger = logging.getLogger(__name__)


# Usage syntax for this module's commands: rendered by the handlers and
# passed to their @command declarations, so the two cannot drift apart.
SYSTEMPROMPT_USAGE = '<text>'
MODEL_USAGE = '<model_id>'
SUBAGENT_MODEL_USAGE = '<model_id>'
TOP_P_USAGE = '<0.0-1.0>'
TEMPERATURE_USAGE = '<0.0-2.0>'
TIMEZONE_USAGE = '<IANA timezone, e.g. Australia/Sydney>'
USER_PROFILE_USAGE = '<text>'


@command('start', 'Start a new conversation')
async def start_restart_command(ctx: ChatUpdateContext):
    stop_generation(ctx)
    # Best-effort: a failure here must not block the new conversation, but
    # staying silent is how subagent tasks end up orphaned.
    try:
        from tools.subagent import cancel_all_subagents
        cancel_all_subagents(ctx)
    except Exception:
        logger.error('Could not cancel subagents for user %d before /start',
                     ctx.user_id, exc_info=True)
    try:
        # A wakeup already queued in the worker carries the old epoch and
        # skips instead of injecting into the fresh conversation. Its own
        # block, so it runs even when the cancel above failed.
        from agent_inbox import bump_conversation_epoch
        bump_conversation_epoch(ctx)
    except Exception:
        logger.error('Could not bump the conversation epoch for user %d '
                     'before /start', ctx.user_id, exc_info=True)
    ctx.conversation = Conversation()
    # Persist before announcing: a failed write must surface through the
    # handler error boundary, never as a message claiming a start that
    # did not happen.
    await save_conversation(ctx.user_id, ctx.message_thread_id,
                            ctx.conversation)
    await ctx.messaging.send_message(f"Starting conversation with model `{ctx.model}`")


def _drop_queued(queue: "asyncio.Queue") -> int:
    """Release every still-queued work item so its submitter resumes.

    Nothing is closed: a queued item's job is a thunk, so no coroutine was
    created for it. Items the worker already dequeued belong to
    ``stop_generation``, not the queue.
    """
    dropped = 0
    while not queue.empty():
        queue_item = queue.get_nowait()
        if queue_item.done.is_set():
            continue
        queue_item.abandon()
        dropped += 1
    return dropped


@command('stop', 'Stop the current AI response')
async def stop_command(ctx: ChatUpdateContext):
    stopped = stop_generation(ctx)
    # Clear anything waiting behind the running unit: queued items are
    # released so their submitters resume instead of waiting forever.
    dropped = _drop_queued(generation_queue(ctx))

    if stopped and dropped:
        await ctx.messaging.send_message(
            f"Stopped the current response and dropped "
            f"{dropped} queued request(s).")
    elif stopped:
        await ctx.messaging.send_message("Stopped the current response.")
    elif dropped:
        await ctx.messaging.send_message(f"Dropped {dropped} queued request(s).")
    else:
        await ctx.messaging.send_message("No response is currently being generated.")


@command('systemprompt', 'Set a custom system prompt',
         usage=SYSTEMPROMPT_USAGE)
@settings
async def systemprompt_command(ctx: ChatUpdateContext):
    new_prompt = ctx.command_text

    if not new_prompt:
        await ctx.messaging.send_message(f'Current system prompt. {usage_line(ctx.command_name, SYSTEMPROMPT_USAGE)}')
        await ctx.messaging.send_preformatted_message(ctx.settings.system_prompt)
        return

    ctx.settings.system_prompt = new_prompt
    await ctx.messaging.send_message("System prompt updated.")

@command('systemprompt_reset', 'Reset system prompt to default')
@settings
async def systemprompt_reset_command(ctx: ChatUpdateContext):
    ctx.settings.system_prompt = None
    await ctx.messaging.send_message("System prompt reset to default.")

# One lazily created sync client, shared across /model_list calls: the
# OpenAI client owns an httpx pool, and building one per invocation churned
# sockets. Calls run off the event loop via asyncio.to_thread.
_openai_client: openai.OpenAI | None = None


def _shared_openai_client() -> openai.OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_KEY,
        )
    return _openai_client


def _fetch_model_ids() -> list[str]:
    """The blocking half of /model_list: a synchronous OpenAI HTTP call.

    Never await it directly - it runs inside asyncio.to_thread so the
    event loop keeps serving every user's updates while it waits.
    """
    client = _shared_openai_client()
    return sorted({model.id for model in
                   client.models.list(extra_query={"available": "true"})})


@command('model_list', 'List available NVIDIA NIM models')
async def model_list_command(ctx: ChatUpdateContext):
    try:
        models = await asyncio.to_thread(_fetch_model_ids)
    except Exception as exc:
        await ctx.messaging.send_message(f"Error fetching models: {exc}")
        return
    if ctx.command_text:
        models = [m for m in models if ctx.command_text.lower() in m.lower()]
    message = "\n".join(
        f"<code>/model {model}</code>"
        for model in models
    )
    if not message:
        message = f"No models match filter: {ctx.command_text}"
    await ctx.messaging.send_message(message, parse_mode='HTML')


@command('model', 'Set the model to use', usage=MODEL_USAGE)
@settings
async def model_command(ctx: ChatUpdateContext):
    model = ctx.command_text
    if not model:
        await ctx.messaging.send_message(
            f'Current model is `{ctx.settings.model}`\n'
            f'{usage_line(ctx.command_name, MODEL_USAGE)}')
        return

    ctx.settings.model = model
    await ctx.messaging.send_message(f'Model set to `{model}`.')


@command('model_reset', 'Reset model to default')
@settings
async def model_reset_command(ctx: ChatUpdateContext):
    ctx.settings.model = None
    await ctx.messaging.send_message(f'Model reset to default `{ctx.model}`.')


@command('subagent_model', 'Set the model used by the subagent tool',
         usage=SUBAGENT_MODEL_USAGE)
@settings
async def subagent_model_command(ctx: ChatUpdateContext):
    model = ctx.command_text
    if not model:
        await ctx.messaging.send_message(
            f'Current subagent model is `{ctx.settings.subagent_model}`\n'
            f'{usage_line(ctx.command_name, SUBAGENT_MODEL_USAGE)}')
        return

    ctx.settings.subagent_model = model
    await ctx.messaging.send_message(f'Subagent model set to `{model}`.')


@command('subagent_model_reset', 'Reset subagent model to default')
@settings
async def subagent_model_reset_command(ctx: ChatUpdateContext):
    ctx.settings.subagent_model = None
    await ctx.messaging.send_message(f'Subagent model reset to default `{ctx.settings.subagent_model}`.')

@command('top_p', 'Set top_p sampling parameter', usage=TOP_P_USAGE)
@settings
async def top_p_command(ctx: ChatUpdateContext):
    val = ctx.command_text
    if not val:
        await ctx.messaging.send_message(
            f'Current top\\_p is `{ctx.settings.top_p}`\n'
            f'{usage_line(ctx.command_name, TOP_P_USAGE)}')
        return
    try:
        ctx.settings.top_p = float(val)
        await ctx.messaging.send_message(f'top\\_p set to `{ctx.settings.top_p}`.')
    except ValueError as e:
        await ctx.messaging.send_message(str(e))
        return


@command('top_p_reset', 'Reset top_p to default')
@settings
async def top_p_reset_command(ctx: ChatUpdateContext):
    ctx.settings.top_p = None
    await ctx.messaging.send_message(f'top\\_p reset to default `{ctx.settings.top_p}`.')


@command('temperature', 'Set temperature parameter',
         usage=TEMPERATURE_USAGE)
@settings
async def temperature_command(ctx: ChatUpdateContext):
    val = ctx.command_text
    if not val:
        await ctx.messaging.send_message(
            f'Current temperature is `{ctx.settings.temperature}`\n'
            f'{usage_line(ctx.command_name, TEMPERATURE_USAGE)}')
        return
    try:
        ctx.settings.temperature = float(val)
        await ctx.messaging.send_message(f'Temperature set to `{ctx.settings.temperature}`.')
    except ValueError as e:
        await ctx.messaging.send_message(str(e))
        return


@command('temperature_reset', 'Reset temperature to default')
@settings
async def temperature_reset_command(ctx: ChatUpdateContext):
    ctx.settings.temperature = None
    await ctx.messaging.send_message(f'Temperature reset to default `{ctx.settings.temperature}`.')


@command('timezone', 'Set your timezone (e.g. Australia/Sydney)',
         usage=TIMEZONE_USAGE)
@settings
async def timezone_command(ctx: ChatUpdateContext):
    zone = ctx.command_text.strip()
    if not zone:
        await ctx.messaging.send_message(
            f'Current timezone is `{ctx.settings.timezone}`\n'
            f'{usage_line(ctx.command_name, TIMEZONE_USAGE)}')
        return
    try:
        ctx.settings.timezone = zone
        await ctx.messaging.send_message(f'Timezone set to `{ctx.settings.timezone}`.')
    except ValueError as e:
        await ctx.messaging.send_message(str(e))
        return


@command('settings_reset', 'Reset all user settings to default')
@settings
async def settings_reset_command(ctx: ChatUpdateContext):
    ctx.settings = ctx.settings.default()
    await ctx.messaging.send_message('Settings reset to default.')


@command('tool_output', 'Toggle tool call output on/off')
@settings
async def tool_output_command(ctx: ChatUpdateContext):
    current = ctx.settings.show_tool_output
    ctx.settings.show_tool_output = not current
    state = "on" if not current else "off"
    await ctx.messaging.send_message(f'Tool output is now {state}.')


@command('thinking_keep', 'Toggle thinking output on/off')
@settings
async def thinking_keep_command(ctx: ChatUpdateContext):
    current = ctx.settings.thinking_keep
    ctx.settings.thinking_keep = not current
    if not current:
        await ctx.messaging.send_message('Thinking will be kept.')
    else:
        await ctx.messaging.send_message('Thinking will be discarded.')


@command('user_profile', 'Set your profile/biography',
         usage=USER_PROFILE_USAGE)
@settings
async def user_profile_command(ctx: ChatUpdateContext):
    profile = ctx.command_text

    if not profile:
        current = ctx.settings.user_profile
        if current:
            await ctx.messaging.send_message(
                f"Your current profile:\n\n{current}\n\n"
                f"{usage_line(ctx.command_name, USER_PROFILE_USAGE)}")
        else:
            await ctx.messaging.send_message(f"No profile set. {usage_line(ctx.command_name, USER_PROFILE_USAGE)}")
        return

    ctx.settings.user_profile = profile
    await ctx.messaging.send_message("Profile updated.")


@command('user_profile_reset', 'Reset your profile to empty')
@settings
async def user_profile_reset_command(ctx: ChatUpdateContext):
    ctx.settings.user_profile = None
    await ctx.messaging.send_message("Profile reset.")
