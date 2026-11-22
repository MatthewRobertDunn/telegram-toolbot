import asyncio
import logging

import openai
from chat_update_context import ChatUpdateContext
from command_spec import usage_line
from conversation import Conversation
from handlers.middleware import handler, settings
from handlers.generation_queue import generation_queue
from handlers.message_handlers import stop_generation
from persistence.settings_store import save_conversation
from config import OPENAI_KEY, OPENAI_BASE_URL

logger = logging.getLogger(__name__)


# Usage syntax for the commands in this module. Each constant is both
# rendered here and declared in the command table (commands.COMMANDS), so
# a handler's usage line and its /help entry cannot drift apart.
SYSTEMPROMPT_USAGE = '<text>'
MODEL_USAGE = '<model_id>'
SUBAGENT_MODEL_USAGE = '<model_id>'
TOP_P_USAGE = '<0.0-1.0>'
TEMPERATURE_USAGE = '<0.0-2.0>'
TIMEZONE_USAGE = '<IANA timezone, e.g. Australia/Sydney>'
USER_PROFILE_USAGE = '<text>'


@handler
async def start_restart_command(ctx: ChatUpdateContext):
    stop_generation(ctx)
    # Best-effort teardown of the old conversation's background work. A
    # bug here must not stop the fresh conversation from being built, but
    # it must also not stay silent: silent failure is exactly how subagent
    # tasks end up orphaned.
    try:
        from tools.subagent import cancel_all_subagents
        cancel_all_subagents(ctx)
    except Exception:
        logger.error('Could not cancel subagents for user %d before /start',
                     ctx.user_id, exc_info=True)
    try:
        # Any wakeup already queued in the worker carries the old epoch
        # and skips instead of injecting into the fresh conversation. This
        # must happen even if the cancel above failed, so the bump is its
        # own best-effort block.
        from agent_inbox import bump_conversation_epoch
        bump_conversation_epoch(ctx)
    except Exception:
        logger.error('Could not bump the conversation epoch for user %d '
                     'before /start', ctx.user_id, exc_info=True)
    ctx.conversation = Conversation()
    # Persist before announcing: the status message must not outlive its
    # own save. A failed write now surfaces through the handler error
    # boundary instead of being followed by a message claiming a start
    # that did not happen.
    await save_conversation(ctx.user_id, ctx.message_thread_id,
                            ctx.conversation)
    await ctx.messaging.send_message(f"Starting conversation with model `{ctx.model}`")


def _drop_queued(queue: "asyncio.Queue") -> int:
    """Release every still-queued work item so its submitter resumes.

    Nothing is closed: a queued item's job is a thunk, so no coroutine
    was ever created for an item the worker did not dequeue. Items the
    worker already dequeued are deliberately not touched: their job is
    mid-flight, and stopping that is the generation task cancel's job
    (``stop_generation``), not the queue's.
    """
    dropped = 0
    while not queue.empty():
        queue_item = queue.get_nowait()
        if queue_item.done.is_set():
            continue
        queue_item.abandon()
        dropped += 1
    return dropped


@handler
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


@handler
@settings
async def systemprompt_command(ctx: ChatUpdateContext):
    new_prompt = ctx.command_text

    if not new_prompt:
        await ctx.messaging.send_message(f'Current system prompt. {usage_line(ctx.command_name, SYSTEMPROMPT_USAGE)}')
        await ctx.messaging.send_preformatted_message(ctx.settings.system_prompt)
        return

    ctx.settings.system_prompt = new_prompt
    await ctx.messaging.send_message("System prompt updated.")

@handler
@settings
async def systemprompt_reset_command(ctx: ChatUpdateContext):
    ctx.settings.system_prompt = None
    await ctx.messaging.send_message("System prompt reset to default.")

# One lazily created sync client, shared across /model_list calls: the
# OpenAI client owns an httpx pool, and building one per invocation
# churned sockets for a list that rarely changes. (The async HTTP layer
# being added for generation does not cover this path, so the sync client
# stays and is simply run off the event loop.)
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


@handler
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


@handler
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


@handler
@settings
async def model_reset_command(ctx: ChatUpdateContext):
    ctx.settings.model = None
    await ctx.messaging.send_message(f'Model reset to default `{ctx.model}`.')


@handler
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


@handler
@settings
async def subagent_model_reset_command(ctx: ChatUpdateContext):
    ctx.settings.subagent_model = None
    await ctx.messaging.send_message(f'Subagent model reset to default `{ctx.settings.subagent_model}`.')

@handler
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


@handler
@settings
async def top_p_reset_command(ctx: ChatUpdateContext):
    ctx.settings.top_p = None
    await ctx.messaging.send_message(f'top\\_p reset to default `{ctx.settings.top_p}`.')


@handler
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


@handler
@settings
async def temperature_reset_command(ctx: ChatUpdateContext):
    ctx.settings.temperature = None
    await ctx.messaging.send_message(f'Temperature reset to default `{ctx.settings.temperature}`.')


@handler
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


@handler
@settings
async def settings_reset_command(ctx: ChatUpdateContext):
    ctx.settings = ctx.settings.default()
    await ctx.messaging.send_message('Settings reset to default.')


@handler
@settings
async def tool_output_command(ctx: ChatUpdateContext):
    current = ctx.settings.show_tool_output
    ctx.settings.show_tool_output = not current
    state = "on" if not current else "off"
    await ctx.messaging.send_message(f'Tool output is now {state}.')


@handler
@settings
async def thinking_keep_command(ctx: ChatUpdateContext):
    current = ctx.settings.thinking_keep
    ctx.settings.thinking_keep = not current
    if not current:
        await ctx.messaging.send_message('Thinking will be kept.')
    else:
        await ctx.messaging.send_message('Thinking will be discarded.')


@handler
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


@handler
@settings
async def user_profile_reset_command(ctx: ChatUpdateContext):
    ctx.settings.user_profile = None
    await ctx.messaging.send_message("Profile reset.")
