import asyncio
import logging

import openai
from chat_update_context import ChatUpdateContext
from conversation import Conversation
from handlers.middleware import handler, settings
from handlers.generation_queue import generation_queue
from handlers.message_handlers import stop_generation
from settings_store import save_conversation
from config import OPENAI_KEY, OPENAI_BASE_URL



@handler
async def start_restart_command(ctx: ChatUpdateContext):
    stop_generation(ctx)
    try:
        from agent_inbox import bump_conversation_epoch
        from tools.subagent import cancel_all_subagents
        cancel_all_subagents(ctx)
        # Any wakeup already queued in the worker carries the old epoch
        # and skips instead of injecting into the fresh conversation.
        bump_conversation_epoch(ctx)
    except Exception:
        pass
    await ctx.messaging.send_message(f"Starting conversation with model `{ctx.model}`")
    ctx.conversation = Conversation()
    await save_conversation(ctx.user_id, ctx.message_thread_id,
                            ctx.conversation)


@handler
async def stop_command(ctx: ChatUpdateContext):
    stopped = stop_generation(ctx)
    # Clear anything waiting behind the running unit: queued items are
    # released so their submitters resume instead of waiting forever.
    queue = generation_queue(ctx)
    dropped = 0
    while not queue.empty():
        item = queue.get_nowait()
        if not item.done.is_set():
            item.done.set()
            dropped += 1

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
        await ctx.messaging.send_message(f'Current system prompt. Use `/{ctx.command_name} <text>` to update it.')
        await ctx.messaging.send_preformatted_message(ctx.settings.system_prompt)
        return

    ctx.settings.system_prompt = new_prompt
    await ctx.messaging.send_message("System prompt updated.")

@handler
@settings
async def systemprompt_reset_command(ctx: ChatUpdateContext):
    ctx.settings.system_prompt = None
    await ctx.messaging.send_message("System prompt reset to default.")

@handler
async def model_list_command(ctx: ChatUpdateContext):
    try:
        client = openai.OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_KEY,
        )
        models = sorted({model.id for model in client.models.list(extra_query={"available": "true"})})
        if ctx.command_text:
            models = [m for m in models if ctx.command_text.lower() in m.lower()]
        message = "\n".join(
            f"<code>/model {model}</code>"
            for model in models
        )
        if not message:
            message = f"No models match filter: {ctx.command_text}"
        await ctx.messaging.send_message(message, parse_mode='HTML')
    except Exception as exc:
        await ctx.messaging.send_message(f"Error fetching models: {exc}")


@handler
@settings
async def model_command(ctx: ChatUpdateContext):
    model = ctx.command_text
    if not model:
        await ctx.messaging.send_message(f'Current model is `{ctx.settings.model}`\n Usage: `/{ctx.command_name} <model_id>`')
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
        await ctx.messaging.send_message(f'Current subagent model is `{ctx.settings.subagent_model}`\n Usage: `/{ctx.command_name} <model_id>`')
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
        await ctx.messaging.send_message(f'Current top\\_p is `{ctx.settings.top_p}`\n Usage: `/{ctx.command_name} <0.0-1.0>`')
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
        await ctx.messaging.send_message(f'Current temperature is `{ctx.settings.temperature}`\n Usage: `/{ctx.command_name} <0.0-2.0>`')
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
        await ctx.messaging.send_message(f'Current timezone is `{ctx.settings.timezone}`\n Usage: `/{ctx.command_name} <IANA timezone, e.g. Australia/Sydney>`')
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
            await ctx.messaging.send_message(f"Your current profile:\n\n{current}\n\nUse `/{ctx.command_name} <text>` to update it.")
        else:
            await ctx.messaging.send_message(f"No profile set. Use `/{ctx.command_name} <text>` to set your profile.")
        return

    ctx.settings.user_profile = profile
    await ctx.messaging.send_message("Profile updated.")


@handler
@settings
async def user_profile_reset_command(ctx: ChatUpdateContext):
    ctx.settings.user_profile = None
    await ctx.messaging.send_message("Profile reset.")