import asyncio

import openai
from chat_update_context import ChatUpdateContext
from compaction import (DEFAULT_KEEP_TOKENS, CompactionError, compact_context)
from conversation import Conversation
from handlers.middleware import handler, settings
from handlers.message_handlers import stop_generation
from messaging import send_preformatted_message, send_message, run_with_typing
from config import OPENAI_KEY, OPENAI_BASE_URL



@handler
async def start_restart_command(ctx: ChatUpdateContext):
    stop_generation(ctx)
    await send_message(ctx, f"Starting conversation with model `{ctx.model}`")
    ctx.conversation = Conversation()


@handler
async def stop_command(ctx: ChatUpdateContext):
    if stop_generation(ctx):
        await send_message(ctx, "Stopped the current response.")
    else:
        await send_message(ctx, "No response is currently being generated.")


@handler
async def compact_command(ctx: ChatUpdateContext):
    keep_tokens = DEFAULT_KEEP_TOKENS
    if ctx.command_text:
        try:
            keep_tokens = int(ctx.command_text)
            if keep_tokens < 1:
                raise ValueError
        except ValueError:
            await send_message(
                ctx,
                f'Usage: `/{ctx.command_name} [tokens of recent history to keep]`')
            return

    try:
        result = await run_with_typing(
            ctx,
            compact_context(
                ctx,
                keep_tokens=keep_tokens,
                on_start=lambda: send_message(ctx, 'Starting Compaction...')))
    except asyncio.CancelledError:
        await send_message(ctx, 'Compaction stopped.')
        return
    except CompactionError as e:
        await send_message(ctx, str(e))
        return

    await send_message(ctx, result.summary, parse_mode='LLM')
    await send_message(
        ctx,
        f'Compaction complete: {result.tokens_before:,} tokens '
        f'-> {result.tokens_after:,} tokens')


@handler
@settings
async def systemprompt_command(ctx: ChatUpdateContext):
    new_prompt = ctx.command_text

    if not new_prompt:
        await send_message(ctx, f'Current system prompt. Use `/{ctx.command_name} <text>` to update it.')
        await send_preformatted_message(ctx, ctx.settings.system_prompt)
        return

    ctx.settings.system_prompt = new_prompt
    await send_message(ctx, "System prompt updated.")

@handler
@settings
async def systemprompt_reset_command(ctx: ChatUpdateContext):
    ctx.settings.system_prompt = None
    await send_message(ctx, "System prompt reset to default.")

@handler
async def model_list_command(ctx: ChatUpdateContext):
    try:
        client = openai.OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_KEY,
        )
        models = sorted({model.id for model in client.models.list()})
        if ctx.command_text:
            models = [m for m in models if ctx.command_text.lower() in m.lower()]
        message = "\n".join(
            f"<code>/model {model}</code>"
            for model in models
        )
        if not message:
            message = f"No models match filter: {ctx.command_text}"
        await send_message(ctx, message, parse_mode='HTML')
    except Exception as exc:
        await send_message(ctx, f"Error fetching models: {exc}")


@handler
@settings
async def model_command(ctx: ChatUpdateContext):
    model = ctx.command_text
    if not model:
        await send_message(ctx, f'Current model is `{ctx.settings.model}`\n Usage: `/{ctx.command_name} <model_id>`')
        return

    ctx.settings.model = model
    await send_message(ctx, f'Model set to `{model}`.')


@handler
@settings
async def model_reset_command(ctx: ChatUpdateContext):
    ctx.settings.model = None
    await send_message(ctx, f'Model reset to default `{ctx.model}`.')


@handler
@settings
async def subagent_model_command(ctx: ChatUpdateContext):
    model = ctx.command_text
    if not model:
        await send_message(ctx, f'Current subagent model is `{ctx.settings.subagent_model}`\n Usage: `/{ctx.command_name} <model_id>`')
        return

    ctx.settings.subagent_model = model
    await send_message(ctx, f'Subagent model set to `{model}`.')


@handler
@settings
async def subagent_model_reset_command(ctx: ChatUpdateContext):
    ctx.settings.subagent_model = None
    await send_message(ctx, f'Subagent model reset to default `{ctx.settings.subagent_model}`.')

@handler
@settings
async def top_p_command(ctx: ChatUpdateContext):
    val = ctx.command_text
    if not val:
        await send_message(ctx, f'Current top\\_p is `{ctx.settings.top_p}`\n Usage: `/{ctx.command_name} <0.0-1.0>`')
        return
    try:
        ctx.settings.top_p = float(val)
        await send_message(ctx, f'top\\_p set to `{ctx.settings.top_p}`.')
    except ValueError as e:
        await send_message(ctx, str(e))
        return


@handler
@settings
async def top_p_reset_command(ctx: ChatUpdateContext):
    ctx.settings.top_p = None
    await send_message(ctx, f'top\\_p reset to default `{ctx.settings.top_p}`.')


@handler
@settings
async def temperature_command(ctx: ChatUpdateContext):
    val = ctx.command_text
    if not val:
        await send_message(ctx, f'Current temperature is `{ctx.settings.temperature}`\n Usage: `/{ctx.command_name} <0.0-2.0>`')
        return
    try:
        ctx.settings.temperature = float(val)
        await send_message(ctx, f'Temperature set to `{ctx.settings.temperature}`.')
    except ValueError as e:
        await send_message(ctx, str(e))
        return


@handler
@settings
async def temperature_reset_command(ctx: ChatUpdateContext):
    ctx.settings.temperature = None
    await send_message(ctx, f'Temperature reset to default `{ctx.settings.temperature}`.')


@handler
@settings
async def settings_reset_command(ctx: ChatUpdateContext):
    ctx.settings = ctx.settings.default()
    await send_message(ctx, 'Settings reset to default.')


@handler
@settings
async def tool_output_command(ctx: ChatUpdateContext):
    current = ctx.settings.show_tool_output
    ctx.settings.show_tool_output = not current
    state = "on" if not current else "off"
    await send_message(ctx, f'Tool output is now {state}.')


@handler
@settings
async def user_profile_command(ctx: ChatUpdateContext):
    profile = ctx.command_text

    if not profile:
        current = ctx.settings.user_profile
        if current:
            await send_message(ctx, f"Your current profile:\n\n{current}\n\nUse `/{ctx.command_name} <text>` to update it.")
        else:
            await send_message(ctx, f"No profile set. Use `/{ctx.command_name} <text>` to set your profile.")
        return

    ctx.settings.user_profile = profile
    await send_message(ctx, "Profile updated.")


@handler
@settings
async def user_profile_reset_command(ctx: ChatUpdateContext):
    ctx.settings.user_profile = None
    await send_message(ctx, "Profile reset.")    