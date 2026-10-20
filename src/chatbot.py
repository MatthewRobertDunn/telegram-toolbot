import asyncio
import logging
import sys

import config
from telegram import BotCommand
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, filters
from chat_update_context import callback_adapter
from context_store import create_store, create_settings_store
from file_ops import read_file
from handlers.message_handlers import text_message_handler
from handlers.conversation_handlers import (
    compact_command,
    regenerate_command,
    retry_command,
    undo_command,
)
from handlers.command_handlers import (
    start_restart_command,
    stop_command,
    systemprompt_command,
    systemprompt_reset_command,
    model_list_command,
    model_command,
    model_reset_command,
    subagent_model_command,
    subagent_model_reset_command,
    top_p_command,
    top_p_reset_command,
    temperature_command,
    temperature_reset_command,
    settings_reset_command,
    tool_output_command,
    user_profile_command,
    user_profile_reset_command,
)
from handlers.upload_handler import upload_handler
from handlers.sandbox_handlers import sandbox_reset_command
from handlers.store_handlers import (
    store_command,
    store_list_command,
    store_delete_command,
    store_delete_all_command,
)
from tool_registration import register_tools
from sandbox import manager, start_reaper, stop_reaper
import user_settings

logger = logging.getLogger(__name__)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)


async def post_init(application: Application):
    register_tools()
    async with create_store() as store:
        await store.create_schema()
    async with create_settings_store() as store:
        await store.create_schema()
    try:
        await manager.recover_leftover_sandboxes()
    except Exception as e:
        logger.warning("Could not recover leftover sandbox containers: %s", e)
    await start_reaper()
    await application.bot.set_my_commands(
        [
            BotCommand("start", "Start a new conversation"),
            BotCommand("stop", "Stop the current AI response"),
            BotCommand("regenerate", "Regenerate the last response"),
            BotCommand("retry", "Retry the conversation"),
            BotCommand("undo", "Undo the last message"),
            BotCommand("compact", "Compact the conversation context"),
            BotCommand("systemprompt", "Set a custom system prompt"),
            BotCommand("systemprompt_reset", "Reset system prompt to default"),
            BotCommand("model", "Set the model to use"),
            BotCommand("model_list", "List available NVIDIA NIM models"),
            BotCommand("model_reset", "Reset model to default"),
            BotCommand("subagent_model", "Set the model used by the subagent tool"),
            BotCommand("subagent_model_reset", "Reset subagent model to default"),
            BotCommand("top_p", "Set top_p sampling parameter"),
            BotCommand("top_p_reset", "Reset top_p to default"),
            BotCommand("temperature", "Set temperature parameter"),
            BotCommand("temperature_reset", "Reset temperature to default"),
            BotCommand("store", "Get or set a stored value"),
            BotCommand("store_list", "List all stored keys"),
            BotCommand("store_delete", "Delete a stored key"),
            BotCommand("store_delete_all", "Delete all stored keys"),
            BotCommand("tool_output", "Toggle tool call output on/off"),
            BotCommand("settings_reset", "Reset all user settings to default"),
            BotCommand("sandbox_reset", "Destroy your sandbox and all its files"),
            BotCommand("user_profile", "Set your profile/biography"),
            BotCommand("user_profile_reset", "Reset your profile to empty"),
        ]
    )


def _validate_config():
    if not config.API_TOKEN:
        raise ValueError("TELEGRAM_API_TOKEN environment variable is not set")
    if not config.OPENAI_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set")


async def post_stop(application: Application):
    stop_reaper()


def main():
    configure_logging()
    _validate_config()
    logger.info("Starting bot...")
    user_settings.DEFAULT_SYSTEM_PROMPT = read_file("./system_prompt") or 'You are a helpful assistant in a Telegram chat bot platform.'

    application = (
        ApplicationBuilder()
        .token(config.API_TOKEN)
        .post_init(post_init)
        .post_stop(post_stop)
        .concurrent_updates(True)
        .build()
    )
    application.bot_data.update({
        'allowed_ids': config.ALLOWED_IDS,
    })

    application.add_handler(CommandHandler('start', callback_adapter(start_restart_command)))
    application.add_handler(CommandHandler('stop', callback_adapter(stop_command)))
    application.add_handler(CommandHandler('regenerate', callback_adapter(regenerate_command)))
    application.add_handler(CommandHandler('retry', callback_adapter(retry_command)))
    application.add_handler(CommandHandler('undo', callback_adapter(undo_command)))
    application.add_handler(CommandHandler('compact', callback_adapter(compact_command)))
    application.add_handler(CommandHandler('systemprompt', callback_adapter(systemprompt_command)))
    application.add_handler(CommandHandler('systemprompt_reset', callback_adapter(systemprompt_reset_command)))
    application.add_handler(CommandHandler('model_list', callback_adapter(model_list_command)))
    application.add_handler(CommandHandler('model', callback_adapter(model_command)))
    application.add_handler(CommandHandler('model_reset', callback_adapter(model_reset_command)))
    application.add_handler(CommandHandler('subagent_model', callback_adapter(subagent_model_command)))
    application.add_handler(CommandHandler('subagent_model_reset', callback_adapter(subagent_model_reset_command)))
    application.add_handler(CommandHandler('top_p', callback_adapter(top_p_command)))
    application.add_handler(CommandHandler('top_p_reset', callback_adapter(top_p_reset_command)))
    application.add_handler(CommandHandler('temperature', callback_adapter(temperature_command)))
    application.add_handler(CommandHandler('temperature_reset', callback_adapter(temperature_reset_command)))
    application.add_handler(CommandHandler('store_list', callback_adapter(store_list_command)))
    application.add_handler(CommandHandler('store', callback_adapter(store_command)))
    application.add_handler(CommandHandler('store_delete', callback_adapter(store_delete_command)))
    application.add_handler(CommandHandler('store_delete_all', callback_adapter(store_delete_all_command)))
    application.add_handler(CommandHandler('tool_output', callback_adapter(tool_output_command)))
    application.add_handler(CommandHandler('settings_reset', callback_adapter(settings_reset_command)))
    application.add_handler(CommandHandler('sandbox_reset', callback_adapter(sandbox_reset_command)))
    application.add_handler(CommandHandler('user_profile', callback_adapter(user_profile_command)))
    application.add_handler(CommandHandler('user_profile_reset', callback_adapter(user_profile_reset_command)))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), callback_adapter(text_message_handler)))
    application.add_handler(MessageHandler(filters.ATTACHMENT, callback_adapter(upload_handler)))

    application.run_polling()


if __name__ == '__main__':
    main()