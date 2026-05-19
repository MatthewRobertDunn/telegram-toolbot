import config
from telegram import BotCommand
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, filters
from chat_update_context import callback_adapter
from context_store import create_store, create_settings_store
from file_ops import read_file
from handlers.message_handlers import text_message_handler
from handlers.command_handlers import (
    start_restart_command,
    systemprompt_command,
    systemprompt_reset_command,
    model_list_command,
    model_command,
    model_reset_command,
    top_p_command,
    top_p_reset_command,
    temperature_command,
    temperature_reset_command,
    settings_reset_command,
    tool_output_command,
    user_profile_command,
    user_profile_reset_command,
)
from handlers.text_handler import text_file_handler
from handlers.image_handler import image_file_handler
from handlers.store_handlers import (
    store_command,
    store_list_command,
    store_delete_command,
    store_delete_all_command,
)
from tool_registration import register_tools
import user_settings


async def post_init(application: Application):
    register_tools()
    async with create_store() as store:
        await store.create_schema()
    async with create_settings_store() as store:
        await store.create_schema()
    await application.bot.set_my_commands(
        [
            BotCommand("start", "Start a new conversation"),
            BotCommand("systemprompt", "Set a custom system prompt"),
            BotCommand("systemprompt_reset", "Reset system prompt to default"),
            BotCommand("model", "Set the model to use"),
            BotCommand("model_list", "List available NVIDIA NIM models"),
            BotCommand("model_reset", "Reset model to default"),
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
            BotCommand("user_profile", "Set your profile/biography"),
            BotCommand("user_profile_reset", "Reset your profile to empty"),
        ]
    )


def _validate_config():
    if not config.API_TOKEN:
        raise ValueError("TELEGRAM_API_TOKEN environment variable is not set")
    if not config.OPENAI_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set")


def main():
    _validate_config()
    print("Starting bot...")
    user_settings.DEFAULT_SYSTEM_PROMPT = read_file("./system_prompt") or 'You are a helpful assistant in a Telegram chat bot platform.'

    application = (
        ApplicationBuilder()
        .token(config.API_TOKEN)
        .post_init(post_init)
        .concurrent_updates(True)
        .build()
    )
    application.bot_data.update({
        'allowed_ids': config.ALLOWED_IDS,
    })

    application.add_handler(CommandHandler('start', callback_adapter(start_restart_command)))
    application.add_handler(CommandHandler('systemprompt', callback_adapter(systemprompt_command)))
    application.add_handler(CommandHandler('systemprompt_reset', callback_adapter(systemprompt_reset_command)))
    application.add_handler(CommandHandler('model_list', callback_adapter(model_list_command)))
    application.add_handler(CommandHandler('model', callback_adapter(model_command)))
    application.add_handler(CommandHandler('model_reset', callback_adapter(model_reset_command)))
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
    application.add_handler(CommandHandler('user_profile', callback_adapter(user_profile_command)))
    application.add_handler(CommandHandler('user_profile_reset', callback_adapter(user_profile_reset_command)))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), callback_adapter(text_message_handler)))
    application.add_handler(MessageHandler(filters.ATTACHMENT & (~filters.PHOTO), callback_adapter(text_file_handler)))
    application.add_handler(MessageHandler(filters.PHOTO, callback_adapter(image_file_handler)))

    application.run_polling()


if __name__ == '__main__':
    main()