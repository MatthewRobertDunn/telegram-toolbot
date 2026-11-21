import logging
import sys

import config
import http_client
from telegram.ext import Application, ApplicationBuilder, MessageHandler, filters
from chat_update_context import callback_adapter
from commands import bot_commands, register_commands
from memory_store import create_memory_store
from settings_store import create_settings_store
from journal_store import create_journal_store
from file_ops import read_file
from handlers.message_handlers import text_message_handler
from handlers.upload_handler import upload_handler
from tool_registration import register_tools
from sandbox import manager, start_reaper, stop_reaper
from skills_store import load_skills
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
    load_skills()
    async with create_memory_store() as memory:
        await memory.create_schema()
    async with create_settings_store() as settings_store:
        await settings_store.create_schema()
    async with create_journal_store() as journal_store:
        await journal_store.create_schema()
    try:
        await manager.recover_leftover_sandboxes()
    except Exception as e:
        logger.warning("Could not recover leftover sandbox containers: %s", e)
    await start_reaper()
    # The command menu is generated from the same table the handlers are
    # registered from (see commands.py), so the two cannot disagree.
    await application.bot.set_my_commands(bot_commands())


def _validate_config():
    if not config.API_TOKEN:
        raise ValueError("TELEGRAM_API_TOKEN environment variable is not set")
    if not config.OPENAI_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set")


async def post_stop(application: Application):
    stop_reaper()
    # The shared aiohttp session outlives any single request, so it needs
    # closing here: otherwise shutdown leaves its connector (and the
    # transport underneath) open for the lifetime of the process.
    #
    # PTB awaits post_stop unguarded, before its own shutdown(); an
    # exception escaping here would skip Bot/Updater shutdown and the
    # persistence flush, so a failed close must not escape.
    try:
        await http_client.aclose()
    except Exception:
        logger.warning('Could not close the HTTP session', exc_info=True)


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

    # Every command handler: one row in commands.COMMANDS.
    register_commands(application)
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), callback_adapter(text_message_handler)))
    application.add_handler(MessageHandler(filters.ATTACHMENT, callback_adapter(upload_handler)))

    application.run_polling()


if __name__ == '__main__':
    main()
