import random

from telegram.error import TelegramError

from chat_update_context import ChatUpdateContext
from handlers.middleware import handler
from words import WORDS

# Telegram caps forum topic titles at 128 characters.
MAX_TOPIC_NAME = 128


@handler
async def window_new_command(ctx: ChatUpdateContext):
    """Open a new forum topic (thread), titled by the command text.

    Bare /window_new picks a random two-word title.
    """
    name = ctx.command_text.strip()
    if not name:
        name = " ".join(random.sample(WORDS, 2))
    if len(name) > MAX_TOPIC_NAME:
        await ctx.messaging.send_message(
            f'Topic titles are capped at {MAX_TOPIC_NAME} characters.')
        return
    try:
        topic = await ctx.bot.create_forum_topic(
            chat_id=ctx.chat_id, name=name)
    except TelegramError as e:
        await ctx.messaging.send_message(
            f'Could not open a new thread: {e.message}')
        return
    await ctx.messaging.send_message(f'Opened “{topic.name}”.')
