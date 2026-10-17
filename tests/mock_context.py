"""Shared ChatUpdateContext factory for unit tests.

Every part of the context is a plain dict or a SimpleNamespace stub, so
tests can poke at it freely without Telegram objects.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chat_update_context import ChatUpdateContext
from conversation import Conversation


def mock_ctx(*, user_id: int = 1, bot=None,
             conversation: list | Conversation | None = None,
             settings=None) -> ChatUpdateContext:
    """Build a ChatUpdateContext over stubs.

    bot: the fake bot tests want to capture sends with.
    conversation: preset conversation history.
    settings: the LLM settings namespace (model/temperature/top_p/
        show_tool_output) for tests that reach for ctx.settings.
    """
    context = SimpleNamespace(
        bot=bot, bot_data={}, user_data={}, chat_data={})
    message = SimpleNamespace(
        text='', caption=None,
        chat=SimpleNamespace(id=user_id, type='private'),
        from_user=SimpleNamespace(id=user_id, first_name='T', last_name='U'))
    ctx = ChatUpdateContext(
        update=None,  # type: ignore[arg-type]
        context=context,  # type: ignore[arg-type]
        bot=bot,  # type: ignore[arg-type]
        chat=message.chat,
        user=message.from_user,
        message=message,  # type: ignore[arg-type]
        user_data=context.user_data,
        chat_data=None,
        command_text='',
        command_name='')
    if conversation is not None:
        if isinstance(conversation, Conversation):
            ctx.conversation = conversation
        else:
            ctx.conversation = Conversation(list(conversation))
    if settings is not None:
        ctx.user_data['settings'] = settings
    return ctx
