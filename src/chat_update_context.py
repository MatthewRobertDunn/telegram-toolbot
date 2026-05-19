from dataclasses import dataclass
from typing import TYPE_CHECKING

from telegram import Update, Bot, Chat, User, Message
from telegram.ext import ContextTypes

if TYPE_CHECKING:
    from user_settings import UserSettings


@dataclass
class ChatUpdateContext:
    update: Update
    context: ContextTypes.DEFAULT_TYPE
    bot: Bot
    chat: Chat
    user: User
    message: Message
    user_data: dict
    command_text: str
    command_name: str

    @property
    def message_text(self) -> str:
        text = self.message.text or self.message.caption
        if text is None:
            raise ValueError("Message has no text or caption")
        return text

    @property
    def chat_id(self) -> int:
        return self.chat.id

    @property
    def user_id(self) -> int:
        return self.user.id

    @property
    def conversation(self) -> list:
        return self.user_data['conversation']

    @conversation.setter
    def conversation(self, value: list) -> None:
        self.user_data['conversation'] = value

    @property
    def settings(self) -> "UserSettings":
        return self.user_data['settings']

    @settings.setter
    def settings(self, value: "UserSettings") -> None:
        self.user_data['settings'] = value

    @property
    def settings_loaded(self) -> bool:
        return 'settings' in self.user_data

    @property
    def model(self) -> str:
        return self.settings.model

    @model.setter
    def model(self, value: str) -> None:
        self.settings.model = value

    @property
    def system_prompt(self) -> str:
        base = self.settings.system_prompt
        profile = self.settings.user_profile
        if profile:
            return (
                f"{base}\n\n"
                "The following is user-provided background information. Use it only when relevant.\n"
                "<user_profile>\n"
                f"{profile.strip()}\n"
                "</user_profile>"
            )
        return base


def _parse_command_text(text: str) -> str:
    _, _, args = text.partition(' ')
    return args.strip()


def _parse_command_name(text: str) -> str:
    if not text.startswith("/"):
        return ""

    command = text[1:].split(maxsplit=1)[0]
    return command


def create_chat_update_context(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> ChatUpdateContext:
    chat = update.effective_chat
    assert chat is not None
    user = update.effective_user
    assert user is not None
    message = update.message
    assert message is not None
    user_data = context.user_data
    assert user_data is not None
    text = message.text or message.caption or ''
    return ChatUpdateContext(
        update=update,
        context=context,
        bot=context.bot,
        chat=chat,
        user=user,
        message=message,
        user_data=user_data,
        command_text=_parse_command_text(text),
        command_name=_parse_command_name(text),
    )


def context_tool(func):
    func._requires_context = True
    return func


def callback_adapter(func):
    import functools

    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        ctx = create_chat_update_context(update, context)
        return await func(ctx)

    return wrapper
