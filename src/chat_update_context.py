"""The chat an agent acts for.

Two levels of access live here:

``AgentContext``
    The agent's own view of a chat: who it acts for, its conversation and
    its settings. Deliberately small - it holds no bot, no chat object and
    no ``messaging`` handle, so nothing reached through it can write to
    the user-visible chat.

``ChatUpdateContext``
    The interactive context built for each Telegram update. It is an
    ``AgentContext`` plus the Telegram objects, the ``messaging`` handle
    that can actually post, and the parent's generation bookkeeping.

Subagents are given an ``AgentContext`` built by
``ChatUpdateContext.new_subagent_context``; see that method for the state
isolation rules.
"""

import asyncio
import copy
from typing import TYPE_CHECKING

from telegram import Update, Bot, Chat, User, Message
from telegram.ext import ContextTypes

from conversation import Conversation
from messaging import Messaging
from skills_store import get_active_skills

if TYPE_CHECKING:
    from user_settings import UserSettings


class AgentContext:
    """The agent's own view of a chat, with no way to write to it.

    Deliberately minimal: who the agent acts for plus the state of its own
    work (its conversation and settings). Everything geared towards
    talking to the user - the Telegram objects, the messaging handle, the
    slash-command text, the parent's generation bookkeeping - lives on the
    interactive subclass.

    Every tool receives this type, so a tool cannot post to the chat, send
    files or typing indicators: the capability is simply absent.
    """

    def __init__(self, *, user_id: int, user_data: dict) -> None:
        self.user_id = user_id
        # Per-agent state. For a subagent this is its own dict (settings
        # snapshot + its conversation), never the parent's.
        self.user_data = user_data

    @property
    def conversation(self) -> "Conversation":
        return self.user_data['conversation']

    @conversation.setter
    def conversation(self, value: "Conversation") -> None:
        self.user_data['conversation'] = value

    @property
    def settings(self) -> "UserSettings":
        return self.user_data['settings']

    @settings.setter
    def settings(self, value: "UserSettings") -> None:
        self.user_data['settings'] = value


class ChatUpdateContext(AgentContext):
    """The interactive per-update context: the chat plus a Messaging handle.

    Built for each Telegram update (see ``create_chat_update_context``).
    Subagents never receive one; they receive the silent context returned
    by ``new_subagent_context``.
    """

    def __init__(self, update: Update,
                 context: ContextTypes.DEFAULT_TYPE,
                 bot: Bot,
                 chat: Chat,
                 user: User,
                 message: Message,
                 user_data: dict,
                 command_text: str,
                 command_name: str) -> None:
        super().__init__(user_id=user.id, user_data=user_data)
        self.update = update
        self.context = context
        self.bot = bot
        self.chat = chat
        self.user = user
        self.message = message
        self.command_text = command_text
        self.command_name = command_name
        self.messaging = Messaging(self)

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
    def generation_task(self) -> asyncio.Task | None:
        return self.user_data.get('generation_task')

    @generation_task.setter
    def generation_task(self, value: asyncio.Task | None) -> None:
        self.user_data['generation_task'] = value

    @property
    def generation_worker(self) -> "asyncio.Task | None":
        return self.user_data.get('generation_worker')

    @generation_worker.setter
    def generation_worker(self, value: "asyncio.Task | None") -> None:
        self.user_data['generation_worker'] = value

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
        parts = [base]
        profile = self.settings.user_profile
        if profile:
            parts.append(
                "The following is user-provided background information. Use it only when relevant.\n"
                "<user_profile>\n"
                f"{profile.strip()}\n"
                "</user_profile>"
            )
        skills = get_active_skills()
        if skills:
            blocks = [
                f"## {skill.name}\n"
                f"- Path: {skill.sandbox_path}\n"
                f"{skill.description}".rstrip()
                for skill in skills
            ]
            parts.append("# Available skills\n\n" + "\n\n".join(blocks))
        return "\n\n".join(parts)

    def new_subagent_context(self,
                             conversation: Conversation) -> AgentContext:
        """Build the isolated context a subagent runs against.

        The clone deliberately shares no mutable state with the parent:
        it gets its own ``user_data`` holding only a settings snapshot and
        the subagent's own conversation. The parent's coordination state -
        inbox, wakeup flags, conversation epoch, generation queue, sandbox
        read marks - is not reachable from here, which is what makes an
        accidental mutation impossible rather than merely discouraged.
        """
        user_data: dict = {'conversation': conversation}
        settings = self.user_data.get('settings')
        if settings is not None:
            user_data['settings'] = copy.deepcopy(settings)
        return AgentContext(
            user_id=self.user_id,
            user_data=user_data,
        )


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
