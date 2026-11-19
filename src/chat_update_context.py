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

 ``UserData``
    Per-user state: settings and sandbox read marks. A single instance
    lives in PTB's ``context.user_data`` dict
    (see ``create_chat_update_context``).

 ``ThreadData``
    Per-chat/thread state: the conversation plus all generation and
    subagent coordination (queue, tasks, inbox, epoch, wakeup, records).
    Stored in PTB's ``context.user_data`` dict keyed by
    ``message_thread_id`` (``None`` for unthreaded chats), so each thread
    gets its own isolated state while sharing the user's ``UserData``.

Subagents are given an ``AgentContext`` built by
``ChatUpdateContext.new_subagent_context``; see that method for the state
isolation rules.
"""

import asyncio
import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from telegram import Update, Bot, Chat, User, Message
from telegram.ext import ContextTypes

from conversation import Conversation
from messaging import Messaging
from skills_store import get_active_skills

if TYPE_CHECKING:
    from agent_inbox import SubagentInbox
    from tools.subagent import SubagentRecord
    from user_settings import UserSettings


@dataclass
class ThreadData:
    """Per-chat/thread state: the conversation, its settings and its coordination.

    Everything here belongs to one chat/thread, not to the user: the
    conversation, its own settings copy (cloned from the user's template
    on first touch), the serialized generation machinery (queue, worker
    and tracked task), the subagent inbox plus its wakeup bookkeeping,
    the epoch guarding against stale wakeups, and the subagent records
    that report back to this thread.
    """

    conversation: Conversation | None = None
    settings: "UserSettings | None" = None
    generation_task: asyncio.Task | None = None
    generation_worker: asyncio.Task | None = None
    generation_queue: asyncio.Queue | None = None
    main_inbox: "SubagentInbox | None" = None
    main_loop_active: bool = False
    conversation_epoch: int = 0
    wakeup_task: asyncio.Task | None = None
    subagents: dict[str, "SubagentRecord"] = field(default_factory=dict)


@dataclass
class UserData:
    """Per-user state, shared by every chat/thread of that user.

    Only genuinely user-scoped things live here: the settings template
    that new threads clone on first touch (kept in sync with the most
    recently changed thread), and the sandbox read marks (the sandbox
    itself is per-user, so a file read in any thread counts everywhere).
    """

    settings: "UserSettings | None" = None
    sandbox_read_files: set[str] = field(default_factory=set)


# The single key holding the UserData instance inside PTB's per-user dict.
_USER_DATA_KEY = '__user_data__'

# The key holding the per-thread state map inside PTB's per-user dict:
# thread id (message_thread_id, None when unthreaded) -> ThreadData.
_THREADS_KEY = '__threads__'


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

    def __init__(self, *, user_id: int, user_data: UserData,
                 thread_data: ThreadData) -> None:
        self.user_id = user_id
        # Per-agent state. For a subagent this is its own UserData (settings
        # snapshot) plus its own ThreadData (its conversation), never the
        # parent's.
        self.user_data = user_data
        self.thread_data = thread_data

    @property
    def conversation(self) -> "Conversation":
        assert self.thread_data.conversation is not None
        return self.thread_data.conversation

    @conversation.setter
    def conversation(self, value: "Conversation") -> None:
        self.thread_data.conversation = value

    @property
    def settings(self) -> "UserSettings":
        # Per-thread copy once the thread has been initialised; the
        # user's template otherwise (keeps stub-built contexts working).
        if self.thread_data.settings is not None:
            return self.thread_data.settings
        assert self.user_data.settings is not None
        return self.user_data.settings

    @settings.setter
    def settings(self, value: "UserSettings") -> None:
        self.thread_data.settings = value


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
                 user_data: UserData,
                 thread_data: ThreadData,
                 message_thread_id: int | None,
                 command_text: str,
                 command_name: str) -> None:
        super().__init__(user_id=user.id, user_data=user_data,
                         thread_data=thread_data)
        self.update = update
        self.context = context
        self.bot = bot
        self.chat = chat
        self.user = user
        self.message = message
        self.message_thread_id = message_thread_id
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
        return self.thread_data.generation_task

    @generation_task.setter
    def generation_task(self, value: asyncio.Task | None) -> None:
        self.thread_data.generation_task = value

    @property
    def generation_worker(self) -> "asyncio.Task | None":
        return self.thread_data.generation_worker

    @generation_worker.setter
    def generation_worker(self, value: "asyncio.Task | None") -> None:
        self.thread_data.generation_worker = value

    @property
    def settings_loaded(self) -> bool:
        return self.user_data.settings is not None

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
        it gets its own ``UserData`` (empty: subagents need no template)
        plus its own ``ThreadData`` holding the subagent's conversation
        and a deepcopy of the parent's effective settings. The
        parent thread's coordination state - inbox, wakeup flags,
        conversation epoch, generation queue, subagents - and its sandbox
        read marks are not reachable from here (fresh empty set), which is
        what makes an accidental mutation impossible rather than merely
        discouraged.
        """
        effective = self.thread_data.settings
        if effective is None:
            effective = self.user_data.settings
        return AgentContext(
            user_id=self.user_id,
            user_data=UserData(),
            thread_data=ThreadData(
                conversation=conversation,
                settings=(copy.deepcopy(effective)
                          if effective is not None else None),
            ),
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
    user_data_dict = context.user_data
    assert user_data_dict is not None
    user_data = user_data_dict.get(_USER_DATA_KEY)
    if not isinstance(user_data, UserData):
        user_data = UserData()
        user_data_dict[_USER_DATA_KEY] = user_data
    threads = user_data_dict.get(_THREADS_KEY)
    if not isinstance(threads, dict):
        threads = {}
        user_data_dict[_THREADS_KEY] = threads
    # Unthreaded chats have no thread id, so they share the None thread.
    effective = update.effective_message
    thread_id = getattr(effective, 'message_thread_id', None)
    thread_data = threads.get(thread_id)
    if not isinstance(thread_data, ThreadData):
        thread_data = ThreadData()
        threads[thread_id] = thread_data
    text = message.text or message.caption or ''
    return ChatUpdateContext(
        update=update,
        context=context,
        bot=context.bot,
        chat=chat,
        user=user,
        message=message,
        user_data=user_data,
        thread_data=thread_data,
        message_thread_id=thread_id,
        command_text=_parse_command_text(text),
        command_name=_parse_command_name(text),
    )


def callback_adapter(func):
    import functools

    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        ctx = create_chat_update_context(update, context)
        return await func(ctx)

    return wrapper
