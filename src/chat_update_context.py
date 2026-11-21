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
    One deliberate exception to that isolation: an unthreaded chat's
    ``settings`` aliases the user's template object (see
    ``ThreadData.settings_is_template``), so a change made without a
    thread also updates the template every new thread clones.

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
    and tracked task), the subagent inbox, the epoch guarding against
    stale wakeups, and the subagent records that report back to this
    thread.
    """

    conversation: Conversation | None = None
    settings: "UserSettings | None" = None
    # Names the settings aliasing: True when ``settings`` is not the
    # thread's own object but the user's template shared in place - the
    # unthreaded/'all' category, where a write through the thread must
    # also be a write to the template (``middleware.init_conversation``
    # sets it; the AgentContext.settings setter keeps it true after a
    # command replaces the object). Threaded chats own a deep copy instead
    # and always run with False, so their changes stay local.
    settings_is_template: bool = False
    generation_task: asyncio.Task | None = None
    generation_worker: asyncio.Task | None = None
    generation_queue: asyncio.Queue | None = None
    main_inbox: "SubagentInbox | None" = None
    conversation_epoch: int = 0
    subagents: dict[str, "SubagentRecord"] = field(default_factory=dict)


@dataclass
class UserData:
    """Per-user state, shared by every chat/thread of that user.

    Only genuinely user-scoped things live here: the settings template
    that new threads clone on first touch (only changes made in the
    unthreaded/'all' category update it), and the sandbox read marks (the sandbox
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
        # The aliasing flag is a fact about object identity, so it is
        # maintained here rather than only at init: a command that swaps
        # the object in (e.g. /settings_reset) changes whether the thread
        # shares the template, even mid-command (the unthreaded branch of
        # the settings decorator then re-shares the new object).
        self.thread_data.settings_is_template = (
            value is not None and value is self.user_data.settings)


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
        parent thread's coordination state - inbox, conversation epoch,
        generation queue, subagents - and its sandbox
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


def _ensure_user_data(user_data_dict: dict) -> tuple[UserData, dict]:
    """Fetch (or lazily create) this user's state inside PTB's per-user
    dict: the ``UserData`` itself plus the thread-keyed map of
    ``ThreadData``. PTB hands us a plain dict, so missing entries are
    normal on a user's first update.
    """
    user_data = user_data_dict.get(_USER_DATA_KEY)
    if not isinstance(user_data, UserData):
        user_data = UserData()
        user_data_dict[_USER_DATA_KEY] = user_data
    threads = user_data_dict.get(_THREADS_KEY)
    if not isinstance(threads, dict):
        threads = {}
        user_data_dict[_THREADS_KEY] = threads
    return user_data, threads


def _thread_data_for(threads: dict, thread_id: int | None) -> ThreadData:
    """Fetch (or lazily create) the state of one chat/thread. Unthreaded
    chats have no thread id, so they share the None thread."""
    thread_data = threads.get(thread_id)
    if not isinstance(thread_data, ThreadData):
        thread_data = ThreadData()
        threads[thread_id] = thread_data
    return thread_data


def _parse_command_text(text: str) -> str:
    """Everything after the command token (the arguments), stripped.

    Private because it has exactly one consumer: the per-update parse in
    ``create_chat_update_context`` below.
    """
    _, _, args = text.partition(' ')
    return args.strip()


def _parse_command_name(text: str) -> str:
    """The bare, lowercased command name of a message text ('' when it is
    not one).

    Lowercased because that is how PTB dispatches: registered names are
    lowercased and the incoming token is matched lowercased, so ``/MODEL``
    runs the ``model`` handler and would otherwise miss the table. The
    ``@mention`` is stripped too: ``/start@MyBot arg`` yields ``start`` -
    the mention addresses the bot, not the command, so carrying it would
    render usage lines like ``/start@MyBot <text>``.

    Private because it has exactly one consumer: the per-update parse in
    ``create_chat_update_context`` below.
    """
    if not text.startswith("/"):
        return ""
    token = text[1:].split(maxsplit=1)[0]
    return token.partition('@')[0].lower()


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
    user_data, threads = _ensure_user_data(user_data_dict)
    effective = update.effective_message
    thread_id = getattr(effective, 'message_thread_id', None)
    thread_data = _thread_data_for(threads, thread_id)
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
