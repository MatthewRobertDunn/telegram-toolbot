import json
import logging

from context_store import create_settings_store
from conversation import Conversation
from user_settings import UserSettings

logger = logging.getLogger(__name__)


SETTINGS_KEY = "user_settings"
# Conversation rows live in the settings table too, partitioned per user
# by the user_id column. The unthreaded conversation keeps the plain
# CONVERSATION_KEY (so pre-thread databases keep working untouched);
# each thread id gets its own suffixed key row carrying that thread_id.
CONVERSATION_KEY = "conversation"


def _conversation_key(thread_id: int | None) -> str:
    if thread_id is None:
        return CONVERSATION_KEY
    return f"{CONVERSATION_KEY}:thread:{thread_id}"


async def load_settings(user_id: int) -> UserSettings:
    async with create_settings_store() as store:
        raw = await store.get(user_id, SETTINGS_KEY)
    if raw is None:
        return UserSettings.default()
    return UserSettings.from_json(raw)


async def save_settings(user_id: int, settings: UserSettings) -> None:
    async with create_settings_store() as store:
        await store.set(user_id, SETTINGS_KEY, settings.to_json())


async def save_conversation(user_id: int, thread_id: int | None,
                            conversation: Conversation) -> None:
    """Persist the conversation under the thread's key.

    The single save point is the generation worker (which covers every
    queued unit of work); the few direct mutations (/start reset, ambient
    group chatter) save explicitly at their call site."""
    payload = json.dumps({'id': conversation.id,
                          'messages': list(conversation.messages)})
    async with create_settings_store() as store:
        await store.set_with_thread(user_id, _conversation_key(thread_id),
                                    payload, thread_id)


async def load_conversation(user_id: int,
                            thread_id: int | None) -> Conversation | None:
    """Load the persisted conversation for the thread, or None when
    absent/corrupt."""
    async with create_settings_store() as store:
        raw = await store.get(user_id, _conversation_key(thread_id))
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        return Conversation(messages=data.get('messages') or [],
                            id=data.get('id'))
    except (ValueError, AttributeError, TypeError) as e:
        logger.warning('Could not load conversation for user %d: %s',
                       user_id, e)
        return None