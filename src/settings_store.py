import copy
import json
import logging

from sqlite_store import AsyncSQLiteStore
from conversation import IMAGE_PLACEHOLDER, MAX_IMAGES, Conversation
from user_settings import UserSettings

logger = logging.getLogger(__name__)


SETTINGS_KEY = "user_settings"
# Conversation rows live in the settings table too: one CONVERSATION_KEY
# row per (user, thread), told apart by the thread_id column.
CONVERSATION_KEY = "conversation"

SETTINGS_SCHEMA = '''
    CREATE TABLE IF NOT EXISTS settings (
        user_id INTEGER NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        thread_id INTEGER DEFAULT NULL,
        last_updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (user_id, key, thread_id)
    )
'''


class SettingsStore(AsyncSQLiteStore):
    """The settings table: every row carries its thread id.

    Settings rows share SETTINGS_KEY and conversation rows share
    CONVERSATION_KEY; both are told apart by thread_id. ``IS`` matches
    NULL thread ids too.
    """

    def __init__(self, db_path: str | None = None):
        super().__init__(db_path, "settings", SETTINGS_SCHEMA)

    async def get(self, user_id: int, key: str,
                  thread_id: int | None) -> str | None:
        db = await self._ready()
        cursor = await db.execute(
            'SELECT value FROM settings WHERE user_id = ? '
            'AND key = ? AND thread_id IS ?',
            (user_id, key, thread_id))
        row = await cursor.fetchone()
        return row[0] if row else None

    async def set(self, user_id: int, key: str, value: str,
                  thread_id: int | None) -> None:
        """UPDATE-then-INSERT (not INSERT OR REPLACE): rows are keyed by
        (user_id, key, thread_id), and SQLite never treats two NULL
        thread ids as conflicting, so REPLACE would duplicate the
        unthreaded/template row instead of replacing it.
        """
        db = await self._ready()
        cursor = await db.execute(
            'UPDATE settings SET value = ?, thread_id = ?, '
            'last_updated = CURRENT_TIMESTAMP '
            'WHERE user_id = ? AND key = ? AND thread_id IS ?',
            (value, thread_id, user_id, key, thread_id))
        if cursor.rowcount == 0:
            await db.execute(
                'INSERT INTO settings '
                '(user_id, key, value, thread_id) VALUES (?, ?, ?, ?)',
                (user_id, key, value, thread_id))
        await db.commit()

    # get_all_keys is inherited from AsyncSQLiteStore.


def _prune_inline_images(messages: list) -> list:
    """Messages with all but the newest ``MAX_IMAGES`` inline image
    parts replaced by placeholder text.

    Applied before a conversation is stored and again on load: the row
    is rewritten whole on every queued turn, so inline images would
    otherwise be re-serialized forever, and a payload written with more
    images than the cap (an older version, or an oversized stored blob)
    must not resurrect them. Returns a deep copy; the caller's
    conversation is untouched.
    """
    pruned = copy.deepcopy(list(messages))
    image_parts = [
        part
        for message in pruned
        if isinstance(message, dict)
        and isinstance(message.get('content'), list)
        for part in message['content']
        if isinstance(part, dict) and part.get('type') == 'image_url'
    ]
    for part in image_parts[:-MAX_IMAGES]:
        part.clear()
        part['type'] = 'text'
        part['text'] = IMAGE_PLACEHOLDER
    return pruned


def create_settings_store() -> SettingsStore:
    # The db path is resolved when the store connects (config.sql_db_path),
    # so patching the config redirects stores created here.
    return SettingsStore()


async def load_settings(user_id: int,
                        thread_id: int | None = None) -> UserSettings | None:
    """Load one thread's settings, or None when absent/corrupt.

    Thread ``None`` is the template every new thread clones: the legacy
    global row keeps working untouched.
    """
    async with create_settings_store() as store:
        raw = await store.get(user_id, SETTINGS_KEY, thread_id)
    if raw is None:
        return None
    try:
        return UserSettings.from_json(raw)
    except (ValueError, AttributeError, TypeError) as e:
        logger.warning('Could not load settings for user %d: %s',
                       user_id, e)
        return None


async def save_settings(user_id: int, thread_id: int | None,
                        settings: UserSettings) -> None:
    async with create_settings_store() as store:
        await store.set(user_id, SETTINGS_KEY,
                        settings.to_json(), thread_id)


async def save_conversation(user_id: int, thread_id: int | None,
                            conversation: Conversation) -> None:
    """Persist the conversation under the thread's row.

    The single save point is the generation worker (which covers every
    queued unit of work); the few direct mutations (/start reset, ambient
    group chatter) save explicitly at their call site. Inline images
    beyond the newest ``MAX_IMAGES`` are pruned to placeholders: the row
    is rewritten whole each time, and the payloads are the bulk of it."""
    payload = json.dumps({'id': conversation.id,
                          'messages': _prune_inline_images(
                              list(conversation.messages))})
    async with create_settings_store() as store:
        await store.set(user_id, CONVERSATION_KEY,
                        payload, thread_id)


async def load_conversation(user_id: int,
                            thread_id: int | None) -> Conversation | None:
    """Load the persisted conversation for the thread, or None when
    absent/corrupt. Stored payloads are pruned on the way out too, so a
    blob carrying more images than the cap cannot resurrect them."""
    async with create_settings_store() as store:
        raw = await store.get(user_id, CONVERSATION_KEY,
                              thread_id)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        return Conversation(
            messages=_prune_inline_images(data.get('messages') or []),
            id=data.get('id'))
    except (ValueError, AttributeError, TypeError) as e:
        logger.warning('Could not load conversation for user %d: %s',
                       user_id, e)
        return None
