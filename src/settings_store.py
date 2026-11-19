import json
import logging

from config import sql_db_path
from sqlite_store import AsyncSQLiteStore
from conversation import Conversation
from user_settings import UserSettings

logger = logging.getLogger(__name__)


SETTINGS_KEY = "user_settings"
# Conversation rows live in the settings table too, partitioned per user
# by the user_id column. The unthreaded conversation keeps the plain
# CONVERSATION_KEY (so pre-thread databases keep working untouched);
# each thread id gets its own suffixed key row carrying that thread_id.
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

    Settings rows share SETTINGS_KEY and are told apart by thread_id;
    conversation rows keep suffixed keys. ``IS`` matches NULL thread
    ids too.
    """

    def __init__(self, db_path: str):
        super().__init__(db_path, "settings", SETTINGS_SCHEMA)

    async def get(self, user_id: int, key: str,
                  thread_id: int | None) -> str | None:
        db = await self._get_conn()
        cursor = await db.execute(
            f'SELECT value FROM {self.table_name} WHERE user_id = ? '
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
        db = await self._get_conn()
        cursor = await db.execute(
            f'UPDATE {self.table_name} SET value = ?, thread_id = ?, '
            'last_updated = CURRENT_TIMESTAMP '
            'WHERE user_id = ? AND key = ? AND thread_id IS ?',
            (value, thread_id, user_id, key, thread_id))
        if cursor.rowcount == 0:
            await db.execute(
                f'INSERT INTO {self.table_name} '
                '(user_id, key, value, thread_id) VALUES (?, ?, ?, ?)',
                (user_id, key, value, thread_id))
        await db.commit()

    async def get_all_keys(self, user_id: int) -> list[str]:
        db = await self._get_conn()
        cursor = await db.execute(
            f'SELECT key FROM {self.table_name} WHERE user_id = ?', (user_id,))
        rows = await cursor.fetchall()
        return [r[0] for r in rows]


def create_settings_store() -> SettingsStore:
    return SettingsStore(sql_db_path)


def _conversation_key(thread_id: int | None) -> str:
    if thread_id is None:
        return CONVERSATION_KEY
    return f"{CONVERSATION_KEY}:thread:{thread_id}"


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
    """Persist the conversation under the thread's key.

    The single save point is the generation worker (which covers every
    queued unit of work); the few direct mutations (/start reset, ambient
    group chatter) save explicitly at their call site."""
    payload = json.dumps({'id': conversation.id,
                          'messages': list(conversation.messages)})
    async with create_settings_store() as store:
        await store.set(user_id, _conversation_key(thread_id),
                        payload, thread_id)


async def load_conversation(user_id: int,
                            thread_id: int | None) -> Conversation | None:
    """Load the persisted conversation for the thread, or None when
    absent/corrupt."""
    async with create_settings_store() as store:
        raw = await store.get(user_id, _conversation_key(thread_id),
                              thread_id)
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
