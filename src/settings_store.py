import json

from context_store import create_settings_store
from user_settings import UserSettings


SETTINGS_KEY = "user_settings"


async def load_settings(user_id: int) -> UserSettings:
    async with create_settings_store() as store:
        raw = await store.get(user_id, SETTINGS_KEY)
    if raw is None:
        return UserSettings.default()
    return UserSettings.from_json(raw)


async def save_settings(user_id: int, settings: UserSettings) -> None:
    async with create_settings_store() as store:
        await store.set(user_id, SETTINGS_KEY, settings.to_json())