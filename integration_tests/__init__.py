import asyncio
import logging
import os
import sys

_HERE = os.path.dirname(__file__)

# Make `mocks` and other test modules importable.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Load .env BEFORE importing any src module that transitively imports config
# (which reads os.environ at import time). python-dotenv only loads from CWD
# upward by default, so we point it at the file explicitly.
from dotenv import load_dotenv
load_dotenv(os.path.join(_HERE, '.env'))

# Silence all logging for the duration of integration tests so the test
# runner output isn't drowned in tool-call chatter.
logging.disable(logging.CRITICAL)

# Add src to sys.path AFTER .env is loaded so config picks up the env vars.
_SRC = os.path.join(_HERE, '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Mirror chatbot.post_init: wipe and recreate the test database so each test
# run starts from a clean store. STORE_DB_PATH is set by .env to test_store.db.
from context_store import create_store, create_settings_store


def _reset_test_db():
    db_path = os.environ.get("STORE_DB_PATH", "store.db")
    try:
        os.remove(db_path)
    except FileNotFoundError:
        pass

    async def _create():
        async with create_store() as store:
            await store.create_schema()
        async with create_settings_store() as store:
            await store.create_schema()

    asyncio.run(_create())


_reset_test_db()