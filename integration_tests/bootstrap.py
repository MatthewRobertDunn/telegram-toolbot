"""One-time test bootstrap: path setup, env loading, logging, DB reset.

Import this first in every test file that relies on src modules:
    import bootstrap  # noqa: F401
"""
import asyncio
import logging
import os
import sys

_HERE = os.path.dirname(__file__)

# ------------------------------------------------------------------ paths
_SRC = os.path.join(_HERE, '..', 'src')
for _p in (_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ------------------------------------------------------------------ .env
from dotenv import load_dotenv
load_dotenv(os.path.join(_HERE, '.env'))

# ------------------------------------------------------------------ logging
logging.disable(logging.CRITICAL)

# ------------------------------------------------------------------ database
from context_store import create_store, create_settings_store

_db_reset_done = False


def _reset_test_db():
    global _db_reset_done
    if _db_reset_done:
        return
    _db_reset_done = True

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
