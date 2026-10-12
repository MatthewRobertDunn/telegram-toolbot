"""One-time test bootstrap: path setup, env loading, logging, DB reset.

Import this first in every test file that relies on src modules:
    import bootstrap  # noqa: F401
"""
import asyncio
import functools
import logging
import os
import subprocess
import sys
import unittest

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

# ------------------------------------------------------------------ uid
_next_uid = 3000


def next_uid() -> int:
    """Return a unique user_id for each call, shared across all test files."""
    global _next_uid
    _next_uid += 1
    return _next_uid

# ------------------------------------------------------------------ docker


@functools.lru_cache(maxsize=1)
def _docker_cli(args: tuple[str, ...]) -> bool:
    try:
        result = subprocess.run(
            ["docker", *args], capture_output=True, timeout=15)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def docker_available() -> bool:
    return _docker_cli(("version", "--format", "{{.Server.Version}}"))


def image_available() -> bool:
    import config
    return _docker_cli(("image", "inspect", config.SANDBOX_IMAGE))


def requires_docker(test_cls):
    """Class decorator: skip tests when the Docker daemon is unavailable."""
    return unittest.skipUnless(
        docker_available(), "Docker daemon not available")(test_cls)


def requires_sandbox_image(test_cls):
    """Class decorator: skip tests when Docker or the sandbox image is missing."""
    import config
    return unittest.skipUnless(
        docker_available(), "Docker daemon not available")(
        unittest.skipUnless(
            image_available(),
            f"sandbox image {config.SANDBOX_IMAGE} not built locally")(
            test_cls))

