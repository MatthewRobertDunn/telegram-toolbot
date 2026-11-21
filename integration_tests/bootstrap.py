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
from memory_store import create_memory_store
from settings_store import create_settings_store

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
        async with create_memory_store() as memory:
            await memory.create_schema()
        async with create_settings_store() as settings_store:
            await settings_store.create_schema()

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


def nvidia_key_available() -> bool:
    """True when an image-generation API key is configured.

    Mirrors src/config.py: NVIDIA_API_KEY, falling back to OPENAI_API_KEY.
    By the time this runs, bootstrap has already loaded
    integration_tests/.env into the process environment (load_dotenv above
    precedes every config import), so a key added there is picked up
    with no further edits.
    """
    return bool(os.environ.get("NVIDIA_API_KEY")
                or os.environ.get("OPENAI_API_KEY"))


def requires_nvidia_key(test_cls):
    """Class decorator: skip tests that call the NVIDIA image endpoint
    when no API key is configured."""
    return unittest.skipUnless(
        nvidia_key_available(),
        "NVIDIA_API_KEY (or OPENAI_API_KEY) not configured")(test_cls)

