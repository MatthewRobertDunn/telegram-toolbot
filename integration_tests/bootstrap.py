"""Shared setup for integration tests.

Importing this module establishes the isolated test environment (same
harness as the unit tier, see ``tests/_bootstrap.py``). Import does
have one filesystem effect: it creates the isolated per-run temporary
database (and its schema). What it never touches is the repo's real
``store.db`` — the old import-time delete/recreate of
``integration_tests/test_store.db`` is gone, and a guard forbids
opening any real database.

* ``integration_tests/.env`` is loaded first, before the shared harness
  imports any application module: ``config.py`` snapshots the
  environment once at import time, so keys that only exist in that file
  (image-generation API keys, for example) must already be in
  ``os.environ`` by then.
* ``STORE_DB_PATH`` points at a fresh per-run temporary database
  outside the checkout, so integration tests never touch the repo's
  ``store.db``. One directory per process, so parallel runs cannot
  collide. The harness sets it unconditionally, so a ``STORE_DB_PATH``
  in ``integration_tests/.env`` cannot redirect the suite either.
* A guard forbids opening the repo's real databases.

Test modules import this module for its helpers only::

    import bootstrap  # noqa: F401  (env + guard)
    ...
    uid = bootstrap.next_uid()
    @bootstrap.requires_docker
    class TestSandbox(...):
"""
import functools
import os
import subprocess
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# ------------------------------------------------------------------ .env
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(_HERE, '.env'))

# ------------------------------------------------------------------ harness
# Imported after the .env load above and before any src module is
# imported: config.py reads the environment (including STORE_DB_PATH)
# when it is first imported, and the harness pulls in the store modules
# that import config.
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests import _bootstrap  # noqa: E402

_bootstrap.setup()

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
