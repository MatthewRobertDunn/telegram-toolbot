"""Shared setup for integration tests.

Importing this module establishes the isolated test environment (same
harness as the unit tier, see ``tests/_bootstrap.py``):

* ``STORE_DB_PATH`` points at a fresh per-run temporary database
  outside the checkout, so integration tests never touch the repo's
  ``store.db``. The database file and schema are created lazily on
  first connect, so importing a test module has no filesystem side
  effects and parallel runs (one process each) cannot collide.
* A guard forbids opening the repo's real databases.
* The old import-time delete/recreate of
  ``integration_tests/test_store.db`` is gone.

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

# Import the shared harness. It must run before any src module is
# imported (config.py reads STORE_DB_PATH on first import), which holds
# because test modules import this module before anything from src.
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests import _bootstrap  # noqa: E402

_bootstrap.setup()

# ------------------------------------------------------------------ .env
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(_HERE, '.env'))

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
