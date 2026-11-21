"""Test runner: sets up the isolated test environment, then discovers.

Usage:
    python tests/run.py                     # unit tier (tests/)
    python tests/run.py integration_tests   # integration tier

``pake unittest`` / ``pake integration_test`` call this runner. It
exists so the isolated database is in place before any application
module is imported:

1. ``tests._bootstrap.setup()`` exports ``STORE_DB_PATH`` at a fresh
   per-run temp directory (config.py reads it at import), puts the repo
   root, ``src/`` and ``tests/`` on sys.path, installs the guard that
   forbids opening the repo's real databases, and creates the schema.
2. unittest discovery runs for the requested start directory.

The runner also self-checks the isolation (see ``--check``): if
``config.sql_db_path`` ever resolves to the repo database, the run
fails loudly instead of silently testing against dev data.
"""
import os
import sys
import unittest

# Make the repo root importable so `tests` resolves as a package no
# matter the caller's working directory.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests import _bootstrap  # noqa: E402

_bootstrap.setup()


def check_isolation() -> None:
    """Fail the run immediately if the test DB is not isolated."""
    import config

    if _bootstrap.is_forbidden_path(config.sql_db_path):
        raise RuntimeError(
            'Test isolation broken: config.sql_db_path points at a real '
            f'development database ({config.sql_db_path!r}). The runner '
            'must export STORE_DB_PATH before config is imported.')
    if os.path.abspath(config.sql_db_path) != os.path.abspath(
            _bootstrap.db_path()):
        raise RuntimeError(
            'Test isolation broken: config.sql_db_path '
            f'({config.sql_db_path!r}) is not the isolated test database '
            f'({_bootstrap.db_path()!r}).')


def main(argv: list[str]) -> int:
    start_dir = argv[0] if argv else 'tests'
    pattern = argv[1] if len(argv) > 1 else 'test_*.py'
    verbosity = 2 if '-v' in argv or '--verbose' in argv else 1

    check_isolation()

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir, pattern=pattern, top_level_dir=_ROOT)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
