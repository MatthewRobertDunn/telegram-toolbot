"""Test runner: sets up the isolated test environment, then discovers.

Usage (from any working directory):

    python tests/run.py                     # unit tier (tests/)
    python tests/run.py integration_tests   # integration tier

``pake unittest`` / ``pake integration_test`` call this runner; it runs
``tests._bootstrap.setup()`` first so the isolated database exists before
any application module is imported.

``check_isolation`` fails the run if ``config.sql_db_path`` resolves to a
repo database instead of the per-run one.

Plain unittest discovery works too, but the top-level directory must be
the repo root so the ``tests`` package (and its isolation) is imported:
``python -m unittest discover -s tests -t .``. Without ``-t .`` the
modules load top-level, ``tests/__init__.py`` never runs, and the real
database gets touched.
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
    args = [a for a in argv if not a.startswith('-')]
    start_dir = args[0] if args else 'tests'
    pattern = args[1] if len(args) > 1 else 'test_*.py'
    verbosity = 2 if {'-v', '--verbose'} & set(argv) else 1

    check_isolation()

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir, pattern=pattern, top_level_dir=_ROOT)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
