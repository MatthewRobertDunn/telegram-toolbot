"""Isolated-database harness for the test tiers.

Every supported entry point routes through :func:`setup` before any
application module is imported:

* ``python tests/run.py`` (what ``pake unittest`` / ``pake
  integration_test`` invoke) — explicit setup, then unittest discovery.
* ``python -m unittest discover -s tests -t .`` (from the repo root) —
  ``-t .`` makes unittest import test modules as the ``tests`` package,
  so ``tests/__init__.py`` runs and calls :func:`setup`. Without
  ``-t .`` modules load top-level, the package never imports, and the
  isolation never happens.
* ``python -m unittest discover -s integration_tests -t .`` — each test
  module imports ``bootstrap``, which calls :func:`setup`.

What :func:`setup` does, in order:

1. Exports ``STORE_DB_PATH`` pointing at a fresh per-run temporary
   directory. ``config.py`` reads that env var when it is first
   imported, which only happens after this line, so the whole process —
   application code included — uses the isolated database. The repo's
   ``store.db`` is never read, written, or created.
2. Puts the repo root, ``src/`` and ``tests/`` on ``sys.path``
   (idempotently, so test files that still carry their own
   ``sys.path.insert`` lines keep working).
3. Installs a hard guard around ``aiosqlite.connect`` and
   ``sqlite3.connect``: opening a path that resolves to the repo's
   ``store.db`` (or the integration tier's old ``test_store.db``)
   raises immediately, so a future test that hardcodes the real
   database fails loudly instead of silently reading dev data.
4. Creates the store schemas once (memory, settings, journal — mirroring
   ``chatbot.post_init``) so ``init_conversation`` works, and registers
   cleanup that deletes the temporary directory at exit.

Why a per-run temporary *file* rather than an in-memory database: the
stores open a new ``aiosqlite`` connection per operation and
``persistence.sqlite_store.AsyncSQLiteStore`` calls ``aiosqlite.connect(path)``
without ``uri=True``, so a shared-cache memory URI cannot be passed
through ``config.sql_db_path`` — each connection would get its own
empty database. A per-run file gives every connection the same content
while staying isolated and parallel-safe (one directory per process).
"""
import asyncio
import atexit
import logging
import os
import shutil
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS = os.path.join(_ROOT, 'tests')
_SRC = os.path.join(_ROOT, 'src')

# Per-run state.
_db_dir: str | None = None
_schema_ready = False
_creating_schema = False
_setup_done = False
_cleanup_registered = False


def _uri_to_path(candidate: str) -> str:
    """Strip a sqlite ``file:`` URI down to its path part.

    SQLite accepts several spellings — ``file:/abs``, ``file:///abs``,
    ``file://localhost/abs``, all optionally followed by ``?query`` — so
    a real database referenced as a URI must not slip past the guard by
    spelling. Only the path form is handled: percent-encoding of
    forbidden-path characters is not honoured (none of the real paths
    contain characters that would need it).
    """
    if not candidate.startswith('file:'):
        return candidate
    rest = candidate[len('file:'):]
    if rest.startswith('//'):
        rest = rest[2:]
        # file://localhost/... is the same file as /...
        if rest[:9].casefold() == 'localhost':
            rest = rest[9:]
    if (len(rest) >= 3 and rest[0] == '/'
            and rest[1].isascii() and rest[1].isalpha() and rest[2] == ':'):
        # file:///C:/... on Windows: drop the leading slash so abspath
        # treats the drive-qualified path as absolute instead of
        # drive-relative to the CWD.
        rest = rest[1:]
    # The query (e.g. ?mode=ro) is not part of the path.
    return rest.split('?', 1)[0]


def _realpath(path: object) -> str:
    """Absolute, symlink-resolved, case-normalised path.

    Accepts plain paths and ``file:`` URI spellings (see
    :func:`_uri_to_path`).
    """
    return os.path.normcase(os.path.realpath(os.path.abspath(
        _uri_to_path(str(path)))))


def repo_path(*parts: str) -> str:
    """An absolute path inside this checkout."""
    return os.path.join(_ROOT, *parts)


def forbidden_db_paths() -> set[str]:
    """The real databases tests must never open."""
    return {
        _realpath(repo_path('store.db')),
        _realpath(repo_path('integration_tests', 'test_store.db')),
    }


def is_forbidden_path(path: object) -> bool:
    """True if opening ``path`` would touch a real development database."""
    return _realpath(path) in forbidden_db_paths()


def db_dir() -> str:
    """The per-run directory holding the isolated test database."""
    if _db_dir is None:
        raise RuntimeError(
            'Test DB bootstrap has not run; import tests/_bootstrap and '
            'call setup() before importing application modules.')
    return _db_dir


def db_path() -> str:
    """The isolated database file backing ``config.sql_db_path``."""
    return os.environ['STORE_DB_PATH']


def _add_path(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


def _export_env() -> None:
    # Forced, not setdefault: a stale STORE_DB_PATH from the surrounding
    # environment must not be able to redirect tests at a real database.
    os.environ['STORE_DB_PATH'] = os.path.join(_db_dir or '', 'store.db')


def _assert_not_repo_db(database: object) -> None:
    if is_forbidden_path(database):
        raise RuntimeError(
            'Test attempted to open the real development database '
            f'({database!r}); use the isolated test DB (config.sql_db_path). '
            'See tests/_bootstrap.py.')


def _is_test_db(database: object) -> bool:
    return _db_dir is not None and _realpath(database) == _realpath(db_path())


def _install_connect_guard() -> None:
    """Wrap connect() so the real databases cannot be opened."""
    import sqlite3

    import aiosqlite

    if getattr(aiosqlite.connect, '_toolbot_test_guard', False):
        return
    real_async_connect = aiosqlite.connect
    real_sync_connect = sqlite3.connect

    async def guarded_async_connect(database, *args, **kwargs):
        _assert_not_repo_db(database)
        # Safety net for entry points that skip the runner's explicit
        # ensure_schema(): materialise the schema on first use. Not
        # re-entrant: while it runs, inner connects pass straight
        # through (the store factories below route through this guard).
        if _is_test_db(database) and not _schema_ready and not _creating_schema:
            await ensure_schema_async()
        return await real_async_connect(database, *args, **kwargs)

    guarded_async_connect._toolbot_test_guard = True  # type: ignore[attr-defined]
    aiosqlite.connect = guarded_async_connect  # type: ignore[assignment]

    def guarded_sync_connect(database, *args, **kwargs):
        _assert_not_repo_db(database)
        return real_sync_connect(database, *args, **kwargs)

    guarded_sync_connect._toolbot_test_guard = True  # type: ignore[attr-defined]
    sqlite3.connect = guarded_sync_connect  # type: ignore[assignment]


async def ensure_schema_async() -> None:
    """Create the store schemas in the isolated DB (idempotent)."""
    global _schema_ready, _creating_schema
    if _schema_ready or _creating_schema:
        return
    _creating_schema = True
    try:
        from persistence.memory_store import create_memory_store
        from persistence.settings_store import create_settings_store
        from persistence.journal_store import create_journal_store

        # The factories resolve config.sql_db_path, which the env export
        # above has pointed at the isolated database for this process.
        async with create_memory_store() as memory:
            await memory.create_schema()
        async with create_settings_store() as settings_store:
            await settings_store.create_schema()
        async with create_journal_store() as journal_store:
            await journal_store.create_schema()
    finally:
        _creating_schema = False
    _schema_ready = True


def ensure_schema() -> None:
    """Create the schemas from sync context (no running event loop)."""
    if _schema_ready:
        return
    asyncio.run(ensure_schema_async())


def _cleanup() -> None:
    if _db_dir is None:
        return
    # Best effort: on Windows an aiosqlite thread still closing keeps
    # the file locked; the OS temp dir cleans up leftovers regardless.
    shutil.rmtree(_db_dir, ignore_errors=True)


def setup() -> None:
    """Prepare the isolated test environment. Idempotent.

    Call before importing any application module (config reads
    STORE_DB_PATH at import time).
    """
    global _db_dir, _setup_done, _cleanup_registered
    if _setup_done:
        return
    _setup_done = True

    _db_dir = tempfile.mkdtemp(prefix='toolbot-tests-')
    _export_env()
    _add_path(_ROOT)
    _add_path(_TESTS)
    _add_path(_SRC)
    if not _cleanup_registered:
        _cleanup_registered = True
        atexit.register(_cleanup)

    # Application modules are imported from here on.
    _install_connect_guard()
    ensure_schema()

    # aiosqlite logs every open/close at DEBUG; pin that one logger to
    # WARNING so suite output stays readable. The root logger keeps its
    # default configuration, so warnings from the code under test are
    # still visible and assertLogs keeps working.
    logging.getLogger('aiosqlite').setLevel(logging.WARNING)
