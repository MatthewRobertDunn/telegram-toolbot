"""Unit-test package.

Importing this package runs ``tests/_bootstrap.setup()`` once, which
isolates the database (STORE_DB_PATH -> per-run temp file), puts the
repo root, ``src/`` and ``tests/`` on sys.path, and installs the guard
that forbids opening the repo's real databases. See ``tests/run.py``
for the runner ``pake unittest`` uses.
"""
from . import _bootstrap

_bootstrap.setup()
