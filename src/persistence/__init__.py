"""SQLite-backed persistence: one module per table, plus the shared
connection/schema base class and the startup housekeeping task.

Each ``*_store`` module owns exactly one table and extends
``sqlite_store.AsyncSQLiteStore``, so no caller deals with columns
its table does not have. ``cleanup`` holds the startup purge task.
"""
