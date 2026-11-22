"""Startup database housekeeping.

``purge_stale_thread_rows`` runs once at service startup (from
``chatbot.post_init``, after the schemas are created) and deletes the
threaded settings/conversation rows nothing has touched for a week.
It returns the number of rows deleted, so the caller can log it per
process start.
"""

import logging

from .settings_store import create_settings_store

logger = logging.getLogger(__name__)

# Rows in a (user, thread) group whose newest ``last_updated`` is older
# than this are deleted wholesale on the next startup.
STALE_THREAD_AGE_DAYS = 7


async def purge_stale_thread_rows(
        max_age_days: int = STALE_THREAD_AGE_DAYS) -> int:
    """Purge thread-scoped rows untouched for ``max_age_days``.

    See ``SettingsStore.purge_stale_threads`` for what counts as stale;
    unthreaded rows (the settings template and the global conversation)
    are never touched. Returns the number of rows deleted.
    """
    async with create_settings_store() as store:
        deleted = await store.purge_stale_threads(max_age_days)
    if deleted:
        logger.info('Startup cleanup: purged %d stale thread row(s) '
                    'older than %d day(s)', deleted, max_age_days)
    return deleted
