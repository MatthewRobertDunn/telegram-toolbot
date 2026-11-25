"""Tracks files recently uploaded to the chat.

When a user uploads a file, Telegram already shows it in the chat
window, so read_image re-displaying the same file is redundant. The
handler records each stored upload here and read_image checks it before
sending a photo to the chat. Entries expire after UPLOAD_TTL seconds.
"""

import logging
import posixpath
import time

from cachetools import TTLCache

logger = logging.getLogger(__name__)

UPLOAD_TTL = 60 * 60
UPLOAD_CACHE_MAXSIZE = 1024

# (user_id, path) -> time.time() when stored; keys expire after
# UPLOAD_TTL. Module-level so the handler and the sandbox tools share it.
_recent_uploads: TTLCache = TTLCache(
    maxsize=UPLOAD_CACHE_MAXSIZE, ttl=UPLOAD_TTL)


def _key(path: str) -> str:
    """Normalise a sandbox path so equivalent spellings share an entry."""
    return posixpath.normpath(path)


def record_upload(user_id: int, path: str) -> None:
    """Note that the file at `path` was just uploaded by `user_id`.

    `path` is the file's full path in the sandbox (e.g.
    ``/workspace/uploads/report.pdf``); entries are keyed on that whole
    path so same-named files in different directories stay distinct.
    """
    key = _key(path)
    _recent_uploads[(user_id, key)] = time.time()
    logger.debug("Cached recent upload %s for user %d", key, user_id)


def was_recently_uploaded(user_id: int, path: str) -> bool:
    return (user_id, _key(path)) in _recent_uploads
