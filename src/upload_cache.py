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

# (user_id, filename) -> time.time() when stored; keys expire after
# UPLOAD_TTL. Module-level so the handler and the sandbox tools share it.
_recent_uploads: TTLCache = TTLCache(
    maxsize=UPLOAD_CACHE_MAXSIZE, ttl=UPLOAD_TTL)


def record_upload(user_id: int, path: str) -> None:
    """Note that the file at `path` was just uploaded by `user_id`."""
    filename = posixpath.basename(path)
    _recent_uploads[(user_id, filename)] = time.time()
    logger.debug("Cached recent upload %s for user %d", filename, user_id)


def was_recently_uploaded(user_id: int, path: str) -> bool:
    """True if a file with `path`'s filename was recently uploaded by `user_id`."""
    return (user_id, posixpath.basename(path)) in _recent_uploads
