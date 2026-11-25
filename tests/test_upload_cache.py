"""Unit tests for the recent-upload cache keyed on (user_id, full path)."""

import unittest
import unittest.mock

import upload_cache


class _FakeClock:
    """A deterministic stand-in for ``TTLCache``'s monotonic clock."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class TestUploadCacheKeys(unittest.TestCase):

    def setUp(self):
        patcher = unittest.mock.patch.object(
            upload_cache, '_recent_uploads',
            upload_cache.TTLCache(maxsize=8, ttl=60))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_same_name_in_different_directories_do_not_collide(self):
        # Entries are keyed on the full path, not the basename, so a
        # same-named file elsewhere — or a relative spelling of the same
        # name — is a different file.
        upload_cache.record_upload(1, '/workspace/uploads/a/pic.png')
        self.assertFalse(
            upload_cache.was_recently_uploaded(1, '/workspace/uploads/b/pic.png'))
        upload_cache.record_upload(1, '/workspace/uploads/pic.png')
        self.assertFalse(
            upload_cache.was_recently_uploaded(1, 'uploads/pic.png'))

    def test_same_file_path_hits(self):
        # Equivalent spellings of one full path normalise to one entry.
        upload_cache.record_upload(1, '/workspace/uploads/pic.png')
        self.assertTrue(
            upload_cache.was_recently_uploaded(1, '/workspace/uploads/pic.png'))
        self.assertTrue(
            upload_cache.was_recently_uploaded(1, '/workspace/uploads/./pic.png'))
        self.assertTrue(
            upload_cache.was_recently_uploaded(1, '/workspace/uploads/pic.png/'))

    def test_other_user_misses(self):
        # Per-user scoping: the same file recorded by one user is not a
        # hit for another, and a different file is a miss even for the
        # recording user.
        upload_cache.record_upload(1, '/workspace/uploads/pic.png')
        self.assertFalse(
            upload_cache.was_recently_uploaded(2, '/workspace/uploads/pic.png'))
        self.assertFalse(
            upload_cache.was_recently_uploaded(1, '/workspace/uploads/other.png'))

    def test_entry_expires_after_the_ttl(self):
        # The cache's only time-dependent behaviour. Drive TTLCache with
        # a fake clock rather than sleeping: the entry hits before the
        # TTL elapses and misses once the TTL has passed.
        clock = _FakeClock()
        with unittest.mock.patch.object(
                upload_cache, '_recent_uploads',
                upload_cache.TTLCache(maxsize=8, ttl=60, timer=clock)):
            upload_cache.record_upload(1, '/workspace/uploads/pic.png')
            self.assertTrue(
                upload_cache.was_recently_uploaded(1, '/workspace/uploads/pic.png'))
            clock.now += 61
            self.assertFalse(
                upload_cache.was_recently_uploaded(1, '/workspace/uploads/pic.png'))


if __name__ == '__main__':
    unittest.main()
