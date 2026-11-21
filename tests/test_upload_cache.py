"""Unit tests for the recent-upload cache keyed on (user_id, full path)."""

import unittest
import unittest.mock

import upload_cache


class TestUploadCacheKeys(unittest.TestCase):

    def setUp(self):
        patcher = unittest.mock.patch.object(
            upload_cache, '_recent_uploads',
            upload_cache.TTLCache(maxsize=8, ttl=60))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_same_name_in_different_directories_do_not_collide(self):
        upload_cache.record_upload(1, '/workspace/uploads/a/pic.png')
        self.assertFalse(
            upload_cache.was_recently_uploaded(1, '/workspace/uploads/b/pic.png'))

    def test_same_file_path_hits(self):
        upload_cache.record_upload(1, '/workspace/uploads/pic.png')
        self.assertTrue(
            upload_cache.was_recently_uploaded(1, '/workspace/uploads/pic.png'))

    def test_relative_path_is_a_different_file(self):
        upload_cache.record_upload(1, '/workspace/uploads/pic.png')
        self.assertFalse(
            upload_cache.was_recently_uploaded(1, 'uploads/pic.png'))

    def test_equivalent_path_spelling_hits(self):
        upload_cache.record_upload(1, '/workspace/uploads/pic.png')
        self.assertTrue(
            upload_cache.was_recently_uploaded(1, '/workspace/uploads/./pic.png'))
        self.assertTrue(
            upload_cache.was_recently_uploaded(1, '/workspace/uploads/pic.png/'))

    def test_other_user_misses(self):
        upload_cache.record_upload(1, '/workspace/uploads/pic.png')
        self.assertFalse(
            upload_cache.was_recently_uploaded(2, '/workspace/uploads/pic.png'))

    def test_other_file_misses(self):
        upload_cache.record_upload(1, '/workspace/uploads/pic.png')
        self.assertFalse(
            upload_cache.was_recently_uploaded(1, '/workspace/uploads/other.png'))

    def test_rerecording_updates_the_entry(self):
        upload_cache.record_upload(1, '/workspace/uploads/pic.png')
        upload_cache.record_upload(1, '/workspace/uploads/pic.png')
        self.assertTrue(
            upload_cache.was_recently_uploaded(1, '/workspace/uploads/pic.png'))


if __name__ == '__main__':
    unittest.main()
