import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # noqa: F401

from tools.web import web_fetch, web_search


class TestWebTools(unittest.IsolatedAsyncioTestCase):

    async def test_web_search_returns_results(self):
        result = await web_search("python programming language", max_results=3)
        self.assertNotIn("Web search error", result)
        self.assertIn("1. ", result[:10])

    async def test_web_fetch_returns_page_content(self):
        result = await web_fetch("https://example.com", max_chars=2000)
        self.assertNotIn("Web fetch error", result)
        self.assertNotIn("Failed to fetch", result)
        self.assertIn("https://example.com", result)
        self.assertGreater(len(result), 50)


if __name__ == "__main__":
    unittest.main()
