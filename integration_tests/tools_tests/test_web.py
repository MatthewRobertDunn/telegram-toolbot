import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # noqa: F401

import http_client
from tools.web import web_fetch, web_search


class TestWebTools(unittest.IsolatedAsyncioTestCase):

    async def asyncTearDown(self):
        # Each test ran on its own event loop, so the shared session must
        # not leak into the next one (see http_client.session).
        await http_client.aclose()

    async def test_web_search_returns_results(self):
        result = await web_search("python programming language", max_results=3)
        self.assertNotIn("Web search error", result)
        self.assertIn("1. ", result[:10])

    async def test_web_search_clamps_max_results(self):
        # max_results is documented as max 10 and is clamped, not passed
        # through.
        result = await web_search("python programming language", max_results=50)
        self.assertNotIn("Web search error", result)
        numbered = [line for line in result.splitlines()
                    if re.match(r"^\d+\. ", line)]
        self.assertGreater(len(numbered), 0)
        self.assertLessEqual(len(numbered), 10)

    async def test_web_fetch_returns_page_content(self):
        result = await web_fetch("https://example.com", max_chars=2000)
        self.assertNotIn("Web fetch error", result)
        self.assertNotIn("Failed to fetch", result)
        self.assertIn("https://example.com", result)
        self.assertGreater(len(result), 50)

    async def test_web_fetch_truncates_to_max_chars(self):
        # The body is capped during the read, and the returned text honors
        # a small max_chars exactly.
        result = await web_fetch("https://example.com", max_chars=50)
        self.assertNotIn("Web fetch error", result)
        self.assertNotIn("Failed to fetch", result)
        self.assertLessEqual(len(result), 50)

    async def test_web_fetch_zero_max_chars_is_clamped_not_empty(self):
        # max_chars=0 must not produce the misleading "page content was
        # empty" message.
        result = await web_fetch("https://example.com", max_chars=0)
        self.assertNotIn("page content was empty", result)
        self.assertNotIn("Web fetch error", result)
        self.assertLessEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
