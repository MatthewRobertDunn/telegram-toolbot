"""Unit tests for the shared async HTTP client (fully hermetic).

The aiohttp layer is faked at the ``session()`` seam, so no test here
touches the network. The web_fetch tests live here too because they
exercise the same seam through tools.web.
"""
import asyncio
import os
import sys
import threading
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import aiohttp

import http_client
from tools.web import web_fetch


class _FakeContent:
    """Minimal StreamReader stand-in: serves the body in one read."""

    def __init__(self, data: bytes):
        self.data = data
        self.requested: list[int] = []

    async def read(self, n: int) -> bytes:
        self.requested.append(n)
        return self.data[:n]


class _FakeResponse:
    """Minimal ClientResponse stand-in for request() and get_text()."""

    def __init__(self, status: int = 200, data: bytes = b'',
                 url: str = 'https://unit.test/x'):
        self.status = status
        self.headers = {'Content-Type': 'text/plain'}
        self.content = _FakeContent(data)
        self.url = url

    async def read(self) -> bytes:
        return self.content.data


class _FakeContext:
    def __init__(self, response: _FakeResponse):
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeSession:
    """Stand-in for aiohttp.ClientSession: returns the built response,
    or raises, when a request is started."""

    def __init__(self, response: _FakeResponse | None = None,
                 exc: Exception | None = None):
        self._response = response
        self._exc = exc

    def request(self, method: str, url: str, **kwargs: object) -> _FakeContext:
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return _FakeContext(self._response)

    def get(self, url: str, **kwargs: object) -> _FakeContext:
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return _FakeContext(self._response)


def _patched(fake: _FakeSession):
    return unittest.mock.patch.object(http_client, 'session', lambda: fake)


def _make_and_close_loop_session() -> aiohttp.ClientSession:
    """Create a session on a fresh event loop, then close the loop.

    Runs in a worker thread: the result is a session whose owning loop is
    dead, like the per-test-case loops the module must cope with.
    """
    loop = asyncio.new_event_loop()
    try:
        async def _make():
            return http_client.session()
        return loop.run_until_complete(_make())
    finally:
        loop.close()


class TestGetText(unittest.IsolatedAsyncioTestCase):

    async def asyncTearDown(self):
        await http_client.aclose()

    async def test_multibyte_content_still_yields_max_chars(self):
        # UTF-8 costs 3 bytes for these characters: the byte cap must never
        # cut the text below max_chars while content is available.
        data = ('x' + '漢' * 2000).encode('utf-8')
        with _patched(_FakeSession(_FakeResponse(200, data))):
            text = await http_client.get_text('https://unit.test/x',
                                              timeout=5, max_chars=10)
        self.assertGreaterEqual(len(text), 10)

    async def test_body_is_capped_during_the_read(self):
        # A huge body is not buffered whole: the read is limited to roughly
        # max_chars worth of bytes.
        response = _FakeResponse(200, b'x' * 1_000_000)
        with _patched(_FakeSession(response)):
            await http_client.get_text('https://unit.test/x',
                                       timeout=5, max_chars=10)
        self.assertEqual(response.content.requested, [10 * 4 + 1024])

    async def test_non_2xx_raises_status_error_with_details(self):
        fake = _FakeSession(_FakeResponse(404, b'missing',
                                          url='https://unit.test/x'))
        with _patched(fake):
            with self.assertRaises(http_client.HttpStatusError) as ctx:
                await http_client.get_text('https://unit.test/x',
                                           timeout=5, max_chars=100)
        self.assertEqual(ctx.exception.status, 404)
        self.assertEqual(ctx.exception.url, 'https://unit.test/x')
        self.assertEqual(ctx.exception.body, 'missing')

    async def test_timeout_becomes_readable_runtime_error(self):
        for entry_point in ('get_text', 'request'):
            with self.subTest(entry_point=entry_point):
                with _patched(_FakeSession(exc=asyncio.TimeoutError())):
                    with self.assertRaises(RuntimeError) as ctx:
                        if entry_point == 'get_text':
                            await http_client.get_text('https://unit.test/x',
                                                       timeout=7,
                                                       max_chars=100)
                        else:
                            await http_client.request(
                                'GET', 'https://unit.test/x', timeout=7)
                self.assertIn('timed out after 7s', str(ctx.exception))
                if entry_point == 'request':
                    self.assertIn('https://unit.test/x', str(ctx.exception))


class TestRequest(unittest.IsolatedAsyncioTestCase):

    async def asyncTearDown(self):
        await http_client.aclose()

    async def test_response_is_buffered_with_status_and_headers(self):
        with _patched(_FakeSession(_FakeResponse(201, b'hello'))):
            resp = await http_client.request('GET', 'https://unit.test/x')
        self.assertEqual(resp.status, 201)
        self.assertEqual(resp.content, b'hello')
        self.assertEqual(resp.url, 'https://unit.test/x')
        self.assertEqual(resp.header('content-TYPE'), 'text/plain')


class TestSessionLifecycle(unittest.IsolatedAsyncioTestCase):

    async def asyncTearDown(self):
        await http_client.aclose()

    async def test_aclose_clears_the_session(self):
        # The session is reused across calls on the same loop.
        self.assertIs(http_client.session(), http_client.session())
        first = http_client.session()
        await http_client.aclose()
        self.assertTrue(first.closed)
        self.assertIsNone(http_client._session)
        # A later call creates a fresh, usable session.
        second = http_client.session()
        self.assertIsNot(second, first)
        self.assertFalse(second.closed)

    async def test_session_from_a_dead_loop_is_replaced(self):
        # A session made on a loop that has since closed must be replaced,
        # not reused (this is what per-test-case loops look like to the
        # module). The stale loop is spun up in a worker thread because a
        # second loop cannot run inside the running one.
        old = await asyncio.get_running_loop().run_in_executor(
            None, _make_and_close_loop_session)
        fresh = http_client.session()
        self.assertIsNot(fresh, old)
        self.assertFalse(fresh.closed)

    async def test_session_from_a_live_loop_is_closed_not_leaked(self):
        # The old loop is still running in another thread, so the stale
        # session's close must be scheduled there instead of dropped.
        bg = asyncio.new_event_loop()
        thread = threading.Thread(target=bg.run_forever, daemon=True)
        thread.start()
        try:
            async def _make():
                return http_client.session()
            old = await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(_make(), bg))
            fresh = http_client.session()  # loop mismatch: old is discarded
            self.assertIsNot(fresh, old)
            for _ in range(100):
                if old.closed:
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(old.closed, 'stale session was never closed')
        finally:
            bg.call_soon_threadsafe(bg.stop)
            thread.join(timeout=5)
            await http_client.aclose()


class TestWebFetch(unittest.IsolatedAsyncioTestCase):

    async def asyncTearDown(self):
        await http_client.aclose()

    async def test_result_never_exceeds_max_chars(self):
        for max_chars, body in ((10, '漢' * 100), (0, 'hello')):
            with self.subTest(max_chars=max_chars):
                async def fake_get_text(url, *, timeout, max_chars):
                    return body

                with unittest.mock.patch.object(http_client, 'get_text',
                                                fake_get_text):
                    result = await web_fetch('https://example.com',
                                             max_chars=max_chars)
                if max_chars == 0:
                    # Zero is clamped to a usable limit, not "empty page".
                    self.assertNotIn('page content was empty', result)
                    self.assertLessEqual(len(result), 1)
                else:
                    self.assertEqual(len(result), 10)

    async def test_fragment_is_percent_encoded_into_the_path(self):
        # '#' must not be a safe character: yarl would strip it and the
        # fragment from the URL proxied to the reader.
        captured = {}

        async def fake_get_text(url, *, timeout, max_chars):
            captured['url'] = url
            return 'hello'

        with unittest.mock.patch.object(http_client, 'get_text',
                                        fake_get_text):
            await web_fetch('https://example.com/page#section',
                            max_chars=10)
        self.assertIn('%23', captured['url'])
        self.assertNotIn('#', captured['url'])


if __name__ == '__main__':
    unittest.main()
