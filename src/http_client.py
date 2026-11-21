"""Shared async HTTP client for the tool backends.

Every remote call made by tool code (image providers, page fetch) goes
through one lazily created ``aiohttp.ClientSession``: connections are
pooled across calls instead of a per-call TLS handshake, every request
carries a timeout, and nothing here blocks the event loop.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import aiohttp

# Per-request budget: a single request may take at most this long whatever
# the remote does; connect is bounded separately so a dead host fails fast
# instead of eating the whole budget.
REQUEST_TIMEOUT = 120.0
CONNECT_TIMEOUT = 15.0

_session: aiohttp.ClientSession | None = None
# The loop the session was created on; a session must only be used there.
_session_loop: asyncio.AbstractEventLoop | None = None


def _discard(old: aiohttp.ClientSession | None,
             old_loop: asyncio.AbstractEventLoop | None) -> None:
    """Best-effort cleanup of a session whose loop is not the running one.

    ``ClientSession.close()`` is a coroutine that must run on the loop that
    owns the session's transports: while that loop is still alive, the
    close is scheduled on it (from a foreign loop the transports must not
    be touched). Once the old loop has stopped, nothing can execute the
    coroutine any more, so the session is merely dropped for the GC to
    collect — the shape of every per-test loop in the test suite.
    """
    if old is None or old.closed:
        return
    if old_loop is not None and old_loop.is_running():

        def _close() -> None:
            asyncio.ensure_future(old.close())

        try:
            old_loop.call_soon_threadsafe(_close)
        except RuntimeError:
            pass  # the loop stopped between the check and the scheduling


class HttpStatusError(Exception):
    """A remote endpoint answered with a non-2xx status.

    Carries the response body: endpoint error bodies (content-policy
    rejections, bad-parameter details) are the useful part for the model
    when the failure is surfaced as a tool result.
    """

    def __init__(self, status: int, url: str, body: str = ""):
        self.status = status
        self.url = url
        self.body = body
        super().__init__(f"HTTP {status} from {url}: {body.strip()[:200]}")


@dataclass(frozen=True)
class Response:
    """A buffered HTTP response: status, headers, and the full body."""

    status: int
    headers: Mapping[str, str]
    content: bytes
    url: str

    def header(self, name: str, default: str = "") -> str:
        """Case-insensitive header lookup."""
        for key, value in self.headers.items():
            if key.lower() == name.lower():
                return value
        return default

    def text(self) -> str:
        """The body decoded as UTF-8 (every backend here is UTF-8/ASCII)."""
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        """The body parsed as JSON."""
        return json.loads(self.content)

    def raise_for_status(self) -> None:
        """Raise :class:`HttpStatusError` unless the status is 2xx."""
        if not 200 <= self.status < 300:
            raise HttpStatusError(self.status, self.url, self.text())


def _timeout(total: float) -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(total=total, connect=CONNECT_TIMEOUT)


def session() -> aiohttp.ClientSession:
    """The shared ``ClientSession``, created on first use and reused.

    The session is bound to the event loop that created it; when that loop
    is no longer the running one (each test case runs on a fresh loop), the
    stale session is cleaned up (:func:`_discard`) and a new one is opened
    for the running loop.
    """
    global _session, _session_loop
    loop = asyncio.get_running_loop()
    if _session is not None and not _session.closed and _session_loop is loop:
        return _session
    _discard(_session, _session_loop)
    _session = aiohttp.ClientSession(timeout=_timeout(REQUEST_TIMEOUT))
    _session_loop = loop
    return _session


async def aclose() -> None:
    """Close the shared session (graceful shutdown; tests use it too)."""
    global _session, _session_loop
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None
    _session_loop = None


async def request(method: str, url: str, *,
                  headers: Mapping[str, str] | None = None,
                  json_body: Any = None,
                  timeout: float | None = None) -> Response:
    """Send one request on the shared session and buffer the response.

    ``json_body`` is sent as a JSON request body when not None. ``timeout``
    is the total budget in seconds, defaulting to REQUEST_TIMEOUT. A
    request-level timeout surfaces as ``RuntimeError`` with a readable
    message; a deadline held by the caller arrives as ``CancelledError``
    and is not converted.
    """
    total = REQUEST_TIMEOUT if timeout is None else timeout
    try:
        async with session().request(method, url, headers=headers,
                                     json=json_body,
                                     timeout=_timeout(total)) as resp:
            return Response(status=resp.status,
                            headers=dict(resp.headers),
                            content=await resp.read(),
                            url=str(resp.url))
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"{method} {url} timed out after {total:g}s") from None


async def get(url: str, **kwargs: Any) -> Response:
    """``request("GET", ...)``."""
    return await request("GET", url, **kwargs)


async def post(url: str, **kwargs: Any) -> Response:
    """``request("POST", ...)``."""
    return await request("POST", url, **kwargs)


async def get_text(url: str, *, timeout: float, max_chars: int) -> str:
    """GET a URL and return its body as text.

    Reads at most roughly ``max_chars`` characters' worth of bytes rather
    than the whole body, so an enormous page cannot be buffered just to
    truncate it. Raises :class:`HttpStatusError` on a non-2xx response and
    ``RuntimeError`` on timeout.
    """
    # UTF-8 costs at most 4 bytes per character; the slack absorbs a read
    # that stops mid-character.
    byte_cap = max_chars * 4 + 1024
    try:
        async with session().get(url, timeout=_timeout(timeout)) as resp:
            if not 200 <= resp.status < 300:
                body = await resp.content.read(4096)
                raise HttpStatusError(
                    resp.status, str(resp.url),
                    body.decode("utf-8", errors="replace"))
            raw = await resp.content.read(byte_cap)
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"GET {url} timed out after {timeout:g}s") from None
    return raw.decode("utf-8", errors="replace")
