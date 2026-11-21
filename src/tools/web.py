"""Web tools: DuckDuckGo search and reader-proxied page fetching.

Neither tool may block the event loop: the search runs ddgs' blocking
client in a worker thread (it has no async API), and the page fetch goes
through the shared async client in :mod:`http_client`.
"""
import asyncio
import logging
import re
from urllib.parse import quote

from ddgs import DDGS

import http_client
from tool_registry import BOTH, agent_tool

logger = logging.getLogger(__name__)

# DuckDuckGo returns at most 10 results per text search; clamp instead of
# passing an oversized request through.
MAX_SEARCH_RESULTS = 10
# max_chars is clamped into 1..MAX_FETCH_CHARS: non-positive values would
# otherwise produce a misleading "page content was empty" result, and huge
# values would defeat the point of truncating what the model receives.
MAX_FETCH_CHARS = 1_000_000
# Cap on the exception text echoed to the model as the tool result. The
# text itself is returned verbatim — it is the model's only diagnostic
# signal (proxy host, status, TLS/socket and timeout detail all help it
# decide whether to retry, change the URL, or give up) — only its length
# is capped, so a pathological error cannot flood the context.
MAX_ERROR_CHARS = 500

READER_URL = "https://r.jina.ai/"
FETCH_TIMEOUT = 15.0


@agent_tool(
    scope=BOTH,
    description=("Web search via DuckDuckGo. Returns titles, URLs, snippets."),
    parameters={
        "query": {
            "type": "string",
            "description": "Search query.",
        },
        "max_results": {
            "type": "integer",
            "description": ("Max results (default 5, max 10; clamped)."),
        },
    })
async def web_search(query: str, max_results: int = 5) -> str:
    max_results = max(1, min(max_results, MAX_SEARCH_RESULTS))

    def search() -> list[dict[str, str]]:
        return DDGS().text(query, max_results=max_results)

    try:
        # ddgs has no async API: run its blocking client in a worker thread
        # so the shared event loop keeps serving other updates.
        results = await asyncio.to_thread(search)
        if not results:
            return "No search results found."
        output = []
        for i, r in enumerate(results):
            output.append(f"{i+1}. {r['title']}\n URL: {r['href']}\n {r['body']}")
        return "\n\n".join(output)
    except Exception as e:
        # Exception text verbatim (see MAX_ERROR_CHARS above), length-capped.
        logger.warning("web_search failed: %s", e)
        return f"Web search error: {str(e)[:MAX_ERROR_CHARS]}"


@agent_tool(
    scope=BOTH,
    description=("Fetch a web page's text content as markup, given its URL."),
    parameters={
        "url": {
            "type": "string",
            "description": ("Full webpage URL, including https://."),
        },
        "max_chars": {
            "type": "integer",
            "description": ("Max characters to return (default 100000; "
                            "clamped to 1-1000000)."),
        },
    })
async def web_fetch(url: str, max_chars: int = 100000) -> str:
    if not re.match(r'^https?://', url):
        return f"Invalid URL: {url}. Must start with http:// or https://"
    max_chars = max(1, min(max_chars, MAX_FETCH_CHARS))
    # The reader service takes the target URL as its path; quote it so
    # spaces, unicode and query strings survive the round trip. The
    # fragment separator is deliberately NOT safe: a '#' would make yarl
    # strip everything after it from the proxied URL, so it is
    # percent-encoded and preserved in the path instead.
    reader_url = READER_URL + quote(url, safe=":/?[]@!$&'()*+,;=~-._")
    try:
        text = await http_client.get_text(reader_url, timeout=FETCH_TIMEOUT,
                                          max_chars=max_chars)
    except http_client.HttpStatusError as e:
        return f"Failed to fetch {url}: HTTP {e.status}"
    except Exception as e:
        # Exception text verbatim (see MAX_ERROR_CHARS above), length-capped.
        logger.warning("web_fetch failed for %s: %s", url, e)
        return f"Web fetch error for {url}: {str(e)[:MAX_ERROR_CHARS]}"
    return text[:max_chars] if text else "(page content was empty after parsing)"
