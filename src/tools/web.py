import re

import aiohttp
from ddgs import DDGS

from tool_registry import BOTH, agent_tool


@agent_tool(
    scope=BOTH,
    description=("Search the web using DuckDuckGo. Returns titles, URLs, "
                 "and snippets for each result."),
    parameters={
        "query": {
            "type": "string",
            "description": "The search query string.",
        },
        "max_results": {
            "type": "integer",
            "description": ("Maximum number of results to return "
                            "(default 5, max 10)."),
        },
    })
async def web_search(query: str, max_results: int = 5) -> str:
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return "No search results found."
        output = []
        for i, r in enumerate(results):
            output.append(f"{i+1}. {r['title']}\n URL: {r['href']}\n {r['body']}")
        return "\n\n".join(output)
    except Exception as e:
        return f"Web search error: {e}"


@agent_tool(
    scope=BOTH,
    description=("Fetch and extract the text content of a web page converted "
                 "to markup given its URL."),
    parameters={
        "url": {
            "type": "string",
            "description": ("The full URL of the webpage to fetch, including "
                            "https://."),
        },
        "max_chars": {
            "type": "integer",
            "description": ("Maximum characters to return from the page "
                            "(default 100000)."),
        },
    })
async def web_fetch(url: str, max_chars: int = 100000) -> str:
    if not re.match(r'^https?://', url):
        return f"Invalid URL: {url}. Must start with http:// or https://"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://r.jina.ai/{url}", timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return f"Failed to fetch {url}: HTTP {resp.status}"
                text = await resp.text()
                return text[:max_chars] if text else "(page content was empty after parsing)"
    except Exception as e:
        return f"Web fetch error for {url}: {e}"
