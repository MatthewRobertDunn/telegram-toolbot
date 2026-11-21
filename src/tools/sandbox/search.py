"""Finding files and content in the sandbox."""

from chat_update_context import AgentContext
from tool_registry import BOTH, agent_tool

from .common import _resolve_path, _run_helper


@agent_tool(
    scope=BOTH,
    description=(
        "Fast content search via ripgrep in sandbox, any codebase size. "
        "Usage:\n"
        "- Regex search file contents (e.g. \"log.*Error\", "
        "\"function\\s+\\w+\")\n"
        "- Filter with include (e.g. \"*.js\", \"*.{ts,tsx}\")\n"
        "- Returns paths, line numbers, matching lines\n"
        "- Use to find files by content pattern\n"
        "- To count matches, use bash with `rg` directly, NOT `grep`."),
    parameters={
        "pattern": {
            "type": "string",
            "description": ("Regex to search file contents"),
        },
        "path": {
            "type": "string",
            "description": ("Directory to search (default working dir)."),
        },
        "include": {
            "type": "string",
            "description": ("File pattern filter (e.g. "
                            "\"*.js\", \"*.{ts,tsx}\")"),
        },
    })
async def grep(ctx: AgentContext, pattern: str,
               path: str | None = None, include: str | None = None) -> str:
    if not pattern:
        return "Error: pattern is required"
    args = ["--pattern", pattern]
    if path:
        args += ["--path", _resolve_path(path)]
    if include:
        args += ["--include", include]
    payload = await _run_helper(ctx.user_id, "toolbot_grep.py", args)
    if not payload.get("ok"):
        return f"Error: {payload.get('error', 'grep failed')}"
    data = payload["data"]
    matches = data["matches"]
    if not matches:
        return "No files found"

    output = [f"Found {len(matches)} matches"
              + (" (more matches available)" if data["truncated"] else "")]
    current = ""
    for match in matches:
        if match["path"] != current:
            if current:
                output.append("")
            current = match["path"]
            output.append(f"{current}:")
        output.append(f"  Line {match['line']}: {match['text']}")
    if data["truncated"]:
        output += ["", "(Results truncated. Consider using a more specific "
                       "path or pattern.)"]
    return "\n".join(output)


@agent_tool(
    scope=BOTH,
    description=(
        "Fast file search via ripgrep in sandbox, any codebase size. "
        "Returns matches newest-first. Usage:\n"
        "- Glob patterns like \"**/*.js\", \"src/**/*.ts\"\n"
        "- Use to find files by name\n"
        "- Batch multiple speculative searches when useful."),
    parameters={
        "pattern": {
            "type": "string",
            "description": "Glob pattern to match files",
        },
        "path": {
            "type": "string",
            "description": ("Directory to search (default workspace root)."),
        },
    })
async def glob(ctx: AgentContext, pattern: str,
               path: str | None = None) -> str:
    args = ["--pattern", pattern]
    if path:
        args += ["--path", _resolve_path(path)]
    payload = await _run_helper(ctx.user_id, "toolbot_glob.py", args)
    if not payload.get("ok"):
        return f"Error: {payload.get('error', 'glob failed')}"
    files = payload["data"]["files"]
    if not files:
        return "No files found"
    output = list(files)
    if payload["data"]["truncated"]:
        output += ["", "(Results are truncated: showing first 100 results. "
                       "Consider using a more specific path or pattern.)"]
    return "\n".join(output)
