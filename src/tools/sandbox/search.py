"""Finding files and content in the sandbox."""

import json
import shlex

from chat_update_context import AgentContext
from sandbox import manager
from tool_registry import BOTH, agent_tool

from .common import _resolve_path, _run_helper

MAX_GREP_MATCHES = 100
MAX_GREP_BYTES = 512 * 1024


def _parse_rg_json(stdout: str) -> list[dict]:
    matches = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue
        data = event.get("data", {})
        matches.append({
            "path": (data.get("path") or {}).get("text", ""),
            "line": data.get("line_number"),
            "text": ((data.get("lines") or {}).get("text") or "").rstrip("\n"),
        })
    return matches


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
    search_path = _resolve_path(path) if path else "."
    parts = ["rg", "--json", "-e", shlex.quote(pattern)]
    if include:
        parts += ["--glob", shlex.quote(include)]
    parts.append(shlex.quote(search_path))
    command = " ".join(parts) + f" | head -c {MAX_GREP_BYTES}"
    result = await manager.exec(ctx.user_id, command)
    all_matches = _parse_rg_json(result["stdout"])
    if not all_matches and result["stderr"].strip():
        return f"Error: {result['stderr'].strip()[:500]}"
    truncated = len(all_matches) >= MAX_GREP_MATCHES
    matches = all_matches[:MAX_GREP_MATCHES]
    if not matches:
        return "No files found"

    def abs_path(p: str) -> str:
        return p if p.startswith("/") else _resolve_path(p)

    output = [f"Found {len(matches)} matches"
              + (" (more matches available)" if truncated else "")]
    current = ""
    for match in matches:
        if match["path"] != current:
            if current:
                output.append("")
            current = match["path"]
            output.append(f"{abs_path(current)}:")
        output.append(f"  Line {match['line']}: {match['text']}")
    if truncated:
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
