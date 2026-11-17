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
        "Fast content search tool that works with any codebase "
        "size, powered by ripgrep in the sandbox. Usage:\n"
        "- Searches file contents using regular expressions\n"
        "- Supports full regex syntax (eg. \"log.*Error\", "
        "\"function\\s+\\w+\", etc.)\n"
        "- Filter files by pattern with the include parameter "
        "(eg. \"*.js\", \"*.{ts,tsx}\")\n"
        "- Returns file paths and line numbers with matching lines\n"
        "- Use this tool when you need to find files containing "
        "specific patterns\n"
        "- If you need to identify/count the number of matches "
        "within files, use the bash tool with `rg` (ripgrep) "
        "directly. Do NOT use `grep`."),
    parameters={
        "pattern": {
            "type": "string",
            "description": ("The regex pattern to search for in file "
                            "contents"),
        },
        "path": {
            "type": "string",
            "description": ("The directory to search in. Defaults to the "
                            "current working directory."),
        },
        "include": {
            "type": "string",
            "description": ("File pattern to include in the search (e.g. "
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
        "Fast file pattern matching tool that works with any "
        "codebase size, powered by ripgrep in the sandbox. "
        "Returns matching file paths sorted by modification time "
        "(newest first). Usage:\n"
        "- Supports glob patterns like \"**/*.js\" or "
        "\"src/**/*.ts\"\n"
        "- Use this tool when you need to find files by name "
        "patterns\n"
        "- It is always better to speculatively perform multiple "
        "searches as a batch that are potentially useful."),
    parameters={
        "pattern": {
            "type": "string",
            "description": "The glob pattern to match files against",
        },
        "path": {
            "type": "string",
            "description": ("The directory to search in. If not specified, "
                            "the workspace root will be used."),
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
