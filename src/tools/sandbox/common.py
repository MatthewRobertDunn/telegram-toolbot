"""Helpers shared by the sandbox tools."""

import json
import posixpath
import shlex

from chat_update_context import AgentContext
from sandbox import manager

WORKSPACE_ROOT = "/workspace"
HELPER_DIR = "/opt/toolbot"
BASH_OUTPUT_LIMIT = 6000


def _resolve_path(path: str) -> str:
    if not path.startswith("/"):
        path = posixpath.join(WORKSPACE_ROOT, path)
    return posixpath.normpath(path)


# The "_sandbox_read_files" user_data key tracks the files read so far.
# Private to this package; its tests may reach in, nothing else should use it.
def _mark_read(ctx: AgentContext, path: str) -> None:
    ctx.user_data.setdefault("_sandbox_read_files", set()).add(path)


def _ensure_read(ctx: AgentContext, path: str, tool: str) -> str | None:
    if path not in ctx.user_data.get("_sandbox_read_files", set()):
        return (f"Error: file has not been read. Use the read tool on "
                f"{path} before using the {tool} tool on it.")
    return None


def _cap_output(text: str) -> str:
    if len(text) <= BASH_OUTPUT_LIMIT:
        return text
    return (text[:BASH_OUTPUT_LIMIT]
            + f"\n... (output truncated, {len(text) - BASH_OUTPUT_LIMIT} more characters)")


async def _run_helper(user_id: int, helper: str, args: list[str],
                      payload: dict | None = None) -> dict:
    command = f"python3 {HELPER_DIR}/{helper}"
    if args:
        command += " " + " ".join(shlex.quote(arg) for arg in args)
    stdin_data = json.dumps(payload).encode() if payload is not None else None
    result = await manager.exec(user_id, command,
                                stdin_data=stdin_data)
    if result["timeout"]:
        return {"ok": False, "error": "helper timed out"}
    if result["returncode"] != 0:
        return {"ok": False, "error": (result["stderr"].strip()[:500]
                or f"helper exited with code {result['returncode']}")}
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {"ok": False,
                "error": f"helper returned invalid JSON: {result['stdout'][:200]}"}
