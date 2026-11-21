"""Helpers shared by the sandbox tools."""

import json
import posixpath
import shlex
from collections.abc import Mapping
from typing import Any

from chat_update_context import AgentContext
from sandbox import manager

WORKSPACE_ROOT = "/workspace"
HELPER_DIR = "/opt/toolbot"
BASH_OUTPUT_LIMIT = 6000
MAX_ERROR_CHARS = 500

# Must match the helpers' PROTOCOL_VERSION (sandbox_image/helpers/
# toolbot_protocol.py). A mismatch means the image is older than the bot.
HELPER_PROTOCOL_VERSION = 1


def _resolve_path(path: str) -> str:
    if not path.startswith("/"):
        path = posixpath.join(WORKSPACE_ROOT, path)
    return posixpath.normpath(path)


# The "sandbox_read_files" user_data field tracks the files read so far.
# Private to this package; its tests may reach in, nothing else should use
# it. The manager learns where the marks live (watch_read_marks) so a
# destroy/reset can clear them along with the files they refer to.
def _mark_read(ctx: AgentContext, path: str) -> None:
    manager.watch_read_marks(ctx.user_id, ctx.user_data)
    ctx.user_data.sandbox_read_files.add(path)


def _ensure_read(ctx: AgentContext, path: str, tool: str) -> str | None:
    if path not in ctx.user_data.sandbox_read_files:
        return (f"Error: file has not been read. Use the read tool on "
                f"{path} before using the {tool} tool on it.")
    return None


def _cap_output(text: str) -> str:
    if len(text) <= BASH_OUTPUT_LIMIT:
        return text
    return (text[:BASH_OUTPUT_LIMIT]
            + f"\n... (output truncated, {len(text) - BASH_OUTPUT_LIMIT} more characters)")


def format_command_result(result: Mapping[str, Any]) -> str:
    """Render a manager.exec() result as model-visible text.

    The one formatter for the command-running tools (bash, python), so
    both present the same shape: an optional timeout notice, the return
    code, then capped stdout and stderr sections.
    """
    sections = []
    if result.get("timeout"):
        sections.append("Command timed out.")
    if result.get("returncode") is not None:
        sections.append(f"Return code: {result['returncode']}")
    if result.get("stdout"):
        sections.append(f"stdout:\n{_cap_output(result['stdout'])}")
    if result.get("stderr"):
        sections.append(f"stderr:\n{_cap_output(result['stderr'])}")
    return "\n".join(sections) or "(executed with no output)"


async def _run_helper(user_id: int, helper: str, args: list[str],
                      payload: dict | None = None) -> dict:
    """Run a container helper and return its validated response.

    Helpers speak the toolbot protocol (toolbot_protocol.py): one JSON
    line {"v": 1, "ok": ...}. A helper that crashes, times out, or speaks
    an unknown protocol degrades to {"ok": False, "error": ...} instead
    of raising or leaking its raw internals to the model.
    """
    command = f"python3 {HELPER_DIR}/{helper}"
    if args:
        command += " " + " ".join(shlex.quote(arg) for arg in args)
    stdin_data = json.dumps(payload).encode() if payload is not None else None
    result = await manager.exec(user_id, command, stdin_data=stdin_data)
    if result["timeout"]:
        return {"ok": False, "error": f"{helper} timed out."}
    # Parse the payload before the exit-code check: a failed helper still
    # speaks the protocol (emit() prints and exits 1), and its error
    # message beats a generic "exited with code N".
    response: Any = None
    if result["stdout"].strip():
        try:
            response = json.loads(result["stdout"])
        except json.JSONDecodeError:
            response = None
    if response is not None and not isinstance(response, dict):
        return {"ok": False, "error": f"{helper} response is not an object."}
    if isinstance(response, dict):
        version = response.get("v")
        if version != HELPER_PROTOCOL_VERSION:
            return {"ok": False,
                    "error": (f"{helper} spoke protocol version "
                              f"{version!r}, expected "
                              f"{HELPER_PROTOCOL_VERSION}.")}
        ok = response.get("ok")
        if ok is True and result["returncode"] == 0:
            if "data" not in response:
                return {"ok": False,
                        "error": f"{helper} response is missing 'data'."}
            return response
        if ok is False:
            error = response.get("error")
            if isinstance(error, str) and error:
                return {"ok": False, "error": error[:MAX_ERROR_CHARS]}
            return {"ok": False,
                    "error": f"{helper} failed without an error message."}
        if ok is not False:
            return {"ok": False, "error": f"{helper} response has no 'ok' flag."}
        # else: ok=True on a failed run is untrustworthy; report the run.
    if result["returncode"] != 0:
        detail = result["stderr"].strip()[:MAX_ERROR_CHARS]
        return {"ok": False,
                "error": detail or
                f"{helper} exited with code {result['returncode']}."}
    return {"ok": False, "error": f"{helper} returned an unreadable response."}
