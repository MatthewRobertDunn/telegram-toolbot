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

# The helpers in sandbox_image emit protocol version 1
# (toolbot_protocol.py's PROTOCOL_VERSION); a missing version field is
# read as version 1 (helpers from before the protocol existed), and
# versions up to MAX_HELPER_PROTOCOL_VERSION are accepted - both keep a
# code-only `pake publish` working against an older sandbox image. Only a
# helper newer than this code is rejected, with a clear message.
MAX_HELPER_PROTOCOL_VERSION = 2


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
    a protocol newer than this code understands degrades to
    {"ok": False, "error": ...} instead of raising or leaking its raw
    internals to the model.
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
        # A missing version field means a helper from before the protocol
        # existed: treat it as version 1, so a code-only deploy keeps
        # working against an older sandbox image.
        version = response.get("v", 1)
        if (not isinstance(version, int) or isinstance(version, bool)
                or version < 1):
            return {"ok": False,
                    "error": (f"{helper} spoke unparseable protocol "
                              f"version {version!r}.")}
        if version > MAX_HELPER_PROTOCOL_VERSION:
            return {"ok": False,
                    "error": (f"{helper} speaks protocol v{version}, newer "
                              f"than this bot understands "
                              f"(v{MAX_HELPER_PROTOCOL_VERSION}); the "
                              f"sandbox image needs updating.")}
        ok = response.get("ok")
        if ok is True:
            if result["returncode"] == 0:
                if "data" not in response:
                    return {"ok": False,
                            "error": f"{helper} response is missing 'data'."}
                return response
            # A "success" payload on a failed run is untrustworthy; report
            # the run itself below.
        elif ok is False:
            error = response.get("error")
            if isinstance(error, str) and error:
                return {"ok": False, "error": error[:MAX_ERROR_CHARS]}
            return {"ok": False,
                    "error": f"{helper} failed without an error message."}
        else:
            return {"ok": False,
                    "error": f"{helper} response has no 'ok' flag."}
    if result["returncode"] != 0:
        detail = result["stderr"].strip()[:MAX_ERROR_CHARS]
        return {"ok": False,
                "error": detail or
                f"{helper} exited with code {result['returncode']}."}
    return {"ok": False, "error": f"{helper} returned an unreadable response."}
