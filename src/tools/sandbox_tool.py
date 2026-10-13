import base64
import json
import logging
import posixpath
import shlex

from chat_update_context import ChatUpdateContext, context_tool
from sandbox import manager, UPLOADS_DIR
from .sandbox_tool_definitions import TOOL_DEFINITIONS, WORKSPACE_ROOT
from tool_result import ImageAttachment, ToolCallResult

logger = logging.getLogger(__name__)

HELPER_DIR = "/opt/toolbot"
BASH_OUTPUT_LIMIT = 6000
MAX_GREP_MATCHES = 100
MAX_GREP_BYTES = 512 * 1024


def _resolve_path(path: str) -> str:
    if not path.startswith("/"):
        path = posixpath.join(WORKSPACE_ROOT, path)
    return posixpath.normpath(path)


def _mark_read(ctx: ChatUpdateContext, path: str) -> None:
    ctx.user_data.setdefault("sandbox_read_files", set()).add(path)


def _ensure_read(ctx: ChatUpdateContext, path: str, tool: str) -> str | None:
    if path not in ctx.user_data.get("sandbox_read_files", set()):
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


@context_tool
async def bash(ctx: ChatUpdateContext, command: str) -> str:
    if not command.strip():
        return "Error: empty command."
    result = await manager.exec(ctx.user_id, command)
    output = []
    if result["timeout"]:
        output.append("Command timed out.")
    if result["returncode"] is not None:
        output.append(f"Return code: {result['returncode']}")
    if result["stdout"]:
        output.append(f"stdout:\n{_cap_output(result['stdout'])}")
    if result["stderr"]:
        output.append(f"stderr:\n{_cap_output(result['stderr'])}")
    if not output:
        return "(executed with no output)"
    return "\n".join(output)


@context_tool
async def read(ctx: ChatUpdateContext, filePath: str,
               offset: int | None = None, limit: int | None = None) -> str:
    path = _resolve_path(filePath)
    args = ["--path", path]
    if offset:
        args += ["--offset", str(int(offset))]
    if limit:
        args += ["--limit", str(int(limit))]
    payload = await _run_helper(ctx.user_id, "toolbot_read.py", args)
    if not payload.get("ok"):
        return f"Error: {payload.get('error', 'read failed')}"

    data = payload["data"]
    if data["type"] == "directory":
        output = [f"<path>{path}</path>", "<type>directory</type>",
                  "<entries>"]
        output.extend(data["entries"])
        if data["truncated"]:
            shown = len(data["entries"])
            output.append(
                f"\n(Showing {shown} of {data['totalEntries']} entries. "
                f"Use 'offset' parameter to read beyond entry "
                f"{(offset or 1) + shown})")
        else:
            output.append(f"\n({data['totalEntries']} entries)")
        output.append("</entries>")
        return "\n".join(output)

    lines = data["lines"]
    last = data["lineEnd"]
    nxt = last + 1
    output = [f"<path>{path}</path>", "<type>file</type>", "<content>\n"]
    output.append("\n".join(f"{line['n']}: {line['text']}" for line in lines))
    if data["cut"]:
        output.append(
            f"\n\n(Output capped at 50 KB. Showing lines "
            f"{data['lineStart']}-{last}. Use offset={nxt} to continue.)")
    elif data["more"]:
        output.append(
            f"\n\n(Showing lines {data['lineStart']}-{last} of "
            f"{data['totalLines']}. Use offset={nxt} to continue.)")
    else:
        output.append(f"\n\n(End of file - total {data['totalLines']} lines)")
    output.append("\n</content>")
    _mark_read(ctx, path)
    return "\n".join(output)


@context_tool
async def write(ctx: ChatUpdateContext, filePath: str, content: str) -> str:
    path = _resolve_path(filePath)
    check = await manager.exec(
        ctx.user_id, f"test -e {shlex.quote(path)}")
    if check["returncode"] == 0:
        error = _ensure_read(ctx, path, "write")
        if error:
            return error
    parent = posixpath.dirname(path)
    command = (f"mkdir -p {shlex.quote(parent)} && "
               f"cat > {shlex.quote(path)}")
    result = await manager.exec(ctx.user_id, command,
                                           stdin_data=content.encode())
    if result["timeout"]:
        return "Error: write timed out."
    if result["returncode"] != 0:
        return f"Error: {result['stderr'].strip()[:500]}"
    _mark_read(ctx, path)
    return "Wrote file successfully."


@context_tool
async def edit(ctx: ChatUpdateContext, filePath: str, oldString: str,
               newString: str, replaceAll: bool | None = None) -> str:
    path = _resolve_path(filePath)
    error = _ensure_read(ctx, path, "edit")
    if error:
        return error
    if oldString == newString:
        return "Error: No changes to apply: oldString and newString are identical."
    payload = await _run_helper(
        ctx.user_id, "toolbot_edit.py", [],
        {"path": path, "old_string": oldString, "new_string": newString,
         "replace_all": bool(replaceAll)})
    if not payload.get("ok"):
        return f"Error: {payload.get('error', 'edit failed')}"
    _mark_read(ctx, path)
    return "Edit applied successfully."


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


@context_tool
async def grep(ctx: ChatUpdateContext, pattern: str,
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


@context_tool
async def glob(ctx: ChatUpdateContext, pattern: str,
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


@context_tool
async def apply_patch(ctx: ChatUpdateContext, patchText: str) -> str:
    result = await manager.exec(
        ctx.user_id, "apply-patch-py", stdin_data=patchText.encode())
    if result["timeout"]:
        return "Error: apply_patch timed out."
    if result["returncode"] != 0:
        return f"Error: {result['stderr'].strip()[:1000]}"
    summary = result["stdout"].strip()
    for line in summary.splitlines():
        if line.startswith(("A ", "M ")):
            _mark_read(ctx, _resolve_path(line[2:].strip()))
    return summary or "Success. Updated the following files:"


MAX_SEND_FILE_BYTES = 45 * 1024 * 1024

MAX_IMAGE_DISPLAY_BYTES = 5 * 1024 * 1024

IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@context_tool
async def read_image(ctx: ChatUpdateContext, filePath: str) -> str | ToolCallResult:
    path = _resolve_path(filePath)
    mime = IMAGE_MIME_TYPES.get(posixpath.splitext(path)[1].lower())
    if mime is None:
        return ("Error: not a supported image type. Supported formats: "
                "png, jpg, jpeg, gif, webp.")
    stat = await manager.exec(
        ctx.user_id, f"stat -c %s {shlex.quote(path)}")
    if stat["returncode"] != 0:
        return f"Error: File not found: {path}"
    size = int(stat["stdout"].strip() or 0)
    if size > MAX_IMAGE_DISPLAY_BYTES:
        return (f"Error: image is too large to display "
                f"({size:,} bytes; limit is {MAX_IMAGE_DISPLAY_BYTES:,}).")

    encoded = await manager.exec(
        ctx.user_id, f"base64 -w0 {shlex.quote(path)}")
    if encoded["returncode"] != 0:
        return f"Error: {encoded['stderr'].strip()[:500]}"
    try:
        data = base64.b64decode(encoded["stdout"].strip())
    except ValueError:
        return "Error: could not read the image contents."

    if not path.startswith(f"{UPLOADS_DIR}/"):
        try:
            await ctx.bot.send_photo(ctx.chat_id, photo=data)
        except Exception as e:
            logger.warning("Could not send image to chat: %s", e)

    return ToolCallResult(
        text=(f"Image shown: {path} ({len(data):,} bytes). "
              "The image is attached to this result."),
        image=ImageAttachment(data=data, mime=mime),
    )


@context_tool
async def send_file(ctx: ChatUpdateContext, filePath: str) -> str:
    path = _resolve_path(filePath)
    stat = await manager.exec(
        ctx.user_id, f"stat -c %s {shlex.quote(path)}")
    if stat["returncode"] != 0:
        return f"Error: File not found: {path}"
    size = int(stat["stdout"].strip() or 0)
    if size > MAX_SEND_FILE_BYTES:
        return (f"Error: file is too large to send "
                f"({size:,} bytes; limit is {MAX_SEND_FILE_BYTES:,}).")
    encoded = await manager.exec(
        ctx.user_id, f"base64 -w0 {shlex.quote(path)}")
    if encoded["returncode"] != 0:
        return f"Error: {encoded['stderr'].strip()[:500]}"
    try:
        data = base64.b64decode(encoded["stdout"].strip())
    except ValueError:
        return "Error: could not decode the file contents."
    filename = posixpath.basename(path)
    await ctx.bot.send_document(ctx.chat_id, document=data, filename=filename)
    return f"Sent {filename} ({len(data):,} bytes)."


BASH = "bash"
READ = "read"
WRITE = "write"
EDIT = "edit"
GREP = "grep"
GLOB = "glob"
APPLY_PATCH = "apply_patch"
SEND_FILE = "send_file"
READ_IMAGE = "read_image"


def register(tools_list: list[dict], tool_map: dict):
    tools_list.extend(TOOL_DEFINITIONS)
    tool_map[BASH] = bash
    tool_map[READ] = read
    tool_map[WRITE] = write
    tool_map[EDIT] = edit
    tool_map[GREP] = grep
    tool_map[GLOB] = glob
    tool_map[APPLY_PATCH] = apply_patch
    tool_map[SEND_FILE] = send_file
    tool_map[READ_IMAGE] = read_image
