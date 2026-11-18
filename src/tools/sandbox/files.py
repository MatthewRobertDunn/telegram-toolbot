"""Reading and changing files in the sandbox."""

import posixpath
import shlex

from chat_update_context import AgentContext
from sandbox import manager
from tool_registry import BOTH, agent_tool

from .common import _ensure_read, _mark_read, _resolve_path, _run_helper


@agent_tool(
    scope=BOTH,
    description=(
        "Read a file or directory in sandbox. Errors if missing (possibly "
        "with similar-file suggestions). Usage:\n"
        "- Returns up to 2000 lines from start by default.\n"
        "- offset is 1-indexed start line; re-call with larger offset for "
        "later sections.\n"
        "- Use grep for content in large/long-line files; glob to find paths "
        "when unsure.\n"
        "- Files return lines as `<line>: <content>`; directories list one "
        "entry per line (`/` suffix for subdirs).\n"
        "- Lines over 2000 chars are truncated.\n"
        "- Read known multiple files in parallel.\n"
        "- Prefer larger windows over tiny 30-line slices."),
    parameters={
        "filePath": {
            "type": "string",
            "description": "Path to file or directory to read.",
        },
        "offset": {
            "type": "number",
            "description": ("1-indexed line number to start from."),
        },
        "limit": {
            "type": "number",
            "description": ("Max lines to read (default 2000)."),
        },
    })
async def read(ctx: AgentContext, filePath: str,
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


@agent_tool(
    scope=BOTH,
    description=(
        "Write (create/overwrite) a file in sandbox. Usage:\n"
        "- Overwrites existing files.\n"
        "- For existing files, you MUST read first; otherwise fails.\n"
        "- Parent directories are auto-created.\n"
        "- Prefer edit for existing files. Never write new files unless "
        "required.\n"
        "- Never create *.md/README docs unless user requested."),
    parameters={
        "filePath": {
            "type": "string",
            "description": "Path of file to write.",
        },
        "content": {
            "type": "string",
            "description": "Content to write.",
        },
    })
async def write(ctx: AgentContext, filePath: str, content: str) -> str:
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


@agent_tool(
    scope=BOTH,
    description=(
        "Exact string replacement in sandbox files. Usage:\n"
        "- Must read file first; errors otherwise.\n"
        "- Match exact indentation from read output (after `N: ` prefix); "
        "never include line-number prefix in oldString/newString.\n"
        "- Prefer editing over writing new files; no emojis unless "
        "requested.\n"
        "- Fails if oldString missing or matches multiple times; add context "
        "to disambiguate or use replaceAll.\n"
        "- Use replaceAll for renames across file; prefer patch for "
        "large/multi-file changes."),
    parameters={
        "filePath": {
            "type": "string",
            "description": "Path of file to modify",
        },
        "oldString": {
            "type": "string",
            "description": "Text to replace",
        },
        "newString": {
            "type": "string",
            "description": ("Replacement text (must differ from oldString)"),
        },
        "replaceAll": {
            "type": "boolean",
            "description": ("Replace all matches (default false)"),
        },
    })
async def edit(ctx: AgentContext, filePath: str, oldString: str,
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


@agent_tool(
    scope=BOTH,
    description=(
        "Apply file patches in sandbox for diffs and multi-file changes. "
        "Patch format:\n"
        "\n"
        "*** Begin Patch\n"
        "[ one or more file sections ]\n"
        "*** End Patch\n"
        "\n"
        "Each file section starts with one of:\n"
        "- `*** Add File: <path>` - create new file. Every following line "
        "is a + line.\n"
        "- `*** Delete File: <path>` - remove file. Nothing follows.\n"
        "- `*** Update File: <path>` - patch existing file (optionally with "
        "`*** Move to: <new_path>`).\n"
        "\n"
        "Update sections use `@@` hunks (optionally with context anchor). "
        "Within a hunk, `+` adds, `-` removes, leading space is context. "
        "Example:\n"
        "\n"
        "*** Begin Patch\n"
        "*** Add File: hello.txt\n"
        "+Hello world\n"
        "*** Update File: src/app.py\n"
        "@@ def greet():\n"
        "-print(\"Hi\")\n"
        "+print(\"Hello, world!\")\n"
        "*** Delete File: obsolete.txt\n"
        "*** End Patch\n"
        "\n"
        "Rules:\n"
        "- Include Add/Delete/Update header\n"
        "- Prefix new lines with `+` even for new files\n"
        "- Paths relative to /workspace, must stay inside it."),
    parameters={
        "patchText": {
            "type": "string",
            "description": ("Full patch text describing all changes"),
        },
    })
async def patch(ctx: AgentContext, patchText: str) -> str:
    result = await manager.exec(
        ctx.user_id, "apply-patch-py", stdin_data=patchText.encode())
    if result["timeout"]:
        return "Error: patch timed out."
    if result["returncode"] != 0:
        return f"Error: {result['stderr'].strip()[:1000]}"
    summary = result["stdout"].strip()
    for line in summary.splitlines():
        if line.startswith(("A ", "M ")):
            _mark_read(ctx, _resolve_path(line[2:].strip()))
    return summary or "Success. Updated the following files:"
