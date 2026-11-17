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
        "Read a file or directory in the sandbox. If the path "
        "does not exist, an error is returned (possibly with "
        "similar file suggestions). Usage:\n"
        "- By default, this tool returns up to 2000 lines from the "
        "start of the file.\n"
        "- The offset parameter is the line number to start from "
        "(1-indexed).\n"
        "- To read later sections, call this tool again with a "
        "larger offset.\n"
        "- Use the grep tool to find specific content in large "
        "files or files with long lines.\n"
        "- If you are unsure of the correct file path, use the "
        "glob tool to look up filenames by glob pattern.\n"
        "- Contents are returned with each line prefixed by its "
        "line number as `<line>: <content>`. For directories, "
        "entries are returned one per line (without line numbers) "
        "with a trailing `/` for subdirectories.\n"
        "- Any line longer than 2000 characters is truncated.\n"
        "- Call this tool in parallel when you know there are "
        "multiple files you want to read.\n"
        "- Avoid tiny repeated slices (30 line chunks). If you "
        "need more context, read a larger window."),
    parameters={
        "filePath": {
            "type": "string",
            "description": "The path to the file or directory to read.",
        },
        "offset": {
            "type": "number",
            "description": ("The line number to start reading from "
                            "(1-indexed)."),
        },
        "limit": {
            "type": "number",
            "description": ("The maximum number of lines to read "
                            "(defaults to 2000)."),
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
        "Write a file in the sandbox, creating it or "
        "overwriting an existing one. Usage:\n"
        "- This tool will overwrite the existing file if there is "
        "one at the provided path.\n"
        "- If this is an existing file, you MUST use the read tool "
        "first to read the file's contents. This tool will fail if "
        "you did not read the file first.\n"
        "- Parent directories are created automatically.\n"
        "- ALWAYS prefer editing existing files with the edit "
        "tool. NEVER write new files unless explicitly required.\n"
        "- NEVER proactively create documentation files (*.md) or "
        "README files. Only create documentation files if "
        "explicitly requested by the User."),
    parameters={
        "filePath": {
            "type": "string",
            "description": "The path to the file to write.",
        },
        "content": {
            "type": "string",
            "description": "The content to write to the file.",
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
        "Performs exact string replacements in files in the "
        "sandbox. Usage:\n"
        "- You must use your read tool at least once in the "
        "conversation before editing. This tool will error if you "
        "attempt an edit without reading the file.\n"
        "- When editing text from read tool output, ensure you "
        "preserve the exact indentation (tabs/spaces) as it "
        "appears AFTER the line number prefix. The line number "
        "prefix format is: line number + colon + space (e.g., "
        "`1: `). Everything after that space is the actual file "
        "content to match. Never include any part of the line "
        "number prefix in the oldString or newString.\n"
        "- ALWAYS prefer editing existing files in the codebase. "
        "NEVER write new files unless explicitly required.\n"
        "- Only use emojis if the user explicitly requests it. "
        "Avoid adding emojis to files unless asked.\n"
        "- The edit will FAIL if `oldString` is not found in the "
        "file with an error \"oldString not found in content\".\n"
        "- The edit will FAIL if `oldString` is found multiple "
        "times in the file with an error \"Found multiple matches "
        "for oldString. Provide more surrounding lines in "
        "oldString to identify the correct match.\" Either provide "
        "a larger string with more surrounding context to make it "
        "unique or use `replaceAll` to change every instance of "
        "`oldString`.\n"
        "- Use `replaceAll` for replacing and renaming strings "
        "across the file. This parameter is useful if you want to "
        "rename a variable for instance. For larger or more "
        "complex multi-file changes, prefer the patch tool."),
    parameters={
        "filePath": {
            "type": "string",
            "description": "The path to the file to modify",
        },
        "oldString": {
            "type": "string",
            "description": "The text to replace",
        },
        "newString": {
            "type": "string",
            "description": ("The text to replace it with (must be different "
                            "from oldString)"),
        },
        "replaceAll": {
            "type": "boolean",
            "description": ("Replace all occurrences of oldString "
                            "(default false)"),
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
        "Apply patches to files in the sandbox. Useful for "
        "applying diffs and making multi-file changes. The patch "
        "language is a stripped-down, file-oriented diff format:\n"
        "\n"
        "*** Begin Patch\n"
        "[ one or more file sections ]\n"
        "*** End Patch\n"
        "\n"
        "Each file section starts with one of three headers:\n"
        "- `*** Add File: <path>` - create a new file. Every "
        "following line is a + line (the initial contents).\n"
        "- `*** Delete File: <path>` - remove an existing file. "
        "Nothing follows.\n"
        "- `*** Update File: <path>` - patch an existing file in "
        "place (optionally with a `*** Move to: <new_path>` line).\n"
        "\n"
        "Update sections contain hunks introduced by `@@` lines "
        "(optionally followed by a context anchor line). Within a "
        "hunk, lines starting with `+` are added, `-` are removed, "
        "and a leading space is context. Example:\n"
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
        "It is important to remember:\n"
        "- You must include a header with your intended action "
        "(Add/Delete/Update)\n"
        "- You must prefix new lines with `+` even when creating a "
        "new file\n"
        "- Paths are relative to the sandbox workspace root "
        "/workspace and must stay inside it."),
    parameters={
        "patchText": {
            "type": "string",
            "description": ("The full patch text that describes all changes "
                            "to be made"),
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
