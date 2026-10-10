import base64
import json
import posixpath
import shlex

from chat_update_context import ChatUpdateContext, context_tool
from sandbox import manager

WORKSPACE_ROOT = "/workspace"
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
    result = await manager.exec_in_sandbox(user_id, command,
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
    result = await manager.exec_in_sandbox(ctx.user_id, command)
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
    check = await manager.exec_in_sandbox(
        ctx.user_id, f"test -e {shlex.quote(path)}")
    if check["returncode"] == 0:
        error = _ensure_read(ctx, path, "write")
        if error:
            return error
    parent = posixpath.dirname(path)
    command = (f"mkdir -p {shlex.quote(parent)} && "
               f"cat > {shlex.quote(path)}")
    result = await manager.exec_in_sandbox(ctx.user_id, command,
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
    result = await manager.exec_in_sandbox(ctx.user_id, command)
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
    result = await manager.exec_in_sandbox(
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


@context_tool
async def send_file(ctx: ChatUpdateContext, filePath: str) -> str:
    path = _resolve_path(filePath)
    stat = await manager.exec_in_sandbox(
        ctx.user_id, f"stat -c %s {shlex.quote(path)}")
    if stat["returncode"] != 0:
        return f"Error: File not found: {path}"
    size = int(stat["stdout"].strip() or 0)
    if size > MAX_SEND_FILE_BYTES:
        return (f"Error: file is too large to send "
                f"({size:,} bytes; limit is {MAX_SEND_FILE_BYTES:,}).")
    encoded = await manager.exec_in_sandbox(
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

SANDBOX_COMMON = (
    "Targets the user's personal sandbox (Ubuntu, persistent between "
    "calls, network access, root privileges). Files the user uploads "
    "arrive in /workspace/uploads/. Paths are relative to the "
    "sandbox workspace root /workspace, or absolute. All sandbox state is "
    "permanently lost when the sandbox is destroyed, which happens one hour "
    "after the last command executed on it."
)

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": BASH,
            "strict": False,
            "description": (
                "Execute a bash shell command in the user's sandbox and "
                "return the output. The sandbox persists between calls: files "
                "created and packages installed remain available in later "
                f"calls. Commands run from the workspace root {WORKSPACE_ROOT}. "
                "The sandbox has full network access and root privileges, so "
                "software can be installed, e.g. 'apt-get update && "
                "apt-get install -y <package>'. Prefer the dedicated read, "
                "write, edit, grep and glob tools for file work; use bash "
                "for everything else (git, npm, running code, etc.). All "
                "state is permanently lost when the sandbox is destroyed, which "
                "happens one hour after the last command executed on it. "
                "Long output is truncated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute."
                    }
                },
                "required": ["command"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": READ,
            "strict": False,
            "description": (
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
                "need more context, read a larger window. "
                + SANDBOX_COMMON
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "type": "string",
                        "description": "The path to the file or directory to read."
                    },
                    "offset": {
                        "type": "number",
                        "description": "The line number to start reading from (1-indexed)."
                    },
                    "limit": {
                        "type": "number",
                        "description": "The maximum number of lines to read (defaults to 2000)."
                    }
                },
                "required": ["filePath"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": WRITE,
            "strict": False,
            "description": (
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
                "explicitly requested by the User. "
                + SANDBOX_COMMON
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "type": "string",
                        "description": "The path to the file to write."
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file."
                    }
                },
                "required": ["filePath", "content"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": EDIT,
            "strict": False,
            "description": (
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
                "complex multi-file changes, prefer the apply_patch tool. "
                + SANDBOX_COMMON
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "type": "string",
                        "description": "The path to the file to modify"
                    },
                    "oldString": {
                        "type": "string",
                        "description": "The text to replace"
                    },
                    "newString": {
                        "type": "string",
                        "description": "The text to replace it with (must be different from oldString)"
                    },
                    "replaceAll": {
                        "type": "boolean",
                        "description": "Replace all occurrences of oldString (default false)"
                    }
                },
                "required": ["filePath", "oldString", "newString"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": GREP,
            "strict": False,
            "description": (
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
                "directly. Do NOT use `grep`. "
                + SANDBOX_COMMON
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regex pattern to search for in file contents"
                    },
                    "path": {
                        "type": "string",
                        "description": "The directory to search in. Defaults to the current working directory."
                    },
                    "include": {
                        "type": "string",
                        "description": "File pattern to include in the search (e.g. \"*.js\", \"*.{ts,tsx}\")"
                    }
                },
                "required": ["pattern"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": GLOB,
            "strict": False,
            "description": (
                "Fast file pattern matching tool that works with any "
                "codebase size, powered by ripgrep in the sandbox. "
                "Returns matching file paths sorted by modification time "
                "(newest first). Usage:\n"
                "- Supports glob patterns like \"**/*.js\" or "
                "\"src/**/*.ts\"\n"
                "- Use this tool when you need to find files by name "
                "patterns\n"
                "- It is always better to speculatively perform multiple "
                "searches as a batch that are potentially useful. "
                + SANDBOX_COMMON
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The glob pattern to match files against"
                    },
                    "path": {
                        "type": "string",
                        "description": "The directory to search in. If not specified, the workspace root will be used."
                    }
                },
                "required": ["pattern"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": APPLY_PATCH,
            "strict": False,
            "description": (
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
                f"{WORKSPACE_ROOT} and must stay inside it. "
                + SANDBOX_COMMON
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patchText": {
                        "type": "string",
                        "description": "The full patch text that describes all changes to be made"
                    }
                },
                "required": ["patchText"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": SEND_FILE,
            "strict": False,
            "description": (
                "Send a file from the sandbox to the user as a Telegram "
                "document. Usage:\n"
                "- The file can be anywhere in the sandbox; paths are "
                "relative to the sandbox workspace root /workspace, or "
                "absolute.\n"
                "- Use this to share files generated in the sandbox with "
                "the user (reports, data files, images, binaries, etc.).\n"
                "- Files larger than 45 MB cannot be sent. "
                + SANDBOX_COMMON
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "type": "string",
                        "description": "The path to the file to send."
                    }
                },
                "required": ["filePath"],
                "additionalProperties": False
            }
        }
    },
]


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
