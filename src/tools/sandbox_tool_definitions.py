"""Tool definitions for the sandbox tools, shared via sandbox_tool.py."""

WORKSPACE_ROOT = "/workspace"

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "strict": False,
            "description": (
                "Execute a bash shell command in the user's sandbox and "
                "return the output. The sandbox persists between calls: files "
                "created and packages installed remain available in later "
                "calls. Commands always run from the workspace root /workspace. "
                "You don't need to prefix commands with `cd /workspace &&`."
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
            "name": "read",
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
                "need more context, read a larger window."
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
            "name": "write",
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
                "explicitly requested by the User."
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
            "name": "edit",
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
                "complex multi-file changes, prefer the patch tool."
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
            "name": "grep",
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
                "directly. Do NOT use `grep`."
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
            "name": "glob",
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
                "searches as a batch that are potentially useful."
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
            "name": "patch",
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
                "/workspace and must stay inside it."
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
            "name": "send_file",
            "strict": False,
            "description": (
                "Send a file from the sandbox to the user as a Telegram "
                "document. Usage:\n"
                "- The file can be anywhere in the sandbox; paths are "
                "relative to the sandbox workspace root /workspace, or "
                "absolute.\n"
                "- Use this to share files generated in the sandbox with "
                "the user (reports, data files, images, binaries, etc.).\n"
                "- Files larger than 45 MB cannot be sent."
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
    {
        "type": "function",
        "function": {
            "name": "read_image",
            "strict": False,
            "description": (
                "Read an image file (png, jpg, jpeg, gif, webp) from the "
                "sandbox and attach it to the conversation so you can see "
                "it. Requires a vision-capable model. The image is also "
                "sent to the user's chat, unless send_to_chat is false. "
                "Use this instead of the read tool for image "
                "files; the read tool rejects binary files. Images larger "
                "than 5 MB cannot be displayed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "type": "string",
                        "description": "The path to the image file to view."
                    },
                    "send_to_chat": {
                        "type": "boolean",
                        "description": ("Whether to also send the image to the user in the chat. "
                                        "Defaults to true. Set to false to inspect the image "
                                        "yourself without showing it to the user.")
                    }
                },
                "required": ["filePath"],
                "additionalProperties": False
            }
        }
    },
]
