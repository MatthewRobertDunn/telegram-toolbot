"""Glob tool backend for the toolbot sandbox.

Finds files by glob pattern using ripgrep (respects .gitignore), sorted by
modification time (newest first). Prints a JSON result to stdout:
{"ok": true, "data": {"files": [...], "truncated": ...}} or
{"ok": false, "error": "..."}.
"""
import argparse
import json
import os
import subprocess
import sys

LIMIT = 100


def _rg_files(path: str, pattern: str) -> list[str]:
    try:
        proc = subprocess.run(
            ["rg", "--files", path, "-g", pattern],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"ripgrep failed: {e}")
    if proc.returncode not in (0, 1):
        raise RuntimeError(proc.stderr.strip() or "ripgrep failed")
    return [line for line in proc.stdout.splitlines() if line]


def find_files(path: str, pattern: str) -> dict:
    files = _rg_files(path, pattern)
    entries = []
    for file in files:
        try:
            mtime = os.path.getmtime(file)
        except OSError:
            mtime = 0.0
        entries.append((mtime, file))
    entries.sort(key=lambda entry: entry[0], reverse=True)
    return {
        "files": [os.path.abspath(file) for _, file in entries[:LIMIT]],
        "truncated": len(entries) > LIMIT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--path", default=".")
    args = parser.parse_args()

    try:
        if not os.path.isdir(args.path):
            raise ValueError(f"glob path must be a directory: {args.path}")
        print(json.dumps({"ok": True, "data": find_files(args.path, args.pattern)}))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))


if __name__ == "__main__":
    sys.exit(main())
