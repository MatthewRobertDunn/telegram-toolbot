"""Grep tool backend for the toolbot sandbox.

Searches file contents with ripgrep (respects .gitignore), reading its
JSON event stream so paths, line numbers and text stay separated. Stops
as soon as the match cap is exceeded, so a huge result set is neither
buffered whole nor cut mid-event: the caller gets the matches up to the
cap plus {"truncated": true}. Prints the shared helper protocol
(see toolbot_protocol.py).
"""
import argparse
import json
import os
import subprocess
import sys
from typing import Any

from toolbot_protocol import emit

LIMIT = 100
MAX_LINE_CHARS = 500


def _start_rg(path: str, pattern: str,
              include: str | None) -> subprocess.Popen:
    """Spawn ripgrep's JSON event stream for the search.

    stderr joins stdout: rg's warnings (binary file matches, unreadable
    files) would otherwise deadlock an unread pipe, and the parser
    tolerates non-JSON lines anyway. Typed as Any so tests can substitute
    a fake stream.
    """
    command = ["rg", "--json", "-e", pattern]
    if include:
        command += ["-g", include]
    command.append(path)
    return subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace")


def _parse_match(event: dict) -> dict | None:
    """Extract one match record from an rg --json event, or None."""
    if event.get("type") != "match":
        return None
    data = event.get("data", {})
    text = ((data.get("lines") or {}).get("text") or "").rstrip("\n")
    if len(text) > MAX_LINE_CHARS:
        text = text[:MAX_LINE_CHARS] + "... (line truncated)"
    return {
        # Absolute: the bot formats paths without knowing the exec cwd.
        "path": os.path.abspath((data.get("path") or {}).get("text", "")),
        "line": data.get("line_number"),
        "text": text,
    }


def search(path: str, pattern: str, include: str | None) -> dict:
    """Run the search, capping matches and reporting truncation exactly."""
    proc = _start_rg(path, pattern, include)
    if proc.stdout is None:  # pragma: no cover - PIPE is always set
        raise RuntimeError("ripgrep produced no output stream")
    matches: list[dict] = []
    total = 0
    notes: list[str] = []
    for line in proc.stdout:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = None
        if not isinstance(event, dict):
            if line.strip() and len(notes) < 5:
                notes.append(line.strip()[:200])
            continue
        match = _parse_match(event)
        if match is None:
            continue
        total += 1
        if len(matches) < LIMIT:
            matches.append(match)
        if total > LIMIT:
            break  # Enough to know it truncated; do not buffer the rest.
    proc.kill()
    proc.wait()
    if total <= LIMIT and proc.returncode not in (0, 1):
        # rg only fails loudly after we read it to the end.
        raise RuntimeError(
            "\n".join(notes) or
            f"ripgrep failed with exit code {proc.returncode}")
    return {
        "matches": matches,
        "totalMatches": total,
        "truncated": total > LIMIT,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--path", default=".")
    parser.add_argument("--include", default=None)
    args = parser.parse_args()

    try:
        if not os.path.isdir(args.path):
            raise ValueError(f"grep path must be a directory: {args.path}")
        return emit(True, search(args.path, args.pattern, args.include))
    except Exception as e:
        return emit(False, error=str(e))


if __name__ == "__main__":
    sys.exit(main())
