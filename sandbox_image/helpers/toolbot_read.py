"""Read tool backend for the toolbot sandbox.

Reads a file (with line numbers, offset/limit, truncation caps, binary
detection) or lists a directory. Prints a JSON result to stdout:
{"ok": true, "data": {...}} or {"ok": false, "error": "..."}.
"""
import argparse
import json
import os
import sys

DEFAULT_READ_LIMIT = 2000
MAX_LINE_LENGTH = 2000
MAX_LINE_SUFFIX = f"... (line truncated to {MAX_LINE_LENGTH} chars)"
MAX_BYTES = 50 * 1024
SAMPLE_BYTES = 4096

BINARY_EXTENSIONS = {
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".exe", ".dll", ".so",
    ".class", ".jar", ".war", ".7z", ".doc", ".docx", ".xls", ".xlsx",
    ".ppt", ".pptx", ".odt", ".ods", ".odp", ".bin", ".dat", ".obj", ".o",
    ".a", ".lib", ".wasm", ".pyc", ".pyo", ".pdf", ".png", ".jpg", ".jpeg",
    ".gif", ".webp", ".ico", ".mp3", ".mp4", ".avi", ".mov", ".sqlite",
    ".db", ".whl", ".ttf", ".otf",
}


def is_binary_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    if ext in BINARY_EXTENSIONS:
        return True
    try:
        with open(path, "rb") as f:
            sample = f.read(SAMPLE_BYTES)
    except OSError:
        return False
    if not sample:
        return False
    non_printable = 0
    for b in sample:
        if b == 0:
            return True
        if b < 9 or (b > 13 and b < 32):
            non_printable += 1
    return non_printable / len(sample) > 0.3


def strip_eol(raw: str) -> str:
    if raw.endswith("\n"):
        raw = raw[:-1]
    if raw.endswith("\r"):
        raw = raw[:-1]
    return raw


def read_file_lines(path: str, offset: int, limit: int) -> dict:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    count = 0
    lines = []
    bytes_used = 0
    cut = False
    more = False
    start = offset - 1

    for raw in raw_lines:
        count += 1
        if count <= start:
            continue
        if len(lines) >= limit:
            more = True
            continue
        text = strip_eol(raw)
        display = (text[:MAX_LINE_LENGTH] + MAX_LINE_SUFFIX
                   if len(text) > MAX_LINE_LENGTH else text)
        size = len(display.encode("utf-8")) + (1 if lines else 0)
        if bytes_used + size <= MAX_BYTES:
            lines.append(display)
            bytes_used += size
        else:
            cut = True
            more = True
            break

    if count < offset and not (count == 0 and offset == 1):
        raise ValueError(
            f"Offset {offset} is out of range for this file ({count} lines)")

    return {
        "type": "file",
        "path": path,
        "lines": [{"n": i + offset, "text": t} for i, t in enumerate(lines)],
        "lineStart": offset,
        "lineEnd": offset + len(lines) - 1,
        "totalLines": count,
        "more": more,
        "cut": cut,
    }


def list_directory(path: str, offset: int, limit: int) -> dict:
    entries = []
    for name in sorted(os.listdir(path)):
        if os.path.isdir(os.path.join(path, name)):
            entries.append(name + "/")
        else:
            entries.append(name)
    start = offset - 1
    sliced = entries[start:start + limit]
    return {
        "type": "directory",
        "path": path,
        "entries": sliced,
        "offset": offset,
        "totalEntries": len(entries),
        "truncated": start + len(sliced) < len(entries),
    }


def similar_files(path: str) -> list[str]:
    directory = os.path.dirname(path) or "."
    base = os.path.basename(path).lower()
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    matches = [os.path.join(directory, name) for name in names
               if base in name.lower() or name.lower() in base]
    return matches[:3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    try:
        path = args.path
        if not os.path.exists(path):
            suggestions = similar_files(path)
            if suggestions:
                raise ValueError(
                    f"File not found: {path}\n\nDid you mean one of these?\n"
                    + "\n".join(suggestions))
            raise ValueError(f"File not found: {path}")
        offset = args.offset or 1
        limit = args.limit or DEFAULT_READ_LIMIT
        if os.path.isdir(path):
            data = list_directory(path, offset, limit)
        else:
            if is_binary_file(path):
                raise ValueError(f"Cannot read binary file: {path}")
            data = read_file_lines(path, offset, limit)
        print(json.dumps({"ok": True, "data": data}))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))


if __name__ == "__main__":
    sys.exit(main())
