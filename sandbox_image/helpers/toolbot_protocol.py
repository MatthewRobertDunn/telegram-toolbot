"""Shared plumbing for the sandbox tool helpers.

Every helper speaks one protocol: it prints a single JSON line to stdout
and exits 0 only on success.

    {"v": 1, "ok": true, "data": {...}}    exit 0
    {"v": 1, "ok": false, "error": "..."}  exit 1

The version lets the bot-side runner (tools/sandbox/common.py) reject a
response it does not understand — from a helper older or newer than the
bot — with a clear error instead of misparsing whatever shape the other
side happened to produce.
"""

import json

PROTOCOL_VERSION = 1


def emit(ok: bool, data: object = None, error: str = "") -> int:
    """Print the protocol response and return the process exit code."""
    payload: dict[str, object] = {"v": PROTOCOL_VERSION, "ok": ok}
    if ok:
        payload["data"] = data if data is not None else {}
    else:
        payload["error"] = error or "helper failed"
    print(json.dumps(payload))
    return 0 if ok else 1
