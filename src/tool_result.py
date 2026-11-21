"""Structured tool results.

Tools normally return a plain string. Tools that need to attach binary
payloads (e.g. images for multi-modal models) return a ToolCallResult
instead; run_llm turns it into a string tool message plus a
follow-up user message carrying the attachment.

Both are frozen: a tool builds its result once and hands it on, and
nothing downstream edits it.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageAttachment:
    data: bytes
    mime: str


@dataclass(frozen=True)
class ToolCallResult:
    text: str
    image: ImageAttachment | None = None
