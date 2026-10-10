"""Structured tool results.

Tools normally return a plain string. Tools that need to attach binary
payloads (e.g. images for multi-modal models) return a ToolCallResult
instead; run_with_tools turns it into a string tool message plus a
follow-up user message carrying the attachment.
"""
from dataclasses import dataclass


@dataclass
class ImageAttachment:
    data: bytes
    mime: str


@dataclass
class ToolCallResult:
    text: str
    image: ImageAttachment | None = None
