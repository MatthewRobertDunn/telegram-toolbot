"""The transcript: conversation messages rendered as plain text.

Compaction summarizes and checkpoints conversations in this form (one
`message_to_text` line per message), so the format is part of the
summarizing prompt's contract: tests pin it, and a rendered transcript is
what the summarizing model actually reads. Changes here are deliberate,
never incidental refactors.

Rendering reads messages only through the messages accessors, so a plain
dict and a ConversationMessage render identically.
"""

from messages import (has_image, is_thinking, iter_content_text, iter_parts,
                      text_of, tool_calls_of)

# Text left behind where an image was pruned or rendered to text.
IMAGE_PLACEHOLDER = '[image omitted]'
# Long tool outputs are cut to this many characters in the transcript.
TOOL_OUTPUT_MAX_CHARS = 1_250


def message_to_text(message: dict) -> str:
    """One conversation message as a transcript line.

    Empty for messages the transcript leaves out (e.g. system prompts and
    kept thinking blocks - reasoning is ephemeral, the summary keeps
    outcomes). Images become placeholder text and long tool outputs are
    truncated.
    """
    if is_thinking(message):
        return ''
    role = message.get('role')
    if role == 'system':
        return ''
    if role == 'user':
        content = message.get('content')
        if isinstance(content, str):
            return f'[User]: {content}'
        if not has_image(message):
            # Text-only: one line per text part, no placeholder to weave in.
            return '\n'.join(f'[User]: {text}'
                             for text in iter_content_text(message))
        lines = []
        for part in iter_parts(message):
            if part.get('type') == 'text':
                lines.append(f"[User]: {part.get('text') or ''}")
            elif part.get('type') == 'image_url':
                lines.append(IMAGE_PLACEHOLDER)
        return '\n'.join(lines)
    if role == 'assistant':
        lines = []
        text = text_of(message)
        if text:
            lines.append(f'[Assistant]: {text}')
        for tool_call in tool_calls_of(message):
            function = (tool_call or {}).get('function') or {}
            name = function.get('name') or ''
            arguments = function.get('arguments') or ''
            lines.append(f'[Assistant tool call]: {name}({arguments})')
        return '\n'.join(lines)
    if role == 'tool':
        content = message.get('content') or ''
        return f'[Tool result]: {_truncate_tool_output(content)}'
    return ''


def _truncate_tool_output(value: str) -> str:
    if len(value) <= TOOL_OUTPUT_MAX_CHARS:
        return value
    return value[:TOOL_OUTPUT_MAX_CHARS] + '\n[truncated]'
