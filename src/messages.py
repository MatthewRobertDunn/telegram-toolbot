"""Accessors over the raw OpenAI-style message dicts.

A message is a plain dict everywhere it is stored and serialized, so every
read of its local-only metadata (the ``custom`` key) or of its content
shapes (plain string vs multimodal parts vs tool calls) is otherwise a
hand-rolled walk repeated at each call site. This module is the one place
those walks live: ``ConversationMessage`` (conversation.py) is a thin dict
subclass delegating to these functions, and the token estimator (tokens.py),
the transcript renderer (transcript.py) and the handlers read messages only
through them.

The metadata vocabulary lives here too: the ``custom`` key name and
``ConversationMessageType``, the tag recording what produced a message.
"""

import enum

# Where local-only metadata (e.g. Telegram ids) lives on a raw message
# dict; never sent to the endpoint (see Conversation.to_openai()).
CUSTOM = 'custom'


class ConversationMessageType(enum.Enum):
    """What produced a message, recorded in its custom metadata so
    commands can re-run the right flow over it (e.g. /regenerate redoing
    a /impersonation turn instead of answering as usual). Stripped by
    to_openai() like the rest of the metadata."""
    IMPERSONATION = 'impersonation'
    # A turn injected from an agent inbox (subagent report, parent
    # steering): model-visible as a user turn, but /undo skips it when
    # looking for the human exchange to remove.
    AGENT_MESSAGE = 'agent_message'
    # Model thinking kept for display: role=assistant in the history so
    # /undo and /regenerate treat it like any assistant turn, but never
    # sent to the endpoint (see to_openai()).
    THINKING = 'thinking'


def custom_of(message: dict) -> dict:
    """The message's custom-metadata dict, or {} when absent."""
    return message.get(CUSTOM) or {}


def message_type_of(message: dict) -> ConversationMessageType | None:
    """What produced the message (see ConversationMessageType), or None
    when untagged. Works for plain dicts and ConversationMessage alike.

    An unknown persisted tag (a hand-edited or older history) reads as
    None - the tag is not coerced to a known type - so comparisons like
    "is this the impersonation turn?" fail safe the way the old string
    comparison did, instead of failing to load the history."""
    raw = custom_of(message).get('message_type')
    if not raw:
        return None
    try:
        return ConversationMessageType(raw)
    except ValueError:
        return None


def set_message_type(message: dict,
                     value: ConversationMessageType | None) -> None:
    """Tag (or, with None, untag) what produced a message. Untagging
    leaves no empty `custom` container behind."""
    if value is None:
        custom = message.get(CUSTOM)
        if custom is not None:
            custom.pop('message_type', None)
            if not custom:
                del message[CUSTOM]
        return
    message.setdefault(CUSTOM, {})['message_type'] = value.value


def telegram_ids_of(message: dict) -> list:
    """The Telegram message ids linked to the message, or []."""
    return custom_of(message).get('telegram_message_ids') or []


def add_telegram_ids(message: dict,
                     message_ids: int | list[int] | None) -> None:
    """Link Telegram messages to this message: a single id or a list of
    them (reply chunks, tool call announcements). Nothing is linked for
    no ids; everything is stored as a list."""
    if not message_ids:
        return
    if isinstance(message_ids, int):
        message_ids = [message_ids]
    message.setdefault(CUSTOM, {}).setdefault(
        'telegram_message_ids', []).extend(message_ids)


def is_thinking(message: dict) -> bool:
    """Whether the message is a kept thinking block: tagged THINKING in
    its custom metadata. Tag-only on purpose (fail-closed): anything so
    tagged is display-only and never sent to the endpoint, whatever role
    it carries. Works for plain dicts and ConversationMessage alike.

    An unknown message_type value is treated as not-thinking: it can
    never be display-only, so it must reach the endpoint like any
    ordinary message. The unknown tag is not coerced to a known type -
    message_type_of reports it as None."""
    return message_type_of(message) is ConversationMessageType.THINKING


def tool_calls_of(message: dict) -> list:
    """The message's tool calls ([] for every other message)."""
    return message.get('tool_calls') or []


def tool_call_ids(message: dict) -> set:
    """Every tool call id the message references: the ids in an assistant
    message's tool_calls, or a tool result's tool_call_id."""
    if message.get('role') == 'tool':
        return {message.get('tool_call_id')} - {None}
    return {(call or {}).get('id')
            for call in tool_calls_of(message)} - {None}


def iter_parts(message: dict):
    """The dict parts of a multimodal content list, one at a time.

    Yields nothing for a plain string content; junk (non-dict) parts are
    skipped, mirroring every reader's tolerance for malformed content."""
    content = message.get('content')
    if not isinstance(content, list):
        return
    for part in content:
        if isinstance(part, dict):
            yield part


def iter_content_text(message: dict):
    """Yield the message's text pieces: a string content yields itself;
    a multimodal content yields each text part's text (image parts carry
    no text and yield nothing)."""
    content = message.get('content')
    if isinstance(content, str):
        yield content
        return
    for part in iter_parts(message):
        if part.get('type') == 'text':
            yield part.get('text') or ''


def text_of(message: dict) -> str:
    """The message's text: its string content, or the text parts joined
    with newlines. Empty for image-only messages."""
    return '\n'.join(iter_content_text(message))


def has_image(message: dict) -> bool:
    """Whether a multimodal content carries an inline image part."""
    return any(part.get('type') == 'image_url'
               for part in iter_parts(message))
