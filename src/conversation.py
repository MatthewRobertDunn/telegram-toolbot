"""The assistant/LLM conversation history object.

A Conversation owns the raw OpenAI-style message dicts and an identity: a
fresh uuid assigned at construction. Whoever creates a Conversation gets a
session id for the x-opencode-session header for free - no hashing of
contents, no id(...) bookkeeping.

The class also owns the small behaviors the message list needs everywhere
(append helpers, system-prompt head management, token estimation).
"""

import base64
import uuid
from dataclasses import dataclass, field

# Rough token estimate: about a quarter token per character.
TOKENS_PER_CHAR = 0.25
# What an inline image is assumed to cost before it is dropped.
IMAGE_TOKEN_ESTIMATE = 1_500
# Where local-only metadata (e.g. Telegram ids) lives on a raw message
# dict; never sent to the endpoint (see `to_openai()`).
CUSTOM = 'custom'
# Text left behind where an image was pruned.
IMAGE_PLACEHOLDER = '[image omitted]'
# How many of the most recent images a conversation holds onto.
MAX_IMAGES = 8


def _tool_call_ids(message: dict) -> set:
    """Every tool call id a message references: the ids in an assistant
    message's tool_calls, or a tool result's tool_call_id."""
    if message.get('role') == 'tool':
        return {message.get('tool_call_id')} - {None}
    return {((call or {}).get('id'))
            for call in message.get('tool_calls') or []} - {None}


def estimate_conversation_tokens(conversation) -> int:
    """Estimated tokens of a whole conversation, images included."""
    return sum(estimate_message_tokens(message) for message in conversation)


def token_count(text: str) -> int:
    """Rough token estimate for text: about a quarter token per character."""
    return int(len(text) * TOKENS_PER_CHAR)


def estimate_message_tokens(message: dict) -> int:
    total = 0
    content = message.get('content')
    if isinstance(content, str):
        total += token_count(content)
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get('type') == 'text':
                total += token_count(part.get('text') or '')
            elif part.get('type') == 'image_url':
                total += IMAGE_TOKEN_ESTIMATE
    for tool_call in message.get('tool_calls') or []:
        function = (tool_call or {}).get('function') or {}
        total += token_count(function.get('name') or '')
        total += token_count(function.get('arguments') or '')
    return total


class ConversationMessage(dict):
    """A conversation message: the OpenAI-style dict the endpoint sees,
    plus a helper for the local-only Telegram linkage. Plain dicts stay
    valid wherever a message is only read; `ConversationMessage` is what
    the add_ helpers hand back so callers can attach ids as replies are
    sent."""

    def add_telegram_message_id(self,
                                message_ids: int | list[int] | None) -> None:
        """Link Telegram messages to this conversation message: a single
        id or a list of them (reply chunks, tool call announcements).
        Nothing is linked for no ids; everything is stored as a list."""
        if not message_ids:
            return
        if isinstance(message_ids, int):
            message_ids = [message_ids]
        self.setdefault(CUSTOM, {}).setdefault(
            'telegram_message_ids', []).extend(message_ids)


@dataclass(eq=False)
class Conversation:
    """A model conversation: the message list plus a stable identity.

    Session tracking is implicit: creating a Conversation is what starts a
    new session (fresh uuid), and mutating `messages` - appending turns,
    rewriting the system head, compacting in place - keeps the same id.
    Compares equal to a plain list of the same messages so the old
    list-based call sites and tests keep working against `.messages` and
    against the object itself.
    """

    messages: list = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Cap on retained inline images. Unannotated, so it stays a plain
    # class attribute rather than a dataclass field: instances may
    # shrink it (tests do) without touching the constructor.
    max_images = MAX_IMAGES

    def add(self, role: str, content, **fields) -> ConversationMessage:
        """Append a message and return it; extra kwargs become message keys."""
        message = ConversationMessage({'role': role, 'content': content, **fields})
        self.messages.append(message)
        return message

    def add_user(self, content,
                 telegram_message_ids: int | list[int] | None = None,
                 **fields) -> ConversationMessage:
        message = self.add('user', content, **fields)
        message.add_telegram_message_id(telegram_message_ids)
        return message

    def add_assistant(self, content,
                      telegram_message_ids: int | list[int] | None = None,
                      **fields) -> ConversationMessage:
        message = self.add('assistant', content, **fields)
        message.add_telegram_message_id(telegram_message_ids)
        return message

    def add_tool(self, content, tool_call_id: str) -> dict:
        """A tool result message answering the tool call with that id."""
        return self.add('tool', content, tool_call_id=tool_call_id)

    def add_image(self, data: bytes, mime: str, caption: str = '') -> dict:
        """Append a user message carrying an inline base64 image.

        Keeps only the `max_images` most recent images: each addition
        first replaces the oldest ones with an '[image omitted]'
        placeholder merged into the message's text, so any text around
        the image - its caption, or a user's own words - survives.
        """
        self._prune_images(self.max_images - 1)
        b64 = base64.b64encode(data).decode()
        content = []
        if caption:
            content.append({'type': 'text', 'text': caption})
        content.append({'type': 'image_url',
                        'image_url': {'url': f'data:{mime};base64,{b64}'}})
        return self.add_user(content)

    def _prune_images(self, keep: int) -> None:
        """Replace the oldest image parts with placeholder text until
        only `keep` images remain.

        The placeholder merges into the message's preceding text part,
        so a pruned message stays a single text utterance.
        """
        image_parts = [
            (message, index)
            for message in self.messages
            if isinstance(message.get('content'), list)
            for index, part in enumerate(message['content'])
            if isinstance(part, dict) and part.get('type') == 'image_url'
        ]
        for message, index in reversed(
                image_parts[:max(len(image_parts) - keep, 0)]):
            previous = message['content'][index - 1] if index else None
            if isinstance(previous, dict) and previous.get('type') == 'text':
                text = previous.get('text') or ''
                previous['text'] = f'{text}\n{IMAGE_PLACEHOLDER}'
                del message['content'][index]
            else:
                message['content'][index] = {'type': 'text',
                                             'text': IMAGE_PLACEHOLDER}

    def remove_tool_calls(self, message: dict) -> list:
        """Remove every message referencing the tool calls `message`
        references - its own tool_calls, or the results answering its
        tool_call_id - and return the removed messages, so the caller
        can clean up anything attached to them (e.g. Telegram messages)
        and preview what went.

        Removal cascades: an assistant message removed for one id drags
        the results of its other parallel calls along, since a call left
        without its results (or vice versa) can't be replayed."""
        ids = _tool_call_ids(message)
        removed = []
        while True:
            batch = [message for message in self.messages
                     if _tool_call_ids(message) & ids
                     and not any(message is old for old in removed)]
            if not batch:
                break
            removed.extend(batch)
            for message in batch:
                ids |= _tool_call_ids(message)
        if removed:
            self.messages[:] = [message for message in self.messages
                                if not any(message is old for old in removed)]
        return removed

    def to_openai(self) -> list:
        """The message list as the endpoint should see it: `custom` local
        metadata (e.g. Telegram ids) removed, everything else passed through."""
        return [{key: value for key, value in message.items()
                 if key != CUSTOM}
                for message in self.messages]

    def set_system_head(self, prompt: str) -> dict:
        """Put this system prompt at position 0, creating the head if needed.

        Call in place every turn, so the conversation keeps its identity
        while its system prompt stays in sync with the current settings.
        """
        if not self.messages or self.messages[0].get('role') != 'system':
            self.messages.insert(0, {'role': 'system', 'content': prompt})
        else:
            self.messages[0] = {'role': 'system', 'content': prompt}
        return self.messages[0]

    @property
    def head_is_system(self) -> bool:
        """Whether messages[0] is a system prompt message."""
        return bool(self.messages) and \
            self.messages[0].get('role') == 'system'

    @property
    def token_estimate(self) -> int:
        """Estimated tokens of the whole conversation, images included."""
        return sum(estimate_message_tokens(message) for message in self.messages)

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)

    def __getitem__(self, index):
        return self.messages[index]

    def __eq__(self, other):
        if isinstance(other, Conversation):
            return self.messages == other.messages
        if isinstance(other, list):
            return self.messages == other
        return NotImplemented

    def __repr__(self) -> str:
        return (f'Conversation(id={self.id!r}, '
                f'messages={self.messages!r})')
