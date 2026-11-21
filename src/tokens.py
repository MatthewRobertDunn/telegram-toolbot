"""Token accounting for conversations.

Two metrics live here and must not be mixed up:

- `estimate_message_tokens` / `estimate_conversation_tokens` measure the
  *raw messages*: content characters at TOKENS_PER_CHAR plus a flat
  IMAGE_TOKEN_ESTIMATE per inline image. This is what /compact reports
  before and after a compaction.
- `token_count` measures *rendered transcript* text
  (transcript.message_to_text), which drops system prompts and thinking
  blocks and adds role prefixes. The compaction keep-budget and its
  thresholds (compaction.MIN_COMPACT_TOKENS, the trivial-summary
  fraction) are enforced against this metric, consistently on both sides
  of every comparison. The raw estimate and the transcript size are close
  but not equal; comparing one against the other is how /compact used to
  end up refusing nothing, summarizing nothing, and reporting a token
  increase.

Thinking blocks are display-only (never sent to the endpoint), so every
estimate skips them exactly like to_openai() does.
"""

from collections.abc import Iterable

from messages import is_thinking, iter_parts, tool_calls_of

# Rough token estimate: about a quarter token per character.
TOKENS_PER_CHAR = 0.25
# What an inline image is assumed to cost before it is dropped.
IMAGE_TOKEN_ESTIMATE = 1_500


def token_count(text: str) -> int:
    """Rough token estimate for text: about a quarter token per character."""
    return int(len(text) * TOKENS_PER_CHAR)


def estimate_message_tokens(message: dict) -> int:
    """Rough token estimate of one raw message, tool calls included."""
    total = 0
    content = message.get('content')
    if isinstance(content, str):
        total += token_count(content)
    elif isinstance(content, list):
        for part in iter_parts(message):
            if part.get('type') == 'text':
                total += token_count(part.get('text') or '')
            elif part.get('type') == 'image_url':
                total += IMAGE_TOKEN_ESTIMATE
    for call in tool_calls_of(message):
        function = (call or {}).get('function') or {}
        total += token_count(function.get('name') or '')
        total += token_count(function.get('arguments') or '')
    return total


def estimate_conversation_tokens(conversation: Iterable[dict]) -> int:
    """Estimated tokens of a whole conversation, images included.

    Thinking blocks are display-only (never sent to the endpoint), so
    they are excluded like the endpoint excludes them."""
    return sum(estimate_message_tokens(message) for message in conversation
               if not is_thinking(message))
