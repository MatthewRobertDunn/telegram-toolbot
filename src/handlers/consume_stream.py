import re
from dataclasses import dataclass

from .draft_streamer import DraftStreamer

# Some backends stream reasoning inline in the content field (instead of
# the reasoning_content field), ending in a bare close tag; split there so
# chain-of-thought never reaches Telegram in the final message.
_THINK_CLOSE = '</th' + 'ink>'
_THINK_RE = re.compile(r'^(.*?)(?:' + _THINK_CLOSE + r')(.*)$', flags=re.DOTALL)


@dataclass
class FunctionCall:
    name: str
    arguments: str


@dataclass
class ToolCall:
    id: str
    function: FunctionCall


@dataclass
class ConsumeStreamResult:
    """A model reply assembled from one streamed completion.

    All fields are always present; None when empty.
    """
    content: str | None
    tool_calls: list[ToolCall] | None
    reasoning_content: str | None


def _split_thinking(text: str) -> tuple[str, str]:
    """Split LLM content at the first think close tag.

    Returns (visible, reasoning). Drafts are streamed raw while generating;
    the thinking is removed here when the final message is assembled.
    """
    match = _THINK_RE.match(text)
    if match:
        return match.group(2).strip(), match.group(1).strip()
    return text.strip(), ''


async def consume_stream(stream, streamer: DraftStreamer,
                         strip_thinking: bool) -> ConsumeStreamResult:
    """Read a chat completion stream, forwarding content to live drafts.

    Returns a ConsumeStreamResult; all fields are always present
    (None when empty).
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_acc: dict[int, dict] = {}
    # Which accumulated entry each streamed index currently points at.
    # Normally identical to tool_acc, but a backend may split off a new
    # call onto a higher key while still sending index on the wire.
    active: dict[int, dict] = {}

    async for chunk in stream:
        choices = getattr(chunk, 'choices', None)
        if not choices:
            continue
        delta = choices[0].delta
        if delta is None:
            continue
        text = getattr(delta, 'content', None)
        if text:
            content_parts.append(text)
            await streamer.update(text)
        reasoning = getattr(delta, 'reasoning_content', None)
        if reasoning:
            reasoning_parts.append(reasoning)
            # stream thinking into the draft too, so all models behave the
            # same (big-pickle inlines it in content); it is stripped from
            # the draft when the final message is assembled
            await streamer.update(reasoning)
        for part in getattr(delta, 'tool_calls', None) or []:
            idx = getattr(part, 'index', None)
            if not isinstance(idx, int):
                idx = max(tool_acc, default=-1) + 1
            pid = getattr(part, 'id', None)
            entry = active.get(idx)
            if entry is not None and pid and entry['id'] \
                    and pid != entry['id']:
                # Some backends (observed: step-5-preview on NVIDIA NIM)
                # emit follow-up tool calls reusing the same index
                # instead of incrementing it. A different id arriving
                # while this stream slot is occupied means the index
                # should have been incremented: start a fresh call for
                # it rather than concatenating into the current entry.
                entry = tool_acc[max(tool_acc, default=-1) + 1] = {
                    'id': pid, 'name': '', 'arguments': ''}
            elif entry is None:
                entry = tool_acc.setdefault(
                    idx, {'id': '', 'name': '', 'arguments': ''})
            if pid:
                entry['id'] = pid
            # This stream slot now streams this entry, so later id-less
            # fragments keep appending to the right call even though the
            # slot may have been split off to a higher tool_acc key.
            active[idx] = entry
            fn = getattr(part, 'function', None)
            if fn is not None:
                if getattr(fn, 'name', None):
                    entry['name'] += fn.name
                entry['arguments'] += getattr(fn, 'arguments', None) or ''

    await streamer.flush(force=True)

    tool_calls = []
    for idx in sorted(tool_acc):
        entry = tool_acc[idx]
        if entry['id'] or entry['name']:
            tool_calls.append(ToolCall(
                id=entry['id'],
                function=FunctionCall(name=entry['name'],
                                      arguments=entry['arguments'])))
    if strip_thinking:
        visible, streamed_reasoning = _split_thinking(''.join(content_parts))
    else:
        visible = ''.join(content_parts).strip()
        streamed_reasoning = ''
    return ConsumeStreamResult(
        content=visible or None,
        tool_calls=tool_calls or None,
        reasoning_content=(''.join(reasoning_parts) or streamed_reasoning
                           or None),
    )
