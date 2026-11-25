"""Conversation context compaction.

Summarizes the older part of a conversation into a compact checkpoint and
keeps only the newest messages verbatim, so long conversations stay well
inside the model's context window. The summary is produced by a nested
run_llm call with tools disabled and streamed to the user like any reply.

The split is decided on the *rendered transcript* (transcript.py): the
newest `keep_tokens` worth of transcript is kept verbatim (15000 by
default), the rest is summarized, and no tool call is ever separated from
its results across the boundary. Every size check here is transcript
tokens (token_count over message_to_text lines); the raw message estimates
(tokens.estimate_conversation_tokens) are only for the before/after
report. Images are dropped everywhere and replaced by placeholder text.

The logic is deliberately independent of command handling so compaction can
later run automatically once the context grows past a threshold (see
split_for_compaction for the size checks).
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from chat_update_context import ChatUpdateContext
from conversation import Conversation
from handlers.run_llm import (NO_CONTENT_MESSAGE, NO_RESPONSE_MESSAGE,
                              run_llm)
from messages import tool_call_ids
from tokens import token_count, estimate_conversation_tokens
from transcript import message_to_text

# Tokens of recent conversation kept verbatim beside the summary.
DEFAULT_KEEP_TOKENS = 15_000
# Contexts whose transcript is smaller than this are left alone: a
# summary costs a model call, and below about twice the keep budget the
# compaction cannot even halve the context. Deliberately distinct from
# DEFAULT_KEEP_TOKENS so the size check and the keep-budget check stay
# independent.
MIN_COMPACT_TOKENS = 2 * DEFAULT_KEEP_TOKENS
# The summarized part must be at least this fraction of the transcript,
# else compaction is refused: summarizing a sliver of the context cannot
# pay for the model call it needs, and the "compaction" would report a
# token increase. This is what guards custom /compact keeps larger than
# half the context.
MIN_SUMMARY_FRACTION = 0.25


class CompactionError(Exception):
    """Base error for failed or refused compactions."""


@dataclass
class CompactionSplit:
    """A conversation divided into the part to summarize and the part to keep.

    Both parts are flattened transcripts: the summary request and the
    compacted checkpoint are pure text, with images replaced by placeholders.
    """
    older_text: str
    recent_text: str
    tokens_before: int


@dataclass
class CompactionResult:
    summary: str
    tokens_before: int
    tokens_after: int


COMPACTION_SYSTEM_PROMPT = (
    "You are a conversation summarizer. You are shown a transcript of a "
    "conversation between a user and an assistant, and produce a structured "
    "summary that lets the assistant continue the conversation as if it still "
    "had the full history. Do not answer the user and do not call tools.")

SUMMARY_TEMPLATE = """You MUST use this format for your response (you may omit sections that aren't applicable). Do not include the <template> tags in your response.
<template>
## Objective
- [one or two brief sentences describing what the user is trying to accomplish]
## Requirements
- [constraints, preferences, requirements, and scope boundaries stated by the user, or "(none)"]
## Decisions
- [decisions already made and why, or "(none)"]
## Work State
Break the objective into smaller goals and report which are completed, which are being worked on, and which are blocked.
### Completed
- [goals that have been completed; otherwise "(none)"]
### Active
- [goals currently being worked on; otherwise "(none)"]
### Blocked
- [anything blocking progress, and why; otherwise "(none)"]
## Next Move
1. [ordered list of next actions, or "(none)"]
## Relevant Resources
List files, URLs, stored values, and other resources another assistant would need to continue this work. Include at most 15, most important first. If none, write "(none)".
- `[resource]`: [brief reason it matters]
## Important Context
- [facts the next assistant cannot continue without and cannot easily find again; or "(none)"]
</template>"""

SUMMARY_RULES = """Rules:
- Keep each section concise. Use terse, single-line bullets, not prose paragraphs or nested lists.
- Prefer short references over detailed restatement. It is fine to leave out information that can be recovered from the resources listed above.
- Preserve exact file paths, symbols, commands, error strings, URLs, and identifiers.
- Carry forward only user questions or requests that remain unanswered or require further action. Do not repeat ones that newer history has answered or resolved. Preserve exact wording when carrying one forward.
- Do not mention the summary process or that the conversation was compacted."""

SUMMARY_PROMPT = "\n\n".join([
    "You MUST summarize the conversation transcript above into a structured "
    "summary that will be given to another assistant to resume the work.",
    "Summarize only what the user and the assistant said and did. Leave out "
    "instructions and setup the assistant was given rather than told by the "
    "user, such as system prompt text and environment details.",
    "If the conversation transcript contains a compaction summary, build the "
    "new summary on top of it, carrying forward only what is still relevant.",
    SUMMARY_TEMPLATE,
    SUMMARY_RULES,
    "Do not continue the task or call tools.",
    "Return only the structured summary in the requested format. Do not "
    "include a preamble, explanation, or other commentary.",
])


@dataclass(frozen=True)
class _Entry:
    """One transcript line of a conversation, with its size."""
    message: dict
    text: str
    tokens: int


def choose_boundary(entries: list[_Entry], *, keep_tokens: int,
                    transcript_tokens: int) -> int | None:
    """Where the verbatim part starts: an index into `entries`, or None
    when compaction is refused.

    Rules, in order:
    - The kept suffix never exceeds the keep budget: the boundary is the
      oldest position whose suffix still fits.
    - A tool call and its results are never separated across the
      boundary: one that would split a pair moves forward past the whole
      batch.
    - The boundary is at least 1 - there is always something to
      summarize - and the summarized part is at least
      MIN_SUMMARY_FRACTION of the transcript, else None: compacting a
      trivial remainder is a model call that saves nothing.
    """
    suffix = 0
    boundary = len(entries)
    for position in range(len(entries) - 1, -1, -1):
        if suffix + entries[position].tokens > keep_tokens:
            break
        suffix += entries[position].tokens
        boundary = position
    # Push the boundary past any tool-call batch it would split: every
    # entry absorbed into the older side may answer (or be answered by)
    # the next one. A kept part also never starts on a tool result - a
    # well-formed result is caught by its id, and a malformed one without
    # an id still always belongs to an older call.
    older_ids: set = set()
    for position in range(boundary):
        older_ids |= tool_call_ids(entries[position].message)
    while boundary < len(entries) and (
            entries[boundary].message.get('role') == 'tool'
            or older_ids & tool_call_ids(entries[boundary].message)):
        older_ids |= tool_call_ids(entries[boundary].message)
        boundary += 1
    summarized = transcript_tokens - sum(
        entry.tokens for entry in entries[boundary:])
    if summarized < MIN_SUMMARY_FRACTION * transcript_tokens:
        return None
    return boundary


def split_for_compaction(
        conversation: Conversation | list,
        keep_tokens: int = DEFAULT_KEEP_TOKENS) -> CompactionSplit | None:
    """Split a conversation into the part to summarize and the part to keep.

    The newest messages worth about `keep_tokens` of transcript are kept
    verbatim; the rest is summarized. Returns None when there is nothing
    worth compacting: the transcript is small (MIN_COMPACT_TOKENS), all
    of it fits in `keep_tokens`, or the summarized part would be trivial
    (choose_boundary refuses).
    """
    split = _split(conversation, keep_tokens)
    return None if isinstance(split, str) else split


def _split(conversation: Conversation | list,
           keep_tokens: int) -> "CompactionSplit | str":
    """split_for_compaction, with the user-facing refusal message in
    place of None: a transcript below the floor is genuinely too small,
    while a custom keep budget can be refused because keeping that much
    would leave too little to summarize."""
    entries = []
    for message in conversation:
        text = message_to_text(message)
        if text:
            entries.append(_Entry(message, text, token_count(text)))
    if not entries:
        return 'Context too small to compact'
    transcript_tokens = sum(entry.tokens for entry in entries)
    if transcript_tokens < MIN_COMPACT_TOKENS or \
            transcript_tokens <= keep_tokens:
        return 'Context too small to compact'

    boundary = choose_boundary(entries, keep_tokens=keep_tokens,
                               transcript_tokens=transcript_tokens)
    if boundary is None:
        return (f'Nothing to compact: keeping {keep_tokens:,} tokens '
                'would summarize too little of the context')

    return CompactionSplit(
        older_text='\n\n'.join(entry.text for entry in entries[:boundary]),
        recent_text='\n\n'.join(entry.text for entry in entries[boundary:]),
        tokens_before=estimate_conversation_tokens(conversation))


async def compact_context(
        ctx: ChatUpdateContext,
        keep_tokens: int = DEFAULT_KEEP_TOKENS,
        on_start: Callable[[], Awaitable[None]] | None = None
) -> CompactionResult:
    """Compact ctx.conversation in place and report the token savings.

    The summary is requested with tools disabled and arrives through run_llm,
    so it is streamed to the user like any other reply. The conversation is
    then replaced by a single checkpoint message holding the summary plus the
    newest `keep_tokens` of messages verbatim.

    Runs inline - whoever awaits it owns the task. The /compact handler
    enqueues it onto the per-thread work queue like any other unit of
    work, so it never overlaps a running generation and cannot be busy.

    Raises CompactionError when there is nothing to compact - the context
    is too small, or the keep budget would summarize too little - or the
    summary fails. `on_start`, when given, is awaited just before the
    summarizing call begins.
    """
    split = _split(ctx.conversation, keep_tokens)
    if isinstance(split, str):
        raise CompactionError(split)

    return await _run_compaction(ctx, split, on_start)


async def _run_compaction(
        ctx: ChatUpdateContext,
        split: CompactionSplit,
        on_start: Callable[[], Awaitable[None]] | None) -> CompactionResult:
    """The summarizing call and the checkpoint rebuild."""
    if on_start is not None:
        await on_start()

    summary = await run_llm(
        ctx,
        conversation=Conversation([
            {'role': 'system', 'content': COMPACTION_SYSTEM_PROMPT},
            {'role': 'user',
             'content': f'Conversation transcript:\n\n{split.older_text}'
                        f'\n\n{SUMMARY_PROMPT}'},
        ]))
    summary = (summary or '').strip()
    if not summary or summary in (NO_RESPONSE_MESSAGE, NO_CONTENT_MESSAGE):
        raise CompactionError('Compaction failed: no summary was produced.')

    checkpoint = (
        '<conversation-checkpoint>\n'
        'The earlier conversation was compacted into this checkpoint. '
        'Treat it as the conversation so far and continue from it.\n'
        '<summary>\n'
        f'{summary}\n'
        '</summary>\n')
    if split.recent_text:
        checkpoint += (
            '<recent-context>\n'
            f'{split.recent_text}\n'
            '</recent-context>\n')
    checkpoint += '</conversation-checkpoint>'
    head = (ctx.conversation[0]
            if ctx.conversation.head_is_system
            else {'role': 'system', 'content': ''})
    # Mutate in place: the compacted conversation is the same conversation,
    # so its session id survives compaction.
    ctx.conversation.set_messages(
        [head, {'role': 'user', 'content': checkpoint}])

    return CompactionResult(
        summary=summary,
        tokens_before=split.tokens_before,
        tokens_after=estimate_conversation_tokens(ctx.conversation))
