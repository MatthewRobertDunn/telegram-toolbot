"""Conversation context compaction.

Summarizes the older part of a conversation into a compact checkpoint and
keeps only the newest messages verbatim, so long conversations stay well
inside the model's context window. The summary is produced by a nested
run_llm call with tools disabled and streamed to the user like any reply.

Only the newest `keep_tokens` of messages are kept verbatim (15000 by
default); images are dropped everywhere and replaced by placeholder text.

The logic is deliberately independent of command handling so compaction can
later run automatically once the context grows past a threshold (see
estimate_conversation_tokens / split_for_compaction for the size checks).
"""
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from conversation import (Conversation, TOKENS_PER_CHAR, IMAGE_TOKEN_ESTIMATE,
                          IMAGE_PLACEHOLDER, token_count,
                          estimate_message_tokens, estimate_conversation_tokens)
from chat_update_context import ChatUpdateContext
from handlers.run_llm import (NO_CONTENT_MESSAGE, NO_RESPONSE_MESSAGE,
                              run_llm)

# Tokens of recent conversation kept verbatim beside the summary.
DEFAULT_KEEP_TOKENS = 15_000
# Contexts smaller than this are left alone; compaction would save little.
MIN_COMPACT_TOKENS = 15_000
# Long tool outputs are cut to this many characters in the summary source.
TOOL_OUTPUT_MAX_CHARS = 1_250


class CompactionError(Exception):
    """Base error for failed or refused compactions."""


class CompactionBusyError(CompactionError):
    """Raised when compaction is requested while the bot is busy."""


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
    SUMMARY_TEMPLATE,
    SUMMARY_RULES,
    "Do not continue the task or call tools.",
    "Return only the structured summary in the requested format. Do not "
    "include a preamble, explanation, or other commentary.",
])


def message_to_text(message: dict) -> str:
    """One conversation message as a transcript line.

    Empty for messages the transcript leaves out (e.g. system prompts).
    Images become placeholder text and long tool outputs are truncated.
    """
    role = message.get('role')
    if role == 'system':
        return ''
    if role == 'user':
        content = message.get('content')
        if isinstance(content, str):
            return f'[User]: {content}'
        lines = []
        for part in content or []:
            if not isinstance(part, dict):
                continue
            if part.get('type') == 'text':
                lines.append(f"[User]: {part.get('text') or ''}")
            elif part.get('type') == 'image_url':
                lines.append(IMAGE_PLACEHOLDER)
        return '\n'.join(lines)
    if role == 'assistant':
        lines = []
        content = message.get('content')
        if content:
            lines.append(f'[Assistant]: {content}')
        for tool_call in message.get('tool_calls') or []:
            function = (tool_call or {}).get('function') or {}
            name = function.get('name') or ''
            arguments = function.get('arguments') or ''
            lines.append(f'[Assistant tool call]: {name}({arguments})')
        return '\n'.join(lines)
    if role == 'tool':
        return f"[Tool result]: {_truncate_tool_output(message.get('content') or '')}"
    return ''


def _truncate_tool_output(value: str) -> str:
    if len(value) <= TOOL_OUTPUT_MAX_CHARS:
        return value
    return value[:TOOL_OUTPUT_MAX_CHARS] + '\n[truncated]'


def split_for_compaction(
        conversation: Conversation | list,
        keep_tokens: int = DEFAULT_KEEP_TOKENS) -> CompactionSplit | None:
    """Split a conversation into the part to summarize and the part to keep.

    The newest messages worth about `keep_tokens` are kept verbatim; the rest
    is summarized. Returns None when there is nothing worth compacting: the
    conversation is small, or all of it fits in `keep_tokens`.
    """
    tokens_before = estimate_conversation_tokens(conversation)
    if tokens_before < MIN_COMPACT_TOKENS or tokens_before <= keep_tokens:
        return None

    entries = []
    for message in conversation:
        text = message_to_text(message)
        if text:
            entries.append((message, text, token_count(text)))
    if not entries:
        return None

    # Drop oldest entries until the rest fit the allowance, but always keep
    # the newest one.
    kept = 0
    kept_tokens = 0
    for _, _, tokens in reversed(entries):
        if kept and kept_tokens + tokens > keep_tokens:
            break
        kept_tokens += tokens
        kept += 1
    first_kept = len(entries) - kept

    # Start the verbatim part at a user message so an assistant's tool calls
    # and their results are never separated across the summary boundary.
    user_positions = [pos for pos, (message, _, _) in enumerate(entries)
                      if pos > 0 and message.get('role') == 'user']
    if user_positions:
        first_kept = max((pos for pos in user_positions if pos <= first_kept),
                         default=user_positions[-1])
    else:
        # A single exchange cannot be split at a user message: summarize it
        # all and keep nothing verbatim.
        first_kept = len(entries)

    return CompactionSplit(
        older_text='\n\n'.join(text for _, text, _ in entries[:first_kept]),
        recent_text='\n\n'.join(text for _, text, _ in entries[first_kept:]),
        tokens_before=tokens_before)


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

    The work runs as its own task, tracked in ctx.compaction_task the same way
    generation is tracked in ctx.generation_task, so cancelling it stops just
    the compaction and never its caller's task.

    Raises CompactionError when the context is too small to compact or the
    summary fails, and CompactionBusyError when a reply is being generated or
    another compaction is already running. `on_start`, when given, is awaited
    just before the summarizing call begins.
    """
    if ctx.is_busy:
        raise CompactionBusyError(
            'Cannot compact the context while a response is being generated '
            'or a compaction is already running.')

    split = split_for_compaction(ctx.conversation, keep_tokens)
    if split is None:
        raise CompactionError('Context too small to compact')

    task = asyncio.create_task(_run_compaction(ctx, split, on_start))
    ctx.compaction_task = task
    try:
        return await task
    finally:
        ctx.compaction_task = None


async def _run_compaction(
        ctx: ChatUpdateContext,
        split: CompactionSplit,
        on_start: Callable[[], Awaitable[None]] | None) -> CompactionResult:
    """The summarizing call and the checkpoint rebuild; runs as its own task."""
    if on_start is not None:
        await on_start()

    summary = await run_llm(
        ctx,
        conversation=Conversation([
            {'role': 'system', 'content': COMPACTION_SYSTEM_PROMPT},
            {'role': 'user',
             'content': f'Conversation transcript:\n\n{split.older_text}'
                        f'\n\n{SUMMARY_PROMPT}'},
        ]),
        allow_tools=False)
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
    ctx.conversation.messages = [head, {'role': 'user', 'content': checkpoint}]

    return CompactionResult(
        summary=summary,
        tokens_before=split.tokens_before,
        tokens_after=estimate_conversation_tokens(ctx.conversation))

