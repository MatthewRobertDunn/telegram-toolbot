"""Telegram delivery: everything that writes to the user-visible chat.

A ``Messaging`` handle exists only on ``ChatUpdateContext`` (the
interactive context). Subagents work through ``AgentContext``,
which has no ``messaging`` attribute and no bot reference at all, so a
subagent tool has no code path to the chat even if it wanted one.
"""

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from telegram import constants
from telegram.error import BadRequest
from telegram.helpers import escape_markdown
import html

if TYPE_CHECKING:
    from chat_update_context import ChatUpdateContext

logger = logging.getLogger(__name__)
MAX_MSG_LEN = 4000
# Telegram's sendRichMessage limit is 32,768 UTF-8 chars / 500 blocks per
# message. Chunk well below to stay clear of the block-count cap.
MAX_RICH_MSG_LEN = 32000


class Messaging:
    """The chat-writing half of an interactive context.

    Constructed by ``ChatUpdateContext`` and exposed as ``ctx.messaging``.
    Everything that mutates the chat (messages, media, typing, deletes)
    lives here, so "can this code reach the user?" is answered by "does it
    hold an interactive context?".
    """

    def __init__(self, ctx: "ChatUpdateContext") -> None:
        self._ctx = ctx

    async def send_rich_message(self, text: str) -> list:
        """Send markdown via sendRichMessage, chunking if needed. Returns
        the message_ids of all posted chunks, like send_message, so callers
        can link them back to the conversation."""
        ctx = self._ctx
        message_ids = []
        # PTB doesn't expose sendRichMessage yet (Bot API 10.1); we use the
        # raw request layer as recommended by the PTB maintainers until it
        # ships.
        for chunk in _split_at_linebreak(text, MAX_RICH_MSG_LEN):
            data = {
                "chat_id": ctx.chat_id,
                "rich_message": {"markdown": chunk},
            }
            if ctx.message_thread_id is not None:
                data["message_thread_id"] = ctx.message_thread_id
            result = await ctx.bot._post(  # type: ignore[attr-defined]
                "sendRichMessage",
                data=data,
            )
            if isinstance(result, dict) and result.get('message_id') is not None:
                message_ids.append(result['message_id'])
        return message_ids

    async def send_collapsible(self, summary: str, body: str) -> list:
        """Post a collapsible rich-text block, chunking long bodies.

        Each posted message is a self-contained <details> block; bodies
        over the rich-message limit are split at line breaks across
        continuations ("summary (cont. N)"). Returns the message_ids of all
        posted chunks, like send_message, so callers can link them back to
        the conversation.
        """
        overhead = len("<details>\n<summary></summary>\n\n\n\n</details>")
        budget = max(1024, MAX_RICH_MSG_LEN - len(summary) - overhead)
        message_ids = []
        for i, chunk in enumerate(_split_at_linebreak(body, budget)):
            label = summary if i == 0 else f"{summary} (cont. {i + 1})"
            message_ids.extend(await self.send_rich_message(
                f"<details>\n<summary>{label}</summary>\n\n{chunk}\n\n</details>"))
        return message_ids

    async def send_json(self, summary: str, json_text: str) -> list:
        """Post a JSON payload as a collapsible pretty-printed code block.

        Malformed input falls back to the raw text, so model output never
        fails to display. Returns the message_ids, like send_collapsible.
        """
        try:
            pretty = json.dumps(json.loads(json_text), indent=2,
                                ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            pretty = json_text
        return await self.send_collapsible(summary, f"```json\n{pretty}\n```")

    async def _send_chunk_maybe_fallback(self, text: str, parse_mode,
                                         disable_web_page_preview):
        ctx = self._ctx
        try:
            return await ctx.bot.send_message(
                chat_id=ctx.chat_id, text=text, parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview,
                message_thread_id=ctx.message_thread_id)
        except BadRequest as e:
            if parse_mode and 'parse entities' in str(e):
                escaped = escape_markdown(text, version=1)
                return await ctx.bot.send_message(
                    chat_id=ctx.chat_id, text=escaped, parse_mode=parse_mode,
                    disable_web_page_preview=disable_web_page_preview,
                    message_thread_id=ctx.message_thread_id)
            raise

    async def send_message(self, text: str, parse_mode='Markdown',
                           disable_web_page_preview=True) -> list:
        """Send text, chunking if needed. Returns the message_ids of all
        posted chunks, so callers can link them back to the conversation."""
        message_ids = []
        if parse_mode == 'LLM':
            message_ids.extend(await self.send_rich_message(text))
        else:
            for chunk in _split_at_linebreak(text, MAX_MSG_LEN):
                message = await self._send_chunk_maybe_fallback(
                    chunk, parse_mode, disable_web_page_preview)
                if getattr(message, 'message_id', None) is not None:
                    message_ids.append(message.message_id)
        return message_ids

    async def send_preformatted_message(self, text: str) -> list[int]:
        """Send monospaced text (e.g. /memory dumps) chunked to fit.

        Returns the message ids of every posted chunk, like send_message.
        """
        overhead = len("<pre></pre>")
        chunk_len = max(1, MAX_MSG_LEN - overhead)
        message_ids = []
        for chunk in _split_at_linebreak(text, chunk_len):
            message = await self._ctx.bot.send_message(
                chat_id=self._ctx.chat_id,
                text=f"<pre>{html.escape(chunk)}</pre>",
                parse_mode="HTML",
                message_thread_id=self._ctx.message_thread_id,
            )
            if getattr(message, 'message_id', None) is not None:
                message_ids.append(message.message_id)
        return message_ids

    async def send_image(self, photo: bytes, caption: str = "") -> list[int]:
        """Post a photo. Returns its message id, so the caller can link it
        to the conversation turn it belongs to."""
        message = await self._ctx.bot.send_photo(
            chat_id=self._ctx.chat_id,
            photo=photo,
            caption=caption,
            message_thread_id=self._ctx.message_thread_id,
        )
        if getattr(message, 'message_id', None) is None:
            return []
        return [message.message_id]

    async def send_document(self, document: bytes,
                            filename: str) -> list[int]:
        """Post a file. Returns its message id, like send_image."""
        message = await self._ctx.bot.send_document(
            chat_id=self._ctx.chat_id, document=document, filename=filename,
            message_thread_id=self._ctx.message_thread_id)
        if getattr(message, 'message_id', None) is None:
            return []
        return [message.message_id]

    async def delete_message(self, message_id: int) -> None:
        await self._ctx.bot.delete_message(self._ctx.chat_id, message_id)

    async def _send_typing_periodically(self,
                                        stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await self._ctx.bot.send_chat_action(
                    chat_id=self._ctx.chat_id,
                    action=constants.ChatAction.TYPING,
                    message_thread_id=self._ctx.message_thread_id)
            except Exception as e:
                logger.exception(
                    "Error occurred while sending typing action: %s", e)
            await asyncio.sleep(5)

    async def run_with_typing(self, coro):
        """Run coro with a typing indicator until it finishes.

        The indicator task is cancelled and then reaped through gather:
        cancelling alone would leave it un-reaped, and awaiting it
        directly would raise its CancelledError here and mask whatever
        the wrapped coroutine did (including a /stop cancellation it must
        propagate - see generate_response).
        """
        typing_stop = asyncio.Event()
        typing_task = asyncio.create_task(
            self._send_typing_periodically(typing_stop))
        try:
            return await coro
        finally:
            typing_stop.set()
            typing_task.cancel()
            await asyncio.gather(typing_task, return_exceptions=True)


def _split_at_linebreak(text: str, limit: int) -> list[str]:
    """Split into chunks of at most ``limit`` chars, preferring line breaks.

    A degenerate limit (0 or negative) would spin forever - no chunking
    could ever shrink the text - so anything that cannot make progress is
    returned whole instead of hanging the bot. Chunks are never empty: a
    text starting with a newline would otherwise yield a blank leading
    chunk and post an empty message.
    """
    if limit <= 0 or len(text) <= limit:
        return [text] if text else []
    chunks = []
    while len(text) > limit:
        break_at = text.rfind("\n", 0, limit)
        if break_at == -1:
            break_at = limit
        chunk = text[:break_at]
        if chunk:
            chunks.append(chunk)
        text = text[break_at:]
        if text.startswith("\n"):
            text = text[1:]
    if text:
        chunks.append(text)
    return chunks
