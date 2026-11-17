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

    async def send_preformatted_message(self, text: str) -> None:
        overhead = len("<pre></pre>")
        chunk_len = MAX_MSG_LEN - overhead
        for chunk in _split_at_linebreak(text, chunk_len):
            await self._ctx.bot.send_message(
                chat_id=self._ctx.chat_id,
                text=f"<pre>{html.escape(chunk)}</pre>",
                parse_mode="HTML",
                message_thread_id=self._ctx.message_thread_id,
            )

    async def send_image(self, photo: bytes, caption: str = "") -> None:
        await self._ctx.bot.send_photo(
            chat_id=self._ctx.chat_id,
            photo=photo,
            caption=caption,
            message_thread_id=self._ctx.message_thread_id,
        )

    async def send_document(self, document: bytes, filename: str) -> None:
        await self._ctx.bot.send_document(
            chat_id=self._ctx.chat_id, document=document, filename=filename,
            message_thread_id=self._ctx.message_thread_id)

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
        typing_stop = asyncio.Event()
        typing_task = asyncio.create_task(
            self._send_typing_periodically(typing_stop))
        try:
            return await coro
        finally:
            typing_stop.set()
            await typing_task


def _split_at_linebreak(text: str, limit: int) -> list[str]:
    chunks = []
    while len(text) > limit:
        break_at = text.rfind("\n", 0, limit)
        if break_at == -1:
            break_at = limit
        chunks.append(text[:break_at])
        text = text[break_at:]
        if text.startswith("\n"):
            text = text[1:]
    if text:
        chunks.append(text)
    return chunks
