import asyncio
import logging
import random

import config
from chat_update_context import ChatUpdateContext
from telegram.constants import ChatType
from telegram.error import RetryAfter, TelegramError

logger = logging.getLogger(__name__)

# Live draft streaming (https://core.telegram.org/api/bots/ai). Server-side
# ratelimits for messages.setTyping are 20 calls/5s and 40 calls/30s per
# peer; 0.8s between draft updates stays under both tiers.
DRAFT_MIN_INTERVAL = 0.8
DRAFT_MAX_TEXT = 8000  # rich draft cap is 32768 UTF-8 chars; stay well below
THINKING_DRAFT = "<tg-thinking>Thinking...</tg-thinking>"
PLAIN_DRAFT_MAX = 4096


class DraftStreamer:
    """Streams partial model output as Telegram live drafts.

    Live drafts (Bot API counterparts of sendMessageTextDraftAction) are
    only rendered in private chats; the final sendRichMessage that the
    caller sends afterwards replaces the draft. The rich draft is attempted
    first; if a partial markdown snippet can't be parsed it falls back to a
    plain draft, and disables itself entirely if the server won't accept
    drafts in this chat (e.g. pre-Bot-API-10.1 server).
    """

    def __init__(self, ctx: ChatUpdateContext):
        self._ctx = ctx
        self._chat_id = ctx.chat_id
        thread_id = getattr(ctx.message, 'message_thread_id', None)
        self._thread_id = thread_id if isinstance(thread_id, int) else None
        self._draft_id = random.getrandbits(31) + 1
        self._round_text = ''
        self._sent_text = ''
        self._last_flush = 0.0
        self._prefer_plain = False
        self.enabled = (
            config.STREAM_DRAFTS
            and hasattr(ctx.bot, '_post')  # type: ignore[attr-defined]
            and ctx.chat.type == ChatType.PRIVATE  # type: ignore[union-attr]
        )

    async def start(self) -> None:
        """Send the initial empty draft (shows a "Thinking." placeholder)."""
        self.new_round()
        if self.enabled:
            await self._send_draft(THINKING_DRAFT)

    def new_round(self) -> None:
        """Reset per-model-round state before (re)creating a completion."""
        self._round_text = ''
        self._sent_text = ''

    async def update(self, delta: str) -> None:
        if not self.enabled:
            return
        self._round_text += delta
        await self._maybe_flush()

    async def flush(self, force: bool = False) -> None:
        await self._maybe_flush(force=force)

    async def _maybe_flush(self, force: bool = False) -> None:
        if self._round_text == self._sent_text:
            return
        now = asyncio.get_event_loop().time()
        if not force and now - self._last_flush < DRAFT_MIN_INTERVAL:
            return
        self._last_flush = now
        text = self._round_text
        if len(text) > DRAFT_MAX_TEXT:
            text = '…' + text[-(DRAFT_MAX_TEXT - 1):]
        await self._send_draft(text)
        self._sent_text = text

    async def _send_draft(self, text: str) -> None:
        if not self._prefer_plain:
            if await self._try_rich(text):
                return
            self._prefer_plain = True
        if await self._try_plain(text):
            return
        logger.warning('draft streaming not available in chat %s, disabling',
                       self._chat_id)
        self.enabled = False

    async def _try_rich(self, text: str) -> bool:
        data: dict = {
            'chat_id': self._chat_id,
            'draft_id': self._draft_id,
            'rich_message': {'markdown': text},
        }
        if self._thread_id is not None:
            data['message_thread_id'] = self._thread_id
        return await self._try_post('sendRichMessageDraft', data)

    async def _try_plain(self, text: str) -> bool:
        plain = text.replace(THINKING_DRAFT, '')
        if len(plain) > PLAIN_DRAFT_MAX:
            plain = '…' + plain[-(PLAIN_DRAFT_MAX - 1):]
        data = {'chat_id': self._chat_id, 'draft_id': self._draft_id,
                'text': plain}
        if self._thread_id is not None:
            data['message_thread_id'] = self._thread_id
        return await self._try_post('sendMessageDraft', data)

    async def _try_post(self, method: str, data: dict) -> bool:
        for attempt in range(2):
            try:
                await self._ctx.bot._post(method, data=data)  # type: ignore[attr-defined]
                return True
            except RetryAfter as e:
                delay = min(getattr(e, 'retry_after', None) or 3.0, 5.0)
                logger.warning('%s flood wait %ss, backing off', method, delay)
                await asyncio.sleep(delay + 0.1)
            except TelegramError as e:
                logger.debug('%s failed: %s', method, e)
                return False
            except Exception as e:
                logger.debug('%s unexpected failure: %r', method, e)
                return False
        return False
