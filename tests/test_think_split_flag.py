import asyncio
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chat_update_context import ChatUpdateContext
from handlers.consume_stream import consume_stream
from handlers.draft_streamer import DraftStreamer

TAG_OPEN = '<' + 'think' + '>'
TAG_CLOSE = '</' + 'think' + '>'


def _chunk(content=None, reasoning=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=content, reasoning_content=reasoning,
        tool_calls=None), finish_reason=None)])


async def _fake_stream(chunks):
    for c in chunks:
        yield c


class RecordingStreamer(DraftStreamer):
    def __init__(self, ctx):  # type: ignore[no-untyped-call]
        super().__init__(ctx)
        self.updates = []

    async def update(self, delta):
        self.updates.append(delta)


def _mock_ctx() -> ChatUpdateContext:
    message = SimpleNamespace(
        text='', caption=None,
        chat=SimpleNamespace(id=1, type='private'),
        from_user=SimpleNamespace(id=1, first_name='T', last_name='U'))
    return ChatUpdateContext(
        update=None,  # type: ignore[arg-type]
        context=None,  # type: ignore[arg-type]
        bot=None,  # type: ignore[arg-type]
        chat=message.chat,
        user=message.from_user,
        message=message,  # type: ignore[arg-type]
        user_data={},
        chat_data=None,
        command_text='',
        command_name='')  # type: ignore[arg-type]


class TestStripThinkingFlag(unittest.IsolatedAsyncioTestCase):

    async def _run(self, chunks, strip):
        streamer = RecordingStreamer(_mock_ctx())
        streamer.enabled = True
        msg = await consume_stream(_fake_stream(chunks), streamer,
                                   strip_thinking=strip)
        return streamer, msg

    async def test_strip_enabled(self):
        streamer, msg = await self._run(
            [_chunk(reasoning='Plan it.'),
             _chunk(TAG_OPEN + 'inlined thoughts' + TAG_CLOSE),
             _chunk('Answer.')], strip=True)
        self.assertEqual(msg.content, 'Answer.')
        self.assertEqual(msg.reasoning_content, 'Plan it.')
        self.assertEqual(''.join(streamer.updates),
                         'Plan it.' + TAG_OPEN + 'inlined thoughts'
                         + TAG_CLOSE + 'Answer.')

    async def test_strip_disabled_keeps_tag_in_content(self):
        streamer, msg = await self._run(
            [_chunk(reasoning='Plan it.'),
             _chunk('<code>' + TAG_CLOSE + 'shown</code>')], strip=False)
        self.assertEqual(msg.content, '<code>' + TAG_CLOSE + 'shown</code>')
        self.assertEqual(msg.reasoning_content, 'Plan it.')

    async def test_drafts_stream_raw_regardless(self):
        # drafts receive deltas verbatim whether or not the split applies
        for strip in (True, False):
            streamer, _ = await self._run(
                [_chunk(TAG_CLOSE + 'tail')], strip=strip)
            self.assertEqual(''.join(streamer.updates), TAG_CLOSE + 'tail')


class TestDraftIdRotation(unittest.TestCase):

    def test_new_round_regenerates_draft_id(self):
        for _ in range(3):
            streamer = DraftStreamer(_mock_ctx())  # type: ignore[arg-type]
            streamer.enabled = True
            first_id = streamer._draft_id
            streamer.new_round()
            self.assertNotEqual(streamer._draft_id, first_id)
            self.assertNotEqual(streamer._draft_id, 0)


if __name__ == '__main__':
    unittest.main()
