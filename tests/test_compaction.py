import asyncio
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import compaction
from mock_context import mock_ctx


def _text(tokens: int) -> str:
    """A string that token_count() estimates at exactly `tokens` tokens."""
    return 'x' * int(tokens / compaction.TOKENS_PER_CHAR)


def _big_conversation() -> list:
    """Two exchanges totalling just over 30k estimated tokens."""
    return [
        {'role': 'system', 'content': 'system prompt'},
        {'role': 'user', 'content': _text(8_000) + ' first question'},
        {'role': 'assistant', 'content': _text(8_000) + ' first answer'},
        {'role': 'user', 'content': _text(8_000) + ' second question'},
        {'role': 'assistant', 'content': _text(8_000) + ' second answer'},
    ]


class TestTokenCount(unittest.TestCase):

    def test_quarter_token_per_character(self):
        self.assertEqual(compaction.token_count(''), 0)
        self.assertEqual(compaction.token_count('x' * 400), 100)

    def test_estimates_text_and_images(self):
        conversation = [
            {'role': 'system', 'content': 'x' * 400},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': 'x' * 400},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}},
            ]},
        ]
        self.assertEqual(
            compaction.estimate_conversation_tokens(conversation),
            100 + 100 + compaction.IMAGE_TOKEN_ESTIMATE)

    def test_estimates_tool_calls(self):
        message = {
            'role': 'assistant', 'content': '',
            'tool_calls': [{
                'id': 'call_1', 'type': 'function',
                'function': {'name': 'toolname', 'arguments': 'x' * 400},
            }],
        }
        self.assertEqual(compaction.estimate_message_tokens(message), 102)


class TestMessageToText(unittest.TestCase):

    def test_system_messages_are_left_out(self):
        self.assertEqual(
            compaction.message_to_text({'role': 'system', 'content': 'prompt'}),
            '')

    def test_images_become_placeholder_text(self):
        text = compaction.message_to_text({
            'role': 'user', 'content': [
                {'type': 'text', 'text': 'look at this'},
                {'type': 'image_url',
                 'image_url': {'url': 'data:image/png;base64,AAAA'}},
            ],
        })
        self.assertEqual(text, '[User]: look at this\n[image omitted]')
        self.assertNotIn('base64', text)

    def test_assistant_tool_calls(self):
        text = compaction.message_to_text({
            'role': 'assistant', 'content': 'checking',
            'tool_calls': [{
                'id': 'call_1', 'type': 'function',
                'function': {'name': 'read', 'arguments': '{"path": "a.txt"}'},
            }],
        })
        self.assertEqual(
            text,
            '[Assistant]: checking\n'
            '[Assistant tool call]: read({"path": "a.txt"})')

    def test_tool_output_is_truncated(self):
        text = compaction.message_to_text(
            {'role': 'tool', 'content': 'y' * 5_000})
        self.assertTrue(text.startswith('[Tool result]: '))
        self.assertTrue(text.endswith('\n[truncated]'))
        self.assertLess(len(text), 5_000)


class TestSplitForCompaction(unittest.TestCase):

    def test_small_conversation_is_not_split(self):
        conversation = [
            {'role': 'system', 'content': 'prompt'},
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'hello'},
        ]
        self.assertIsNone(compaction.split_for_compaction(conversation))

    def test_context_that_fits_in_keep_tokens_is_not_split(self):
        self.assertIsNone(compaction.split_for_compaction(
            _big_conversation(), keep_tokens=40_000))

    def test_nothing_to_transcribe_is_not_split(self):
        self.assertIsNone(compaction.split_for_compaction(
            [{'role': 'system', 'content': 'x' * 100_000}]))

    def test_newest_messages_are_kept_verbatim(self):
        split = compaction.split_for_compaction(_big_conversation(),
                                                keep_tokens=15_000)
        assert split is not None
        self.assertIn('first question', split.older_text)
        self.assertIn('first answer', split.older_text)
        self.assertNotIn('second question', split.older_text)
        self.assertIn('second question', split.recent_text)
        self.assertIn('second answer', split.recent_text)
        self.assertGreater(split.tokens_before, compaction.MIN_COMPACT_TOKENS)

    def test_split_snaps_to_a_user_message(self):
        conversation = [
            {'role': 'user', 'content': _text(5_000) + ' question one'},
            {'role': 'assistant', 'content': '',
             'tool_calls': [{'id': 'c1', 'type': 'function',
                             'function': {'name': 'read',
                                          'arguments': _text(3_000)}}]},
            {'role': 'tool', 'content': 'tool output ' + _text(5_000)},
            {'role': 'user', 'content': _text(5_000) + ' question two'},
            {'role': 'assistant', 'content': _text(5_000) + ' answer two'},
        ]
        # The raw split lands on the tool result, mid-exchange.
        split = compaction.split_for_compaction(conversation,
                                                keep_tokens=11_000)
        assert split is not None
        # The tool call and its result must end up on the same side of the
        # boundary.
        self.assertIn('[Assistant tool call]: read', split.older_text)
        self.assertIn('tool output', split.older_text)
        self.assertIn('question two', split.recent_text)

    def test_single_exchange_is_summarized_with_nothing_kept(self):
        conversation = [
            {'role': 'user', 'content': _text(10_000) + ' question'},
            {'role': 'assistant', 'content': _text(10_000) + ' answer'},
        ]
        split = compaction.split_for_compaction(conversation,
                                                keep_tokens=1_000)
        assert split is not None
        self.assertIn('question', split.older_text)
        self.assertIn('answer', split.older_text)
        self.assertEqual(split.recent_text, '')


class TestCompactContext(unittest.IsolatedAsyncioTestCase):

    async def test_compacts_and_rebuilds_conversation(self):
        ctx = mock_ctx(conversation=_big_conversation())
        calls = []

        async def fake_run_llm(ctx, **kwargs):
            calls.append(kwargs)
            return 'THE SUMMARY'

        with unittest.mock.patch.object(compaction, 'run_llm', fake_run_llm):
            result = await compaction.compact_context(ctx)

        self.assertEqual(result.summary, 'THE SUMMARY')
        self.assertGreater(result.tokens_before,
                           compaction.MIN_COMPACT_TOKENS)
        self.assertLess(result.tokens_after, result.tokens_before)
        self.assertEqual(result.tokens_after,
                         compaction.estimate_conversation_tokens(
                             ctx.conversation))

        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]['allow_tools'])
        request = calls[0]['conversation']
        self.assertEqual(request[0]['role'], 'system')
        self.assertIn('first question', request[1]['content'])
        self.assertIn(compaction.SUMMARY_PROMPT, request[1]['content'])

        self.assertEqual(ctx.conversation[0]['role'], 'system')
        checkpoint = ctx.conversation[1]['content']
        self.assertIn('<conversation-checkpoint>', checkpoint)
        self.assertIn('THE SUMMARY', checkpoint)
        self.assertIn('second question', checkpoint)
        self.assertNotIn('first question', checkpoint)

    async def test_compacts_without_recent_context(self):
        conversation = [
            {'role': 'user', 'content': _text(10_000) + ' question'},
            {'role': 'assistant', 'content': _text(10_000) + ' answer'},
        ]
        ctx = mock_ctx(conversation=conversation)

        async def fake_run_llm(ctx, **kwargs):
            return 'THE SUMMARY'

        with unittest.mock.patch.object(compaction, 'run_llm', fake_run_llm):
            await compaction.compact_context(ctx)

        checkpoint = ctx.conversation[1]['content']
        self.assertNotIn('<recent-context>', checkpoint)
        self.assertIn('THE SUMMARY', checkpoint)

    async def test_small_context_is_left_untouched(self):
        conversation = [
            {'role': 'system', 'content': 'prompt'},
            {'role': 'user', 'content': 'hi'},
        ]
        ctx = mock_ctx(conversation=conversation)

        async def fail_run_llm(ctx, **kwargs):
            self.fail('run_llm must not be called for a small context')

        with unittest.mock.patch.object(compaction, 'run_llm', fail_run_llm):
            with self.assertRaisesRegex(compaction.CompactionError,
                                        'Context too small to compact'):
                await compaction.compact_context(ctx)

        self.assertEqual(ctx.conversation, conversation)

    async def test_empty_summary_fails_and_keeps_conversation(self):
        conversation = _big_conversation()
        ctx = mock_ctx(conversation=conversation)

        async def fake_run_llm(ctx, **kwargs):
            return ''

        with unittest.mock.patch.object(compaction, 'run_llm', fake_run_llm):
            with self.assertRaises(compaction.CompactionError):
                await compaction.compact_context(ctx)

        self.assertEqual(ctx.conversation, conversation)


    async def test_on_start_runs_before_the_summary_call(self):
        ctx = mock_ctx(conversation=_big_conversation())
        events = []

        async def on_start():
            events.append('start')

        async def fake_run_llm(ctx, **kwargs):
            events.append('llm')
            return 'THE SUMMARY'

        with unittest.mock.patch.object(compaction, 'run_llm', fake_run_llm):
            await compaction.compact_context(ctx, on_start=on_start)

        self.assertEqual(events, ['start', 'llm'])

    async def test_compaction_runs_inline_under_the_caller(self):
        ctx = mock_ctx(conversation=_big_conversation())
        seen = {}

        async def fake_run_llm(ctx, **kwargs):
            seen['caller'] = asyncio.current_task()
            return 'THE SUMMARY'

        with unittest.mock.patch.object(compaction, 'run_llm', fake_run_llm):
            result = await compaction.compact_context(ctx)

        # No internal task is spawned: the caller awaits the work itself,
        # and whoever calls in (a queued unit) tracks it in
        # ctx.generation_task.
        self.assertIs(seen['caller'], asyncio.current_task())
        self.assertEqual(result.summary, 'THE SUMMARY')


if __name__ == '__main__':
    unittest.main()
