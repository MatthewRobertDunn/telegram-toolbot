import asyncio
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import compaction
from conversation import Conversation
from messages import tool_call_ids
from mock_context import mock_ctx
from tokens import (IMAGE_TOKEN_ESTIMATE, TOKENS_PER_CHAR, estimate_message_tokens,
                    token_count)
from transcript import message_to_text


def _text(tokens: int) -> str:
    """A string that token_count() estimates at exactly `tokens` tokens."""
    return 'x' * int(tokens / TOKENS_PER_CHAR)


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
        self.assertEqual(token_count(''), 0)
        self.assertEqual(token_count('x' * 400), 100)

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
            100 + 100 + IMAGE_TOKEN_ESTIMATE)

    def test_estimates_tool_calls(self):
        message = {
            'role': 'assistant', 'content': '',
            'tool_calls': [{
                'id': 'call_1', 'type': 'function',
                'function': {'name': 'toolname', 'arguments': 'x' * 400},
            }],
        }
        self.assertEqual(estimate_message_tokens(message), 102)


class TestSummaryPrompt(unittest.TestCase):
    """The assembled prompt is coherent prose: two complete instructions,
    no wedged sentence fragments, no misspellings."""

    def test_omission_instruction_is_one_complete_sentence(self):
        self.assertIn(
            'Leave out instructions and setup the assistant was given '
            'rather than told by the user, such as system prompt text and '
            'environment details.',
            compaction.SUMMARY_PROMPT)

    def test_checkpoint_instruction_is_one_complete_sentence(self):
        self.assertIn(
            'If the conversation transcript contains a compaction summary, '
            'build the new summary on top of it, carrying forward only what '
            'is still relevant.',
            compaction.SUMMARY_PROMPT)

    def test_no_misspelled_transcript(self):
        self.assertNotIn('trnacript', compaction.SUMMARY_PROMPT)


class TestThresholds(unittest.TestCase):
    """The size floor and the keep budget are different checks with
    different jobs; equal values made /compact fire only once the keep
    budget was already exhausted."""

    def test_min_compact_tokens_exceeds_the_default_keep_budget(self):
        self.assertGreater(compaction.MIN_COMPACT_TOKENS,
                           compaction.DEFAULT_KEEP_TOKENS)


class TestMessageToText(unittest.TestCase):

    def test_system_messages_are_left_out(self):
        self.assertEqual(
            message_to_text({'role': 'system', 'content': 'prompt'}),
            '')

    def test_images_become_placeholder_text(self):
        text = message_to_text({
            'role': 'user', 'content': [
                {'type': 'text', 'text': 'look at this'},
                {'type': 'image_url',
                 'image_url': {'url': 'data:image/png;base64,AAAA'}},
            ],
        })
        self.assertEqual(text, '[User]: look at this\n[image omitted]')
        self.assertNotIn('base64', text)

    def test_assistant_tool_calls(self):
        text = message_to_text({
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
        text = message_to_text(
            {'role': 'tool', 'content': 'y' * 5_000})
        self.assertTrue(text.startswith('[Tool result]: '))
        self.assertTrue(text.endswith('\n[truncated]'))
        self.assertLess(len(text), 5_000)

    def test_thinking_blocks_are_left_out(self):
        conversation = Conversation()
        thinking = conversation.add_thinking()
        self.assertEqual(message_to_text(thinking), '')


def _user(marker: str, tokens: int) -> dict:
    return {'role': 'user', 'content': _text(tokens) + f' {marker}'}


def _assistant(marker: str, tokens: int,
               tool_call_id: str | None = None) -> dict:
    if tool_call_id is None:
        return {'role': 'assistant', 'content': _text(tokens) + f' {marker}'}
    return {'role': 'assistant', 'content': '',
            'tool_calls': [{'id': tool_call_id, 'type': 'function',
                            'function': {'name': f'tool_{marker}',
                                         'arguments': _text(tokens)}}]}


def _tool(marker: str, tokens: int, tool_call_id: str) -> dict:
    return {'role': 'tool', 'tool_call_id': tool_call_id,
            'content': f'{marker} output ' + _text(tokens)}


class _Case:
    """One table row: a conversation, a keep budget, and which message
    markers must land on which side of the boundary (or a refusal)."""

    def __init__(self, name: str, pairs: list, keep: int, *,
                 older: list[str] | None = None,
                 recent: list[str] | None = None,
                 refuse: bool = False):
        self.name = name
        self.pairs = pairs  # (message, marker) - the marker is unique text
        self.keep = keep
        self.older = older or []
        self.recent = recent or []
        self.refuse = refuse

    @property
    def conversation(self) -> list:
        return [message for message, _ in self.pairs]


CASES = [
    # Starts on a user message, no system head: the boundary respects the
    # budget instead of snapping backwards over it.
    _Case('user-first',
          [(_user('m1', 8_000), 'm1'), (_assistant('m2', 8_000), 'm2'),
           (_user('m3', 8_000), 'm3'), (_assistant('m4', 8_000), 'm4')],
          17_000,
          older=['m1', 'm2'], recent=['m3', 'm4']),
    # A tool result renders truncated (TOOL_OUTPUT_MAX_CHARS), so it is
    # small in transcript terms - the raw budget fit starts the kept part
    # right after the call, ON the result; the boundary moves past the
    # whole batch so call and result stay together.
    _Case('tool-call batch',
          [(_user('m1', 9_500), 'm1'),
           (_assistant('m2', 9_500, 'c1'), 'm2'),
           (_tool('m3', 9_000, 'c1'), 'm3'),
           (_user('m4', 9_500), 'm4'), (_assistant('m5', 9_500), 'm5')],
          20_000,
          older=['m1', 'm2', 'm3'], recent=['m4', 'm5']),
    # The batch sits at the very end: pushing past it keeps nothing, and
    # the whole exchange is summarized instead.
    _Case('boundary pushed past the batch',
          [(_user('m1', 16_000), 'm1'),
           (_assistant('m2', 16_000, 'c1'), 'm2'),
           (_tool('m3', 16_000, 'c1'), 'm3')],
          8_000,
          older=['m1', 'm2', 'm3'], recent=[]),
    # One exchange cannot be split at all: it is summarized whole.
    _Case('single exchange',
          [(_user('m1', 16_000), 'm1'), (_assistant('m2', 16_000), 'm2')],
          1_000,
          older=['m1', 'm2'], recent=[]),
    # No user messages anywhere: the budget rule alone decides, and each
    # tool round stays whole on its side.
    _Case('all-assistant',
          [(_assistant('m1', 15_000, 'c1'), 'm1'),
           (_tool('m2', 15_000, 'c1'), 'm2'),
           (_assistant('m3', 15_000, 'c2'), 'm3'),
           (_tool('m4', 15_000, 'c2'), 'm4')],
          17_000,
          older=['m1', 'm2'], recent=['m3', 'm4']),
    # A keep budget close to the whole transcript would leave a sliver to
    # summarize: refused rather than paid for.
    _Case('trivial summarized part refused',
          [(_user('m1', 2_000), 'm1'), (_assistant('m2', 2_000), 'm2'),
           (_user('m3', 14_000), 'm3'), (_assistant('m4', 14_000), 'm4')],
          31_000,
          refuse=True),
]


class TestBoundaryRule(unittest.TestCase):
    """The boundary choice, over every shape of conversation: the kept
    part never exceeds the keep budget, no tool call is ever separated
    from its results, and a trivial summary is refused."""

    def test_table(self):
        for case in CASES:
            with self.subTest(case.name):
                split = compaction.split_for_compaction(
                    case.conversation, keep_tokens=case.keep)
                if case.refuse:
                    self.assertIsNone(split)
                    continue
                assert split is not None
                older_ids: set = set()
                newer_ids: set = set()
                kept = 0
                for message, marker in case.pairs:
                    in_older = marker in split.older_text
                    in_recent = marker in split.recent_text
                    self.assertNotEqual(
                        in_older, in_recent,
                        f'{marker} must be on exactly one side')
                    if in_older:
                        older_ids |= tool_call_ids(message)
                    else:
                        newer_ids |= tool_call_ids(message)
                        kept += token_count(message_to_text(message))
                self.assertFalse(older_ids & newer_ids,
                                 'a tool call/result pair was split')
                self.assertLessEqual(kept, case.keep,
                                     'kept part exceeds the keep budget')
                transcript = sum(token_count(message_to_text(message))
                                 for message in case.conversation)
                self.assertGreaterEqual(
                    transcript - kept,
                    compaction.MIN_SUMMARY_FRACTION * transcript,
                    'summarized part is a trivial fraction')
                for marker in case.older:
                    self.assertIn(marker, split.older_text)
                for marker in case.recent:
                    self.assertIn(marker, split.recent_text)

    def test_something_is_always_summarized(self):
        # The old rule could land the boundary before the first entry:
        # an empty summary request that still cost a model call.
        split = compaction.split_for_compaction(_big_conversation(),
                                                keep_tokens=15_000)
        assert split is not None
        self.assertTrue(split.older_text)
        self.assertLessEqual(token_count(split.recent_text), 15_000)


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
                                                keep_tokens=17_000)
        assert split is not None
        self.assertIn('first question', split.older_text)
        self.assertIn('first answer', split.older_text)
        self.assertNotIn('second question', split.older_text)
        self.assertIn('second question', split.recent_text)
        self.assertIn('second answer', split.recent_text)
        self.assertGreater(split.tokens_before, compaction.MIN_COMPACT_TOKENS)

    def test_tool_call_batch_never_splits_across_the_boundary(self):
        conversation = [
            {'role': 'user', 'content': _text(10_000) + ' question one'},
            {'role': 'assistant', 'content': '',
             'tool_calls': [{'id': 'c1', 'type': 'function',
                             'function': {'name': 'read',
                                          'arguments': _text(8_000)}}]},
            {'role': 'tool', 'tool_call_id': 'c1',
             'content': 'tool output ' + _text(9_000)},
            {'role': 'user', 'content': _text(9_000) + ' question two'},
            {'role': 'assistant', 'content': _text(9_000) + ' answer two'},
        ]
        # The raw budget fit starts the kept part on the (truncated) tool
        # result, mid-exchange; the boundary moves past the whole batch.
        split = compaction.split_for_compaction(conversation,
                                                keep_tokens=19_000)
        assert split is not None
        # The tool call and its result must end up on the same side of the
        # boundary.
        self.assertIn('[Assistant tool call]: read', split.older_text)
        self.assertIn('tool output', split.older_text)
        self.assertIn('question two', split.recent_text)

    def test_single_exchange_is_summarized_with_nothing_kept(self):
        conversation = [
            {'role': 'user', 'content': _text(16_000) + ' question'},
            {'role': 'assistant', 'content': _text(16_000) + ' answer'},
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
            result = await compaction.compact_context(ctx, keep_tokens=17_000)

        self.assertEqual(result.summary, 'THE SUMMARY')
        self.assertGreater(result.tokens_before,
                           compaction.MIN_COMPACT_TOKENS)
        self.assertLess(result.tokens_after, result.tokens_before)
        self.assertEqual(result.tokens_after,
                         compaction.estimate_conversation_tokens(
                             ctx.conversation))

        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0].get('tools'))
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
        # The rebuilt history is replayable at the endpoint.
        self.assertEqual(ctx.conversation.validate_openai_contract(), [])

    async def test_compacts_without_recent_context(self):
        conversation = [
            {'role': 'user', 'content': _text(16_000) + ' question'},
            {'role': 'assistant', 'content': _text(16_000) + ' answer'},
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


    async def test_on_start_runs_inline_before_the_summary_call(self):
        ctx = mock_ctx(conversation=_big_conversation())
        events = []
        seen = {}

        async def on_start():
            events.append('start')

        async def fake_run_llm(ctx, **kwargs):
            events.append('llm')
            seen['caller'] = asyncio.current_task()
            return 'THE SUMMARY'

        with unittest.mock.patch.object(compaction, 'run_llm', fake_run_llm):
            result = await compaction.compact_context(ctx, on_start=on_start)

        self.assertEqual(events, ['start', 'llm'])
        # No internal task is spawned: the caller awaits the work itself,
        # and whoever calls in (a queued unit) tracks it in
        # ctx.generation_task.
        self.assertIs(seen['caller'], asyncio.current_task())
        self.assertEqual(result.summary, 'THE SUMMARY')


if __name__ == '__main__':
    unittest.main()
