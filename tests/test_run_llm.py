import os
import sys
import unittest
import unittest.mock
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from handlers import run_llm as run_llm_mod
from conversation import Conversation
from mock_context import mock_ctx

SETTINGS = SimpleNamespace(
    show_tool_output=False, model='m', temperature=1.0, top_p=0.95)


def _chunk(content=None, reasoning=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=content, reasoning_content=reasoning,
        tool_calls=tool_calls), finish_reason=None)])


def _fake_client(captured: dict, chunks: list):
    async def fake_create(**kwargs):
        captured.update(kwargs)

        async def stream():
            for c in chunks:
                yield c
        return stream()
    return SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=fake_create)))


class TestRunLlmToolToggle(unittest.IsolatedAsyncioTestCase):

    async def test_tools_offered_by_default(self):
        captured: dict = {}
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client(captured, [_chunk('plain reply')])):
            result = await run_llm_mod.run_llm(
                mock_ctx(settings=SETTINGS, conversation=[]))
        self.assertEqual(result, 'plain reply')
        self.assertIn('tools', captured)
        self.assertEqual(captured['tool_choice'], 'auto')

    async def test_allow_tools_false_omits_tools(self):
        captured: dict = {}
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client(captured, [_chunk('plain reply')])):
            result = await run_llm_mod.run_llm(
                mock_ctx(settings=SETTINGS, conversation=[]),
                allow_tools=False)
        self.assertEqual(result, 'plain reply')
        self.assertNotIn('tools', captured)
        self.assertNotIn('tool_choice', captured)


class TestRunLlmStreaming(unittest.IsolatedAsyncioTestCase):

    async def test_always_requests_streaming(self):
        captured: dict = {}
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client(captured, [_chunk('hi')])):
            await run_llm_mod.run_llm(
                mock_ctx(settings=SETTINGS, conversation=[]))
        self.assertIs(captured['stream'], True)

    async def test_empty_reply_reports_no_content(self):
        # Regression: an empty completion (no content, no reasoning, no tool
        # calls) used to raise AttributeError on message.reasoning_content
        # and kill the handler instead of returning the fallback text.
        captured: dict = {}
        with unittest.mock.patch.object(
                run_llm_mod, 'client', _fake_client(captured, [_chunk()])):
            result = await run_llm_mod.run_llm(
                mock_ctx(settings=SETTINGS, conversation=[]))
        self.assertEqual(result, run_llm_mod.NO_CONTENT_MESSAGE)

    async def test_reasoning_only_reply_is_returned(self):
        captured: dict = {}
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client(captured, [_chunk(reasoning='just thinking')])):
            result = await run_llm_mod.run_llm(
                mock_ctx(settings=SETTINGS, conversation=[]))
        self.assertEqual(result, 'just thinking')


class TestRunLlmSessionHeader(unittest.IsolatedAsyncioTestCase):

    async def test_context_conversation_keeps_one_session_id(self):
        # Requests against ctx.conversation must all carry the id of that
        # Conversation object.
        ctx = mock_ctx(settings=SETTINGS, conversation=[])
        captured: dict = {}
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client(captured, [_chunk('a')])):
            await run_llm_mod.run_llm(ctx)
        first = captured['extra_headers']['x-opencode-session']
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client(captured, [_chunk('b')])):
            await run_llm_mod.run_llm(ctx)
        second = captured['extra_headers']['x-opencode-session']
        self.assertEqual(first, second)
        self.assertEqual(first, ctx.conversation.id)

    async def test_nested_conversation_gets_a_fresh_session_id(self):
        # A nested call with its own Conversation (subagent, compaction)
        # must not share the parent chat's session id.
        ctx = mock_ctx(settings=SETTINGS, conversation=[])
        captured: dict = {}
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client(captured, [_chunk('a')])):
            await run_llm_mod.run_llm(ctx, conversation=Conversation([
                {'role': 'user', 'content': 'parent'}]))
        parent = captured['extra_headers']['x-opencode-session']
        self.assertNotEqual(parent, ctx.conversation.id)
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client(captured, [_chunk('b')])):
            await run_llm_mod.run_llm(ctx, conversation=Conversation([
                {'role': 'user', 'content': 'another'}]))
        self.assertNotEqual(
            captured['extra_headers']['x-opencode-session'], parent)

    async def test_explicit_conversation_is_used_as_given(self):
        # Passing a Conversation explicitly sends its messages and its id,
        # and in-place appends stay visible to the caller.
        conversation = Conversation([{'role': 'user', 'content': 'hi'}])
        ctx = mock_ctx(settings=SETTINGS, conversation=[])
        captured: dict = {}
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client(captured, [_chunk('reply')])):
            await run_llm_mod.run_llm(ctx, conversation=conversation)
        self.assertIs(captured['messages'], conversation.messages)
        self.assertEqual(captured['extra_headers']['x-opencode-session'],
                         conversation.id)


if __name__ == '__main__':
    unittest.main()
