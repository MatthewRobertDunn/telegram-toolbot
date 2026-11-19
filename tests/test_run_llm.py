import os
import sys
import unittest
import unittest.mock
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from handlers import run_llm as run_llm_mod
from conversation import Conversation
from messaging import Messaging
from mock_context import mock_ctx

SETTINGS = SimpleNamespace(
    show_tool_output=False, model='m', temperature=1.0, top_p=0.95)

THINKING_SETTINGS = SimpleNamespace(
    show_tool_output=False, model='m', temperature=1.0, top_p=0.95,
    thinking_keep=True)


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


class TestRunLlmReplies(unittest.IsolatedAsyncioTestCase):

    async def test_tools_offered_by_default_and_omitted_on_request(self):
        # Tools are offered only when the caller passes an explicit list;
        # tools=None (the default) means no tools (e.g. nested calls like
        # compaction or impersonation).
        tool_defs = [{"type": "function",
                      "function": {"name": "x", "parameters": {}}}]
        for tools, expected in ((tool_defs, True), (None, False)):
            with self.subTest(tools=bool(tools)):
                captured: dict = {}
                with unittest.mock.patch.object(
                        run_llm_mod, 'client',
                        _fake_client(captured, [_chunk('plain reply')])):
                    result = await run_llm_mod.run_llm(
                        mock_ctx(settings=SETTINGS, conversation=[]),
                        tools=tools)
                self.assertEqual(result, 'plain reply')
                self.assertEqual('tools' in captured, expected)
                self.assertEqual('tool_choice' in captured, expected)
                if expected:
                    self.assertEqual(captured['tool_choice'], 'auto')
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
        # must not share the parent chat's session id. Its message list is
        # used as given, so in-place appends stay visible to the caller.
        ctx = mock_ctx(settings=SETTINGS, conversation=[])
        conversation = Conversation([{'role': 'user', 'content': 'parent'}])
        captured: dict = {}
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client(captured, [_chunk('a')])):
            await run_llm_mod.run_llm(ctx, conversation=conversation)
        parent = captured['extra_headers']['x-opencode-session']
        self.assertNotEqual(parent, ctx.conversation.id)
        # to_openai() builds a fresh sanitized list, but it still reflects
        # the conversation's content at call time.
        self.assertEqual(captured['messages'], conversation.to_openai())
        self.assertEqual(captured['messages'][0]['content'], 'parent')
        self.assertEqual(
            captured['extra_headers']['x-opencode-session'], conversation.id)
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client(captured, [_chunk('b')])):
            await run_llm_mod.run_llm(ctx, conversation=Conversation([
                {'role': 'user', 'content': 'another'}]))
        self.assertNotEqual(
            captured['extra_headers']['x-opencode-session'], parent)


def _tool_chunk(name, arguments, call_id='c1', reasoning=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=None, reasoning_content=reasoning,
        tool_calls=[SimpleNamespace(
            index=0, id=call_id,
            function=SimpleNamespace(name=name, arguments=arguments))]))])


def _fake_client_calls(captured: list, streams: list):
    """A client returning one preset stream per successive create call."""
    async def fake_create(**kwargs):
        index = len(captured)
        captured.append(kwargs)

        async def stream():
            for c in streams[index]:
                yield c
        return stream()
    return SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=fake_create)))


class TestReasoningContentPassback(unittest.IsolatedAsyncioTestCase):
    # DeepSeek/Kimi thinking mode reject later turns when a tool-call
    # assistant message drops the reasoning_content the model produced.
    # Regression: it used to be discarded when the message was appended.

    async def _run_tool_turn(self, reasoning):
        captured: list = []
        streams = [
            [_tool_chunk('unknown_tool', '{}', reasoning=reasoning)],
            [_chunk('done')],
        ]
        tool_defs = [{"type": "function",
                      "function": {"name": "unknown_tool", "parameters": {}}}]
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client_calls(captured, streams)):
            result = await run_llm_mod.run_llm(
                mock_ctx(settings=SETTINGS, conversation=[]),
                tools=tool_defs)
        self.assertEqual(result, 'done')
        return next(m for m in captured[1]['messages']
                    if m.get('role') == 'assistant' and m.get('tool_calls'))

    async def test_reasoning_is_echoed_back(self):
        assistant = await self._run_tool_turn('deliberate reasoning')
        self.assertEqual(assistant['reasoning_content'], 'deliberate reasoning')

    async def test_no_reasoning_adds_no_field(self):
        # Backends that never emit reasoning must not be sent the extra key.
        assistant = await self._run_tool_turn(None)
        self.assertNotIn('reasoning_content', assistant)


class TestToolCallAnnouncements(unittest.IsolatedAsyncioTestCase):
    # The "Tool call:" announcements posted to Telegram must end up
    # linked to the assistant message, or /undo can't clean them up.

    async def test_announcement_ids_attach_to_the_assistant_message(self):
        ctx = mock_ctx(settings=SimpleNamespace(
            show_tool_output=True, model='m', temperature=1.0, top_p=0.95),
            conversation=[])
        send = unittest.mock.AsyncMock(return_value=[101, 102])
        tool_defs = [{"type": "function",
                      "function": {"name": "web_search", "parameters": {}}}]
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client_calls([], [
                    [_tool_chunk('web_search', '{"query": "cheese"}')],
                    [_chunk('done')],
                ])):
            with unittest.mock.patch.object(
                    Messaging, 'send_json', send):
                await run_llm_mod.run_llm(ctx, tools=tool_defs)
        assistant = next(m for m in ctx.conversation.messages
                         if m.get('tool_calls'))
        self.assertEqual(assistant['custom'],
                         {'telegram_message_ids': [101, 102]})
        call = send.await_args
        assert call is not None
        self.assertEqual(call.args[0], 'Tool call: `web_search`')
        self.assertEqual(call.args[1], '{"query": "cheese"}')


class TestThinkingMessages(unittest.IsolatedAsyncioTestCase):
    # Kept thinking lands in history as a first-class assistant message
    # tagged THINKING (never sent to the endpoint), and only for the
    # conversation the user actually chats in.

    def _thinking_ctx(self):
        ctx = mock_ctx(settings=THINKING_SETTINGS, conversation=[])
        posted: list = []

        async def fake_collapsible(title, body):
            posted.append((title, body))
            return [77]

        ctx.messaging.send_collapsible = fake_collapsible  # type: ignore[method-assign]
        return ctx, posted

    async def test_final_thinking_is_a_first_class_message(self):
        from conversation import is_thinking_message
        ctx, posted = self._thinking_ctx()
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client({}, [_chunk('final', reasoning='trace')])):
            result = await run_llm_mod.run_llm(ctx)
        self.assertEqual(result, 'final')
        self.assertEqual(posted, [('Thinking', 'trace')])
        # run_llm records the thinking; the caller appends the answer.
        self.assertEqual(len(ctx.conversation.messages), 1)
        thinking = ctx.conversation.messages[0]
        self.assertTrue(is_thinking_message(thinking))
        self.assertEqual(thinking.get('content'), 'trace')
        self.assertEqual(thinking.telegram_message_ids, [77])

    async def test_nested_conversation_posts_no_thinking(self):
        ctx, posted = self._thinking_ctx()
        clone = Conversation([{'role': 'user', 'content': 'x'}])
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client({}, [_chunk('reply', reasoning='trace')])):
            result = await run_llm_mod.run_llm(ctx, conversation=clone)
        self.assertEqual(result, 'reply')
        self.assertEqual(posted, [])
        self.assertEqual(len(clone), 1)

    async def test_thinking_off_posts_nothing(self):
        from conversation import is_thinking_message
        ctx = mock_ctx(settings=SETTINGS, conversation=[])
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client({}, [_chunk('final', reasoning='trace')])):
            with unittest.mock.patch.object(
                    Messaging, 'send_collapsible',
                    unittest.mock.AsyncMock(
                        side_effect=AssertionError('must not post'))) as send:
                result = await run_llm_mod.run_llm(ctx)
        self.assertEqual(result, 'final')
        send.assert_not_called()
        self.assertFalse(any(is_thinking_message(m)
                             for m in ctx.conversation.messages))


if __name__ == '__main__':
    unittest.main()
