import asyncio
import contextlib
import os
import sys
import unittest
import unittest.mock
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
from handlers import run_llm as run_llm_mod
from tool_registry import ToolSurface
from conversation import Conversation
from messaging import Messaging
from mock_context import mock_ctx

SETTINGS = SimpleNamespace(
    show_tool_output=False, model='m', temperature=1.0, top_p=0.95,
    thinking_keep=False)

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
        # Tools are offered only when the caller passes an explicit
        # surface; tools=None (the default) means no tools (e.g. nested
        # calls like compaction or impersonation).
        tool_defs = [{"type": "function",
                      "function": {"name": "x", "parameters": {}}}]

        def surface() -> ToolSurface:
            return ToolSurface(definitions=tool_defs, functions={})

        for tools, expected in ((surface(), True), (None, False)):
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

    async def test_empty_reply_retries_then_recovers(self):
        # An empty completion (no content, no reasoning, no tool calls)
        # retries instead of returning NO_CONTENT_MESSAGE immediately.
        captured: list = []
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client_calls(captured, [[_chunk()], [_chunk('recovered')]])):
            result = await run_llm_mod.run_llm(
                mock_ctx(settings=SETTINGS, conversation=[]))
        self.assertEqual(result, 'recovered')
        self.assertEqual(len(captured), 2)

    async def test_reasoning_only_reply_is_returned(self):
        captured: dict = {}
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client(captured, [_chunk(reasoning='just thinking')])):
            result = await run_llm_mod.run_llm(
                mock_ctx(settings=SETTINGS, conversation=[]))
        self.assertEqual(result, 'just thinking')


class TestRunLlmSessionHeader(unittest.IsolatedAsyncioTestCase):

    async def test_session_id_comes_from_whichever_conversation_is_in_play(
            self):
        # Requests against ctx.conversation all carry its id; a nested
        # call with its own Conversation (subagent, compaction) must not
        # share the parent chat's session id. Its message list is used
        # as given, so in-place appends stay visible to the caller.
        ctx = mock_ctx(settings=SETTINGS, conversation=[])
        captured: dict = {}

        async def call(conversation=None) -> str:
            with unittest.mock.patch.object(
                    run_llm_mod, 'client',
                    _fake_client(captured, [_chunk('reply')])):
                await run_llm_mod.run_llm(ctx, conversation=conversation)
            return captured['extra_headers']['x-opencode-session']

        # Two requests against ctx.conversation: the id is stable.
        first = await call()
        second = await call()
        self.assertEqual(first, second)
        self.assertEqual(first, ctx.conversation.id)

        # An explicit conversation supplies a fresh id whose messages are
        # sent as given, and a different conversation changes the header.
        conversation = Conversation([{'role': 'user', 'content': 'parent'}])
        parent = await call(conversation)
        self.assertNotEqual(parent, ctx.conversation.id)
        self.assertEqual(parent, conversation.id)
        # to_openai() builds a fresh sanitized list, but it still reflects
        # the conversation's content at call time.
        self.assertEqual(captured['messages'], conversation.to_openai())
        self.assertEqual(captured['messages'][0]['content'], 'parent')
        other = await call(
            Conversation([{'role': 'user', 'content': 'another'}]))
        self.assertNotEqual(other, parent)


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
                tools=ToolSurface(definitions=tool_defs, functions={}))
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
            show_tool_output=True, model='m', temperature=1.0, top_p=0.95,
            thinking_keep=False),
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
                await run_llm_mod.run_llm(
                    ctx, tools=ToolSurface(definitions=tool_defs,
                                           functions={}))
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
        self.assertEqual(thinking.get('content'), '')
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


class TestToolArgumentJson(unittest.IsolatedAsyncioTestCase):
    # Malformed model-provided JSON is answered as a tool result (the
    # model can retry with fixed arguments) instead of killing the
    # whole queued job.

    async def test_bad_json_becomes_a_tool_result(self):
        calls: list = []
        streams = [
            [_tool_chunk('t', '{not json')],
            [_chunk('done')],
        ]
        tool_defs = [{"type": "function",
                      "function": {"name": "t", "parameters": {}}}]

        async def fake_tool():
            return 'unused'

        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client_calls(calls, streams)):
            result = await run_llm_mod.run_llm(
                mock_ctx(settings=SETTINGS, conversation=[]),
                tools=ToolSurface(definitions=tool_defs,
                                  functions={'t': fake_tool}))
        # The loop survived and gave the model a chance to fix it.
        self.assertEqual(result, 'done')
        self.assertEqual(len(calls), 2)
        tool_messages = [m for m in calls[1]['messages']
                         if m.get('role') == 'tool']
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("not valid JSON", tool_messages[0]['content'])
        self.assertIn('call the tool again', tool_messages[0]['content'])

    async def test_unknown_tool_is_reported_not_raised(self):
        calls: list = []
        streams = [
            [_tool_chunk('no_such_tool', '{}')],
            [_chunk('done')],
        ]
        tool_defs = [{"type": "function",
                      "function": {"name": "no_such_tool", "parameters": {}}}]
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client_calls(calls, streams)):
            result = await run_llm_mod.run_llm(
                mock_ctx(settings=SETTINGS, conversation=[]),
                tools=ToolSurface(definitions=tool_defs, functions={}))
        self.assertEqual(result, 'done')
        tool_messages = [m for m in calls[1]['messages']
                         if m.get('role') == 'tool']
        self.assertIn('Unknown tool', tool_messages[0]['content'])


class _FakeStatusError(Exception):
    """A stand-in for openai.APIStatusError carrying a status code."""

    def __init__(self, status: int):
        super().__init__(f'fake status {status}')
        self.status_code = status


class _FakeClock:
    """Deterministic monotonic clock for retry-budget tests."""

    def __init__(self):
        self.now = 100.0

    def time(self) -> float:
        return self.now


def _failing_client(errors: list):
    """A client whose create() raises each error in turn (the last one
    forever), or consumes a chunk list when the entry is a list."""
    calls: list = []

    async def create(**kwargs):
        calls.append(kwargs)
        error = errors[len(calls) - 1 if len(calls) <= len(errors) else -1]
        if isinstance(error, BaseException):
            raise error

        async def stream():
            for c in error:
                yield c
        return stream()

    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    return client, calls


def _completion_cfg() -> run_llm_mod.RunConfig:
    return run_llm_mod.RunConfig(model='m', conversation=Conversation())


class TestRetryPolicy(unittest.IsolatedAsyncioTestCase):
    # The budget is intentionally huge in production (router failover);
    # these tests fake the clock so no test waits in real time.

    def _patched(self, clock: _FakeClock, sleeps: list):
        async def fake_sleep(delay):
            sleeps.append(delay)
            clock.now += delay

        stack = contextlib.ExitStack()
        stack.enter_context(unittest.mock.patch.object(
            run_llm_mod, '_loop_time', clock.time))
        stack.enter_context(unittest.mock.patch.object(
            run_llm_mod, '_sleep', fake_sleep))
        return stack

    async def _run_one(self, errors: list, budget: float,
                       clock: _FakeClock | None = None):
        clock = clock or _FakeClock()
        sleeps: list = []
        client, calls = _failing_client(errors)
        with self._patched(clock, sleeps), \
                unittest.mock.patch.object(run_llm_mod, 'client', client):
            result = await run_llm_mod._one_completion(
                _completion_cfg(),
                SETTINGS,  # type: ignore[arg-type]
                streamer=run_llm_mod.NullDraftSink(),
                strip_thinking=False, budget=budget)
        return result, sleeps, clock

    async def test_client_errors_fail_immediately(self):
        # No point waiting out the budget on a 401 or 403: rerouting
        # cannot fix the key (or the lack of permission).
        for status in (401, 403):
            with self.subTest(status=status):
                result, sleeps, _ = await self._run_one(
                    [_FakeStatusError(status)], budget=1000)
                self.assertIsNone(result)
                self.assertEqual(sleeps, [])

    async def test_request_construction_errors_propagate(self):
        # _request_kwargs is pure assembly, so an exception there is a
        # bug: it must not be swallowed into the retry loop or turned
        # into a "no response" result.
        clock = _FakeClock()
        sleeps: list = []
        client, calls = _failing_client([_FakeStatusError(502)])
        with self._patched(clock, sleeps), \
                unittest.mock.patch.object(run_llm_mod, 'client', client), \
                unittest.mock.patch.object(
                    run_llm_mod, '_request_kwargs',
                    unittest.mock.Mock(
                        side_effect=TypeError('unexpected kwarg'))):
            with self.assertRaises(TypeError):
                await run_llm_mod._one_completion(
                    _completion_cfg(),
                    SETTINGS,  # type: ignore[arg-type]
                    streamer=run_llm_mod.NullDraftSink(),
                    strip_thinking=False, budget=1000)
        self.assertEqual(sleeps, [])
        # The client was never reached: construction failed first.
        self.assertEqual(calls, [])

    async def test_typeerror_from_the_sdk_or_stream_is_retried(self):
        # The same exception types arriving after construction - from the
        # SDK or mid-stream - are presumed transient: a stream ValueError
        # must not become NO_RESPONSE_MESSAGE with zero retries.
        result, sleeps, _ = await self._run_one(
            [TypeError('stream blew up')], budget=1000)
        self.assertIsNone(result)  # budget ran out, not fail-fast
        self.assertGreaterEqual(len(sleeps), 3)

    async def test_zero_or_negative_budget_means_no_retries(self):
        # A misconfigured (<= 0) api_retry_time must not turn into an
        # unbounded retry loop: one attempt runs, the first failure
        # gives up.
        result, sleeps, _ = await self._run_one(
            [_FakeStatusError(502)], budget=0)
        self.assertIsNone(result)
        # No sleep at all: the failure was recognized as terminal at
        # once, not as the start of another wait.
        self.assertEqual(sleeps, [])

    async def test_server_errors_retry_until_budget_spent(self):
        result, sleeps, clock = await self._run_one(
            [_FakeStatusError(500)], budget=10)
        self.assertIsNone(result)
        # Retried (not given up on the first error), and only until the
        # wall-clock budget ran out.
        self.assertGreaterEqual(len(sleeps), 3)
        self.assertGreaterEqual(clock.now, 110.0)

    async def test_backoff_is_clamped_to_the_max_delay(self):
        clock = _FakeClock()
        sleeps: list = []
        client, _ = _failing_client([_FakeStatusError(429)])
        with self._patched(clock, sleeps), \
                unittest.mock.patch.object(
                    run_llm_mod, '_retry_max_delay', lambda: 3.0), \
                unittest.mock.patch.object(run_llm_mod, 'client', client):
            result = await run_llm_mod._one_completion(
                _completion_cfg(),
                SETTINGS,  # type: ignore[arg-type]
                streamer=run_llm_mod.NullDraftSink(),
                strip_thinking=False, budget=100)
        self.assertIsNone(result)
        # 1.2x growth with a hard per-sleep cap.
        self.assertLessEqual(max(sleeps), 3.0)
        self.assertIn(3.0, sleeps)

    async def test_cancellation_interrupts_the_retry_sleep(self):
        # subagent_stop / /stop cancel the task; the cancellation must
        # come out of the retry loop immediately, not be swallowed as a
        # retryable failure.
        clock = _FakeClock()
        sleeps: list = []
        client, _ = _failing_client([_FakeStatusError(502)])

        async def fake_sleep(delay):
            sleeps.append(delay)
            clock.now += delay
            raise asyncio.CancelledError

        with unittest.mock.patch.object(run_llm_mod, '_loop_time',
                                        clock.time), \
                unittest.mock.patch.object(run_llm_mod, '_sleep',
                                           fake_sleep), \
                unittest.mock.patch.object(run_llm_mod, 'client', client):
            with self.assertRaises(asyncio.CancelledError):
                await run_llm_mod._one_completion(
                    _completion_cfg(),
                    SETTINGS,  # type: ignore[arg-type]
                    streamer=run_llm_mod.NullDraftSink(),
                    strip_thinking=False, budget=1000)
        # It was mid-retry (at least one sleep) when the stop landed.
        self.assertGreaterEqual(len(sleeps), 1)

    async def test_run_llm_returns_no_response_when_budget_spent(self):
        ctx = mock_ctx(settings=SETTINGS, conversation=[])
        sleeps: list = []
        clock = _FakeClock()
        client, _ = _failing_client([_FakeStatusError(500)])
        with unittest.mock.patch.object(run_llm_mod, '_loop_time',
                                        clock.time), \
                unittest.mock.patch.object(run_llm_mod, '_sleep',
                                           _advance(clock, sleeps)), \
                unittest.mock.patch.object(run_llm_mod, 'client', client), \
                unittest.mock.patch.object(config, 'api_retry_time', 5):
            result = await run_llm_mod.run_llm(ctx)
        self.assertEqual(result, run_llm_mod.NO_RESPONSE_MESSAGE)

    async def test_silent_sink_absorbs_the_draft_lifecycle(self):
        # run_llm drives the sink identically whether or not drafts are
        # possible, so the subagent sink must take every call as a no-op.
        sink = run_llm_mod.NullDraftSink()
        await sink.start()
        sink.new_round()
        await sink.update('delta')
        await sink.flush(force=True)


def _advance(clock: _FakeClock, sleeps: list):
    async def fake_sleep(delay):
        sleeps.append(delay)
        clock.now += delay
    return fake_sleep


if __name__ == '__main__':
    unittest.main()
