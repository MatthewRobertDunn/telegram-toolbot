"""Unit tests for background subagents (no network)."""

import asyncio
import os
import sys
import unittest
import unittest.mock
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from handlers import run_llm as run_llm_mod
from agent_inbox import SubagentInbox, SubagentMessage
from chat_update_context import (AgentContext,
                                 ChatUpdateContext)
from conversation import Conversation, ConversationMessageType
from messaging import Messaging
from mock_context import mock_ctx
from tool_definitions import TOOL_DEFINITIONS, TOOL_MAP
from tool_scope import BOTH, tool_scope
from tools import subagent as subagent_mod
from tools.subagent import (
    CONTROL_TOOL_DEFINITIONS,
    CONTROL_TOOL_MAP,
    MAX_RECORDS,
    MAX_SUBAGENTS,
    SUBAGENT_TIMEOUT,
    SubagentRecord,
    cancel_all_subagents,
    format_duration,
    get_main_inbox,
    get_parent_tool_defs,
    get_subagent_tool_defs,
    subagent_list,
    subagent_send_to_parent,
    subagent_start,
    subagent_steer,
    subagent_stop,
)

SETTINGS = SimpleNamespace(
    show_tool_output=False, model='m', temperature=1.0, top_p=0.95,
    subagent_model='sub-m')


def _ctx():
    ctx = mock_ctx(settings=SETTINGS, conversation=[])
    return ctx


def _chunk(content=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=content, reasoning_content=None,
        tool_calls=None), finish_reason=None)])


def _fake_client(captured, chunks):
    async def fake_create(**kwargs):
        captured.update(kwargs)

        async def stream():
            for c in chunks:
                yield c
        return stream()
    return SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=fake_create)))


class TestInboxDrain(unittest.IsolatedAsyncioTestCase):

    async def test_inbox_messages_land_in_conversation(self):
        ctx = _ctx()
        inbox = SubagentInbox()
        inbox.put_nowait(SubagentMessage(
            id='abc', state='finished', description='T',
            result='done stuff'))
        captured = {}
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client(captured, [_chunk('reply')])):
            result = await run_llm_mod.run_llm(ctx, inbox=inbox)
        self.assertEqual(result, 'reply')
        users = [m for m in ctx.conversation.messages if m['role'] == 'user']
        self.assertEqual(len(users), 1)
        self.assertIn('done stuff', users[0]['content'])
        # The injected turn was part of the request.
        self.assertIn('done stuff', captured['messages'][-1]['content'])

    async def test_no_inbox_means_no_user_turn(self):
        ctx = _ctx()
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client({}, [_chunk('reply')])):
            result = await run_llm_mod.run_llm(ctx)
        self.assertEqual(result, 'reply')
        self.assertEqual(
            [m for m in ctx.conversation.messages if m['role'] == 'user'], [])

    async def test_no_tools_by_default(self):
        ctx = _ctx()
        captured = {}
        with unittest.mock.patch.object(
                run_llm_mod, 'client',
                _fake_client(captured, [_chunk('reply')])):
            await run_llm_mod.run_llm(ctx)
        self.assertNotIn('tools', captured)


class TestToolSets(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Register the real tool surface once; get_*_tool_defs read the
        # global registry.
        if not TOOL_DEFINITIONS:
            from tool_registration import register_tools
            register_tools()

    def test_parent_tools_include_send_to_parent(self):
        names = {t['function']['name'] for t in get_parent_tool_defs()}
        self.assertIn('subagent_start', names)
        self.assertIn('subagent_list', names)
        self.assertIn('subagent_stop', names)
        self.assertIn('subagent_steer', names)
        # Shared steering tool: offered to the parent too, so role
        # confusion degrades to a working tool instead of 'Unknown tool'.
        self.assertIn('subagent_send_to_parent', names)

    def test_subagent_tools_have_no_subagent_controls(self):
        names = {t['function']['name'] for t in get_subagent_tool_defs()}
        self.assertIn('subagent_send_to_parent', names)
        # Shared toolbox: file, sandbox, web, journal and memory work.
        for name in ('bash', 'read', 'write', 'web_search', 'python',
                     'journal_add', 'memory_set'):
            self.assertIn(name, names)
        # Chat-writing tools are never offered: they would have no way to
        # act (a subagent context has no messaging handle).
        for name in ('subagent_start', 'subagent_list', 'subagent_stop',
                     'subagent_steer', 'send_file', 'read_image',
                     'generate_image'):
            self.assertNotIn(name, names)
        # ...but the silent variants are.
        self.assertIn('read_image_subagent', names)
        self.assertIn('generate_image_subagent', names)

    def test_map_filtered_to_offered_definitions(self):
        # A model hallucinating a non-offered tool cannot execute it.
        active = run_llm_mod._tool_map_for(get_subagent_tool_defs())
        self.assertIn('subagent_send_to_parent', active)
        for name in ('subagent_start', 'subagent_list', 'subagent_stop',
                     'subagent_steer'):
            self.assertNotIn(name, active)
        self.assertEqual(run_llm_mod._tool_map_for([]), {})


class TestSubagentContextWiring(unittest.IsolatedAsyncioTestCase):
    # Subagents run on their own silent context; their mail and wakeups go
    # through the parent's interactive one.

    async def test_run_llm_gets_an_isolated_silent_context(self):
        parent = _ctx()
        record = SubagentRecord(id='sub1', description='Job',
                                conversation=Conversation(), parent=parent)
        captured: dict = {}

        async def fake_run_llm(ctx, **kwargs):
            captured['ctx'] = ctx
            captured['kwargs'] = kwargs
            return 'done'

        with unittest.mock.patch.object(subagent_mod, 'run_llm',
                                        fake_run_llm), \
            unittest.mock.patch.object(subagent_mod, '_notify_parent',
                                       unittest.mock.AsyncMock()):
            await subagent_mod._run_subagent(parent, record, model='sub-m')

        sub_ctx = captured['ctx']
        self.assertIsInstance(sub_ctx, AgentContext)
        self.assertNotIsInstance(sub_ctx, ChatUpdateContext)
        self.assertFalse(hasattr(sub_ctx, 'messaging'))
        self.assertIs(sub_ctx.conversation, record.conversation)
        self.assertIsNot(sub_ctx.user_data, parent.user_data)
        self.assertEqual(captured['kwargs']['model'], 'sub-m')
        self.assertEqual(record.status, 'done')
        self.assertEqual(record.result, 'done')

    async def test_send_to_parent_uses_the_parent_context(self):
        parent = _ctx()
        record = SubagentRecord(id='sub1', description='Job',
                                conversation=Conversation(), parent=parent)
        sub_ctx = parent.new_subagent_context(record.conversation)
        token = subagent_mod._current_subagent.set(record)
        try:
            with unittest.mock.patch.object(
                    subagent_mod, '_notify_parent',
                    unittest.mock.AsyncMock()) as notify:
                result = await subagent_send_to_parent(sub_ctx, 'half done')
        finally:
            subagent_mod._current_subagent.reset(token)
        self.assertIn('sent', result.lower())
        parent_arg, message = notify.call_args.args
        # Not the silent context: running the parent loop against it would
        # write into the subagent's history and post nothing.
        self.assertIs(parent_arg, parent)
        self.assertEqual(message.id, 'sub1')
        self.assertEqual(message.description, 'Job')
        self.assertEqual(message.result, 'half done')


class TestManager(unittest.IsolatedAsyncioTestCase):

    async def test_start_list_send_stop(self):
        ctx = _ctx()

        async def hanging(*args, **kwargs):
            await asyncio.sleep(60)
            return 'never'

        with unittest.mock.patch.object(subagent_mod, 'run_llm', hanging), \
            unittest.mock.patch.object(subagent_mod, '_notify_parent',
                                       unittest.mock.AsyncMock()), \
            unittest.mock.patch.object(Messaging, 'send_message',
                                       unittest.mock.AsyncMock(
                                           return_value=[])):
            started = await subagent_start(ctx, 'Task', 'Do it')
            sid = next(iter(ctx.user_data['subagents']))
            self.assertIn(sid, started)

            listed = await subagent_list(ctx)
            self.assertIn(sid, listed)
            self.assertIn('Task', listed)

            record = ctx.user_data['subagents'][sid]
            self.assertIs(get_main_inbox(ctx),
                          ctx.user_data['main_inbox'])

            result = await subagent_steer(ctx, sid, 'go left')
            self.assertIn('queued', result)
            self.assertIn('go left', (await record.inbox.get()).result)

            self.assertIn('Error', await subagent_steer(ctx, 'nope', 'x'))

            stopped = await subagent_stop(ctx, sid)
            self.assertIn('Stopped', stopped)
            self.assertEqual(record.status, 'stopped')
            self.assertIn('already stopped',
                          await subagent_stop(ctx, sid))
            self.assertIn('Error', await subagent_stop(ctx, 'nope'))

    async def test_start_rejects_empty_description(self):
        ctx = _ctx()
        for bad in ('', '   '):
            result = await subagent_start(ctx, bad, 'Do it')
            self.assertIn('Error', result)
        self.assertEqual(ctx.user_data.get('subagents', {}), {})

    async def test_send_to_parent_queues_for_main(self):
        ctx = _ctx()
        with unittest.mock.patch.object(subagent_mod, '_notify_parent',
                                        unittest.mock.AsyncMock()) as notify:
            result = await subagent_send_to_parent(ctx, 'halfway there')
        self.assertIn('sent', result.lower())
        notify.assert_awaited_once()
        self.assertIn('halfway there', notify.call_args.args[1].result)
        # No subagent context: the parent itself is the sender.
        self.assertEqual(notify.call_args.args[1].id, 'parent')

    async def test_send_to_parent_shared_with_parent_tools(self):
        # The parent loop offers it too: role confusion degrades to a
        # working tool instead of an 'Unknown tool' error.
        names = {t['function']['name'] for t in CONTROL_TOOL_DEFINITIONS}
        self.assertIn('subagent_send_to_parent', names)
        self.assertEqual(tool_scope(CONTROL_TOOL_MAP['subagent_send_to_parent']),
                         BOTH)
        with unittest.mock.patch.object(run_llm_mod, 'TOOL_MAP',
                                        CONTROL_TOOL_MAP):
            active = run_llm_mod._tool_map_for(list(CONTROL_TOOL_DEFINITIONS))
        self.assertIn('subagent_send_to_parent', active)

    async def test_cancel_all(self):
        ctx = _ctx()

        async def hanging(*args, **kwargs):
            await asyncio.sleep(60)
            return 'never'

        with unittest.mock.patch.object(subagent_mod, 'run_llm', hanging), \
            unittest.mock.patch.object(subagent_mod, '_notify_parent',
                                       unittest.mock.AsyncMock()):
            await subagent_start(ctx, 'A', 'x')
            await subagent_start(ctx, 'B', 'y')
            self.assertEqual(len(ctx.user_data['subagents']), 2)
            cancel_all_subagents(ctx)
            self.assertEqual(ctx.user_data['subagents'], {})


def _fake_bot():
    bot = unittest.mock.AsyncMock()
    bot.send_message.side_effect = lambda *a, **k: SimpleNamespace(
        message_id=7)
    return bot


class TestParentWakeup(unittest.IsolatedAsyncioTestCase):

    @staticmethod
    async def _run_job_submit(ctx, job):
        await job

    async def test_two_finishers_cause_one_wakeup(self):
        from handlers import generation_queue as gq
        ctx = _ctx()
        with unittest.mock.patch.object(
                gq, 'submit_task',
                unittest.mock.AsyncMock(
                    side_effect=self._run_job_submit)) as submit, \
            unittest.mock.patch.object(
                gq, 'generate_response',
                unittest.mock.AsyncMock(return_value=None)) as gen:
            await subagent_mod._notify_parent(
                ctx, SubagentMessage(id='x', state='running',
                                     description='T', result='first done'))
            # Let the wakeup task run to completion.
            await ctx.user_data['wakeup_task']
            await subagent_mod._notify_parent(
                ctx, SubagentMessage(id='x', state='running',
                                     description='T', result='second done'))
            wakeup = ctx.user_data.get('wakeup_task')
            if wakeup is not None:
                await wakeup
        # First notify wakes; the second arrives while the loop is active
        # (or its mail is drained by the same generation).
        self.assertTrue(submit.await_count >= 1)
        self.assertTrue(gen.await_count >= 1)

    async def test_stale_epoch_skips_generation(self):
        from agent_inbox import bump_conversation_epoch
        from handlers import generation_queue as gq
        ctx = _ctx()
        get_main_inbox(ctx).put_nowait(SubagentMessage(
            id='x', state='finished', description='T', result='old report'))
        with unittest.mock.patch.object(
                gq, 'submit_task',
                unittest.mock.AsyncMock(
                    side_effect=self._run_job_submit)) as submit, \
            unittest.mock.patch.object(
                gq, 'generate_response',
                unittest.mock.AsyncMock(return_value=None)) as gen:
            bump_conversation_epoch(ctx)  # /start happened meanwhile
            await subagent_mod._wakeup_parent(ctx, epoch=0)
        submit.assert_awaited_once()
        gen.assert_not_awaited()

    async def test_wakeup_skipped_when_inbox_empty(self):
        from handlers import generation_queue as gq
        ctx = _ctx()
        with unittest.mock.patch.object(
                gq, 'submit_task',
                unittest.mock.AsyncMock(
                    side_effect=self._run_job_submit)) as submit, \
            unittest.mock.patch.object(
                gq, 'generate_response',
                unittest.mock.AsyncMock(return_value=None)) as gen:
            await subagent_mod._wakeup_parent(ctx, epoch=0)
        submit.assert_awaited_once()
        gen.assert_not_awaited()


class TestPostReplyDrain(unittest.IsolatedAsyncioTestCase):

    async def test_mail_during_final_round_gets_another_turn(self):
        ctx = _ctx()
        inbox = get_main_inbox(ctx)
        calls = []

        async def fake_create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                inbox.put_nowait(SubagentMessage(
                    id='xyz', state='finished', description='T',
                    result='done stuff'))

            async def stream():
                yield _chunk('reply %d' % len(calls))
            return stream()

        fake_client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create)))
        with unittest.mock.patch.object(
                run_llm_mod, 'client', fake_client):
            result = await run_llm_mod.run_llm(ctx, inbox=inbox)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result, 'reply 2')
        self.assertIn('done stuff', calls[1]['messages'][-1]['content'])


class TestSubagentLimits(unittest.IsolatedAsyncioTestCase):

    async def test_max_subagents_cap(self):
        ctx = _ctx()

        async def hanging(*args, **kwargs):
            await asyncio.sleep(60)
            return 'never'

        with unittest.mock.patch.object(subagent_mod, 'run_llm', hanging), \
            unittest.mock.patch.object(subagent_mod, '_notify_parent',
                                       unittest.mock.AsyncMock()):
            for _ in range(MAX_SUBAGENTS):
                await subagent_start(ctx, 'T', 'x')
            denied = await subagent_start(ctx, 'T', 'one too many')
            self.assertIn('too many', denied)
            self.assertEqual(
                len(ctx.user_data['subagents']), MAX_SUBAGENTS)
            cancel_all_subagents(ctx)

    async def test_finished_records_pruned(self):
        from tools.subagent import SubagentRecord
        from conversation import Conversation
        ctx = _ctx()
        subs = subagent_mod.get_subagents(ctx)
        for i in range(MAX_RECORDS):
            subs[f'old{i}'] = SubagentRecord(
                id=f'old{i}', description='old',
                conversation=Conversation(), status='done', parent=ctx)

        async def hanging(*args, **kwargs):
            await asyncio.sleep(60)
            return 'never'

        with unittest.mock.patch.object(subagent_mod, 'run_llm', hanging), \
            unittest.mock.patch.object(subagent_mod, '_notify_parent',
                                       unittest.mock.AsyncMock()):
            await subagent_start(ctx, 'New', 'x')
            cancel_all_subagents(ctx)
        self.assertEqual(len(subs), MAX_RECORDS)
        self.assertNotIn('old0', subs)

    async def test_failed_subagent_notifies(self):
        ctx = _ctx()

        async def boom(*args, **kwargs):
            raise RuntimeError('kaboom')

        with unittest.mock.patch.object(subagent_mod, 'run_llm', boom), \
            unittest.mock.patch.object(subagent_mod, '_notify_parent',
                                       unittest.mock.AsyncMock()) as notify, \
            unittest.mock.patch.object(Messaging, 'send_message',
                                       unittest.mock.AsyncMock(
                                           return_value=[])) as send:
            await subagent_start(ctx, 'Label', 'Task')
            record = next(iter(ctx.user_data['subagents'].values()))
            await record.task
        self.assertEqual(record.status, 'failed')
        self.assertTrue(record.notified)
        # Single path: nothing goes straight to Telegram; the inbox
        # receipt (carrying the detail) is the only user-visible signal.
        send.assert_not_awaited()
        notify.assert_awaited_once()
        self.assertIn('kaboom', notify.call_args.args[1].result)


class TestSubagentListDurations(unittest.IsolatedAsyncioTestCase):

    def test_format_duration(self):
        self.assertEqual(format_duration(0), '0s')
        self.assertEqual(format_duration(45), '45s')
        self.assertEqual(format_duration(83), '1m23s')
        self.assertEqual(format_duration(2 * 3600 + 4 * 60), '2h04m')

    async def test_running_shows_elapsed(self):
        import time
        from conversation import Conversation
        ctx = _ctx()
        rec = SubagentRecord(
            id='abc123', description='Research',
            conversation=Conversation(), parent=ctx,
            started_at=time.monotonic() - 83)
        ctx.user_data['subagents'] = {'abc123': rec}
        text = await subagent_list(ctx)
        self.assertIn('abc123', text)
        self.assertIn('running 1m23s', text)

    async def test_finished_shows_total(self):
        import time
        from conversation import Conversation
        ctx = _ctx()
        rec = SubagentRecord(
            id='abc123', description='Research',
            conversation=Conversation(), parent=ctx,
            started_at=time.monotonic() - 125)
        rec.finish('done')
        # Pin the end so the assertion is deterministic.
        rec.ended_at = rec.started_at + 125
        ctx.user_data['subagents'] = {'abc123': rec}
        text = await subagent_list(ctx)
        self.assertIn('ran 2m05s', text)


class TestAnnounceTools(unittest.IsolatedAsyncioTestCase):

    async def _run_tool_turn(self, announce):
        from conversation import Conversation

        async def fake_tool(ctx):
            return 'tool output'

        setattr(fake_tool, '_requires_context', True)
        tool_defs = [{'type': 'function',
                      'function': {'name': 't', 'parameters': {}}}]
        chunk_tool = SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(
                content=None, reasoning_content=None,
                tool_calls=[SimpleNamespace(
                    index=0, id='c1',
                    function=SimpleNamespace(
                        name='t', arguments='{}'))]))])
        streams = [[chunk_tool], [_chunk('done')]]
        calls = []

        async def fake_create(**kwargs):
            stream = streams.pop(0)

            async def gen():
                for c in stream:
                    yield c
            return gen()

        fake_client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create)))
        ctx = _ctx()
        ctx.bot = _fake_bot()
        ctx.user_data['settings'] = SimpleNamespace(
            show_tool_output=True, model='m', temperature=1.0, top_p=0.95)
        send = unittest.mock.AsyncMock(return_value=[9])
        with unittest.mock.patch.object(
                run_llm_mod, 'TOOL_MAP', {'t': fake_tool}), \
            unittest.mock.patch.object(
                run_llm_mod, 'client', fake_client), \
            unittest.mock.patch.object(Messaging, 'send_json', send):
            result = await run_llm_mod.run_llm(
                ctx, conversation=Conversation(), tools=tool_defs,
                announce_tools=announce)
        return result, send

    async def test_subagent_turns_stay_silent(self):
        result, send = await self._run_tool_turn(announce=False)
        self.assertEqual(result, 'done')
        send.assert_not_awaited()

    async def test_parent_turns_announce(self):
        result, send = await self._run_tool_turn(announce=True)
        self.assertEqual(result, 'done')
        send.assert_awaited_once()


class TestInboxReceipt(unittest.IsolatedAsyncioTestCase):

    async def _drain(self, message, announce, show_tool_output=True):
        from conversation import Conversation
        ctx = _ctx()
        ctx.bot = _fake_bot()
        ctx.user_data['settings'] = SimpleNamespace(
            show_tool_output=show_tool_output, model='m',
            temperature=1.0, top_p=0.95)
        conversation = Conversation()
        inbox = SubagentInbox()
        inbox.put_nowait(message)
        send = unittest.mock.AsyncMock(return_value=[9])
        ctx.messaging.send_collapsible = send
        drained = await run_llm_mod._drain_inbox(
            ctx, conversation, inbox, ctx if announce else None)
        return drained, conversation, send

    @staticmethod
    def _report(result='report'):
        return SubagentMessage(id='abc', state='finished',
                               description='Job', result=result)

    async def test_receipt_posted_and_linked(self):
        drained, conversation, send = await self._drain(
            self._report(), True)
        self.assertEqual(drained, 1)
        send.assert_awaited_once()
        summary, body = send.call_args.args[0], send.call_args.args[1]
        # Collapsible receipt: summary names sender + state, body is full.
        self.assertIn('Subagent **abc** *Job* — finished', summary)
        self.assertEqual(body, 'report')
        # Its Telegram id is linked to the injected turn, so /undo
        # deletes the post along with the turn.
        user_turn = conversation.messages[-1]
        self.assertEqual(user_turn.get('role'), 'user')
        self.assertEqual(
            user_turn.message_type, ConversationMessageType.AGENT_MESSAGE)
        stored = (user_turn.get('custom') or {}).get('telegram_message_ids')
        self.assertEqual(stored, [9])

    async def test_receipt_carries_full_result(self):
        drained, conversation, send = await self._drain(
            self._report('x' * 2000), True)
        self.assertEqual(drained, 1)
        send.assert_awaited_once()
        body = send.call_args.args[1]
        # No truncation: the details block collapses the full report.
        self.assertIn('x' * 2000, body)
        # The conversation turn keeps the full report for the model.
        self.assertIn('x' * 2000, conversation.messages[-1]['content'])

    async def test_receipt_empty_body_falls_back_to_state(self):
        drained, conversation, send = await self._drain(
            SubagentMessage(id='abc', state='stopped',
                            description='Job'), True)
        self.assertEqual(drained, 1)
        summary, body = send.call_args.args[0], send.call_args.args[1]
        self.assertIn('stopped', summary)
        self.assertEqual(body, '(stopped)')

    async def test_receipt_parent_note_summary(self):
        drained, conversation, send = await self._drain(
            SubagentMessage(id='parent', state='running',
                            result='note'), True)
        self.assertEqual(drained, 1)
        summary, body = send.call_args.args[0], send.call_args.args[1]
        self.assertEqual(summary, 'Parent note')
        self.assertEqual(body, 'note')

    async def test_receipt_running_omits_state_suffix(self):
        drained, conversation, send = await self._drain(
            SubagentMessage(id='abc', state='running',
                            description='Job', result='halfway'), True)
        self.assertEqual(drained, 1)
        summary = send.call_args.args[0]
        self.assertIn('Subagent **abc** *Job*', summary)
        self.assertNotIn('— running', summary)

    async def test_rejects_unstructured_puts(self):
        inbox = SubagentInbox()
        with self.assertRaises(TypeError):
            inbox.put_nowait('just a string')  # type: ignore[arg-type]

    async def test_silent_when_not_announced(self):
        drained, conversation, send = await self._drain(
            self._report('x'), False)
        self.assertEqual(drained, 1)
        send.assert_not_awaited()
        stored = (conversation.messages[-1].get('custom') or {}).get(
            'telegram_message_ids')
        self.assertIsNone(stored)

    async def test_silent_when_show_tool_output_off(self):
        drained, conversation, send = await self._drain(
            self._report('x'), True, show_tool_output=False)
        self.assertEqual(drained, 1)
        send.assert_not_awaited()


class TestAgentMessageUndo(unittest.IsolatedAsyncioTestCase):

    async def test_undo_skips_agent_turns(self):
        from handlers import conversation_handlers
        bot = _fake_bot()
        ctx = mock_ctx(
            bot=bot,
            conversation=[
                {'role': 'user', 'content': 'human question'},
                {'role': 'assistant', 'content': 'human answer'},
            ])
        # A subagent report arrives and gets answered.
        from agent_inbox import SubagentMessage
        ctx.conversation.add_user(
            SubagentMessage(id='abc', state='finished',
                            description='Job', result='report').format(),
            message_type=ConversationMessageType.AGENT_MESSAGE)
        ctx.conversation.add_assistant('noted')
        ctx.messaging.send_message = unittest.mock.AsyncMock(return_value=[1])
        await conversation_handlers.undo_turn(ctx)
        roles = [(m.get('role'), m.get('content'))
                 for m in ctx.conversation.messages]
        # The human exchange is gone; nothing of the agent tail survives
        # as a dangling repliable turn.
        self.assertEqual(roles, [])


if __name__ == '__main__':
    unittest.main()
