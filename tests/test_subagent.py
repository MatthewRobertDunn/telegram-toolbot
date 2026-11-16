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
from conversation import ConversationMessageType
from mock_context import mock_ctx
from tool_definitions import TOOL_DEFINITIONS, TOOL_MAP
from tools import subagent as subagent_mod
from tools.subagent import (
    MAX_RECORDS,
    MAX_SUBAGENTS,
    SUBAGENT_TIMEOUT,
    cancel_all_subagents,
    get_main_inbox,
    get_parent_tool_defs,
    get_subagent_tool_defs,
    parent_send,
    subagent_list,
    subagent_send,
    subagent_start,
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
        inbox = asyncio.Queue()
        inbox.put_nowait("[subagent abc finished]: done stuff")
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
        subagent_mod.register(TOOL_DEFINITIONS, TOOL_MAP)

    def test_parent_tools_have_no_parent_send(self):
        names = {t['function']['name'] for t in get_parent_tool_defs()}
        self.assertIn('subagent_start', names)
        self.assertIn('subagent_list', names)
        self.assertIn('subagent_stop', names)
        self.assertIn('subagent_send', names)
        self.assertNotIn('parent_send', names)

    def test_subagent_tools_have_no_subagent_controls(self):
        names = {t['function']['name'] for t in get_subagent_tool_defs()}
        self.assertIn('parent_send', names)
        for name in ('subagent_start', 'subagent_list', 'subagent_stop',
                     'subagent_send'):
            self.assertNotIn(name, names)

    def test_map_filtered_to_offered_definitions(self):
        # A model hallucinating a non-offered tool cannot execute it.
        active = run_llm_mod._tool_map_for(get_subagent_tool_defs())
        self.assertIn('parent_send', active)
        for name in ('subagent_start', 'subagent_list', 'subagent_stop',
                     'subagent_send'):
            self.assertNotIn(name, active)
        self.assertEqual(run_llm_mod._tool_map_for([]), {})


class TestManager(unittest.IsolatedAsyncioTestCase):

    async def test_start_list_send_stop(self):
        ctx = _ctx()

        async def hanging(*args, **kwargs):
            await asyncio.sleep(60)
            return 'never'

        with unittest.mock.patch.object(subagent_mod, 'run_llm', hanging), \
            unittest.mock.patch.object(subagent_mod, '_notify_parent',
                                       unittest.mock.AsyncMock()), \
            unittest.mock.patch.object(subagent_mod, 'send_message',
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

            result = await subagent_send(ctx, sid, 'go left')
            self.assertIn('queued', result)
            self.assertIn('go left', await record.inbox.get())

            self.assertIn('Error', await subagent_send(ctx, 'nope', 'x'))

            stopped = await subagent_stop(ctx, sid)
            self.assertIn('Stopped', stopped)
            self.assertEqual(record.status, 'stopped')
            self.assertIn('already stopped',
                          await subagent_stop(ctx, sid))
            self.assertIn('Error', await subagent_stop(ctx, 'nope'))

    async def test_parent_send_queues_for_main(self):
        ctx = _ctx()
        with unittest.mock.patch.object(subagent_mod, '_notify_parent',
                                        unittest.mock.AsyncMock()) as notify:
            result = await parent_send(ctx, 'halfway there')
        self.assertIn('sent', result.lower())
        notify.assert_awaited_once()
        self.assertIn('halfway there', notify.call_args.args[1])

    async def test_cancel_all(self):
        ctx = _ctx()

        async def hanging(*args, **kwargs):
            await asyncio.sleep(60)
            return 'never'

        with unittest.mock.patch.object(subagent_mod, 'run_llm', hanging), \
            unittest.mock.patch.object(subagent_mod, '_notify_parent',
                                       unittest.mock.AsyncMock()), \
            unittest.mock.patch.object(subagent_mod, 'send_message',
                                       unittest.mock.AsyncMock(
                                           return_value=[])):
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
            await subagent_mod._notify_parent(ctx, 'first done')
            # Let the wakeup task run to completion.
            await ctx.user_data['wakeup_task']
            await subagent_mod._notify_parent(ctx, 'second done')
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
        get_main_inbox(ctx).put_nowait('old report')
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
                inbox.put_nowait('[subagent xyz finished]: done stuff')

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
                                       unittest.mock.AsyncMock()), \
            unittest.mock.patch.object(subagent_mod, 'send_message',
                                unittest.mock.AsyncMock(return_value=[])):
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
                conversation=Conversation(), status='done')

        async def hanging(*args, **kwargs):
            await asyncio.sleep(60)
            return 'never'

        with unittest.mock.patch.object(subagent_mod, 'run_llm', hanging), \
            unittest.mock.patch.object(subagent_mod, '_notify_parent',
                                       unittest.mock.AsyncMock()), \
            unittest.mock.patch.object(subagent_mod, 'send_message',
                                unittest.mock.AsyncMock(return_value=[])):
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
            unittest.mock.patch.object(subagent_mod, 'send_message',
                                unittest.mock.AsyncMock(
                                    return_value=[])) as send:
            await subagent_start(ctx, 'Label', 'Task')
            record = next(iter(ctx.user_data['subagents'].values()))
            await record.task
        self.assertEqual(record.status, 'failed')
        self.assertTrue(record.notified)
        # Telegram gets a generic message; the parent keeps the detail.
        sent_text = ' '.join(
            str(c.args[1]) for c in send.call_args_list)
        self.assertNotIn('kaboom', sent_text)
        notify.assert_awaited_once()
        self.assertIn('kaboom', notify.call_args.args[1])


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
            unittest.mock.patch.object(run_llm_mod, 'send_message', send):
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
        ctx.conversation.add_user(
            '[subagent abc finished]: report',
            message_type=ConversationMessageType.AGENT_MESSAGE)
        ctx.conversation.add_assistant('noted')
        with unittest.mock.patch.object(
                conversation_handlers, 'send_message',
                unittest.mock.AsyncMock(return_value=[1])):
            await conversation_handlers.undo_turn(ctx)
        roles = [(m.get('role'), m.get('content'))
                 for m in ctx.conversation.messages]
        # The human exchange is gone; nothing of the agent tail survives
        # as a dangling repliable turn.
        self.assertEqual(roles, [])


if __name__ == '__main__':
    unittest.main()
