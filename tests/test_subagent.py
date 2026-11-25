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
from agent_inbox import (SubagentInbox, SubagentMessage, SubagentStatus,
                         bump_conversation_epoch, clear_main_inbox,
                         get_main_inbox)
from chat_update_context import (AgentContext,
                                 ChatUpdateContext)
from conversation import Conversation, ConversationMessageType
from messaging import Messaging
from mock_context import mock_ctx
from tool_registry import PARENT, SUBAGENT, ToolSurface, all_tools, surface_for
from tool_registration import register_tools
from tools import subagent as subagent_mod
from tools.subagent import (
    MAX_RECORDS,
    MAX_SUBAGENTS,
    SubagentRecord,
    cancel_all_subagents,
    format_duration,
    get_main_inbox,
    subagent_list,
    subagent_send_to_parent,
    subagent_start,
    subagent_steer,
    subagent_stop,
)

SETTINGS = SimpleNamespace(
    show_tool_output=False, model='m', temperature=1.0, top_p=0.95,
    subagent_model='sub-m', thinking_keep=False)


def _ctx():
    # A bot stub just for the typing indicator every queued item now
    # sends; anything that really posts still has to be stubbed by the
    # test that cares about it.
    ctx = mock_ctx(settings=SETTINGS, conversation=[],
                   bot=SimpleNamespace(
                       send_chat_action=unittest.mock.AsyncMock()))
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
            id='abc', state=SubagentStatus.FINISHED, description='T',
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
        # Register the real tool surface once; surface_for reads the
        # registry.
        register_tools()

    def test_parent_tools_exclude_send_to_parent(self):
        names = {t['function']['name'] for t in surface_for(PARENT).definitions}
        self.assertIn('subagent_start', names)
        self.assertIn('subagent_list', names)
        self.assertIn('subagent_stop', names)
        self.assertIn('subagent_steer', names)
        # Subagent-only: the parent loop never offers it, so a parent
        # hallucinating the name gets 'Unknown tool' instead of a
        # self-addressed note.
        self.assertNotIn('subagent_send_to_parent', names)

    def test_subagent_tools_have_no_subagent_controls(self):
        names = {t['function']['name'] for t in surface_for(SUBAGENT).definitions}
        self.assertIn('subagent_send_to_parent', names)
        # Shared toolbox: file, sandbox, web, journal and memory work.
        for name in ('bash', 'read', 'write', 'web_search', 'python',
                     'journal_add', 'memory_set'):
            self.assertIn(name, names)
        # Chat-writing tools are never offered: they would have no way to
        # act (a subagent context has no messaging handle).
        for name in ('subagent_start', 'subagent_list', 'subagent_stop',
                     'subagent_steer', 'send_file'):
            self.assertNotIn(name, names)
        # The tools that would post keep their name but swap
        # implementation, so the subagent still has them.
        self.assertIn('read_image', names)
        self.assertIn('generate_image', names)
        self.assertNotIn('read_image_subagent', names)
        self.assertNotIn('generate_image_subagent', names)

    def test_surface_pairs_definitions_with_implementations(self):
        # definitions and functions come from one registry view, so a
        # name the model is offered is always executable and vice versa.
        for role in (PARENT, SUBAGENT):
            surface = surface_for(role)
            names = {t['function']['name'] for t in surface.definitions}
            self.assertEqual(set(surface.functions), names)

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
        self.assertEqual(record.status, SubagentStatus.FINISHED)
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
            sid = next(iter(ctx.thread_data.subagents))
            self.assertIn(sid, started)

            listed = await subagent_list(ctx)
            self.assertIn(sid, listed)
            self.assertIn('Task', listed)

            record = ctx.thread_data.subagents[sid]
            self.assertIs(get_main_inbox(ctx),
                          ctx.thread_data.main_inbox)

            result = await subagent_steer(ctx, sid, 'go left')
            self.assertIn('queued', result)
            self.assertIn('go left', (await record.inbox.get()).result)

            self.assertIn('Error', await subagent_steer(ctx, 'nope', 'x'))

            stopped = await subagent_stop(ctx, sid)
            self.assertIn('Stopped', stopped)
            self.assertEqual(record.status, SubagentStatus.STOPPED)
            self.assertIn('already stopped',
                          await subagent_stop(ctx, sid))
            self.assertIn('Error', await subagent_stop(ctx, 'nope'))

    async def test_start_rejects_empty_description(self):
        ctx = _ctx()
        for bad in ('', '   '):
            result = await subagent_start(ctx, bad, 'Do it')
            self.assertIn('Error', result)
        self.assertEqual(ctx.thread_data.subagents, {})

    async def test_send_to_parent_queues_for_main(self):
        parent = _ctx()
        record = SubagentRecord(id='abc', description='T',
                                conversation=Conversation([]), parent=parent)
        token = subagent_mod._current_subagent.set(record)
        try:
            with unittest.mock.patch.object(
                    subagent_mod, '_notify_parent',
                    unittest.mock.AsyncMock()) as notify:
                result = await subagent_send_to_parent(parent,
                                                       'halfway there')
        finally:
            subagent_mod._current_subagent.reset(token)
        self.assertIn('sent', result.lower())
        notify.assert_awaited_once()
        self.assertIn('halfway there', notify.call_args.args[1].result)
        # The record identifies the sender.
        self.assertEqual(notify.call_args.args[1].id, 'abc')

    async def test_send_to_parent_without_record_is_an_error(self):
        ctx = _ctx()
        result = await subagent_send_to_parent(ctx, 'hello?')
        self.assertIn('Error', result)

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
            self.assertEqual(len(ctx.thread_data.subagents), 2)
            cancel_all_subagents(ctx)
            self.assertEqual(ctx.thread_data.subagents, {})


class TestSubagentStartModel(unittest.IsolatedAsyncioTestCase):
    # subagent_start's optional model overrides the per-user subagent model
    # setting; anything blank is no override, so the setting decides.

    @staticmethod
    async def _started_model(ctx, **kwargs) -> str:
        """Start one subagent, run it to the end, return the model it used."""
        captured: dict = {}

        async def fake_run_llm(ctx, **kw):
            captured.update(kw)
            return 'done'

        with unittest.mock.patch.object(subagent_mod, 'run_llm',
                                        fake_run_llm), \
            unittest.mock.patch.object(subagent_mod, '_notify_parent',
                                       unittest.mock.AsyncMock()):
            await subagent_start(ctx, 'Task', 'Do it', **kwargs)
            record = next(iter(ctx.thread_data.subagents.values()))
            assert record.task is not None
            await record.task
        return captured['model']

    async def test_explicit_model_overrides_the_setting(self):
        ctx = _ctx()
        self.assertEqual(await self._started_model(ctx, model='fast-m'),
                         'fast-m')

    async def test_omitted_model_uses_the_setting(self):
        ctx = _ctx()
        self.assertEqual(await self._started_model(ctx), 'sub-m')

    async def test_blank_model_is_no_override(self):
        # A model can send "" for an optional string; that is not a model
        # id, so the setting stays in charge instead of failing the run.
        ctx = _ctx()
        self.assertEqual(await self._started_model(ctx, model='   '), 'sub-m')

    def test_model_is_declared_optional(self):
        register_tools()
        tool = next(t for t in all_tools() if t.name == 'subagent_start')
        self.assertEqual(tool.properties['model']['type'], 'string')
        self.assertNotIn('model', tool.required)


def _fake_bot():
    bot = unittest.mock.AsyncMock()
    bot.send_message.side_effect = lambda *a, **k: SimpleNamespace(
        message_id=7)
    return bot


class TestParentWakeup(unittest.IsolatedAsyncioTestCase):
    # Always-enqueue wakeup protocol: every delivery schedules one wakeup
    # job; there is no "loop active" flag left to leak. The first job
    # drains the whole inbox, the rest no-op on the empty inbox.

    @staticmethod
    def _draining_run_llm():
        """A run_llm stand-in that drains the inbox like the real one."""
        async def fake_run_llm(ctx, **kwargs):
            inbox = kwargs.get('inbox')
            if inbox is not None:
                while True:
                    try:
                        message = inbox.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    ctx.conversation.add_user(message.format())
            return 'ack'
        return fake_run_llm

    async def _patch_generation(self, run_llm_mock=None):
        from handlers import generation_queue as gq
        # The worker persists after every job; keep tests off sqlite.
        saver = unittest.mock.patch.object(
            gq, 'save_conversation', unittest.mock.AsyncMock())
        run = unittest.mock.patch.object(
            gq, 'run_llm', run_llm_mock or self._draining_run_llm())
        send = unittest.mock.patch.object(
            Messaging, 'send_message',
            unittest.mock.AsyncMock(return_value=[1]))
        return saver, run, send

    async def test_idle_mail_wakes_the_parent_and_drains_it(self):
        from handlers import generation_queue as gq
        ctx = _ctx()
        get_main_inbox(ctx).put_nowait(SubagentMessage(
            id='abc', state=SubagentStatus.FINISHED, description='T',
            result='done stuff'))
        saver, run, send = await self._patch_generation()
        with saver, run, send:
            await subagent_mod._wakeup_parent(ctx, epoch=0)
        # Mail was consumed by a real parent turn.
        self.assertTrue(get_main_inbox(ctx).empty())
        users = [m for m in ctx.conversation.messages if m['role'] == 'user']
        self.assertIn('done stuff', users[0]['content'])

    async def test_busy_mail_queues_behind_the_current_job(self):
        from handlers import generation_queue as gq
        ctx = _ctx()
        release = asyncio.Event()

        async def blocker():
            await release.wait()

        async def main():
            saver, run, send = await self._patch_generation()
            with saver, run, send:
                # The parent is busy: its job is in the worker already.
                busy = asyncio.ensure_future(gq.submit_task(ctx, blocker))
                await asyncio.sleep(0)
                get_main_inbox(ctx).put_nowait(SubagentMessage(
                    id='abc', state=SubagentStatus.FINISHED, description='T',
                    result='late report'))
                wakeup = subagent_mod.ensure_parent_wakeup(ctx)
                await asyncio.sleep(0)  # the wakeup submits its own item
                release.set()
                await busy
                await wakeup
            # The queued wakeup ran after the busy job and drained the
            # mail that arrived mid-job.
            self.assertTrue(get_main_inbox(ctx).empty())
            users = [m for m in ctx.conversation.messages
                     if m['role'] == 'user']
            self.assertIn('late report', users[0]['content'])

        await asyncio.wait_for(main(), 5)

    async def test_five_notifications_drain_without_loss(self):
        ctx = _ctx()
        inbox = get_main_inbox(ctx)
        for i in range(5):
            inbox.put_nowait(SubagentMessage(
                id=f's{i}', state=SubagentStatus.FINISHED, description='T',
                result=f'report {i}'))
        saver, run, send = await self._patch_generation()
        with saver, run, send:
            wakeups = [subagent_mod.ensure_parent_wakeup(ctx)
                       for _ in range(5)]
            await asyncio.gather(*wakeups)
        # Every delivery landed exactly once; the redundant wakeups
        # no-oped instead of generating again.
        self.assertTrue(inbox.empty())
        users = [m for m in ctx.conversation.messages if m['role'] == 'user']
        self.assertEqual(len(users), 5)

    async def test_empty_or_drained_inbox_never_generates(self):
        # Two ways the inbox can be empty when a wakeup job runs: nothing
        # ever arrived, or an earlier wakeup already drained the mail. A
        # drained one still submits its job, but that job must return
        # before any generation instead of replying again.
        from handlers import generation_queue as gq
        for scenario in ('drained', 'never-populated'):
            with self.subTest(scenario=scenario):
                ctx = _ctx()
                submitted: list = []
                run_llm_mock = unittest.mock.AsyncMock(return_value='ack')

                async def fake_submit(ctx, job, _submitted=submitted):
                    _submitted.append(job)

                inbox = get_main_inbox(ctx)
                if scenario == 'drained':
                    inbox.put_nowait(SubagentMessage(
                        id='abc', state=SubagentStatus.FINISHED,
                        description='T', result='report'))
                with unittest.mock.patch.object(gq, 'submit_task',
                                                fake_submit), \
                        unittest.mock.patch.object(gq, 'run_llm',
                                                   run_llm_mock):
                    await subagent_mod._wakeup_parent(ctx, epoch=0)
                    if scenario == 'drained':
                        self.assertEqual(len(submitted), 1)
                        # Drain the inbox before the queued job's turn.
                        inbox.get_nowait()
                        await submitted[0]()
                        # The job resubmitted nothing.
                        self.assertEqual(len(submitted), 1)
                    else:
                        self.assertEqual(submitted, [])
                run_llm_mock.assert_not_awaited()

    async def test_stale_epoch_skips_entirely(self):
        from handlers import generation_queue as gq
        ctx = _ctx()
        get_main_inbox(ctx).put_nowait(SubagentMessage(
            id='abc', state=SubagentStatus.FINISHED, description='T',
            result='old report'))
        submitted: list = []

        async def fake_submit(ctx, job):
            submitted.append(job)

        run_llm_mock = unittest.mock.AsyncMock(return_value='ack')
        with unittest.mock.patch.object(gq, 'submit_task', fake_submit), \
                unittest.mock.patch.object(gq, 'run_llm', run_llm_mock):
            # /start bumped the epoch after this wakeup was scheduled.
            bump_conversation_epoch(ctx)
            await subagent_mod._wakeup_parent(ctx, epoch=0)
        self.assertEqual(submitted, [])
        run_llm_mock.assert_not_awaited()

    async def test_cancel_all_clears_the_inbox(self):
        ctx = _ctx()
        inbox = get_main_inbox(ctx)
        inbox.put_nowait(SubagentMessage(
            id='abc', state=SubagentStatus.FINISHED, description='T',
            result='report'))
        cancel_all_subagents(ctx)
        # Stale mail is dropped, so any wakeup already queued no-ops on
        # the empty inbox instead of replying into a reset conversation.
        # clear_main_inbox drops the queue rather than draining it, so
        # the installed inbox is a fresh (empty) one - either way the
        # report mail is gone; assert both facts so a no-op clear cannot
        # hide behind get_main_inbox's auto-creation.
        self.assertIsNot(get_main_inbox(ctx), inbox)
        self.assertTrue(get_main_inbox(ctx).empty())

    async def test_late_steer_to_finished_subagent_is_dropped(self):
        ctx = _ctx()
        record = SubagentRecord(id='abc', description='Job',
                                conversation=Conversation(), parent=ctx)
        record.finish(SubagentStatus.FINISHED)
        ctx.thread_data.subagents['abc'] = record
        with unittest.mock.patch.object(
                subagent_mod, 'ensure_parent_wakeup') as wake:
            result = await subagent_steer(ctx, 'abc', 'too late')
        self.assertIn('Error', result)
        self.assertTrue(record.inbox.empty())
        wake.assert_not_called()
        # And nothing reached the parent inbox either.
        self.assertTrue(get_main_inbox(ctx).empty())


class TestInboxCutoff(unittest.IsolatedAsyncioTestCase):

    async def test_mail_during_final_round_waits_for_next_turn(self):
        # Cutoff is request time: mail arriving mid-flight stays queued
        # for the next turn instead of extending this one.
        ctx = _ctx()
        inbox = get_main_inbox(ctx)
        calls = []

        async def fake_create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                inbox.put_nowait(SubagentMessage(
                    id='xyz', state=SubagentStatus.FINISHED, description='T',
                    result='done stuff'))

            async def stream():
                yield _chunk('reply %d' % len(calls))
            return stream()

        fake_client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create)))
        with unittest.mock.patch.object(
                run_llm_mod, 'client', fake_client):
            result = await run_llm_mod.run_llm(ctx, inbox=inbox)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result, 'reply 1')
        # Still queued: the next turn's top-of-loop drain picks it up.
        pending = inbox.get_nowait()
        self.assertEqual(pending.result, 'done stuff')


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
                len(ctx.thread_data.subagents), MAX_SUBAGENTS)
            cancel_all_subagents(ctx)

    async def test_finished_records_pruned(self):
        from tools.subagent import SubagentRecord
        from conversation import Conversation
        ctx = _ctx()
        subs = subagent_mod.get_subagents(ctx)
        for i in range(MAX_RECORDS):
            subs[f'old{i}'] = SubagentRecord(
                id=f'old{i}', description='old',
                conversation=Conversation(),
                status=SubagentStatus.FINISHED, parent=ctx)

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
            record = next(iter(ctx.thread_data.subagents.values()))
            assert record.task is not None
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
        ctx.thread_data.subagents = {'abc123': rec}
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
        rec.finish(SubagentStatus.FINISHED)
        # Pin the end so the assertion is deterministic.
        rec.ended_at = rec.started_at + 125
        ctx.thread_data.subagents = {'abc123': rec}
        text = await subagent_list(ctx)
        self.assertIn('ran 2m05s', text)


class TestAnnounceTools(unittest.IsolatedAsyncioTestCase):

    async def _run_tool_turn(self, announce):
        from conversation import Conversation

        async def fake_tool(ctx):
            return 'tool output'

        setattr(fake_tool, '_needs_ctx', True)
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
        ctx.user_data.settings = SimpleNamespace(  # type: ignore[assignment]
            show_tool_output=True, model='m', temperature=1.0, top_p=0.95)
        send = unittest.mock.AsyncMock(return_value=[9])
        with unittest.mock.patch.object(
                run_llm_mod, 'client', fake_client), \
            unittest.mock.patch.object(Messaging, 'send_json', send):
            result = await run_llm_mod.run_llm(
                ctx, conversation=Conversation(),
                tools=ToolSurface(definitions=tool_defs,
                                  functions={'t': fake_tool}),
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
        ctx.user_data.settings = SimpleNamespace(  # type: ignore[assignment]
            show_tool_output=show_tool_output, model='m',
            temperature=1.0, top_p=0.95)
        conversation = Conversation()
        inbox = SubagentInbox()
        inbox.put_nowait(message)
        send = unittest.mock.AsyncMock(return_value=[9])
        ctx.messaging.send_collapsible = send
        drained = await run_llm_mod._drain_inbox(
            conversation, inbox, ctx if announce else None)
        return drained, conversation, send

    @staticmethod
    def _report(result='report'):
        return SubagentMessage(id='abc', state=SubagentStatus.FINISHED,
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
            SubagentMessage(id='abc', state=SubagentStatus.STOPPED,
                            description='Job'), True)
        self.assertEqual(drained, 1)
        summary, body = send.call_args.args[0], send.call_args.args[1]
        self.assertIn('stopped', summary)
        self.assertEqual(body, '(stopped)')

    async def test_receipt_parent_note_summary(self):
        drained, conversation, send = await self._drain(
            SubagentMessage(id='parent', state=SubagentStatus.RUNNING,
                            result='note'), True)
        self.assertEqual(drained, 1)
        summary, body = send.call_args.args[0], send.call_args.args[1]
        self.assertEqual(summary, 'Parent note')
        self.assertEqual(body, 'note')

    async def test_receipt_running_omits_state_suffix(self):
        drained, conversation, send = await self._drain(
            SubagentMessage(id='abc', state=SubagentStatus.RUNNING,
                            description='Job', result='halfway'), True)
        self.assertEqual(drained, 1)
        summary = send.call_args.args[0]
        self.assertIn('Subagent **abc** *Job*', summary)
        self.assertNotIn('— running', summary)

    async def test_rejects_unstructured_puts(self):
        inbox = SubagentInbox()
        with self.assertRaises(TypeError):
            inbox.put_nowait('just a string')  # type: ignore[arg-type]

    async def test_silent_when_receipt_disabled(self):
        # Two independent gates suppress the receipt (and leave the turn
        # unlinked): the subagent path never announces, or tool output is
        # turned off for the chat. Either way the mail is still drained.
        for announce, show_tool_output in ((False, True), (True, False)):
            with self.subTest(announce=announce,
                              show_tool_output=show_tool_output):
                drained, conversation, send = await self._drain(
                    self._report('x'), announce,
                    show_tool_output=show_tool_output)
                self.assertEqual(drained, 1)
                send.assert_not_awaited()
                stored = (conversation.messages[-1].get('custom') or {}).get(
                    'telegram_message_ids')
                self.assertIsNone(stored)


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
            SubagentMessage(id='abc', state=SubagentStatus.FINISHED,
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
