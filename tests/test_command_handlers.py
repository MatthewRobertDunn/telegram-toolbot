"""The command table: invariants and agreement with PTB registration.

``commands.COMMANDS`` is the single source of truth for the Telegram menu,
the ``CommandHandler`` registrations and the usage lines handlers render.
These tests keep those three from drifting apart - historically the menu,
the registrations and the in-handler usage f-strings were maintained by
hand, and forgetting one was invisible.
"""

import asyncio
import os
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import handlers.command_handlers as command_handlers
import handlers.memory_handlers as memory_handlers
from chat_update_context import (_parse_command_name, _parse_command_text,
                                 create_chat_update_context)
from command_spec import usage_line
from commands import COMMANDS, bot_commands, lookup_command, register_commands
from handlers.middleware import handler
from mock_context import mock_ctx


class _Bot:
    def __init__(self):
        self.sent: list = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append(text)
        return SimpleNamespace(message_id=1)

    async def _post(self, method, data):
        return {'message_id': 1}


def _isolated_stores():
    """Stub the persistence layer so @handler's init_conversation and the
    handlers' saves never touch the real database."""
    stack = ExitStack()
    stack.enter_context(mock.patch.object(
        command_handlers, 'save_conversation', mock.AsyncMock()))
    stack.enter_context(mock.patch(
        'handlers.middleware.load_settings',
        mock.AsyncMock(return_value=None)))
    stack.enter_context(mock.patch(
        'handlers.middleware.load_conversation',
        mock.AsyncMock(return_value=None)))
    stack.enter_context(mock.patch(
        'handlers.middleware.save_settings', mock.AsyncMock()))
    return stack


class _RecordingApplication:
    """Just enough of PTB's Application to record registrations."""

    def __init__(self):
        self.handlers: list = []

    def add_handler(self, ptb_handler):
        self.handlers.append(ptb_handler)


class CommandTableTest(unittest.TestCase):

    def test_names_and_aliases_are_unique(self):
        seen: set[str] = set()
        for command in COMMANDS:
            keys = [command.name, *command.aliases]
            for key in keys:
                self.assertNotIn(key, seen,
                                 f'/{key} is declared twice in COMMANDS')
                seen.add(key)
            self.assertEqual(keys, [k.lower() for k in keys],
                             f'/{command.name} is not lowercase')

    def test_every_command_has_non_empty_help(self):
        for command in COMMANDS:
            self.assertTrue(command.help.strip(),
                            f'/{command.name} has empty menu help')

    def test_every_callback_is_callable(self):
        for command in COMMANDS:
            self.assertTrue(callable(command.callback),
                            f'/{command.name} callback is not callable')

    def test_continue_is_an_alias_of_retry_not_a_command(self):
        retry = lookup_command('retry')
        self.assertIsNotNone(retry)
        self.assertIn('continue', retry.aliases)  # type: ignore[union-attr]
        self.assertNotIn('continue', {c.name for c in COMMANDS},
                         '/continue must not be a command of its own')
        self.assertIs(lookup_command('continue'), retry,
                      '/continue must resolve to the /retry command')

    def test_lookup_misses_are_none(self):
        self.assertIsNone(lookup_command(''))
        self.assertIsNone(lookup_command('nosuchcommand'))

    def test_lookup_is_case_insensitive(self):
        self.assertIs(lookup_command('MODEL'), lookup_command('model'))
        self.assertIs(lookup_command('Start'), lookup_command('start'))

    def test_usage_matches_the_handler_module_constants(self):
        # Every table row's usage must be the very constant its handler
        # renders, so a syntax change lands in both or neither.
        expected = {
            'systemprompt': (command_handlers, 'SYSTEMPROMPT_USAGE'),
            'model': (command_handlers, 'MODEL_USAGE'),
            'subagent_model': (command_handlers, 'SUBAGENT_MODEL_USAGE'),
            'top_p': (command_handlers, 'TOP_P_USAGE'),
            'temperature': (command_handlers, 'TEMPERATURE_USAGE'),
            'timezone': (command_handlers, 'TIMEZONE_USAGE'),
            'user_profile': (command_handlers, 'USER_PROFILE_USAGE'),
            'memory': (memory_handlers, 'MEMORY_USAGE'),
            'memory_delete': (memory_handlers, 'MEMORY_DELETE_USAGE'),
        }
        for name, (module, attr) in expected.items():
            command = lookup_command(name)
            self.assertIsNotNone(command)
            self.assertEqual(command.usage, getattr(module, attr),  # type: ignore[union-attr]
                             f'/{name} usage drifted from {module.__name__}.{attr}')

    def test_compact_usage_mirrors_its_handler_text(self):
        # conversation_handlers renders its own usage line; the table row
        # must quote the same syntax.
        command = lookup_command('compact')
        self.assertEqual(
            command.usage, '[tokens of recent history to keep]')  # type: ignore[union-attr]
        self.assertIn(command.usage,  # type: ignore[union-attr]
                      usage_line('compact', command.usage))  # type: ignore[union-attr]

    def test_menu_lists_primary_names_only(self):
        menu = [c.command for c in bot_commands()]
        self.assertEqual(menu, [c.name for c in COMMANDS])
        self.assertNotIn('continue', menu)


class RegistrationAgreementTest(unittest.TestCase):

    def test_every_name_and_alias_is_registered(self):
        app = _RecordingApplication()
        register_commands(app)  # type: ignore[arg-type]
        registered = self._callbacks(app)
        expected = ({c.name for c in COMMANDS}
                    | {a for c in COMMANDS for a in c.aliases})
        self.assertEqual(set(registered), expected)

    def test_registered_callbacks_come_from_the_table(self):
        app = _RecordingApplication()
        register_commands(app)  # type: ignore[arg-type]
        for key, callback in self._callbacks(app).items():
            command = lookup_command(key)
            self.assertIsNotNone(command)
            self.assertIs(
                callback, command.callback,  # type: ignore[union-attr]
                f'/{key} is registered with a foreign callback')

    def test_registration_needs_no_handler_outside_the_table(self):
        app = _RecordingApplication()
        register_commands(app)  # type: ignore[arg-type]
        self.assertEqual(len(app.handlers), len(COMMANDS))

    @staticmethod
    def _callbacks(app: _RecordingApplication) -> dict[str, object]:
        """name/alias -> the table callback wrapped into each handler."""
        found: dict[str, object] = {}
        for ptb_handler in app.handlers:
            for key in ptb_handler.commands:
                assert key not in found, f'/{key} registered twice'
                found[key] = getattr(ptb_handler.callback, '__wrapped__',
                                     ptb_handler.callback)
        return found


class CommandParsingTest(unittest.TestCase):

    @staticmethod
    def _ctx(text: str, bot=None):
        message = SimpleNamespace(text=text, caption=None,
                                  message_thread_id=None)
        chat = SimpleNamespace(id=7, type='private')
        user = SimpleNamespace(id=1, first_name='T', last_name='U')
        update = SimpleNamespace(effective_chat=chat, effective_user=user,
                                 message=message, effective_message=message)
        context = SimpleNamespace(user_data={}, bot=bot, bot_data={})
        return create_chat_update_context(update, context)  # type: ignore[arg-type]

    def test_mentioned_command_resolves_to_the_bare_name(self):
        ctx = self._ctx('/model@MyBot sonnet-4')
        self.assertEqual(ctx.command_name, 'model')
        self.assertEqual(ctx.command_text, 'sonnet-4')
        self.assertIs(lookup_command(ctx.command_name),
                      lookup_command('model'))

    def test_mixed_case_invocations_resolve_to_their_rows(self):
        # PTB matches command tokens lowercased, so these all dispatch:
        # parsing must land on the table rows, not miss them.
        for text, name in (('/MODEL', 'model'), ('/Start@MyBot', 'start'),
                           ('/Timezone', 'timezone')):
            ctx = self._ctx(text)
            self.assertEqual(ctx.command_name, name, text)
            command = lookup_command(ctx.command_name)
            self.assertIsNotNone(command, text)
            self.assertEqual(command.name, name)  # type: ignore[union-attr]

    def test_non_command_text_has_no_command(self):
        ctx = self._ctx('just chatting')
        self.assertEqual(ctx.command_name, '')
        self.assertIsNone(lookup_command(ctx.command_name))

    def test_parse_helpers(self):
        self.assertEqual(_parse_command_name('/stop'), 'stop')
        self.assertEqual(_parse_command_name('hello /stop'), '')
        self.assertEqual(_parse_command_name('/start@MyBot x'), 'start')
        self.assertEqual(_parse_command_name('/MODEL'), 'model')
        self.assertEqual(_parse_command_name('/Top_P@MyBot'), 'top_p')
        self.assertEqual(_parse_command_text('/model@MyBot gpt'), 'gpt')
        self.assertEqual(_parse_command_text('/model'), '')

    def test_usage_line_renders_the_bare_name(self):
        self.assertEqual(
            usage_line('top_p', command_handlers.TOP_P_USAGE),
            'Usage: `/top_p <0.0-1.0>`')
        self.assertNotIn('@MyBot', usage_line('model', 'x'))


class HandlerUsageRenderTest(unittest.IsolatedAsyncioTestCase):
    """Usage lines, end to end: mixed-case invocation -> parse -> the
    handler's own constant, rendered exactly as the table declares it."""

    async def test_mixed_case_timezone_without_args_renders_usage(self):
        bot = _Bot()
        ctx = CommandParsingTest._ctx('/Timezone@MyBot', bot=bot)
        with _isolated_stores():
            await command_handlers.timezone_command(ctx)
        self.assertEqual(len(bot.sent), 1)
        self.assertIn('Current timezone is `UTC`', bot.sent[0])
        self.assertIn(
            'Usage: `/timezone <IANA timezone, e.g. Australia/Sydney>`',
            bot.sent[0])
        self.assertNotIn('@MyBot', bot.sent[0])

    async def test_mixed_case_memory_delete_without_args_renders_usage(self):
        bot = _Bot()
        ctx = CommandParsingTest._ctx('/MEMORY_DELETE', bot=bot)
        with _isolated_stores():
            await memory_handlers.memory_delete_command(ctx)
        self.assertEqual(bot.sent, ['Usage: `/memory_delete <key>`'])

    async def test_mixed_case_memory_without_args_renders_usage(self):
        bot = _Bot()
        ctx = CommandParsingTest._ctx('/Memory', bot=bot)
        with _isolated_stores():
            await memory_handlers.memory_command(ctx)
        self.assertEqual(bot.sent, ['Usage: `/memory <key> [value]`'])


class HandlerErrorBoundaryTest(unittest.IsolatedAsyncioTestCase):

    async def test_crash_is_logged_and_answered_not_raised(self):
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot)

        @handler
        async def boom(inner_ctx):
            raise RuntimeError('kaboom')

        with _isolated_stores():
            await boom(ctx)
        self.assertEqual(len(bot.sent), 1)
        self.assertIn('went wrong', bot.sent[0])

    async def test_cancelled_error_is_not_swallowed(self):
        ctx = mock_ctx(user_id=5, bot=_Bot())

        @handler
        async def cancelled(inner_ctx):
            raise asyncio.CancelledError()

        with _isolated_stores():
            with self.assertRaises(asyncio.CancelledError):
                await cancelled(ctx)


class StartRestartTest(unittest.IsolatedAsyncioTestCase):

    async def test_persists_before_announcing(self):
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot)
        order: list[str] = []

        async def send(text, **kwargs):
            order.append('send')
            return SimpleNamespace(message_id=1)

        save_mock = mock.AsyncMock(
            side_effect=lambda *a, **k: order.append('save'))
        with mock.patch.object(command_handlers, 'save_conversation',
                               save_mock), \
                mock.patch.object(ctx.messaging, 'send_message', send), \
                mock.patch('tools.subagent.cancel_all_subagents',
                           side_effect=RuntimeError('kaboom')), \
                mock.patch.object(command_handlers, 'logger') as log:
            await command_handlers.start_restart_command(ctx)
        # The save lands before the status message: the announcement must
        # never outlive its own persistence.
        self.assertEqual(order, ['save', 'send'])
        save_mock.assert_awaited_once()
        # The subagent teardown failed, and that was logged, not swallowed.
        log.error.assert_called_once()
        self.assertIs(True, log.error.call_args.kwargs.get('exc_info'))

    async def test_cancel_failure_still_bumps_the_epoch(self):
        # The two teardown steps are separate best-effort blocks: a cancel
        # failure must not skip the epoch bump, or old-epoch wakeups stay
        # eligible to inject into the fresh conversation.
        ctx = mock_ctx(user_id=5, bot=_Bot())
        with _isolated_stores(), \
                mock.patch('tools.subagent.cancel_all_subagents',
                           side_effect=RuntimeError('kaboom')), \
                mock.patch.object(command_handlers, 'logger') as log:
            await command_handlers.start_restart_command(ctx)
        self.assertEqual(ctx.thread_data.conversation_epoch, 1)
        log.error.assert_called_once()
        self.assertIn('cancel subagents', log.error.call_args.args[0])

    async def test_bump_failure_is_logged_on_its_own(self):
        ctx = mock_ctx(user_id=5, bot=_Bot())
        with _isolated_stores(), \
                mock.patch('agent_inbox.bump_conversation_epoch',
                           side_effect=RuntimeError('kaboom')), \
                mock.patch.object(command_handlers, 'logger') as log:
            await command_handlers.start_restart_command(ctx)
        log.error.assert_called_once()
        self.assertIn('conversation epoch', log.error.call_args.args[0])
        self.assertEqual(ctx.thread_data.conversation_epoch, 0)


class StopQueuedWorkTest(unittest.IsolatedAsyncioTestCase):

    async def test_drop_closes_unstarted_coroutines(self):
        bot = _Bot()
        ctx = mock_ctx(user_id=5, bot=bot)
        from handlers.generation_queue import WorkItem, generation_queue

        started = False

        async def never_runs():
            nonlocal started
            started = True

        job = never_runs()
        queue = generation_queue(ctx)
        queue.put_nowait(WorkItem(ctx=ctx, job=job))

        with _isolated_stores():
            await command_handlers.stop_command(ctx)
        self.assertTrue(queue.empty())
        # A closed coroutine has no frame left to resume.
        self.assertIsNone(job.cr_frame)
        await asyncio.sleep(0)
        self.assertFalse(started)
        self.assertIn('Dropped 1 queued request', bot.sent[-1])


if __name__ == '__main__':
    unittest.main()
