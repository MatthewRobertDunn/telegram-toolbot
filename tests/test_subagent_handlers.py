"""Unit tests for the user-facing /subagent_list command (no network)."""

import asyncio
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from conversation import Conversation
from handlers import subagent_handlers as handlers_mod
from mock_context import mock_ctx
from tools.subagent import SubagentRecord


def _ctx():
    return mock_ctx(conversation=Conversation())


class TestSubagentListCommand(unittest.IsolatedAsyncioTestCase):

    async def _run(self, ctx):
        with unittest.mock.patch.object(
                handlers_mod, 'send_message',
                unittest.mock.AsyncMock(return_value=[1])) as send:
            await handlers_mod.subagent_list_command(ctx)
        return send

    async def test_empty(self):
        send = await self._run(_ctx())
        send.assert_awaited_once()
        self.assertIn('No subagents', send.call_args.args[1])

    async def test_lists_running_subagent(self):
        ctx = _ctx()
        ctx.user_data['subagents'] = {
            'abc123': SubagentRecord(
                id='abc123', description='Research task',
                conversation=Conversation(), status='running'),
        }
        send = await self._run(ctx)
        send.assert_awaited_once()
        text = send.call_args.args[1]
        self.assertIn('abc123', text)
        self.assertIn('Research task', text)
        self.assertIn('running', text)


if __name__ == '__main__':
    unittest.main()
