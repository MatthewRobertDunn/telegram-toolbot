import asyncio
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # noqa: F401
from mocks import create_mock_context

import config
from user_settings import UserSettings
from tool_registry import BOTH, PARENT
from tools import subagent as subagent_mod
from tools.subagent import (
    subagent_list,
    subagent_start,
    subagent_steer,
    subagent_stop,
)


class TestSubagentTools(unittest.IsolatedAsyncioTestCase):

    @staticmethod
    def _context_with_settings(**settings_overrides):
        ctx = create_mock_context()
        ctx.settings = UserSettings.default()
        for key, value in settings_overrides.items():
            setattr(ctx.settings, key, value)
        return ctx

    async def test_start_returns_id_immediately(self):
        ctx = self._context_with_settings()

        async def hanging_run_llm(*args, **kwargs):
            await asyncio.sleep(60)
            return "never"

        with unittest.mock.patch.object(
                subagent_mod, "run_llm", hanging_run_llm):
            result = await subagent_start(ctx, "Research task", "Do the thing")

        self.assertIn("Started subagent", result)
        subs = ctx.thread_data.subagents
        self.assertEqual(len(subs), 1)
        record = next(iter(subs.values()))
        self.assertEqual(record.description, "Research task")
        self.assertEqual(record.status, "running")
        assert record.task is not None
        self.assertFalse(record.task.done())
        record.task.cancel()
        try:
            await record.task
        except asyncio.CancelledError:
            pass

    async def test_completion_notifies_parent_via_inbox(self):
        ctx = self._context_with_settings()

        async def fake_run_llm(*args, **kwargs):
            return "final report"

        with unittest.mock.patch.object(
                subagent_mod, "run_llm", fake_run_llm), \
            unittest.mock.patch.object(
                subagent_mod, '_notify_parent',
                unittest.mock.AsyncMock()) as notify:
            await subagent_start(ctx, "Label", "Task")
            record = next(iter(ctx.thread_data.subagents.values()))
            assert record.task is not None
            await record.task

        self.assertEqual(record.status, "done")
        self.assertEqual(record.result, "final report")
        # Single path: nothing goes straight to Telegram; the parent
        # inbox receipt is the only user-visible signal.
        self.assertEqual(ctx.sent, [])
        notify.assert_awaited_once()
        self.assertIn("final report", notify.call_args.args[1].result)

    async def test_send_and_stop(self):
        ctx = self._context_with_settings()

        async def hanging_run_llm(*args, **kwargs):
            await asyncio.sleep(60)
            return "never"

        with unittest.mock.patch.object(
                subagent_mod, "run_llm", hanging_run_llm), \
            unittest.mock.patch.object(
                subagent_mod, '_notify_parent',
                unittest.mock.AsyncMock()):
            result = await subagent_start(ctx, "Label", "Task")
            sid = next(iter(ctx.thread_data.subagents))
            self.assertIn(sid, result)

            listed = await subagent_list(ctx)
            self.assertIn(sid, listed)

            sent = await subagent_steer(ctx, sid, "steer left")
            self.assertIn("queued", sent)
            record = ctx.thread_data.subagents[sid]
            queued = await record.inbox.get()
            self.assertIn("steer left", queued.result)

            stopped = await subagent_stop(ctx, sid)
            self.assertIn("Stopped", stopped)
            assert record.task is not None
            try:
                await record.task
            except asyncio.CancelledError:
                pass
            self.assertEqual(record.status, "stopped")

    async def test_model_precedence(self):
        cases = (
            ({}, None, "chat-model"),
            ({}, "config-model", "config-model"),
            ({"subagent_model": "custom-model"}, None, "custom-model"),
            ({"subagent_model": "custom-model"}, "config-model", "custom-model"),
        )
        for overrides, config_model, expected in cases:
            with self.subTest(overrides=overrides, config=config_model):
                ctx = self._context_with_settings(model="chat-model",
                                                  **overrides)
                captured = {}

                async def fake_run_llm(*args, **kwargs):
                    captured.update(kwargs)
                    return "ok"

                with unittest.mock.patch.object(config, "SUBAGENT_MODEL",
                                                config_model), \
                    unittest.mock.patch.object(
                        subagent_mod, "run_llm", fake_run_llm), \
                    unittest.mock.patch.object(
                        subagent_mod, '_notify_parent',
                        unittest.mock.AsyncMock()):
                    await subagent_start(ctx, "Label", "Task")
                    record = next(iter(ctx.thread_data.subagents.values()))
                    assert record.task is not None
                    await record.task
                self.assertEqual(captured["model"], expected)

    def test_tool_definition_shapes(self):
        from tool_registry import all_tools
        declared = {t.name: t for t in all_tools()
                    if t.name.startswith('subagent_')}
        self.assertEqual(
            set(declared),
            {"subagent_start", "subagent_list", "subagent_stop",
             "subagent_steer", "subagent_send_to_parent"})
        # Controls are parent-only; the progress channel is shared so the
        # parent gets a working tool even under role confusion.
        for name in ("subagent_start", "subagent_list", "subagent_stop",
                     "subagent_steer"):
            self.assertEqual(declared[name].scope, PARENT)
        self.assertEqual(declared["subagent_send_to_parent"].scope, BOTH)


if __name__ == "__main__":
    unittest.main()
