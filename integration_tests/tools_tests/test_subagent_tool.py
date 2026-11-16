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
from tools import subagent as subagent_mod
from tools.subagent import (
    PARENT_TOOL_DEFINITIONS,
    SUBAGENT_TOOL_DEFINITIONS,
    subagent_list,
    subagent_send,
    subagent_start,
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
        subs = ctx.user_data['subagents']
        self.assertEqual(len(subs), 1)
        record = next(iter(subs.values()))
        self.assertEqual(record.description, "Research task")
        self.assertEqual(record.status, "running")
        self.assertFalse(record.task.done())
        record.task.cancel()
        try:
            await record.task
        except asyncio.CancelledError:
            pass

    async def test_completion_notifies_telegram_and_parent(self):
        ctx = self._context_with_settings()

        async def fake_run_llm(*args, **kwargs):
            return "final report"

        with unittest.mock.patch.object(
                subagent_mod, "run_llm", fake_run_llm), \
            unittest.mock.patch.object(
                subagent_mod, '_notify_parent',
                unittest.mock.AsyncMock()) as notify:
            await subagent_start(ctx, "Label", "Task")
            record = next(iter(ctx.user_data['subagents'].values()))
            await record.task

        self.assertEqual(record.status, "done")
        self.assertEqual(record.result, "final report")
        # Telegram notice went out via the mock bot.
        self.assertTrue(
            any("final report" in m["text"] for m in ctx.sent),
            f"no completion notice in {ctx.sent}")
        notify.assert_awaited_once()
        args, _ = notify.call_args
        self.assertIn("final report", args[1])

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
            sid = next(iter(ctx.user_data['subagents']))
            self.assertIn(sid, result)

            listed = await subagent_list(ctx)
            self.assertIn(sid, listed)

            sent = await subagent_send(ctx, sid, "steer left")
            self.assertIn("queued", sent)
            record = ctx.user_data['subagents'][sid]
            queued = await record.inbox.get()
            self.assertIn("steer left", queued)

            stopped = await subagent_stop(ctx, sid)
            self.assertIn("Stopped", stopped)
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
                    record = next(iter(ctx.user_data['subagents'].values()))
                    await record.task
                self.assertEqual(captured["model"], expected)

    def test_tool_definition_shapes(self):
        parent_names = {t["function"]["name"]
                        for t in PARENT_TOOL_DEFINITIONS}
        self.assertEqual(parent_names, {"subagent_start", "subagent_list",
                                        "subagent_stop", "subagent_send"})
        self.assertEqual(
            [t["function"]["name"] for t in SUBAGENT_TOOL_DEFINITIONS],
            ["parent_send"])


if __name__ == "__main__":
    unittest.main()
