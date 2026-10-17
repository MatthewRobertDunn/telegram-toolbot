import asyncio
import contextlib
import unittest
import unittest.mock

import bootstrap  # noqa: F401 — path setup, env, logging, DB reset, next_uid
from bootstrap import next_uid as _uid
from conversation import Conversation, TOKENS_PER_CHAR
from mocks import create_mock_context

import compaction
from handlers.command_handlers import compact_command


def _text(tokens: int) -> str:
    return 'x' * int(tokens / TOKENS_PER_CHAR)


def _fill_big_conversation(ctx):
    ctx.conversation = Conversation([
        {'role': 'system', 'content': 'system prompt'},
        {'role': 'user', 'content': _text(8_000) + ' first question'},
        {'role': 'assistant', 'content': _text(8_000) + ' first answer'},
        {'role': 'user', 'content': _text(8_000) + ' second question'},
        {'role': 'assistant', 'content': _text(8_000) + ' second answer'},
    ])


async def _fake_run_llm(ctx, **kwargs):
    return 'THE SUMMARY'


class TestCompactCommand(unittest.IsolatedAsyncioTestCase):

    async def test_small_context_is_left_alone(self):
        ctx = create_mock_context(user_id=_uid(), command_name="compact")
        with unittest.mock.patch("compaction.run_llm", _fake_run_llm):
            await compact_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertEqual(ctx.sent[0]["text"], "Context too small to compact")

    async def test_compacts_and_reports_token_counts(self):
        ctx = create_mock_context(user_id=_uid(), command_name="compact",
                                  command_text="500")
        _fill_big_conversation(ctx)
        with unittest.mock.patch("compaction.run_llm", _fake_run_llm):
            await compact_command(ctx)

        self.assertEqual(len(ctx.sent), 3)
        self.assertEqual(ctx.sent[0]["text"], "Starting Compaction...")
        self.assertEqual(ctx.sent[1]["text"], "THE SUMMARY")
        self.assertEqual(ctx.sent[1]["kind"], "rich_message")
        self.assertRegex(ctx.sent[2]["text"],
                         r"^Compaction complete: [\d,]+ tokens -> [\d,]+ tokens$")
        self.assertIn("THE SUMMARY", ctx.conversation[1]["content"])

    async def test_invalid_argument_is_rejected(self):
        ctx = create_mock_context(user_id=_uid(), command_name="compact",
                                  command_text="not-a-number")
        _fill_big_conversation(ctx)
        with unittest.mock.patch("compaction.run_llm", _fake_run_llm):
            await compact_command(ctx)
        self.assertEqual(len(ctx.sent), 1)
        self.assertIn("Usage:", ctx.sent[0]["text"])

    async def test_refused_while_busy(self):
        for field, reason in (("generation_task", "being generated"),
                              ("compaction_task", "already running")):
            ctx = create_mock_context(user_id=_uid(), command_name="compact")
            _fill_big_conversation(ctx)
            pending = asyncio.create_task(asyncio.sleep(30))
            setattr(ctx, field, pending)
            try:
                with unittest.mock.patch("compaction.run_llm", _fake_run_llm):
                    await compact_command(ctx)
            finally:
                pending.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pending
            self.assertEqual(len(ctx.sent), 1)
            self.assertIn(reason, ctx.sent[0]["text"])
            self.assertIn("first question", ctx.conversation[1]["content"])

    async def test_stopping_compaction_mid_run(self):
        ctx = create_mock_context(user_id=_uid(), command_name="compact")
        _fill_big_conversation(ctx)
        started = asyncio.Event()

        async def slow_run_llm(ctx, **kwargs):
            started.set()
            await asyncio.sleep(30)
            return 'THE SUMMARY'

        with unittest.mock.patch("compaction.run_llm", slow_run_llm):
            command = asyncio.create_task(compact_command(ctx))
            await started.wait()
            task = ctx.compaction_task
            assert task is not None
            task.cancel()
            await command

        self.assertEqual(ctx.sent[-1]["text"], "Compaction stopped.")
        self.assertIn("first question", ctx.conversation[1]["content"])


if __name__ == "__main__":
    unittest.main()
