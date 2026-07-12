import os
import unittest

import bootstrap  # noqa: F401 — path setup, env, logging, DB reset
from mocks import create_mock_context
from handlers.command_handlers import model_command


class TestCommands(unittest.IsolatedAsyncioTestCase):
    # Each test uses a distinct user_id so they don't collide on the test
    # database, which is wiped + recreated every test run by `mocks.py`.

    async def test_model_command(self):
        ctx = create_mock_context(
            user_id=1001,
            command_text="my-org/my-model",
            command_name="model",
        )
        await model_command(ctx)

        self.assertEqual(len(ctx.bot.sent), 1)
        self.assertIn("my-org/my-model", ctx.bot.sent[0]["text"])
        self.assertEqual(ctx.settings.model, "my-org/my-model")

    async def test_model_command_query_without_arg(self):
        # Fresh user with no saved settings -> should fall back to config default.
        ctx = create_mock_context(user_id=1002, command_name="model")
        await model_command(ctx)

        self.assertEqual(len(ctx.bot.sent), 1)
        self.assertIn("Current model", ctx.bot.sent[0]["text"])
        self.assertEqual(
            ctx.settings.model,
            os.environ.get("LLM_MODEL", "minimaxai/minimax-m2.7"),
        )


if __name__ == "__main__":
    unittest.main()