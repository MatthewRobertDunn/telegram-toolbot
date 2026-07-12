import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from handlers.command_handlers import model_command
from mocks import create_mock_context


class TestCommands(unittest.IsolatedAsyncioTestCase):
    # Each test uses a distinct user_id so they don't collide on the real
    # store.db. Tests hit the production database file; isolation is by
    # user_id, not by patching.

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