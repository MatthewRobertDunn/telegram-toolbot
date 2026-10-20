import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # noqa: F401
from mocks import create_mock_context

import config
from user_settings import UserSettings
from tools.subagent import TOOL_DEFINITIONS, TOOL_NAME, subagent


class TestSubagentTool(unittest.IsolatedAsyncioTestCase):

    @staticmethod
    def _context_with_settings(**settings_overrides):
        ctx = create_mock_context()
        ctx.settings = UserSettings.default()
        for key, value in settings_overrides.items():
            setattr(ctx.settings, key, value)
        return ctx

    async def test_returns_report_without_sending_to_user(self):
        ctx = self._context_with_settings()
        captured = {}

        async def fake_run_llm(ctx_arg, **kwargs):
            captured["kwargs"] = kwargs
            return "final report"

        with unittest.mock.patch("tools.subagent.run_llm", fake_run_llm):
            result = await subagent(ctx, "Research task", "Do the thing")

        self.assertEqual(result, "final report")
        # The report goes back to the calling model only, never to the user.
        self.assertEqual(ctx.sent, [])
        self.assertEqual(captured["kwargs"]["excluded_tools"], [TOOL_NAME])
        self.assertEqual(captured["kwargs"]["model"], ctx.settings.subagent_model)
        conversation = captured["kwargs"]["conversation"]
        self.assertEqual(conversation[-1],
                         {"role": "user", "content": "Do the thing"})
        self.assertIn("Research task", conversation[0]["content"])

    async def test_model_precedence(self):
        # The subagent model resolves as: the user's /subagent_model
        # override, else the operator's SUBAGENT_MODEL config, else the
        # user's current /model.
        captured = {}

        async def fake_run_llm(ctx_arg, **kwargs):
            captured["kwargs"] = kwargs
            return "ok"

        cases = (
            # (settings overrides, config.SUBAGENT_MODEL, expected model)
            ({}, None, "chat-model"),
            ({}, "config-model", "config-model"),
            ({"subagent_model": "custom-model"}, None, "custom-model"),
            ({"subagent_model": "custom-model"}, "config-model", "custom-model"),
        )
        for overrides, config_model, expected in cases:
            with self.subTest(overrides=overrides, config=config_model):
                ctx = self._context_with_settings(model="chat-model",
                                                  **overrides)
                with unittest.mock.patch.object(config, "SUBAGENT_MODEL",
                                                config_model), \
                        unittest.mock.patch("tools.subagent.run_llm",
                                            fake_run_llm):
                    await subagent(ctx, "Label", "Task")
                self.assertEqual(captured["kwargs"]["model"], expected)

    def test_tool_definition_shape(self):
        self.assertEqual(len(TOOL_DEFINITIONS), 1)
        function = TOOL_DEFINITIONS[0]["function"]
        self.assertEqual(function["name"], "subagent")
        self.assertEqual(function["parameters"]["required"],
                         ["description", "prompt"])
        self.assertEqual(set(function["parameters"]["properties"]),
                         {"description", "prompt"})


if __name__ == "__main__":
    unittest.main()
