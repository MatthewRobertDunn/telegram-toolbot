import json
import os
import re
import unittest

import bootstrap  # noqa: F401 — path setup, env, logging, DB reset
from mocks import create_mock_context
from handlers.message_handlers import text_message_handler
from tool_registration import register_tools
from tool_definitions import TOOL_MAP

_PROMPT = (
    "You are running an automated self-test of your registered tools. "
    "You MUST exercise EVERY ONE of the tools available to you, with no "
    "exceptions and no shortcuts. Do not skip a tool just because a related "
    "tool already appeared to work; each tool must be invoked at least once. "
    "For each tool, design a minimal invocation that will exercise it, call "
    "it, and check that the result matches your expectation. "
    "Then reply with ONLY a JSON object mapping EVERY tool's name (exactly as "
    "it appears in your tool definitions) to a boolean: true if the tool "
    "worked, false if it did not. Do not include any other text. "
    'Example shape: {"web_search": true, "get_current_datetime": true, ...}. '
    "The JSON keys MUST exactly match your full set of registered tool names."
)


class TestTools(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        if not os.environ.get("OPENAI_API_KEY"):
            self.skipTest(
                "OPENAI_API_KEY not set; meta-test requires a live LLM provider")
        register_tools()

    async def test_all_tools_with_llm(self):
        ctx = create_mock_context(text=_PROMPT)
        try:
            await text_message_handler(ctx)
        except Exception as exc:
            self.fail(f"text_message_handler raised: {exc!r}")

        self.assertTrue(ctx.bot.sent, "no messages were sent by the handler")  # type: ignore[attr-defined]
        last_text = ctx.bot.sent[-1]["text"]  # type: ignore[attr-defined]

        blob = _extract_json_object(last_text)
        self.assertIsNotNone(
            blob, f"could not find a JSON object in handler reply:\n{last_text!r}")
        assert blob is not None
        self.assertIsInstance(blob, dict,
                             f"parsed object is not a dict: {blob!r}")

        registered = set(TOOL_MAP.keys())
        reported = set(blob.keys())
        self.assertEqual(
            registered, reported,
            f"tool set mismatch.\nregistered: {sorted(registered)}\n"
            f"reported:   {sorted(reported)}")

        failures = {k: v for k, v in blob.items() if v is not True}
        self.assertFalse(
            failures, f"tools reported as not working: {failures}")


def _extract_json_object(text: str):
    for match in re.finditer(r"\{[\s\S]*\}", text):
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


if __name__ == "__main__":
    unittest.main()