import os
import sys
import unittest
import unittest.mock
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from handlers import run_llm as run_llm_mod
from mock_context import mock_ctx

SETTINGS = SimpleNamespace(
    show_tool_output=False, model='m', temperature=1.0, top_p=0.95)


class TestRunLlmToolToggle(unittest.IsolatedAsyncioTestCase):

    def _fake_client(self, captured: dict):
        async def fake_create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content='plain reply', tool_calls=None,
                    reasoning_content=None),
                finish_reason='stop')])
        return SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create)))

    async def test_tools_offered_by_default(self):
        captured: dict = {}
        with unittest.mock.patch.object(
                run_llm_mod, 'client', self._fake_client(captured)):
            result = await run_llm_mod.run_llm(
                mock_ctx(settings=SETTINGS), conversation=[])
        self.assertEqual(result, 'plain reply')
        self.assertIn('tools', captured)
        self.assertEqual(captured['tool_choice'], 'auto')

    async def test_allow_tools_false_omits_tools(self):
        captured: dict = {}
        with unittest.mock.patch.object(
                run_llm_mod, 'client', self._fake_client(captured)):
            result = await run_llm_mod.run_llm(
                mock_ctx(settings=SETTINGS), conversation=[],
                allow_tools=False)
        self.assertEqual(result, 'plain reply')
        self.assertNotIn('tools', captured)
        self.assertNotIn('tool_choice', captured)


if __name__ == '__main__':
    unittest.main()
