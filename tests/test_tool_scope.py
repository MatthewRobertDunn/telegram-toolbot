"""Every registered tool declares who may be offered it (see tool_scope).

The failure this guards against is silent: a new tool joins the subagent
surface (or loses its parent visibility) because nobody classified it.
"""

import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chat_update_context import AgentContext
from tool_definitions import TOOL_DEFINITIONS, TOOL_MAP
from tool_registration import register_tools
from tool_scope import BOTH, PARENT, SUBAGENT, tool_scope
from tools.subagent import get_parent_tool_defs, get_subagent_tool_defs

CHAT_WRITING_TOOLS = (
    'send_file',
    'read_image',
    'generate_image',
    'subagent_start',
    'subagent_list',
    'subagent_stop',
    'subagent_steer',
)

SILENT_VARIANTS = ('read_image_subagent', 'generate_image_subagent')


def _names(definitions: list[dict]) -> set[str]:
    return {t['function']['name'] for t in definitions}


class TestToolScope(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not TOOL_DEFINITIONS:
            register_tools()
        cls.parent_names = _names(get_parent_tool_defs())
        cls.subagent_names = _names(get_subagent_tool_defs())

    def test_every_registered_tool_is_classified(self):
        undeclared = sorted(name for name, func in TOOL_MAP.items()
                            if tool_scope(func) is None)
        self.assertEqual(
            undeclared, [],
            'declare a scope for these tools (see tool_scope)')

    def test_scopes_are_known(self):
        for name, func in TOOL_MAP.items():
            with self.subTest(tool=name):
                self.assertIn(tool_scope(func), (PARENT, SUBAGENT, BOTH))

    def test_every_definition_is_executable(self):
        # A definition with no implementation would be offered and fail at
        # call time ('Unknown tool').
        self.assertEqual(_names(TOOL_DEFINITIONS) - set(TOOL_MAP), set())

    def test_subagent_surface_has_no_chat_writing_tools(self):
        for name in CHAT_WRITING_TOOLS:
            with self.subTest(tool=name):
                self.assertIn(name, self.parent_names)
                self.assertNotIn(name, self.subagent_names)

    def test_subagent_surface_has_the_silent_variants(self):
        for name in SILENT_VARIANTS:
            with self.subTest(tool=name):
                self.assertIn(name, self.subagent_names)
                self.assertNotIn(name, self.parent_names)

    def test_unregistered_chat_tool_is_parent_only(self):
        # display_status is not registered right now, but if it comes back
        # it must never reach a subagent.
        from tools.status_tool import display_status
        self.assertEqual(tool_scope(display_status), PARENT)

    def test_shared_tools_are_offered_to_both(self):
        for name in ('bash', 'read', 'write', 'web_search', 'python',
                     'journal_query', 'memory_get'):
            with self.subTest(tool=name):
                self.assertIn(name, self.parent_names)
                self.assertIn(name, self.subagent_names)

    def test_subagent_tools_accept_a_silent_context(self):
        # The structural half of the guarantee: nothing a subagent is
        # offered may demand the interactive context type.
        for name in sorted(self.subagent_names):
            func = TOOL_MAP[name]
            if not getattr(func, '_requires_context', False):
                continue
            params = list(inspect.signature(func).parameters.values())
            with self.subTest(tool=name):
                self.assertIs(params[0].annotation,
                              AgentContext)

    def test_every_tool_is_offered_somewhere(self):
        self.assertEqual(self.parent_names | self.subagent_names,
                         set(TOOL_MAP))


if __name__ == '__main__':
    unittest.main()
