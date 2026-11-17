"""The tool surface: every tool declares itself, and each audience sees
only what it should.

The failure this guards against is silent: a new tool joining the subagent
surface, or a chat-writing tool losing its parent-only marking, because
nobody classified it.
"""

import inspect
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import tool_registry
from chat_update_context import AgentContext, ChatUpdateContext
from tool_registry import (BOTH, PARENT, SUBAGENT, agent_tool, all_tools,
                           definitions_for, tools_for)
from tool_registration import register_tools

CHAT_WRITING_TOOLS = (
    'send_file',
    'subagent_start',
    'subagent_list',
    'subagent_stop',
    'subagent_steer',
)

# Declared once per audience: the model-facing name is shared, the
# implementation is not.
DUAL_IMPLEMENTATION = ('read_image', 'generate_image')


def _names(role: str) -> set[str]:
    return {tool.name for tool in tools_for(role)}


class TestToolSurface(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        register_tools()
        cls.parent_names = _names(PARENT)
        cls.subagent_names = _names(SUBAGENT)

    def test_every_tool_is_offered_somewhere(self):
        self.assertEqual(self.parent_names | self.subagent_names,
                         {tool.name for tool in all_tools()})

    def test_subagent_surface_has_no_chat_writing_tools(self):
        for name in CHAT_WRITING_TOOLS:
            with self.subTest(tool=name):
                self.assertIn(name, self.parent_names)
                self.assertNotIn(name, self.subagent_names)

    def test_shared_tools_are_offered_to_both(self):
        for name in ('bash', 'read', 'write', 'web_search', 'python',
                     'journal_query', 'memory_get'):
            with self.subTest(tool=name):
                self.assertIn(name, self.parent_names)
                self.assertIn(name, self.subagent_names)

    def test_dual_implementation_keeps_one_name(self):
        # A subagent's read_image must not be called read_image_subagent:
        # the two agents share one tool vocabulary.
        for name in DUAL_IMPLEMENTATION:
            with self.subTest(tool=name):
                self.assertIn(name, self.parent_names)
                self.assertIn(name, self.subagent_names)
                self.assertNotIn(f'{name}_subagent', self.parent_names)
                self.assertNotIn(f'{name}_subagent', self.subagent_names)

    def test_dual_implementations_differ(self):
        parent = {t.name: t.func for t in tools_for(PARENT)}
        subagent = {t.name: t.func for t in tools_for(SUBAGENT)}
        for name in DUAL_IMPLEMENTATION:
            with self.subTest(tool=name):
                self.assertIsNot(parent[name], subagent[name])

    def test_subagent_tools_accept_a_silent_context(self):
        # The structural half of the guarantee: nothing a subagent is
        # offered may demand the interactive context type.
        for tool in tools_for(SUBAGENT):
            if not getattr(tool.func, '_needs_ctx', False):
                continue
            params = list(inspect.signature(tool.func).parameters.values())
            with self.subTest(tool=tool.name):
                self.assertIs(params[0].annotation, AgentContext)


class TestDefinitions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        register_tools()

    def test_definition_shape(self):
        definition = next(d for d in definitions_for(PARENT)
                          if d['function']['name'] == 'bash')['function']
        self.assertFalse(definition['strict'])
        self.assertEqual(definition['parameters']['additionalProperties'],
                         False)
        self.assertEqual(definition['parameters']['required'], ['command'])

    def test_required_is_derived_from_the_signature(self):
        # A parameter without a Python default is required, and one with a
        # default is not, so a schema cannot drift from its function.
        for tool in all_tools():
            params = list(inspect.signature(tool.func).parameters.values())
            if getattr(tool.func, '_needs_ctx', False):
                params = params[1:]
            expected = tuple(p.name for p in params
                             if p.default is p.empty)
            with self.subTest(tool=tool.name):
                self.assertEqual(tool.required, expected)

    def test_no_required_key_when_everything_is_optional(self):
        definition = next(d for d in definitions_for(PARENT)
                          if d['function']['name'] == 'journal_query')
        self.assertNotIn('required', definition['function']['parameters'])

    def test_every_parameter_is_declared_and_callable(self):
        # A definition with no implementation, or a schema naming an
        # argument the function does not take, fails at call time. Catch
        # both here instead.
        for tool in all_tools():
            params = {p.name for p in
                      inspect.signature(tool.func).parameters.values()}
            with self.subTest(tool=tool.name):
                self.assertLessEqual(set(tool.properties), params)


class TestDeclaration(unittest.TestCase):

    def setUp(self):
        # Declarations land in a shared module-level list. Isolate these
        # tests so a deliberately valid declaration does not join the real
        # surface for later tests.
        self._declared = unittest.mock.patch.object(
            tool_registry, '_DECLARED', [])
        self._declared.start()
        self.addCleanup(self._declared.stop)

    def test_unknown_scope_is_rejected(self):
        with self.assertRaises(ValueError):
            @agent_tool(description="x", scope="everyone")
            async def nope() -> str:
                return ""

    def test_parameter_not_in_signature_is_rejected(self):
        with self.assertRaises(ValueError):
            @agent_tool(description="x",
                        parameters={"nope": {"type": "string"}})
            async def wrong(y: str) -> str:
                return y

    def test_required_without_declaration_is_rejected(self):
        with self.assertRaises(ValueError):
            @agent_tool(description="x", parameters={})
            async def missing(y: str) -> str:
                return y

    def test_interactive_context_cannot_be_offered_to_subagents(self):
        # The tool would hold a Messaging handle while claiming to be
        # silent-usable, so reject the declaration outright.
        with self.assertRaises(ValueError):
            @agent_tool(description="x", scope=BOTH)
            async def leaky(ctx: ChatUpdateContext) -> str:
                return ""

    def test_interactive_context_is_fine_for_the_parent(self):
        @agent_tool(description="x")
        async def parent_only(ctx: ChatUpdateContext) -> str:
            return ""

        self.assertTrue(getattr(parent_only, '_needs_ctx'))


if __name__ == '__main__':
    unittest.main()
