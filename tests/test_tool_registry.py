"""The tool surface: every tool declares itself, and each audience sees
only what it should.

The failure this guards against is silent: a new tool joining the subagent
surface, or a chat-writing tool losing its parent-only marking, because
nobody classified it.
"""

import importlib
import inspect
import os
import sys
import tempfile
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import tool_registry
from chat_update_context import AgentContext, ChatUpdateContext
from tool_registry import (BOTH, PARENT, SUBAGENT, ToolRegistry, agent_tool,
                           all_tools, definitions_for, tools_for)
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
            if not tool.needs_ctx:
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
            if tool.needs_ctx:
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
        # Declarations land in the shared default registry. Swap in an
        # isolated one so a deliberately valid declaration does not join
        # the real surface for later tests; the module-level functions read
        # tool_registry.registry on every call.
        patcher = unittest.mock.patch.object(
            tool_registry, 'registry',
            ToolRegistry(loader=lambda: None))
        patcher.start()
        self.addCleanup(patcher.stop)

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

    def test_signature_parameter_not_in_schema_is_rejected(self):
        # The model can only pass what the schema declares; a function
        # parameter missing from it is silently undiscoverable.
        with self.assertRaises(ValueError) as caught:
            @agent_tool(description="x",
                        parameters={"known": {"type": "string"}})
            async def incomplete(known: str, hidden: int = 1) -> str:
                return known

        message = str(caught.exception)
        self.assertIn('incomplete', message)
        self.assertIn('hidden', message)

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

    def test_needs_ctx_is_recorded_on_the_tool(self):
        # The same fact, readable without reaching for the function
        # attribute the executor uses.
        @agent_tool(description="x", parameters={"q": {"type": "string"}})
        async def with_ctx(ctx: AgentContext, q: str) -> str:
            return q

        @agent_tool(description="x", parameters={"q": {"type": "string"}})
        async def without_ctx(q: str) -> str:
            return q

        self.assertTrue(all_tools()[0].needs_ctx)
        self.assertFalse(all_tools()[1].needs_ctx)
        self.assertTrue(getattr(with_ctx, '_needs_ctx'))
        self.assertFalse(getattr(without_ctx, '_needs_ctx', False))


class TestDuplicateDeclarations(unittest.TestCase):

    def setUp(self):
        patcher = unittest.mock.patch.object(
            tool_registry, 'registry',
            ToolRegistry(loader=lambda: None))
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _declare(name: str, scope: str, module: str):
        async def tool_impl(q: str) -> str:
            return q

        tool_impl.__name__ = name
        tool_impl.__module__ = module
        return agent_tool(description="x", name=name, scope=scope,
                          parameters={"q": {"type": "string"}})(tool_impl)

    def test_same_scope_is_rejected_with_both_modules(self):
        # What importlib.reload does: runs a module's decorators a second
        # time. The second pass must raise instead of duplicating entries.
        self._declare('ping', PARENT, 'tools.web')
        with self.assertRaises(ValueError) as caught:
            self._declare('ping', PARENT, 'tools.subagent')

        message = str(caught.exception)
        self.assertIn('tools.web', message)
        self.assertIn('tools.subagent', message)
        self.assertIn(PARENT, message)

    def test_same_name_for_disjoint_audiences_is_allowed(self):
        # The read_image / generate_image pattern: one model-facing name,
        # one implementation per audience.
        self._declare('read_image', PARENT, 'tools.sandbox.media')
        self._declare('read_image', SUBAGENT, 'tools.sandbox.media')
        self.assertEqual({tool.name for tool in all_tools()},
                         {'read_image'})

    def test_overlapping_audiences_are_rejected(self):
        # A BOTH declaration plus a PARENT one would leave two candidates
        # for the parent.
        self._declare('ping', PARENT, 'tools.web')
        with self.assertRaises(ValueError) as caught:
            self._declare('ping', BOTH, 'tools.subagent')

        message = str(caught.exception)
        self.assertIn(PARENT, message)
        self.assertIn(BOTH, message)


class TestRegistryLifecycle(unittest.TestCase):
    """Loading, and the failure modes of the module-reload world."""

    def _isolated(self, loader=None) -> ToolRegistry:
        registry = ToolRegistry(
            loader=loader if loader is not None else (lambda: None))
        patcher = unittest.mock.patch.object(
            tool_registry, 'registry', registry)
        patcher.start()
        self.addCleanup(patcher.stop)
        return registry

    def test_loader_runs_once(self):
        calls: list[int] = []
        self._isolated(loader=lambda: calls.append(1))

        tools_for(PARENT)
        tools_for(SUBAGENT)
        all_tools()

        self.assertEqual(len(calls), 1)

    def test_failed_load_is_not_sticky(self):
        calls: list[int] = []

        def failing_loader() -> None:
            calls.append(1)
            raise RuntimeError("tool module import failed")

        self._isolated(loader=failing_loader)
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                tools_for(PARENT)
        self.assertEqual(len(calls), 2)

    def test_successful_load_is_sticky(self):
        calls: list[int] = []
        self._isolated(loader=lambda: calls.append(1))
        tools_for(PARENT)
        tools_for(PARENT)
        self.assertEqual(len(calls), 1)

    def test_declaration_after_load_joins_the_surface(self):
        # A late declaration (a module imported after the first lookup)
        # invalidates the resolved index instead of being invisible.
        self._isolated()

        @agent_tool(description="x", parameters={"q": {"type": "string"}})
        async def early(q: str) -> str:
            return q

        self.assertEqual([t.name for t in tools_for(PARENT)], ['early'])

        @agent_tool(description="x", scope=SUBAGENT,
                    parameters={"q": {"type": "string"}})
        async def late(q: str) -> str:
            return q

        self.assertEqual([t.name for t in tools_for(PARENT)], ['early'])
        self.assertEqual([t.name for t in tools_for(SUBAGENT)], ['late'])

    def test_reload_of_a_tool_module_is_rejected_not_duplicated(self):
        # The concrete failure reload used to cause: the module runs its
        # decorators a second time and every lookup then raised. Now the
        # second declaration itself is the loud failure.
        self._isolated()
        source = "\n".join([
            "from tool_registry import agent_tool",
            "",
            "@agent_tool(description='x', name='reloaded_tool',",
            "            parameters={'q': {'type': 'string'}})",
            "async def reloaded_tool(q: str) -> str:",
            "    return q",
            "",
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'reloadable_tool_module.py')
            with open(path, 'w', encoding='utf-8') as f:
                f.write(source)
            sys.path.insert(0, tmp)
            try:
                module = importlib.import_module('reloadable_tool_module')
                with self.assertRaises(ValueError) as caught:
                    importlib.reload(module)
            finally:
                sys.path.remove(tmp)
                sys.modules.pop('reloadable_tool_module', None)

        self.assertIn('reloadable_tool_module', str(caught.exception))

    def test_unknown_role_is_rejected(self):
        self._isolated()
        with self.assertRaises(ValueError):
            tools_for('everyone')


if __name__ == '__main__':
    unittest.main()
