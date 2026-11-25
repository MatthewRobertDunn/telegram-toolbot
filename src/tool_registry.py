"""The tool surface: what the model is offered, and who gets to use it.

A tool declares itself next to its implementation:

    @agent_tool(
        scope=BOTH,
        description="Search the web and return the top results.",
        parameters={"query": {"type": "string", "description": "..."}})
    async def web_search(query: str, max_results: int = 5) -> str: ...

The decorator records the JSON schema, resolves the name from the function
unless one is given, and derives ``required`` from the signature (a parameter
with no default is required). There is no separate name->function map to keep
in step and no per-module ``register()`` to remember to call.

Two facts are also inferred rather than declared:

``needs_ctx``
    True when the first parameter is annotated ``AgentContext`` (or a
    subclass). A tool that takes a ``ChatUpdateContext`` therefore cannot be
    offered to a subagent, which is rejected when the tool is declared.

``scope``
    Who may be offered the tool. It defaults to ``PARENT``, so a tool cannot
    reach a subagent by accident.

The same name may be declared twice, once per audience, when the two agents
need different behaviour under one name: ``read_image`` posts to the chat for
the parent and stays silent for a subagent. The model sees one tool; which
implementation runs depends on who asked.

The declarations live in a :class:`ToolRegistry` rather than module globals;
the module level functions below delegate to the default ``registry``
instance.
"""

import inspect
from dataclasses import dataclass
from typing import Callable, TypeVar

from chat_update_context import AgentContext, ChatUpdateContext

PARENT = 'parent'
BOTH = 'both'
SUBAGENT = 'subagent'

ROLES = (PARENT, SUBAGENT)

F = TypeVar('F', bound=Callable[..., object])

# Which scopes each audience is offered.
_VISIBLE_TO = {
    PARENT: (PARENT, BOTH),
    SUBAGENT: (SUBAGENT, BOTH),
}


def _audiences(scope: str) -> set[str]:
    """The roles a scope reaches."""
    return {role for role in ROLES if scope in _VISIBLE_TO[role]}


@dataclass(frozen=True)
class Tool:
    """One tool implementation, as seen by one audience."""

    name: str
    scope: str
    func: Callable[..., object]
    description: str
    properties: dict[str, dict]
    required: tuple[str, ...]
    needs_ctx: bool = False

    def definition(self) -> dict:
        """The JSON schema handed to the model."""
        parameters: dict = {
            "type": "object",
            "properties": self.properties,
            "additionalProperties": False,
        }
        if self.required:
            parameters["required"] = list(self.required)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "strict": False,
                "description": self.description,
                "parameters": parameters,
            },
        }


def _default_loader(registry: "ToolRegistry") -> Callable[[], None]:
    """Build the lazy loader that imports the tool modules into *registry*.

    Bound to the instance that asked for it: a fresh ToolRegistry must
    fill itself, not the module-global one.
    """
    def load() -> None:
        from tool_registration import load_tools
        load_tools(registry)

    return load


class ToolRegistry:
    """Owns the declared tools, their audiences, and their resolution.

    A declaration that clashes with an earlier one - the same name visible
    to the same audience - raises immediately, naming both modules, so a
    duplicate is never silently served and a reload is never half-applied.

    ``loader`` populates the registry on first use (the real one imports
    the tool modules). It runs at most once and the registry only counts as
    loaded after it returns, so a failed import is retried on the next
    lookup rather than poisoning every later call.
    """

    def __init__(self, loader: Callable[[], None] | None = None) -> None:
        self._loader: Callable[[], None] = (
            loader if loader is not None else _default_loader(self))
        self._declared: list[Tool] = []
        self._by_role: dict[str, dict[str, Tool]] | None = None
        self._loaded = False
        self._loading = False

    # -- declaration ----------------------------------------------------

    def agent_tool(self, *, description: str,
                   parameters: dict[str, dict] | None = None,
                   name: str | None = None,
                   scope: str = PARENT,
                   required: tuple[str, ...] | list[str] | None = None
                   ) -> Callable[[F], F]:
        """Declare a tool: its schema, its audience, and how to run it.

        ``parameters`` holds the JSON schema for each argument the model may
        pass; the object wrapper and ``additionalProperties: false`` are added
        here. ``required`` defaults to every parameter without a Python default,
        so a schema and a signature cannot drift apart.
        """
        if scope not in (PARENT, BOTH, SUBAGENT):
            raise ValueError(f"unknown tool scope {scope!r}")

        def decorate(func: F) -> F:
            tool = _build(func, description, parameters or {}, name, scope,
                          required)
            self._reject_clash(tool)
            self._declared.append(tool)
            # A tool declared after the first lookup (a test's, or a module
            # imported late) invalidates the resolved index.
            self._by_role = None
            return func

        return decorate

    def _reject_clash(self, tool: Tool) -> None:
        """Raise if an earlier declaration reaches the same audience."""
        for other in self._declared:
            if other.name != tool.name:
                continue
            shared = _audiences(tool.scope) & _audiences(other.scope)
            if not shared:
                continue
            raise ValueError(
                f"tool {tool.name!r} is declared twice for {sorted(shared)}: "
                f"{other.func.__module__} ({other.scope}) and "
                f"{tool.func.__module__} ({tool.scope})")

    # -- lookup -----------------------------------------------------------

    def declared(self) -> list[Tool]:
        """The tools declared so far, without triggering the lazy load."""
        return list(self._declared)

    def tools_for(self, role: str) -> list[Tool]:
        """The tools one audience is offered, in declaration order."""
        if role not in ROLES:
            raise ValueError(f"unknown tool role {role!r}")
        self.ensure_loaded()
        if self._by_role is None:
            self._by_role = self._resolve()
        return list(self._by_role[role].values())

    def definitions_for(self, role: str) -> list[dict]:
        """The JSON tool definitions one audience is offered."""
        return [tool.definition() for tool in self.tools_for(role)]

    def functions_for(self, role: str) -> dict[str, Callable[..., object]]:
        """The implementations one audience may execute, by tool name."""
        return {tool.name: tool.func for tool in self.tools_for(role)}

    def all_tools(self) -> list[Tool]:
        """Every declared tool implementation, both audiences."""
        self.ensure_loaded()
        return list(self._declared)

    # -- loading ----------------------------------------------------------

    def ensure_loaded(self) -> None:
        """Run the loader once, on the first lookup that needs it."""
        if self._loaded or self._loading:
            return
        # A loader that raises must leave nothing behind: its partial
        # declarations would make the retry fail with a misleading
        # "declared twice" clash instead of the real error, and the failed
        # module's tools could never join the surface.
        declared_before = list(self._declared)
        index_before = self._by_role
        self._loading = True
        try:
            self._loader()
        except BaseException:
            self._declared = declared_before
            self._by_role = index_before
            raise
        finally:
            self._loading = False
        self._loaded = True

    def load(self) -> None:
        """Populate and resolve the registry (idempotent)."""
        self.ensure_loaded()
        self._by_role = self._resolve()

    def _resolve(self) -> dict[str, dict[str, Tool]]:
        """Index the declared tools by audience."""
        by_role: dict[str, dict[str, Tool]] = {role: {} for role in ROLES}
        for tool in self._declared:
            for role in ROLES:
                if tool.scope not in _VISIBLE_TO[role]:
                    continue
                by_role[role][tool.name] = tool
        return by_role


# The registry the module-level functions below operate on. Tests replace
# this with a fresh ToolRegistry to keep their declarations isolated.
registry = ToolRegistry()


def agent_tool(*, description: str,
               parameters: dict[str, dict] | None = None,
               name: str | None = None,
               scope: str = PARENT,
               required: tuple[str, ...] | list[str] | None = None
               ) -> Callable[[F], F]:
    """Declare a tool in the default registry (see ToolRegistry.agent_tool)."""
    return registry.agent_tool(description=description, parameters=parameters,
                               name=name, scope=scope, required=required)


def _build(func: Callable, description: str, properties: dict[str, dict],
           name: str | None, scope: str,
           required: tuple[str, ...] | list[str] | None) -> Tool:
    tool_name = name or func.__name__
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    needs_ctx = bool(params) and _is_agent_context(params[0].annotation)
    args = params[1:] if needs_ctx else params

    if required is None:
        required = tuple(p.name for p in args if p.default is p.empty)

    tool = Tool(name=tool_name, scope=scope, func=func,
                description=description, properties=properties,
                required=tuple(required), needs_ctx=needs_ctx)

    # Consumed by run_llm.execute_tool_call. Kept on the function so the
    # executor needs nothing but name -> callable; the same fact is on the
    # Tool itself for everyone else.
    setattr(func, '_needs_ctx', needs_ctx)

    _check(tool, args, needs_ctx)
    return tool


def _check(tool: Tool, args: list[inspect.Parameter], needs_ctx: bool) -> None:
    """Reject a declaration that cannot mean what it says."""
    declared = set(tool.properties)
    known = {p.name for p in args}
    unknown = declared - known
    if unknown:
        raise ValueError(
            f"tool {tool.name!r} declares parameters that are not in its "
            f"signature: {sorted(unknown)}")
    # The reverse direction: a parameter the model cannot express in the
    # schema is silently undiscoverable, so reject it too.
    unexpressible = known - declared
    if unexpressible:
        raise ValueError(
            f"tool {tool.name!r} takes parameters its schema does not "
            f"declare, so the model cannot pass them: "
            f"{sorted(unexpressible)}")
    missing = set(tool.required) - declared
    if missing:
        raise ValueError(
            f"tool {tool.name!r} requires parameters it does not declare: "
            f"{sorted(missing)}")
    if needs_ctx and tool.scope != PARENT and _takes_interactive_context(
            tool.func):
        raise ValueError(
            f"tool {tool.name!r} is offered to subagents but takes a "
            f"ChatUpdateContext; subagents only get an AgentContext")


def _is_agent_context(annotation: object) -> bool:
    if isinstance(annotation, str):
        return annotation in ('AgentContext', 'ChatUpdateContext')
    return (isinstance(annotation, type)
            and issubclass(annotation, AgentContext))


def _takes_interactive_context(func: Callable) -> bool:
    annotation = list(inspect.signature(func).parameters.values())[0].annotation
    if isinstance(annotation, str):
        return annotation == 'ChatUpdateContext'
    return isinstance(annotation, type) and issubclass(annotation,
                                                       ChatUpdateContext)


def tools_for(role: str) -> list[Tool]:
    """The tools one audience is offered, in declaration order."""
    return registry.tools_for(role)


def definitions_for(role: str) -> list[dict]:
    """The JSON tool definitions one audience is offered."""
    return registry.definitions_for(role)


def functions_for(role: str) -> dict[str, Callable[..., object]]:
    """The implementations one audience may execute, by tool name."""
    return registry.functions_for(role)


@dataclass(frozen=True)
class ToolSurface:
    """The tools one agent may call: definitions offered to the model,
    and the executable implementations by name."""

    definitions: list[dict]
    functions: dict[str, Callable[..., object]]


def surface_for(role: str) -> ToolSurface:
    """The registry's view for one audience, as a ToolSurface."""
    return ToolSurface(definitions=definitions_for(role),
                       functions=functions_for(role))


def all_tools() -> list[Tool]:
    """Every declared tool implementation, both audiences."""
    return registry.all_tools()


def load() -> None:
    """Populate and resolve the registry (idempotent)."""
    registry.load()
