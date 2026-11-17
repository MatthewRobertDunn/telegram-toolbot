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


@dataclass(frozen=True)
class Tool:
    """One tool implementation, as seen by one audience."""

    name: str
    scope: str
    func: Callable[..., object]
    description: str
    properties: dict[str, dict]
    required: tuple[str, ...]

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


_DECLARED: list[Tool] = []
# name -> implementation, one entry per audience.
_BY_ROLE: dict[str, dict[str, Tool]] = {}
_loaded = False


def agent_tool(*, description: str,
               parameters: dict[str, dict] | None = None,
               name: str | None = None,
               scope: str = PARENT,
               required: tuple[str, ...] | list[str] | None = None):
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
        _DECLARED.append(tool)
        # A tool declared after the first lookup (a test's, or a module
        # imported late) invalidates the resolved index.
        global _BY_ROLE
        _BY_ROLE = {}
        return func

    return decorate


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
                required=tuple(required))

    # Consumed by run_llm.execute_tool_call. Kept on the function so the
    # executor needs nothing but name -> callable.
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


def role_for(ctx: AgentContext) -> str:
    """Which audience a context belongs to."""
    return PARENT if isinstance(ctx, ChatUpdateContext) else SUBAGENT


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    from tool_registration import load_tools
    load_tools()


def _resolve() -> dict[str, dict[str, Tool]]:
    """Index the declared tools by audience, rejecting a name clash."""
    _ensure_loaded()
    by_role: dict[str, dict[str, Tool]] = {role: {} for role in ROLES}
    for tool in _DECLARED:
        for role in ROLES:
            if tool.scope not in _VISIBLE_TO[role]:
                continue
            existing = by_role[role].get(tool.name)
            if existing is not None:
                raise ValueError(
                    f"tool {tool.name!r} is declared twice for {role} "
                    f"({existing.scope} and {tool.scope})")
            by_role[role][tool.name] = tool
    return by_role


def tools_for(role: str) -> list[Tool]:
    """The tools one audience is offered, in declaration order."""
    global _BY_ROLE
    if not _BY_ROLE:
        _BY_ROLE = _resolve()
    return list(_BY_ROLE[role].values())


def definitions_for(role: str) -> list[dict]:
    """The JSON tool definitions one audience is offered."""
    return [tool.definition() for tool in tools_for(role)]


def functions_for(role: str) -> dict[str, Callable[..., object]]:
    """The implementations one audience may execute, by tool name."""
    return {tool.name: tool.func for tool in tools_for(role)}


def all_tools() -> list[Tool]:
    """Every declared tool implementation, both audiences."""
    _ensure_loaded()
    return list(_DECLARED)


def load() -> None:
    """Populate and resolve the registry (idempotent)."""
    global _BY_ROLE
    _ensure_loaded()
    _BY_ROLE = _resolve()
