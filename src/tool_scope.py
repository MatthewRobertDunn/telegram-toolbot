"""Declares which agents may be offered a tool.

The tool registry is shared by the parent agent and its subagents, so each
tool states its audience next to its implementation:

    @parent_tool         the parent agent only (chat output, controls)
    @subagent_tool       both agents (the shared toolbox)
    @subagent_only_tool  subagents only (the silent variants)

Why not a deny-list of names: a new tool would silently join the subagent
surface. Here the runtime default is conservative - a tool with no
declaration is offered to the parent but never to a subagent - and
``tests/test_tool_scope.py`` asserts every registered tool is declared, so
a forgotten tag is a test failure rather than a leaked capability.
"""

from typing import Callable, TypeVar

PARENT = 'parent'
SUBAGENT = 'subagent'
BOTH = 'both'

F = TypeVar('F', bound=Callable[..., object])


def parent_tool(func: F) -> F:
    """Offer the tool to the parent agent only."""
    setattr(func, '_tool_scope', PARENT)
    return func


def subagent_tool(func: F) -> F:
    """Offer the tool to both the parent agent and its subagents."""
    setattr(func, '_tool_scope', BOTH)
    return func


def subagent_only_tool(func: F) -> F:
    """Offer the tool to subagents only."""
    setattr(func, '_tool_scope', SUBAGENT)
    return func


def tool_scope(func: object) -> str | None:
    """The declared scope of a tool function, or None when undeclared."""
    return getattr(func, '_tool_scope', None)
