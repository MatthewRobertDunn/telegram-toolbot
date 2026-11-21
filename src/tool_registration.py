"""Loads the tool modules, in the order their tools are offered."""

import importlib
import logging

import tool_registry

logger = logging.getLogger(__name__)

# Declaration order is the order the model sees its tools in. It is fixed
# here rather than derived from a package scan so the tool list is stable
# from run to run.
#
# tools.status_tool is deliberately absent: it is not offered at the moment
# (see the module for why).
TOOL_MODULES = (
    'tools.web',
    'tools.datetime_tool',
    'tools.python_runner',
    'tools.memory_tool',
    'tools.journal_tool',
    'tools.image_gen',
    'tools.subagent',
    'tools.sandbox.shell',
    'tools.sandbox.files',
    'tools.sandbox.search',
    'tools.sandbox.media',
)


def load_tools() -> None:
    """Import every tool module, running their @agent_tool declarations.

    Afterwards every declared tool must have come from a module listed
    above: a declaration from anywhere else (a parked tool module imported
    by mistake, or a tool module importing another one) means the surface
    is not what this file says it is, so fail instead of serving it.
    """
    for name in TOOL_MODULES:
        importlib.import_module(name)
    _verify_declared_come_from_listed_modules()


def _verify_declared_come_from_listed_modules() -> None:
    listed = set(TOOL_MODULES)
    strays = [(tool.name, tool.func.__module__)
              for tool in tool_registry.registry.declared()
              if tool.func.__module__ not in listed]
    if strays:
        detail = ", ".join(f"{name!r} ({module})"
                           for name, module in strays)
        raise RuntimeError(
            f"tool(s) declared outside TOOL_MODULES: {detail}. Add the "
            f"module to TOOL_MODULES, or remove its @agent_tool "
            f"declaration if it is parked.")


def register_tools() -> None:
    """Populate and validate the registry. Called once at startup."""
    tool_registry.load()
    logger.info("Registered %d tools", len(tool_registry.all_tools()))
