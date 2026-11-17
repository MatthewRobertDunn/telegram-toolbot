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
    """Import every tool module, running their @agent_tool declarations."""
    for name in TOOL_MODULES:
        importlib.import_module(name)


def register_tools() -> None:
    """Populate and validate the registry. Called once at startup."""
    tool_registry.load()
    logger.info("Registered %d tools", len(tool_registry.all_tools()))
