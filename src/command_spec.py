"""The command vocabulary shared by the table and the handlers.

This module is deliberately dependency-free: it imports nothing from
``commands``, ``handlers`` or ``chat_update_context``. It is the bottom of
the import graph so the command table can be built on top without a
cycle.

- ``usage_line`` renders the usage string a handler sends.
- ``Command`` is the table row type; its ``callback`` is typed loosely
  (``Any``) precisely so this module never needs the handler modules or
  the context type - the table, not this module, wires real callbacks in.
"""


from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


CommandCallback = Callable[[Any], Awaitable[object]]


@dataclass(frozen=True)
class Command:
    """One row of the command table.

    ``name``
        The primary (bare) name: what the menu lists and what usage lines
        render.
    ``help``
        One line of menu help.
    ``callback``
        The handler, already wrapped in the middleware it needs.
    ``usage``
        The argument syntax, without the command name itself (e.g.
        ``<model_id>``); ``None`` for commands that take no input and so
        never render a usage line.
    ``aliases``
        Alternate names that run the same callback (menu entries are not
        generated for them).
    """

    name: str
    help: str
    callback: CommandCallback
    usage: str | None = None
    aliases: tuple[str, ...] = ()


def usage_line(command_name: str, usage: str) -> str:
    """The usage line a handler sends: the bare, already-lowercased
    command name plus its syntax, e.g.
    ``Usage: ``/model <model_id>````.

    Handlers pass their own ``*_USAGE`` module constant, the same constant
    their command table row declares, so the two cannot drift.
    """
    return f'Usage: `/{command_name} {usage}`'
