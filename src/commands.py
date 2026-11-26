"""The command table, derived from the ``@command`` declarations.

``COMMAND_MODULES`` imports every module that declares slash commands, so
their decorators run at startup. ``COMMANDS`` is then built from the
registry and sorted into the menu order:

- ``bot_commands`` builds Telegram's ``set_my_commands`` menu (primary
  names only - aliases are alternate spellings of the same command, not
  menu entries of their own),
- ``register_commands`` registers one PTB ``CommandHandler`` per row
  (name plus aliases),
- ``lookup_command`` maps a name back to its row.

Adding a command is therefore a ``@command`` declaration next to its
handler; nothing here needs an edit unless the handler module is new.
``tests/test_command_handlers.py`` pins the invariants (unique names and
aliases, non-empty help, callable callbacks, ``/continue`` aliasing
``/retry``, usage matching the handler modules' constants) so the menu can
never drift from what is registered.

Parsing lives with the per-update context (``chat_update_context``), whose
factory is the parsers' only consumer; ``usage_line`` and the ``Command``
row type live in the dependency-free ``command_spec``.
"""

import importlib

from telegram import BotCommand
from telegram.ext import Application, CommandHandler

from chat_update_context import callback_adapter
from command_registry import registry
from command_spec import Command


# Declaration order is irrelevant: COMMANDS sorts for the menu. The list
# exists so the decorators run; a module missing here never registers.
COMMAND_MODULES = (
    'handlers.command_handlers',
    'handlers.conversation_handlers',
    'handlers.memory_handlers',
    'handlers.sandbox_handlers',
    'handlers.subagent_handlers',
    'handlers.window_handlers',
)


def _load_commands() -> tuple[Command, ...]:
    """Import the declaring modules, then return the menu-ordered rows.

    Alphabetical, with the shortest spelling first where names share a
    prefix: ``model`` precedes ``model_list`` precedes ``model_reset``.
    """
    for module in COMMAND_MODULES:
        importlib.import_module(module)
    return tuple(sorted(registry.declared(),
                        key=lambda command: (command.name,
                                             len(command.name))))


COMMANDS: tuple[Command, ...] = _load_commands()


def _build_lookup() -> dict[str, Command]:
    lookup: dict[str, Command] = {}
    for command in COMMANDS:
        for key in (command.name, *command.aliases):
            folded = key.lower()
            existing = lookup.get(folded)
            if existing is not None:
                raise ValueError(
                    f'/{key} is declared twice (/{existing.name} and '
                    f'/{command.name})')
            lookup[folded] = command
    return lookup


# Built once at import so a duplicate name is a startup error rather than
# a command silently shadowing another. Keys are folded: PTB lowercases
# command names when registering, so 'Start' and 'start' are the same
# command as far as dispatch is concerned.
_BY_NAME: dict[str, Command] = _build_lookup()


def lookup_command(name: str) -> Command | None:
    """The row a name resolves to, aliases included; None when unknown."""
    return _BY_NAME.get(name.lower())


def bot_commands() -> list[BotCommand]:
    """The Telegram command menu: one entry per row, in table order."""
    return [BotCommand(command.name, command.help) for command in COMMANDS]


def register_commands(application: "Application") -> None:
    """Register every row with PTB: name and aliases in one handler.

    Message handlers go on the application after this call: commands have
    to be matched before they can fall through to plain text.
    """
    for command in COMMANDS:
        application.add_handler(CommandHandler(
            [command.name, *command.aliases],
            callback_adapter(command.callback)))
