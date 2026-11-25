"""The command table: one row per command, feeding every consumer.

``COMMANDS`` is the single source of truth for the bot's command surface:

- ``bot_commands`` builds Telegram's ``set_my_commands`` menu (primary
  names only - aliases are alternate spellings of the same command, not
  menu entries of their own),
- ``register_commands`` registers one PTB ``CommandHandler`` per row
  (name plus aliases),
- ``lookup_command`` maps a name back to its row.

Adding a command is therefore one row here; nothing else needs an edit.
``tests/test_command_handlers.py`` pins the invariants (unique names and
aliases, non-empty help, callable callbacks, ``/continue`` aliasing
``/retry``, usage matching the handler modules' constants) so the menu can
never drift from what is registered.

Parsing lives with the per-update context (``chat_update_context``), whose
factory is the parsers' only consumer; ``usage_line`` and the ``Command``
row type live in the dependency-free ``command_spec``. The table imports
handler callbacks and their usage constants, so no lazy imports are
needed anywhere.
"""

from telegram import BotCommand
from telegram.ext import Application, CommandHandler

from command_spec import Command
from chat_update_context import callback_adapter
from handlers.command_handlers import (
    MODEL_USAGE,
    SUBAGENT_MODEL_USAGE,
    SYSTEMPROMPT_USAGE,
    TEMPERATURE_USAGE,
    TIMEZONE_USAGE,
    TOP_P_USAGE,
    USER_PROFILE_USAGE,
    model_command,
    model_list_command,
    model_reset_command,
    settings_reset_command,
    start_restart_command,
    stop_command,
    subagent_model_command,
    subagent_model_reset_command,
    systemprompt_command,
    systemprompt_reset_command,
    temperature_command,
    temperature_reset_command,
    thinking_keep_command,
    timezone_command,
    tool_output_command,
    top_p_command,
    top_p_reset_command,
    user_profile_command,
    user_profile_reset_command,
)
from handlers.conversation_handlers import (
    COMPACT_USAGE,
    compact_command,
    impersonate_command,
    regenerate_command,
    retry_command,
    undo_command,
)
from handlers.memory_handlers import (
    MEMORY_DELETE_USAGE,
    MEMORY_USAGE,
    memory_command,
    memory_delete_all_command,
    memory_delete_command,
    memory_list_command,
)
from handlers.sandbox_handlers import sandbox_reset_command
from handlers.subagent_handlers import subagent_list_command
from handlers.window_handlers import window_new_command


COMMANDS: tuple[Command, ...] = (
    Command('start', 'Start a new conversation', start_restart_command),
    Command('stop', 'Stop the current AI response', stop_command),
    Command('regenerate', 'Regenerate the last response',
            regenerate_command),
    # /continue is an older spelling of /retry: an alias, not its own row,
    # so it gets no separate menu entry.
    Command('retry', 'Retry the conversation', retry_command,
            aliases=('continue',)),
    Command('undo', 'Undo the last message', undo_command),
    Command('impersonate', 'Impersonate your next message',
            impersonate_command),
    Command('compact', 'Compact the conversation context', compact_command,
            usage=COMPACT_USAGE),
    Command('window_new', 'Open a new thread in this chat',
            window_new_command),
    Command('systemprompt', 'Set a custom system prompt',
            systemprompt_command, usage=SYSTEMPROMPT_USAGE),
    Command('systemprompt_reset', 'Reset system prompt to default',
            systemprompt_reset_command),
    Command('model', 'Set the model to use', model_command,
            usage=MODEL_USAGE),
    Command('model_list', 'List available NVIDIA NIM models',
            model_list_command),
    Command('model_reset', 'Reset model to default', model_reset_command),
    Command('subagent_model', 'Set the model used by the subagent tool',
            subagent_model_command, usage=SUBAGENT_MODEL_USAGE),
    Command('subagent_model_reset', 'Reset subagent model to default',
            subagent_model_reset_command),
    Command('subagent_list', 'List running background subagents',
            subagent_list_command),
    Command('top_p', 'Set top_p sampling parameter', top_p_command,
            usage=TOP_P_USAGE),
    Command('top_p_reset', 'Reset top_p to default', top_p_reset_command),
    Command('temperature', 'Set temperature parameter', temperature_command,
            usage=TEMPERATURE_USAGE),
    Command('temperature_reset', 'Reset temperature to default',
            temperature_reset_command),
    Command('timezone', 'Set your timezone (e.g. Australia/Sydney)',
            timezone_command, usage=TIMEZONE_USAGE),
    Command('memory', 'Get or remember a memory', memory_command,
            usage=MEMORY_USAGE),
    Command('memory_list', 'List everything I remember',
            memory_list_command),
    Command('memory_delete', 'Forget a remembered key',
            memory_delete_command, usage=MEMORY_DELETE_USAGE),
    Command('memory_delete_all', 'Forget everything I remember',
            memory_delete_all_command),
    Command('tool_output', 'Toggle tool call output on/off',
            tool_output_command),
    Command('thinking_keep', 'Toggle thinking output on/off',
            thinking_keep_command),
    Command('settings_reset', 'Reset all user settings to default',
            settings_reset_command),
    Command('sandbox_reset', 'Destroy your sandbox and all its files',
            sandbox_reset_command),
    Command('user_profile', 'Set your profile/biography',
            user_profile_command, usage=USER_PROFILE_USAGE),
    Command('user_profile_reset', 'Reset your profile to empty',
            user_profile_reset_command),
)


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
