"""The command surface: slash commands declare themselves next to their
implementations.

A command declares itself with the decorator, which also applies the
interactive middleware every command needs:

    @command('model', 'Set the model to use', usage=MODEL_USAGE)
    @settings
    async def model_command(ctx: ChatUpdateContext): ...

The decorator wraps the function in ``handlers.middleware.handler``
(authorisation, state initialisation and the error boundary) and records
a :class:`command_spec.Command` row. ``commands.py`` imports the handler
modules so their declarations run, then derives the Telegram menu, the PTB
``CommandHandler`` registrations and the name lookup from these rows: there
is no separate table to keep in step.

``@command`` is deliberately the only decorator a slash command needs. If
a command acquires a different middleware stack later, give it an option
here rather than stacking a second decorator at the call site, so the
registration and the middleware cannot diverge.
"""

from collections.abc import Awaitable, Callable
from typing import TypeVar, cast

from command_spec import Command
from handlers.middleware import handler

F = TypeVar('F', bound=Callable[..., Awaitable[object]])


class CommandRegistry:
    """Owns the declared commands; a fresh one keeps tests isolated."""

    def __init__(self) -> None:
        self._declared: list[Command] = []

    def command(self, name: str, help: str, *,
                usage: str | None = None,
                aliases: tuple[str, ...] = ()
                ) -> Callable[[F], F]:
        """Declare a slash command and wrap its handler in the middleware.

        ``name`` is the primary (bare) name, ``help`` is the one line the
        Telegram menu shows, ``usage`` is the argument syntax rendered by
        the handler, and ``aliases`` are alternate spellings that dispatch
        to the same callback without getting menu entries of their own.
        """

        def decorate(func: F) -> F:
            wrapped = cast(F, handler(func))
            self._declared.append(
                Command(name=name, help=help, callback=wrapped,
                        usage=usage, aliases=tuple(aliases)))
            return wrapped

        return decorate

    def declared(self) -> list[Command]:
        """The commands declared so far, in declaration order."""
        return list(self._declared)


# The registry the module-level decorator operates on. Tests replace this
# with a fresh CommandRegistry to keep their declarations isolated.
registry = CommandRegistry()


def command(name: str, help: str, *,
            usage: str | None = None,
            aliases: tuple[str, ...] = ()
            ) -> Callable[[F], F]:
    """Declare a slash command in the default registry."""
    return registry.command(name, help, usage=usage, aliases=aliases)
