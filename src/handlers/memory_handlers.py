from chat_update_context import ChatUpdateContext
from command_spec import usage_line
from handlers.middleware import handler
from memory_store import create_memory_store

# Usage syntax for this module's commands, also declared in the command
# table (commands.COMMANDS) so the two cannot drift apart.
MEMORY_USAGE = '<key> [value]'
MEMORY_DELETE_USAGE = '<key>'


@handler
async def memory_list_command(ctx: ChatUpdateContext):
    async with create_memory_store() as memory:
        keys = await memory.get_all_keys(ctx.user_id)
    if keys:
        key_list = "\n".join(keys)
        await ctx.messaging.send_preformatted_message(key_list)
    else:
        await ctx.messaging.send_message("I don't remember anything yet.")


@handler
async def memory_command(ctx: ChatUpdateContext):
    parts = ctx.command_text.split(maxsplit=1)
    if not parts:
        await ctx.messaging.send_message(
            usage_line(ctx.command_name, MEMORY_USAGE))
        return
    key = parts[0]
    if len(parts) == 1:
        async with create_memory_store() as memory:
            result = await memory.get_values(ctx.user_id, [key])
        value = result.get(key)
        if value is not None:
            await ctx.messaging.send_preformatted_message(value)
        else:
            await ctx.messaging.send_message(f'I don\'t remember ```{key}```.')
    else:
        value = parts[1]
        async with create_memory_store() as memory:
            await memory.set(ctx.user_id, key, value)
        await ctx.messaging.send_message(f"Remembered '{key}'.")


@handler
async def memory_delete_command(ctx: ChatUpdateContext):
    key = ctx.command_text
    if not key:
        await ctx.messaging.send_message(
            usage_line(ctx.command_name, MEMORY_DELETE_USAGE))
        return
    async with create_memory_store() as memory:
        await memory.delete_values(ctx.user_id, [key])
    await ctx.messaging.send_message(f"Forgot ```{key}```")


@handler
async def memory_delete_all_command(ctx: ChatUpdateContext):
    async with create_memory_store() as memory:
        await memory.delete_all(ctx.user_id)
    await ctx.messaging.send_message("I've forgotten everything.")
