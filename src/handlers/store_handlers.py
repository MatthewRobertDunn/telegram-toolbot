from chat_update_context import ChatUpdateContext
from messaging import send_message, send_preformatted_message
from handlers.middleware import handler
from context_store import create_store


@handler
async def store_list_command(ctx: ChatUpdateContext):
    async with create_store() as store:
        keys = await store.get_all_keys(ctx.user_id)
    if keys:
        key_list = "\n".join(keys)
        await send_preformatted_message(ctx, key_list)
    else:
        await send_message(ctx, 'No keys stored.')


@handler
async def store_command(ctx: ChatUpdateContext):
    parts = ctx.command_text.split(maxsplit=1)
    if not parts:
        await send_message(ctx, f'Usage: `/{ctx.command_name} <key>` or `/{ctx.command_name} <key> <value>`')
        return
    key = parts[0]
    if len(parts) == 1:
        async with create_store() as store:
            result = await store.get_values(ctx.user_id, [key])
        value = result.get(key)
        if value is not None:
            await send_preformatted_message(ctx, value)
        else:
            await send_message(ctx, f'Key ```{key}``` not found.')
    else:
        value = parts[1]
        async with create_store() as store:
            await store.set(ctx.user_id, key, value)
        await send_message(ctx, f"Stored '{key}'.")


@handler
async def store_delete_command(ctx: ChatUpdateContext):
    key = ctx.command_text
    if not key:
        await send_message(ctx, f'Usage: `/{ctx.command_name} <key>`')
        return
    async with create_store() as store:
        await store.delete_values(ctx.user_id, [key])
    await send_message(ctx, f"Deleted ```{key}```.")


@handler
async def store_delete_all_command(ctx: ChatUpdateContext):
    async with create_store() as store:
        await store.delete_all(ctx.user_id)
    await send_message(ctx, 'All keys deleted.')
