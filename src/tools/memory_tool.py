import json

from chat_update_context import AgentContext
from context_store import create_memory_store
from tool_registry import BOTH, agent_tool


@agent_tool(
    name="memory_list",
    scope=BOTH,
    description=("List all keys in your memory. Your memory is private to "
                 "each user and persists across interactions. Use it to "
                 "remember facts, preferences, and details about the user."))
async def memory_get_keys(ctx: AgentContext) -> str:
    async with create_memory_store() as memory:
        keys = await memory.get_all_keys(ctx.user_id)
    return json.dumps(keys) if keys else "I don't remember anything yet."


@agent_tool(
    name="memory_get",
    scope=BOTH,
    description=("Recall values for specified keys from your memory. Returns "
                 "a dict mapping each key to its value (or null for anything "
                 "you don't remember)."),
    parameters={
        "keys": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of keys to recall values for.",
        },
    })
async def memory_get_values(ctx: AgentContext, keys: list[str]) -> str:
    async with create_memory_store() as memory:
        result = await memory.get_values(ctx.user_id, keys)
    return json.dumps(result)


@agent_tool(
    name="memory_set",
    scope=BOTH,
    description=("Save a key-value pair to your memory so you remember it in "
                 "future interactions. Overwrites anything you previously "
                 "remembered under the key."),
    parameters={
        "key": {
            "type": "string",
            "description": "The key under which to remember the value.",
        },
        "value": {
            "type": "string",
            "description": "The value to remember.",
        },
    })
async def memory_set_value(ctx: AgentContext, key: str, value: str) -> str:
    async with create_memory_store() as memory:
        await memory.set(ctx.user_id, key, value)
    return f"Remembered '{key}'."


@agent_tool(
    name="memory_delete",
    scope=BOTH,
    description=("Forget specified keys, permanently erasing them from your "
                 "memory."),
    parameters={
        "keys": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of keys to forget.",
        },
    })
async def memory_delete_values(ctx: AgentContext, keys: list[str]) -> str:
    async with create_memory_store() as memory:
        await memory.delete_values(ctx.user_id, keys)
    return f"Forgot memories: {json.dumps(keys)}"
