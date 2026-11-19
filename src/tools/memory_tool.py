import json

from chat_update_context import AgentContext
from memory_store import create_memory_store
from tool_registry import BOTH, agent_tool


@agent_tool(
    name="memory_list",
    scope=BOTH,
    description=("List all memory keys. Memory is per-user, persistent; use "
                 "for facts, preferences, user details."))
async def memory_get_keys(ctx: AgentContext) -> str:
    async with create_memory_store() as memory:
        keys = await memory.get_all_keys(ctx.user_id)
    return json.dumps(keys) if keys else "I don't remember anything yet."


@agent_tool(
    name="memory_get",
    scope=BOTH,
    description=("Recall values for keys. Returns dict of key to value "
                 "(null if unknown)."),
    parameters={
        "keys": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Keys to recall.",
        },
    })
async def memory_get_values(ctx: AgentContext, keys: list[str]) -> str:
    async with create_memory_store() as memory:
        result = await memory.get_values(ctx.user_id, keys)
    return json.dumps(result)


@agent_tool(
    name="memory_set",
    scope=BOTH,
    description=("Save a key-value pair for future interactions. Overwrites "
                 "existing value."),
    parameters={
        "key": {
            "type": "string",
            "description": "Key to store value under.",
        },
        "value": {
            "type": "string",
            "description": "Value to remember.",
        },
    })
async def memory_set_value(ctx: AgentContext, key: str, value: str) -> str:
    async with create_memory_store() as memory:
        await memory.set(ctx.user_id, key, value)
    return f"Remembered '{key}'."


@agent_tool(
    name="memory_delete",
    scope=BOTH,
    description=("Forget keys, permanently erasing them."),
    parameters={
        "keys": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Keys to forget.",
        },
    })
async def memory_delete_values(ctx: AgentContext, keys: list[str]) -> str:
    async with create_memory_store() as memory:
        await memory.delete_values(ctx.user_id, keys)
    return f"Forgot memories: {json.dumps(keys)}"
