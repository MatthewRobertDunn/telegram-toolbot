import json

from chat_update_context import ChatUpdateContext, context_tool
from context_store import create_memory_store


@context_tool
async def memory_get_keys(ctx: ChatUpdateContext) -> str:
    async with create_memory_store() as memory:
        keys = await memory.get_all_keys(ctx.user_id)
    return json.dumps(keys) if keys else "I don't remember anything yet."


@context_tool
async def memory_get_values(ctx: ChatUpdateContext, keys: list[str]) -> str:
    async with create_memory_store() as memory:
        result = await memory.get_values(ctx.user_id, keys)
    return json.dumps(result)


@context_tool
async def memory_set_value(ctx: ChatUpdateContext, key: str, value: str) -> str:
    async with create_memory_store() as memory:
        await memory.set(ctx.user_id, key, value)
    return f"Remembered '{key}'."


@context_tool
async def memory_delete_values(ctx: ChatUpdateContext, keys: list[str]) -> str:
    async with create_memory_store() as memory:
        await memory.delete_values(ctx.user_id, keys)
    return f"Forgot memories: {json.dumps(keys)}"


MEMORY_LIST = "memory_list"
MEMORY_GET = "memory_get"
MEMORY_SET = "memory_set"
MEMORY_DELETE = "memory_delete"
TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": MEMORY_LIST,
            "strict": False,
            "description": "List all keys in your memory. Your memory is private to each user and persists across interactions. Use it to remember facts, preferences, and details about the user.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": MEMORY_GET,
            "strict": False,
            "description": "Recall values for specified keys from your memory. Returns a dict mapping each key to its value (or null for anything you don't remember).",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of keys to recall values for."
                    }
                },
                "required": ["keys"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": MEMORY_SET,
            "strict": False,
            "description": "Save a key-value pair to your memory so you remember it in future interactions. Overwrites anything you previously remembered under the key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key under which to remember the value."
                    },
                    "value": {
                        "type": "string",
                        "description": "The value to remember."
                    }
                },
                "required": ["key", "value"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": MEMORY_DELETE,
            "strict": False,
            "description": "Forget specified keys, permanently erasing them from your memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of keys to forget."
                    }
                },
                "required": ["keys"],
                "additionalProperties": False
            }
        }
    },
]


def register(tools_list: list[dict], tool_map: dict):
    tools_list.extend(TOOL_DEFINITIONS)
    tool_map[MEMORY_LIST] = memory_get_keys
    tool_map[MEMORY_GET] = memory_get_values
    tool_map[MEMORY_SET] = memory_set_value
    tool_map[MEMORY_DELETE] = memory_delete_values
