import json

from chat_update_context import ChatUpdateContext, context_tool
from context_store import create_store


@context_tool
async def store_get_keys(ctx: ChatUpdateContext) -> str:
    async with create_store() as store:
        keys = await store.get_all_keys(ctx.user_id)
    return json.dumps(keys) if keys else "No keys stored."


@context_tool
async def store_get_values(ctx: ChatUpdateContext, keys: list[str]) -> str:
    async with create_store() as store:
        result = await store.get_values(ctx.user_id, keys)
    return json.dumps(result)


@context_tool
async def store_set_value(ctx: ChatUpdateContext, key: str, value: str) -> str:
    async with create_store() as store:
        await store.set(ctx.user_id, key, value)
    return f"Stored '{key}'."


@context_tool
async def store_delete_values(ctx: ChatUpdateContext, keys: list[str]) -> str:
    async with create_store() as store:
        await store.delete_values(ctx.user_id, keys)
    return f"Deleted keys: {json.dumps(keys)}"

STORE_LIST = "store_list"
STORE_GET = "store_get"
STORE_SET = "store_set"
STORE_DELETE = "store_delete"
TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": STORE_LIST,
            "strict": False,
            "description": "List all keys stored for the current user's store. Returns a list of key names. This store is private to each user and persists across interactions.",
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
            "name": STORE_GET,
            "strict": False,
            "description": "Get values for specified keys from the user's store. Returns a dict mapping each key to its value (or null if not found).",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of keys to retrieve values for."
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
            "name": STORE_SET,
            "strict": False,
            "description": "Store a key-value pair in the user's store. Overwrites any existing value for the key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key under which to store the value."
                    },
                    "value": {
                        "type": "string",
                        "description": "The value."
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
            "name": STORE_DELETE,
            "strict": False,
            "description": "Delete specified keys and their values from the user's persistent store.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of keys to delete."
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
    tool_map[STORE_LIST] = store_get_keys
    tool_map[STORE_GET] = store_get_values
    tool_map[STORE_SET] = store_set_value
    tool_map[STORE_DELETE] = store_delete_values