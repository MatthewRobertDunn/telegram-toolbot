import config
from chat_update_context import ChatUpdateContext, context_tool
from handlers.run_with_tools import run_with_tools

@context_tool
async def ask_strong_llm(ctx: ChatUpdateContext, prompt: str, system_prompt: str = "") -> str:
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": prompt})

    return await run_with_tools(
        ctx,
        model=config.STRONG_LLM_MODEL,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
        conversation=conversation,
        excluded_tools=["ask_strong_llm"],
    )

TOOL_NAME = "ask_strong_llm"
TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "strict": False,
            "description": "Ask a much more powerful LLM a question or prompt. Use this for tasks that require superior reasoning, complex analysis, code generation, mathematical problem-solving, or creative work that the current model may struggle with. The strong model can also use tools like web search and code execution if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The question, instruction, or prompt to send to the strong model."
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional system prompt to override the default behavior of the strong model. Leave empty for default."
                    }
                },
                "required": ["prompt"],
                "additionalProperties": False
            }
        }
    },
]


def register(tools_list: list[dict], tool_map: dict):
    tools_list.extend(TOOL_DEFINITIONS)
    tool_map[TOOL_NAME] = ask_strong_llm