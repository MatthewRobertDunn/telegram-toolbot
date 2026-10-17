from chat_update_context import ChatUpdateContext, context_tool
from handlers.run_llm import run_llm

TOOL_NAME = "subagent"

# The subagent runs in a fresh conversation with none of this chat's history,
# so it gets a small system prompt explaining its role and that only its final
# message is ever seen again by the parent assistant.
SUBAGENT_SYSTEM_PROMPT = (
    "You are a subagent working on behalf of a larger assistant, completing a "
    "single task in isolation.\n"
    "Work through the task independently, using the available tools as needed.\n"
    "When you are done, end with a clear, self-contained final report: the "
    "parent assistant only reads your last message, so it must contain "
    "everything worth keeping - results, findings, and any facts the task "
    "asked for."
)


@context_tool
async def subagent(ctx: ChatUpdateContext, description: str, prompt: str) -> str:
    """Run a task in a fresh conversation and return its final report.

    The subagent's output is streamed to the chat like any other reply (via
    run_llm's draft streaming); the report itself is returned to the calling
    model only and is never sent to the user.
    """
    return await run_llm(
        ctx,
        model=ctx.settings.subagent_model,
        conversation=[
            {"role": "system",
             "content": f"{SUBAGENT_SYSTEM_PROMPT}\n\nTask: {description}"},
            {"role": "user", "content": prompt},
        ],
        excluded_tools=[TOOL_NAME],
    )


TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "strict": False,
            "description": (
                "Spawns a subagent in a separate conversation to work on the "
                "specified task and report back.\n"
                "The subagent starts with fresh context and none of this "
                "conversation's history, so include all relevant context and "
                "instructions in the prompt.\n"
                "It has the same tools as you (except spawning subagents of "
                "its own) and works in the same sandbox: it can read, change, "
                "and run things against the shared sandbox files, and those "
                "changes persist after it finishes.\n"
                "Its intermediate work (tool calls, exploration, drafts) "
                "stays out of this conversation; only its final report is "
                "returned here.\n"
                "Use this for research, multi-step work, or bulky tool output "
                "that would clutter this conversation, and ask it to return a "
                "summary or the specific facts you need."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "A short 3-5 word label for the task, displayed to the user"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The task for the subagent to perform. Include everything it needs: context, constraints, and what its report should contain."
                    }
                },
                "required": ["description", "prompt"],
                "additionalProperties": False
            }
        }
    },
]


def register(tools_list: list[dict], tool_map: dict):
    tools_list.extend(TOOL_DEFINITIONS)
    tool_map[TOOL_NAME] = subagent
