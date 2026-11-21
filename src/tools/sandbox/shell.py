"""Running commands in the sandbox."""

from chat_update_context import AgentContext
from sandbox import manager
from tool_registry import BOTH, agent_tool

from .common import format_command_result


@agent_tool(
    scope=BOTH,
    description=(
        "Run a bash command in sandbox and return output. Sandbox persists "
        "(files/packages remain); commands run from /workspace (no `cd` "
        "needed). Full network + root, e.g. 'apt-get update && apt-get "
        "install -y <package>'. Prefer read/write/edit/grep/glob for file "
        "work; bash for rest (git, npm, running code, etc.). State lost 24h "
        "after last command. Long output truncated."),
    parameters={
        "command": {
            "type": "string",
            "description": "Bash command to execute.",
        },
    })
async def bash(ctx: AgentContext, command: str) -> str:
    if not command.strip():
        return "Error: empty command."
    result = await manager.exec(ctx.user_id, command)
    return format_command_result(result)
