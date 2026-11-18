"""Running commands in the sandbox."""

from chat_update_context import AgentContext
from sandbox import manager
from tool_registry import BOTH, agent_tool

from .common import _cap_output


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
    output = []
    if result["timeout"]:
        output.append("Command timed out.")
    if result["returncode"] is not None:
        output.append(f"Return code: {result['returncode']}")
    if result["stdout"]:
        output.append(f"stdout:\n{_cap_output(result['stdout'])}")
    if result["stderr"]:
        output.append(f"stderr:\n{_cap_output(result['stderr'])}")
    if not output:
        return "(executed with no output)"
    return "\n".join(output)
