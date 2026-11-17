"""Running commands in the sandbox."""

from chat_update_context import AgentContext
from sandbox import manager
from tool_registry import BOTH, agent_tool

from .common import _cap_output


@agent_tool(
    scope=BOTH,
    description=(
        "Execute a bash shell command in the user's sandbox and "
        "return the output. The sandbox persists between calls: files "
        "created and packages installed remain available in later "
        "calls. Commands always run from the workspace root /workspace. "
        "You don't need to prefix commands with `cd /workspace &&`."
        "The sandbox has full network access and root privileges, so "
        "software can be installed, e.g. 'apt-get update && "
        "apt-get install -y <package>'. Prefer the dedicated read, "
        "write, edit, grep and glob tools for file work; use bash "
        "for everything else (git, npm, running code, etc.). All "
        "state is permanently lost when the sandbox is destroyed, which "
        "happens 24 hours after the last command executed on it. "
        "Long output is truncated."),
    parameters={
        "command": {
            "type": "string",
            "description": "The bash command to execute.",
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
