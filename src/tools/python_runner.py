import logging

from chat_update_context import AgentContext
from sandbox import manager
from tool_registry import BOTH, agent_tool
from .sandbox.common import _cap_output

logger = logging.getLogger(__name__)


@agent_tool(
    name="python",
    scope=BOTH,
    description=(
        "Execute a Python script in the user's sandbox and return the "
        "output. The script runs with the working directory "
        "/workspace (continue reading/writing files there with the "
        "other tools afterwards). The sandbox has network access, "
        "and state persists between calls. Use this to run "
        "calculations, process data, or test code. Prefer this tool "
        "over the bash tool for Python scripts."),
    parameters={
        "code": {
            "type": "string",
            "description": "The Python code to execute.",
        },
    })
async def run_python_code(ctx: AgentContext, code: str) -> str:
    if not code.strip():
        return "Error: empty code."
    result = await manager.exec(
        ctx.user_id, "cd /workspace && python3 -",
        stdin_data=code.encode())
    if result["timeout"]:
        return "Execution timed out."
    output = []
    if result["returncode"] is not None:
        output.append(f"Return code: {result['returncode']}")
    if result["stdout"]:
        output.append(f"stdout:\n{_cap_output(result['stdout'])}")
    if result["stderr"]:
        output.append(f"stderr:\n{_cap_output(result['stderr'])}")
    if not output:
        return "(executed with no output)"
    return "\n".join(output)
