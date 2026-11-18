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
        "Run a Python script in sandbox (/workspace, state persists, "
        "network available) and return output. Use for calculations, data "
        "processing, or testing code. Prefer over bash for Python."),
    parameters={
        "code": {
            "type": "string",
            "description": "Python code to execute.",
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
