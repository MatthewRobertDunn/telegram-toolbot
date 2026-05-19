from docker import run_python_in_docker


async def run_python_code(code: str) -> str:
    if not code.strip():
        return "Error: empty code."
    result = await run_python_in_docker(code, timeout=30.0)
    output = []
    if result["timeout"]:
        return "Execution timed out after 30.0 seconds."
    if result["returncode"] is not None:
        output.append(f"Return code: {result['returncode']}")
    if result["stdout"]:
        output.append(f"stdout:\n{result['stdout']}")
    if result["stderr"]:
        output.append(f"stderr:\n{result['stderr']}")
    if not output:
        return "(executed with no output)"
    return "\n".join(output)

TOOL_NAME = "python"
TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "strict": False,
            "description": "Execute a Python script in an isolated Docker container and return the output. The container has no network access, limited memory (128MB), and a timeout. Use this to run calculations, process data, or test code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute."
                    }
                },
                "required": ["code"],
                "additionalProperties": False
            }
        }
    },
]


def register(tools_list: list[dict], tool_map: dict):
    tools_list.extend(TOOL_DEFINITIONS)
    tool_map[TOOL_NAME] = run_python_code
