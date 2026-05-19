import asyncio
import subprocess


DOCKER_TAG = "python:3.14-rc-slim"


async def run_python_in_docker(code: str, timeout: float = 30.0):
    cmd = [
        "docker", "run", "-i", "--rm",

        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp",
        "--pids-limit", "64",
        "--memory", "128m",
        "--cpus", "0.5",
        "--security-opt", "no-new-privileges",
        "--cap-drop", "ALL",
        "--user", "65534:65534",
        DOCKER_TAG,
        "python"
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=code.encode()),
            timeout=timeout
        )

        return {
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "returncode": proc.returncode,
            "timeout": False
        }

    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()

        return {
            "stdout": "",
            "stderr": "Execution timed out",
            "returncode": None,
            "timeout": True
        }


async def self_test():
    result = await run_python_in_docker("print('Hello from Docker!')")
    print("STDOUT:", result["stdout"])
    print("STDERR:", result["stderr"])
    print("Return Code:", result["returncode"])
    print("Timed Out:", result["timeout"])


if __name__ == "__main__":
    asyncio.run(self_test())