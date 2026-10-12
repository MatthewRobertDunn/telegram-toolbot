import asyncio
import logging
import time

import config

logger = logging.getLogger(__name__)

SANDBOX_LABEL = "toolbot.sandbox"
UPLOADS_DIR = "/workspace/uploads"


class SandboxError(Exception):
    pass


class DockerTimeoutError(SandboxError):
    pass


class SandboxStartError(SandboxError):
    pass


async def run_docker_command(args: list[str],
                             timeout: float | None = None,
                             input: bytes | None = None
                             ) -> tuple[int | None, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "docker", *args,
        stdin=asyncio.subprocess.PIPE if input is not None
        else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=input), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise DockerTimeoutError(
            f"docker {' '.join(args[:2])} timed out after {timeout}s")
    return (
        proc.returncode,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


class Sandbox:
    def __init__(self, container_name: str):
        self.container_name = container_name
        self.last_activity = time.time()


class SandboxManager:
    """Manages one ephemeral Docker sandbox container per user.

    A sandbox is destroyed one hour after the last command executed on it
    (managed by reap_idle()), losing all changes. Creation is also treated
    as activity, so a never-used sandbox expires one hour after creation.
    """

    def __init__(self, docker=run_docker_command):
        self._docker = docker
        self._sandboxes: dict[int, Sandbox] = {}
        self._locks: dict[int, asyncio.Lock] = {}

    def _lock_for(self, user_id: int) -> asyncio.Lock:
        lock = self._locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[user_id] = lock
        return lock

    @staticmethod
    def container_name_for(user_id: int) -> str:
        return f"toolbot-sandbox-{user_id}"

    async def _container_running(self, sandbox: Sandbox) -> bool:
        rc, stdout, _ = await self._docker(
            ["inspect", "-f", "{{.State.Running}}", sandbox.container_name])
        return rc == 0 and stdout.strip() == "true"

    async def ensure_sandbox(self, user_id: int) -> Sandbox:
        """Return the user's running sandbox, starting/recreating it if needed.

        Does not refresh the activity timer; only exec_in_sandbox() does.
        """
        async with self._lock_for(user_id):
            sandbox = self._sandboxes.get(user_id)
            if sandbox is not None and await self._container_running(sandbox):
                return sandbox
            name = self.container_name_for(user_id)
            await self._docker(["rm", "-f", name])
            args = [
                "run", "-d",
                "--name", name,
                "--label", f"{SANDBOX_LABEL}=1",
                "--label", f"toolbot.uid={user_id}",
                "--memory", config.SANDBOX_MEMORY,
                "--cpus", config.SANDBOX_CPUS,
                "--pids-limit", str(config.SANDBOX_PIDS_LIMIT),
                "--workdir", "/workspace",
                config.SANDBOX_IMAGE,
                "sleep", "infinity",
            ]
            rc, stdout, stderr = await self._docker(
                args, timeout=config.SANDBOX_START_TIMEOUT)
            if rc != 0:
                logger.warning("Failed to start sandbox for user %d: %s",
                               user_id, stderr.strip())
                raise SandboxStartError(
                    f"Failed to start sandbox container: {stderr.strip()}")
            logger.info("Started sandbox %s for user %d", name, user_id)
            sandbox = Sandbox(container_name=name)
            self._sandboxes[user_id] = sandbox
            return sandbox

    async def exec_in_sandbox(self, user_id: int, command: str,
                              timeout: float | None = None,
                              stdin_data: bytes | None = None) -> dict:
        """Run a shell command in the user's sandbox.

        Refreshes the activity timer. Returns
        {stdout, stderr, returncode, timeout}.
        """
        sandbox = await self.ensure_sandbox(user_id)
        if timeout is None:
            timeout = config.SANDBOX_EXEC_TIMEOUT
        args = ["exec"]
        if stdin_data is not None:
            args.append("-i")
        args += [sandbox.container_name, "bash", "-c", command]
        try:
            rc, stdout, stderr = await self._docker(
                args, timeout=timeout, input=stdin_data)
            timed_out = False
        except DockerTimeoutError:
            rc, stdout, stderr = None, "", (
                f"Command timed out after {timeout} seconds "
                "and was killed; it may have left background processes.")
            timed_out = True
        sandbox.last_activity = time.time()
        return {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": rc,
            "timeout": timed_out,
        }

    async def destroy_sandbox(self, user_id: int) -> bool:
        """Force-remove the user's sandbox, losing all changes.

        Returns True if a container was removed.
        """
        async with self._lock_for(user_id):
            self._sandboxes.pop(user_id, None)
            rc, _, _ = await self._docker(
                ["rm", "-f", self.container_name_for(user_id)])
            if rc == 0:
                logger.info("Destroyed sandbox for user %d", user_id)
            return rc == 0

    async def reap_idle(self) -> list[int]:
        """Destroy sandboxes idle past SANDBOX_IDLE_TIMEOUT.

        Returns the user ids whose sandboxes were destroyed.
        """
        now = time.time()
        destroyed = []
        for user_id, sandbox in list(self._sandboxes.items()):
            if now - sandbox.last_activity >= config.SANDBOX_IDLE_TIMEOUT:
                await self.destroy_sandbox(user_id)
                destroyed.append(user_id)
        return destroyed

    async def cleanup_all_stale(self) -> int:
        """Remove sandbox containers left over from previous bot runs.

        Returns the number of containers removed.
        """
        rc, stdout, _ = await self._docker(
            ["ps", "-a", "--filter", f"label={SANDBOX_LABEL}",
             "--format", "{{.ID}}"])
        if rc != 0:
            return 0
        count = 0
        for container_id in stdout.split():
            r, _, _ = await self._docker(["rm", "-f", container_id])
            if r == 0:
                count += 1
        if count:
            logger.info("Cleaned up %d stale sandbox container(s)", count)
        return count


manager = SandboxManager()

_reaper_task: asyncio.Task | None = None


async def start_reaper() -> asyncio.Task:
    """Start the background task that destroys idle sandboxes."""
    global _reaper_task
    stop_reaper()

    async def _reap_loop():
        while True:
            await asyncio.sleep(config.SANDBOX_REAP_INTERVAL)
            try:
                for user_id in await manager.reap_idle():
                    logger.info("Reaped idle sandbox for user %d", user_id)
            except Exception:
                logger.exception("Sandbox reaper error")

    _reaper_task = asyncio.create_task(_reap_loop())
    return _reaper_task


def stop_reaper():
    global _reaper_task
    if _reaper_task is not None:
        _reaper_task.cancel()
        _reaper_task = None


if __name__ == "__main__":
    async def _self_test():
        result = await manager.exec_in_sandbox(0, "echo Hello from sandbox!")
        print(result)
        print("destroyed:", await manager.destroy_sandbox(0))

    asyncio.run(_self_test())
