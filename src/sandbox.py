import asyncio
import logging
import os
import re
import shlex
import time
from datetime import datetime, timezone
from typing import NamedTuple

import config

logger = logging.getLogger(__name__)

SANDBOX_LABEL = "toolbot.sandbox"
UPLOADS_DIR = "/workspace/uploads"

# Docker timestamp, e.g. 2026-10-14T09:00:00.123456789Z. Docker reports its
# zero time (0001-01-01T00:00:00Z) for "never happened".
_DOCKER_TIME_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?")


def _parse_docker_time(value: str) -> float | None:
    """Parse a Docker timestamp into epoch seconds, or None if unset."""
    match = _DOCKER_TIME_RE.match(value.strip())
    if match is None:
        return None
    year, month, day, hour, minute, second = (
        int(group) for group in match.groups()[:6])
    microsecond = int(((match.group(7) or "0") + "000000")[:6])
    if year <= 1:  # Docker's zero time means "never".
        return None
    try:
        stamp = datetime(year, month, day, hour, minute, second,
                         microsecond, tzinfo=timezone.utc)
    except ValueError:
        return None
    return stamp.timestamp()


class _ContainerState(NamedTuple):
    """Container state read back from docker inspect."""
    running: bool
    finished_at: float | None
    user_id: int | None


def _unique_name(name: str, existing: set[str]) -> str:
    if name not in existing:
        return name
    stem, ext = os.path.splitext(name)
    counter = 1
    while f"{stem}-{counter}{ext}" in existing:
        counter += 1
    return f"{stem}-{counter}{ext}"


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
    def __init__(self, container_name: str,
                 last_activity: float | None = None,
                 running: bool = True):
        self.container_name = container_name
        self.last_activity = (time.time() if last_activity is None
                              else last_activity)
        self.running = running


class SandboxManager:
    """Manages one ephemeral Docker sandbox container per user.

    A sandbox is stopped SANDBOX_IDLE_STOP_TIMEOUT seconds after its last
    access and destroyed SANDBOX_IDLE_DESTROY_TIMEOUT seconds after it
    (managed by reap_idle()). Stopping keeps the container and its files, so
    the next access restarts it where the user left off; destroying loses all
    changes. Creating or restarting a sandbox counts as access, so a
    never-used sandbox expires SANDBOX_IDLE_DESTROY_TIMEOUT after creation.
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

    async def ensure(self, user_id: int) -> Sandbox:
        """Return the user's sandbox, starting or restarting it if needed.

        A stopped container is restarted so its filesystem survives the idle
        timeout. Creating or restarting refreshes the activity timer; a
        sandbox that is already running is returned untouched.
        """
        async with self._lock_for(user_id):
            name = self.container_name_for(user_id)
            sandbox = self._sandboxes.get(user_id) or Sandbox(name)
            if await self._container_running(sandbox):
                self._sandboxes[user_id] = sandbox
                return sandbox
            rc, _, _ = await self._docker(["start", name])
            if rc == 0:
                logger.info("Restarted sandbox %s for user %d", name, user_id)
            else:
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
            sandbox = Sandbox(name)
            self._sandboxes[user_id] = sandbox
            return sandbox

    async def exec(self, user_id: int, command: str,
                   timeout: float | None = None,
                   stdin_data: bytes | None = None) -> dict:
        """Run a shell command in the user's sandbox.

        Refreshes the activity timer. Returns
        {stdout, stderr, returncode, timeout}.
        """
        sandbox = await self.ensure(user_id)
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

    async def store_file(self, user_id: int, data: bytes,
                         name: str) -> tuple[str | None, str | None]:
        """Store bytes in the sandbox uploads dir under a unique name.

        Returns (path, error); exactly one of the two is not None.
        """
        listing = await self.exec(
            user_id, f"mkdir -p {UPLOADS_DIR} && ls -1 {UPLOADS_DIR}")
        if listing["returncode"] != 0:
            error = ("Error preparing the sandbox uploads folder: "
                     f"{listing['stderr'].strip()[:500]}")
            return None, error
        path = f"{UPLOADS_DIR}/{_unique_name(name, set(listing['stdout'].splitlines()))}"
        write = await self.exec(
            user_id, f"cat > {shlex.quote(path)}", stdin_data=data)
        if write["returncode"] != 0:
            error = ("Error storing the file in the sandbox: "
                     f"{write['stderr'].strip()[:500]}")
            return None, error
        return path, None

    async def destroy(self, user_id: int) -> bool:
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

    async def _stop_sandbox(self, sandbox: Sandbox) -> bool:
        """Stop a sandbox container, keeping its files for a later restart.

        Returns True if the container was stopped.
        """
        rc, _, _ = await self._docker(["stop", sandbox.container_name])
        sandbox.running = False
        return rc == 0

    async def reap_idle(self) -> tuple[list[int], list[int]]:
        """Stop sandboxes idle past SANDBOX_IDLE_STOP_TIMEOUT and destroy
        sandboxes idle past SANDBOX_IDLE_DESTROY_TIMEOUT.

        A stopped sandbox keeps its files until it is destroyed and is
        restarted on the next access. Returns (stopped, destroyed) user ids.
        """
        now = time.time()
        stopped: list[int] = []
        destroyed: list[int] = []
        for user_id, sandbox in list(self._sandboxes.items()):
            idle = now - sandbox.last_activity
            if idle >= config.SANDBOX_IDLE_DESTROY_TIMEOUT:
                await self.destroy(user_id)
                destroyed.append(user_id)
            elif idle >= config.SANDBOX_IDLE_STOP_TIMEOUT and sandbox.running:
                await self._stop_sandbox(sandbox)
                stopped.append(user_id)
        return stopped, destroyed

    _INSPECT_STATE = (
        '{{.State.Running}} {{.State.FinishedAt}} '
        '{{index .Config.Labels "toolbot.uid"}}')

    async def _inspect(self, name: str) -> _ContainerState | None:
        """Read a container's state, or return None if it does not exist."""
        rc, stdout, _ = await self._docker(
            ["inspect", "-f", self._INSPECT_STATE, name])
        if rc != 0:
            return None
        fields = stdout.split(None, 2)
        if len(fields) < 2:
            return None
        user_id = None
        if len(fields) == 3:
            try:
                user_id = int(fields[2].strip())
            except ValueError:
                user_id = None
        return _ContainerState(
            running=fields[0] == "true",
            finished_at=_parse_docker_time(fields[1]),
            user_id=user_id)

    async def recover_leftover_sandboxes(self) -> tuple[int, int]:
        """Adopt or remove sandbox containers left over from previous runs.

        Containers idle past SANDBOX_IDLE_DESTROY_TIMEOUT are destroyed; the
        rest are adopted so reap_idle() can manage them and ensure() restarts
        them on the next access. The last access of a stopped container is
        estimated from when it was stopped. Returns (adopted, removed) counts.
        """
        rc, stdout, _ = await self._docker(
            ["ps", "-a", "--filter", f"label={SANDBOX_LABEL}",
             "--format", "{{.Names}}"])
        if rc != 0:
            return 0, 0
        now = time.time()
        adopted = 0
        removed = 0
        for name in stdout.split():
            state = await self._inspect(name)
            if state is None or state.user_id is None:
                # Leftovers without a uid cannot be aged, so treat as stale.
                await self._docker(["rm", "-f", name])
                removed += 1
                continue
            if state.running:
                # A running container may still be in use; assume it was
                # accessed just now so it is never destroyed too early.
                last_activity = now
            elif state.finished_at is not None:
                # Reap stops a container about SANDBOX_IDLE_STOP_TIMEOUT after
                # the last access, so estimate the access time from that.
                last_activity = (state.finished_at
                                 - config.SANDBOX_IDLE_STOP_TIMEOUT)
            else:
                last_activity = None
            if last_activity is None or (
                    now - last_activity >= config.SANDBOX_IDLE_DESTROY_TIMEOUT):
                await self._docker(["rm", "-f", name])
                removed += 1
                continue
            self._sandboxes[state.user_id] = Sandbox(
                name, last_activity=last_activity, running=state.running)
            adopted += 1
        if adopted or removed:
            logger.info("Recovered %d leftover sandbox container(s), "
                        "removed %d", adopted, removed)
        return adopted, removed


manager = SandboxManager()

_reaper_task: asyncio.Task | None = None


async def start_reaper() -> asyncio.Task:
    """Start the background task that stops and destroys idle sandboxes."""
    global _reaper_task
    stop_reaper()

    async def _reap_loop():
        while True:
            await asyncio.sleep(config.SANDBOX_REAP_INTERVAL)
            try:
                stopped, destroyed = await manager.reap_idle()
                for user_id in stopped:
                    logger.info("Stopped idle sandbox for user %d", user_id)
                for user_id in destroyed:
                    logger.info("Destroyed idle sandbox for user %d", user_id)
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
        result = await manager.exec(0, "echo Hello from sandbox!")
        print(result)
        print("destroyed:", await manager.destroy(0))

    asyncio.run(_self_test())
