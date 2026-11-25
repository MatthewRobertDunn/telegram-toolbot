import asyncio
import contextlib
import logging
import os
import re
import shlex
import time
import weakref
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from typing import Any, NamedTuple, Protocol

import config

logger = logging.getLogger(__name__)

SANDBOX_LABEL = "toolbot.sandbox"
UPLOADS_DIR = "/workspace/uploads"

# The sandbox enforces command deadlines itself (see _deadline_command);
# this is only the extra time the docker CLI gets beyond that deadline
# before the client-side backstop kills it and drops whatever output the
# command had produced.
_EXEC_GRACE_SECONDS = 30.0

# GNU timeout's exit status for a command it killed at the deadline; the
# wrapped command exits with it too.
_TIMEOUT_EXIT_CODE = 124

# 128+SIGKILL. Both the deadline's -k escalation and an in-container kill
# (e.g. the OOM killer under the memory limit) exit 137; _exec_locked
# tells them apart by elapsed time (see its timeout computation).
_KILL_EXIT_CODE = 137
_KILL_GRACE_SECONDS = 5.0  # matches the -k 5 in _deadline_command

# Longest a single upload write may run. Uploads are bounded by Telegram's
# bot download limit, so this only catches a wedged container.
_STORE_FILE_TIMEOUT = 120.0

_ERROR_CHARS = 500

# Docker daemon stderr for a sandbox container that is gone: stopped ("is
# not running") or removed ("No such container" / "No such object"). Only
# matched against `docker exec` failures, so a command's own stderr is
# never reinterpreted.
_CONTAINER_GONE_RE = re.compile(
    r"is not running|No such container|No such object|is marked for removal")


def _container_gone(stderr: str) -> bool:
    """True if docker stderr says the sandbox container is gone."""
    return _CONTAINER_GONE_RE.search(stderr) is not None


def _deadline_command(command: str, timeout: float) -> str:
    """Wrap a command so the sandbox enforces the deadline itself.

    Container-side ``timeout`` stops the command at the deadline: SIGTERM
    to the command's process group, escalating to SIGKILL after
    _KILL_GRACE_SECONDS, so partial output survives and nothing keeps
    running past the deadline (docker's --init reaps the remains). The
    client-side wait_for is only a backstop for a hung docker CLI; it is
    granted _EXEC_GRACE_SECONDS beyond the deadline. A command killed at
    the deadline exits with _TIMEOUT_EXIT_CODE; a command that ignored
    SIGTERM dies by the escalation with _KILL_EXIT_CODE (see
    _exec_locked for how the two SIGKILL meanings are told apart).
    """
    return f"timeout -k {_KILL_GRACE_SECONDS:g} {timeout:g} bash -c {shlex.quote(command)}"


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
    running: bool
    finished_at: float | None
    user_id: int | None


# One exec does the mkdir, the name pick and the write, so two concurrent
# uploads cannot race a listing-then-write pair into the same name. stdin
# is the file contents; stdout is the stored path. A taken name falls back
# to stem-1.ext, stem-2.ext, ...
_STORE_SCRIPT = """set -e
d={dir}
n={name}
mkdir -p "$d"
p="$d/$n"
if [ -e "$p" ]; then
    case "$n" in
        ?*.*) stem=${{n%.*}}; ext=.${{n##*.}} ;;
        *) stem=$n; ext= ;;
    esac
    i=1
    while [ -e "$d/$stem-$i$ext" ]; do
        i=$((i + 1))
    done
    p="$d/$stem-$i$ext"
fi
cat > "$p"
printf '%s' "$p"
"""


def _store_script(name: str) -> str:
    return _STORE_SCRIPT.format(dir=shlex.quote(UPLOADS_DIR),
                                name=shlex.quote(name))


class SandboxError(Exception):
    """Base class for sandbox failures reported to callers.

    Subclasses carry the classification; messages are written for whoever
    surfaces them and never quote raw docker stderr (that goes to the log).
    """


class DockerTimeoutError(SandboxError):
    """The docker CLI did not finish within its client-side deadline."""


class SandboxStartError(SandboxError):
    """The sandbox container could not be started or restarted."""


class ReadMarkHolder(Protocol):
    """The per-user object the sandbox tools hang read marks on.

    chat_update_context.UserData satisfies this structurally; the protocol
    keeps sandbox.py from importing the chat layer.
    """

    sandbox_read_files: set[str]


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

    def __init__(self, docker=run_docker_command,
                 clock: Callable[[], float] = time.monotonic):
        self._docker = docker
        self._clock = clock
        self._sandboxes: dict[int, Sandbox] = {}
        self._locks: dict[int, asyncio.Lock] = {}
        self._read_marks: dict[int, list[weakref.ref[ReadMarkHolder]]] = {}

    def _lock_for(self, user_id: int) -> asyncio.Lock:
        lock = self._locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[user_id] = lock
        return lock

    @contextlib.asynccontextmanager
    async def _with_container(self, user_id: int) -> AsyncIterator[None]:
        """Hold the user's sandbox lock for a whole operation.

        One lock per user serializes every operation on that sandbox —
        starting it, running commands, stopping, destroying — so a stop or
        destroy cannot interleave between ensure() and the docker call it
        drives.
        """
        async with self._lock_for(user_id):
            yield

    def watch_read_marks(self, user_id: int, holder: ReadMarkHolder) -> None:
        """Track where a user's read marks live so destroy() can clear them.

        The sandbox tools keep read-before-write marks on the agent's
        per-user state (tools/sandbox/common.py); that object outlives any
        single tool call, so destroy() needs a way back to it. Held weakly:
        a throwaway subagent state object must not be kept alive here.
        """
        holders = self._read_marks.get(user_id)
        if holders is None:
            holders = self._read_marks[user_id] = []
        holders[:] = [ref for ref in holders if ref() is not None]
        if not any(ref() is holder for ref in holders):
            holders.append(weakref.ref(holder))

    def _clear_read_marks(self, user_id: int) -> None:
        """Forget the user's read-before-write marks: the files are gone."""
        for ref in self._read_marks.pop(user_id, []):
            holder = ref()
            if holder is not None:
                holder.sandbox_read_files.clear()

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
        async with self._with_container(user_id):
            return await self._ensure_locked(user_id)

    async def _ensure_locked(self, user_id: int) -> Sandbox:
        """ensure()'s body; the caller holds the user's sandbox lock."""
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
                "--security-opt", "no-new-privileges",
                "--workdir", "/workspace",
            ]
            skills_dir = os.path.abspath(config.SKILLS_DIR)
            if os.path.isdir(skills_dir):
                args += [
                    "--mount",
                    f"type=bind,src={skills_dir},"
                    f"dst={config.SANDBOX_SKILLS_DIR},readonly",
                ]
            # --init gives the container a PID 1 that reaps, so a killed
            # command cannot leave zombies behind.
            args += [
                "--init",
                config.SANDBOX_IMAGE,
                "sleep", "infinity",
            ]
            rc, stdout, stderr = await self._docker(
                args, timeout=config.SANDBOX_START_TIMEOUT)
            if rc != 0:
                logger.warning("Failed to start sandbox for user %d: %s",
                               user_id, stderr.strip())
                raise SandboxStartError(
                    f"Could not start the sandbox container "
                    f"(exit code {rc}).")
            logger.info("Started sandbox %s for user %d", name, user_id)
        sandbox = Sandbox(name)
        self._sandboxes[user_id] = sandbox
        return sandbox

    async def exec(self, user_id: int, command: str,
                   timeout: float | None = None,
                   stdin_data: bytes | None = None) -> dict[str, Any]:
        """Run a shell command in the user's sandbox.

        Refreshes the activity timer. The deadline is enforced inside the
        container (_deadline_command), so a timed-out command's partial
        output survives; the client-side wait is only a backstop for a
        hung docker CLI, and its output is lost with the client. If the
        container vanished mid-command (reset or reaper), the daemon's
        stderr is translated into a plain message instead. Returns
        {stdout, stderr, returncode, timeout}.
        """
        async with self._with_container(user_id):
            return await self._exec_locked(user_id, command, timeout,
                                           stdin_data)

    async def _exec_locked(self, user_id: int, command: str,
                           timeout: float | None,
                           stdin_data: bytes | None) -> dict[str, Any]:
        """exec()'s body; the caller holds the user's sandbox lock."""
        sandbox = await self._ensure_locked(user_id)
        if timeout is None:
            timeout = config.SANDBOX_EXEC_TIMEOUT
        args = ["exec"]
        if stdin_data is not None:
            args.append("-i")
        args += [sandbox.container_name, "bash", "-c",
                 _deadline_command(command, timeout)]
        started = self._clock()
        try:
            rc, stdout, stderr = await self._docker(
                args, timeout=timeout + _EXEC_GRACE_SECONDS,
                input=stdin_data)
        except DockerTimeoutError:
            # Backstop fired: the docker CLI hung past the deadline, so the
            # command's output is lost with it.
            logger.warning(
                "docker exec for user %d hung past %ss; killed the client",
                user_id, timeout + _EXEC_GRACE_SECONDS)
            sandbox.last_activity = time.time()
            return {
                "stdout": "",
                "stderr": (f"Command timed out after {timeout:g} seconds "
                           "and was killed; its output was lost."),
                "returncode": None,
                "timeout": True,
            }
        elapsed = self._clock() - started
        if rc != 0 and _container_gone(stderr):
            # The container vanished mid-command (/sandbox_reset or the
            # reaper). The command never ran; say so instead of surfacing
            # the daemon's stderr.
            logger.info("Sandbox for user %d vanished during exec: %s",
                        user_id, stderr.strip())
            sandbox.last_activity = time.time()
            return {
                "stdout": "",
                "stderr": ("The sandbox was reset or stopped; it will be "
                           "recreated on the next command."),
                "returncode": None,
                "timeout": False,
            }
        sandbox.last_activity = time.time()
        return {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": rc,
            # 124 is timeout(1)'s own deadline status. A SIGKILL death
            # (137) counts as a timeout only once it can only be the -k
            # escalation: the same code within the deadline is an
            # in-container kill such as the OOM killer, which is an
            # ordinary command failure, not a deadline hit.
            "timeout": (rc == _TIMEOUT_EXIT_CODE
                        or (rc == _KILL_EXIT_CODE
                            and elapsed >= timeout + _KILL_GRACE_SECONDS)),
        }

    async def store_file(self, user_id: int, data: bytes,
                         name: str) -> tuple[str | None, str | None]:
        """Store bytes in the sandbox uploads dir under a unique name.

        Returns (path, error); exactly one of the two is not None.
        """
        result = await self.exec(user_id, _store_script(name),
                                 stdin_data=data,
                                 timeout=_STORE_FILE_TIMEOUT)
        if result["timeout"]:
            return None, (f"Error storing the file in the sandbox: timed "
                          f"out after {_STORE_FILE_TIMEOUT:g} seconds.")
        if result["returncode"] != 0:
            detail = (result["stderr"].strip()[:_ERROR_CHARS]
                      or f"exit code {result['returncode']}")
            return None, f"Error storing the file in the sandbox: {detail}"
        path = result["stdout"].strip()
        if not path:
            return None, ("Error storing the file in the sandbox: the "
                          "container did not report the stored path.")
        return path, None

    async def destroy(self, user_id: int) -> bool:
        """Force-remove the user's sandbox, losing all changes.

        The user's read-before-write marks are cleared too - but only when
        the removal actually succeeded: if `docker rm -f` fails the files
        still exist and the marks remain valid. Returns True if a container
        was removed.
        """
        async with self._with_container(user_id):
            return await self._destroy_locked(user_id)

    async def _destroy_locked(self, user_id: int) -> bool:
        """destroy()'s body; the caller holds the user's sandbox lock."""
        self._sandboxes.pop(user_id, None)
        rc, _, _ = await self._docker(
            ["rm", "-f", self.container_name_for(user_id)])
        if rc == 0:
            logger.info("Destroyed sandbox for user %d", user_id)
            self._clear_read_marks(user_id)
        return rc == 0

    async def _stop_sandbox(self, user_id: int, sandbox: Sandbox) -> bool:
        """Stop a sandbox container, keeping its files for a later restart.

        Returns True if the container was stopped. ``running`` tracks the
        real state: a failed stop leaves it True, so reap_idle() retries
        on the next pass.
        """
        async with self._with_container(user_id):
            return await self._stop_locked(user_id, sandbox)

    async def _stop_locked(self, user_id: int, sandbox: Sandbox) -> bool:
        """_stop_sandbox()'s body; the caller holds the sandbox lock."""
        rc, _, _ = await self._docker(["stop", sandbox.container_name])
        if rc == 0:
            sandbox.running = False
        return rc == 0

    async def reap_idle(self) -> tuple[list[int], list[int]]:
        """Stop sandboxes idle past SANDBOX_IDLE_STOP_TIMEOUT and destroy
        sandboxes idle past SANDBOX_IDLE_DESTROY_TIMEOUT.

        A stopped sandbox keeps its files until it is destroyed and is
        restarted on the next access. A sandbox whose stop fails stays
        tracked and running and is retried on the next pass. Returns
        (stopped, destroyed) user ids.

        The idle check is repeated under the user's lock: exec() holds
        that lock for up to a command's timeout plus grace, so the reaper
        can queue behind a long command - and a command that finished
        while it waited has refreshed last_activity. Acting on the
        pre-lock timestamp would stop or destroy a sandbox that was just
        used.
        """
        now = time.time()
        stopped: list[int] = []
        destroyed: list[int] = []
        for user_id, sandbox in list(self._sandboxes.items()):
            if now - sandbox.last_activity < config.SANDBOX_IDLE_STOP_TIMEOUT:
                continue
            async with self._with_container(user_id):
                idle = time.time() - sandbox.last_activity
                if idle >= config.SANDBOX_IDLE_DESTROY_TIMEOUT:
                    if await self._destroy_locked(user_id):
                        destroyed.append(user_id)
                elif (idle >= config.SANDBOX_IDLE_STOP_TIMEOUT
                        and sandbox.running):
                    if await self._stop_locked(user_id, sandbox):
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
