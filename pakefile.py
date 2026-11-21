import os
import tempfile
import time
import pake

# Tasks are registered the the pake.Pake object
# returned by pake's initialization call, using the task decorator.
pk = pake.init()

# Image name used by the sandbox image build tasks and default platforms.
SANDBOX_IMAGE = "elpechos/toolbot-sandbox:latest"
SANDBOX_PLATFORMS = "linux/amd64,linux/arm64"
DEPLOY_HOST = "matty@derg.top"
REMOTE_DIR = "/home/matty/tbot"
ROOT = os.path.dirname(os.path.abspath(__file__))

@pk.task
def build_sandbox_image(ctx):
    """Build the toolbot sandbox image for the local platform only."""

    ctx.call("docker", "build", "-t", SANDBOX_IMAGE, "sandbox_image")

    print(f"Built {SANDBOX_IMAGE}")
    print("To publish, run:")
    print("  pake push_sandbox_image")


@pk.task
def push_sandbox_image(ctx):
    """Build and push a multi-arch sandbox image to Docker Hub."""

    platforms = pk.get_define("PLATFORMS", SANDBOX_PLATFORMS)

    ctx.call("docker", "buildx", "build", "--platform", platforms,
             "-t", SANDBOX_IMAGE, "--push", "sandbox_image")

    print(f"Pushed multi-arch image {SANDBOX_IMAGE} ({platforms})")


@pk.task
def typecheck(ctx):
    """Run pyright type checking over src/, tests/, and integration_tests/."""

    ctx.call("pyright", "src/", "tests/", "integration_tests/")


@pk.task
def unittest(ctx):
    """Run unit tests (pure logic, no network) against an isolated DB."""

    ctx.call("python", "tests/run.py")


@pk.task
def integration_test(ctx):
    """Run integration tests (requires OPENAI_API_KEY in integration_tests/.env)."""

    ctx.call("python", "tests/run.py", "integration_tests")


@pk.task
def publish(ctx):
    """Copy src/ to the server and restart the tbot user service."""

    tar_path = os.path.join(tempfile.gettempdir(), "tbot_src.tar")

    # Ship a tarball without __pycache__ or .env: pyc files on the server
    # may be root-owned (from the service) and scp cannot overwrite them
    # anyway, they are never worth deploying, and secrets travel via
    # pake publish_env instead.
    ctx.call("tar", "-C", ROOT,
             "--exclude", "__pycache__", "--exclude", ".env",
             "-cf", tar_path, "src")
    ctx.call("scp", tar_path, f"{DEPLOY_HOST}:/tmp/")
    # Replace the remote tree rather than extracting over it: a module
    # removed or renamed locally would otherwise linger on the server and
    # keep being imported. Nothing else lives in src, and no secret does:
    # the app reads the project-root .env (see config.py). The chmod undoes
    # read-only directory modes left by earlier archives (see
    # --no-same-permissions below); without write permission matty cannot
    # empty them.
    ctx.call("ssh", DEPLOY_HOST,
             f"chmod -R u+w {REMOTE_DIR}/src 2>/dev/null;"
             f" rm -rf {REMOTE_DIR}/src")
    # Ignore archived modes: bsdtar maps Windows attributes (e.g. OneDrive's
    # read-only flag on dirs) into the tarball, which once left the server
    # tree read-only and broke later deploys. Fresh dirs/files from umask.
    ctx.call("ssh", DEPLOY_HOST,
             f"tar --no-same-permissions -C {REMOTE_DIR} -xf /tmp/tbot_src.tar")
    ctx.call("ssh", DEPLOY_HOST, "rm -f /tmp/tbot_src.tar")


    ctx.call("ssh", DEPLOY_HOST, "systemctl restart tbot --user")
    time.sleep(2)
    # LC_ALL=C keeps output ASCII, since UTF-8 glyphs from systemctl
    # cannot be decoded by pake's Windows locale text streams.
    ctx.call("ssh", DEPLOY_HOST,
             "LC_ALL=C systemctl status tbot --user -l --no-pager")


@pk.task
def publish_env(ctx):
    """Copy the project-root .env to the server, if there is one."""

    env_path = os.path.join(ROOT, ".env")
    if not os.path.isfile(env_path):
        print("No .env at the project root; nothing to copy.")
        return

    # The server reads this one file (see config.py), so there is nowhere
    # else for it to go.
    remote_path = f"{REMOTE_DIR}/.env"

    ctx.call("scp", env_path, f"{DEPLOY_HOST}:{remote_path}")
    print(f"Copied {env_path} to {DEPLOY_HOST}:{remote_path}")
    print("Restart the tbot service (pake publish) to pick up the change.")


@pk.task
def publish_requirements(ctx):
    """Copy requirements.txt to the server and install it into the venv."""

    # The tbot service runs the venv rooted at REMOTE_DIR
    # (REMOTE_DIR/bin/python), so install into that environment.
    requirements = os.path.join(ROOT, "requirements.txt")
    remote_requirements = f"{REMOTE_DIR}/requirements.txt"

    ctx.call("scp", requirements, f"{DEPLOY_HOST}:{remote_requirements}")
    ctx.call("ssh", DEPLOY_HOST,
             f"{REMOTE_DIR}/bin/python -m pip install -r {remote_requirements}")

    print("Restart the tbot service (pake publish) to pick up the change.")


@pk.task
def update_sandbox_image(ctx):
    """Pull the latest sandbox image on the deployment server."""

    ctx.call("ssh", DEPLOY_HOST, f"docker pull {SANDBOX_IMAGE}")


@pk.task
def recreate_sandboxes(ctx):
    """Remove sandbox container(s) on the server so they are recreated.

    Docker cannot add a mount to an existing container, so sandboxes
    created before a `docker run` change (e.g. the /skills mount) must
    be removed; the next access recreates them with the current flags.
    WARNING: this loses all files in the sandbox /workspace.
    Deploy code first (`pake publish`), or combine the two:
      pake publish recreate_sandboxes
    Limit to one user with:
      pake -D USER_ID=123 recreate_sandboxes
    """

    user_id = pk.get_define("USER_ID", "")
    if user_id:
        try:
            uid = int(user_id)
        except ValueError:
            raise ValueError(
                f"USER_ID must be an integer, got {user_id!r}")
        ctx.call("ssh", DEPLOY_HOST, f"docker rm -f toolbot-sandbox-{uid}")
    else:
        ctx.call("ssh", DEPLOY_HOST,
                 "docker ps -aq --filter label=toolbot.sandbox"
                 " | xargs -r docker rm -f")
    print("Removed sandbox container(s); they are recreated on next use.")


pake.run(pk)
