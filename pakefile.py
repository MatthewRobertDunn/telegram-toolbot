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
    """Run unit tests (pure logic, no network)."""

    ctx.call("python", "-m", "unittest", "discover", "-s", "tests",
             "-p", "test_*.py", "-v")


@pk.task
def integration_test(ctx):
    """Run integration tests (requires OPENAI_API_KEY in integration_tests/.env)."""

    ctx.call("python", "-m", "unittest", "discover", "-s", "integration_tests",
             "-p", "test_*.py", "-v")


@pk.task
def deploy(ctx):
    """Copy src/ to the server and restart the tbot user service."""

    remote_dir = "/home/matty/tbot"
    tar_path = os.path.join(tempfile.gettempdir(), "tbot_src.tar")

    # Ship a tarball without __pycache__: pyc files on the server may be
    # root-owned (from the service) and scp cannot overwrite them anyway,
    # plus they are never worth deploying.
    ctx.call("tar", "-C", os.path.dirname(os.path.abspath(__file__)),
             "--exclude", "__pycache__", "-cf", tar_path, "src")
    ctx.call("scp", tar_path, f"{DEPLOY_HOST}:/tmp/")
    ctx.call("ssh", DEPLOY_HOST, f"tar -C {remote_dir} -xf /tmp/tbot_src.tar")
    ctx.call("ssh", DEPLOY_HOST, "rm -f /tmp/tbot_src.tar")


    ctx.call("ssh", DEPLOY_HOST, "systemctl restart tbot --user")
    time.sleep(2)
    # LC_ALL=C keeps output ASCII, since UTF-8 glyphs from systemctl
    # cannot be decoded by pake's Windows locale text streams.
    ctx.call("ssh", DEPLOY_HOST,
             "LC_ALL=C systemctl status tbot --user -l --no-pager")


@pk.task
def update_sandbox_image(ctx):
    """Pull the latest sandbox image on the deployment server."""

    ctx.call("ssh", DEPLOY_HOST, f"docker pull {SANDBOX_IMAGE}")


pake.run(pk)
