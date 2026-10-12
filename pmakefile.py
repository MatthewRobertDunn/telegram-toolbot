import pake

# Tasks are registered the the pake.Pake object
# returned by pake's initialization call, using the task decorator.

pk = pake.init()

# Image name used by the sandbox image build tasks and default platforms.
SANDBOX_IMAGE = "elpechos/toolbot-sandbox:latest"
SANDBOX_PLATFORMS = "linux/amd64,linux/arm64"


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


pake.run(pk)
