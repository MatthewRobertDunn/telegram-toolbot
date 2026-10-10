"""Build (and optionally push) the toolbot sandbox image to Docker Hub.

A plain build targets the local platform only. With --push, a multi-arch
image (amd64 + arm64 by default) is built with buildx and pushed directly.
"""
import argparse
import os
import subprocess
import sys

IMAGE = "elpechos/toolbot-sandbox:latest"
PLATFORMS = "linux/amd64,linux/arm64"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true",
                        help="build the multi-arch image and push it to "
                             "Docker Hub")
    parser.add_argument("--platform", default=PLATFORMS,
                        help="comma-separated platforms used when pushing "
                             f"(default: {PLATFORMS})")
    args = parser.parse_args()

    context_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "sandbox_image")
    if args.push:
        build = subprocess.run([
            "docker", "buildx", "build", "--platform", args.platform,
            "-t", IMAGE, "--push", context_dir,
        ])
        if build.returncode != 0:
            return build.returncode
        print(f"Pushed multi-arch image {IMAGE} ({args.platform})")
        return 0

    build = subprocess.run(["docker", "build", "-t", IMAGE, context_dir])
    if build.returncode != 0:
        return build.returncode
    print(f"Built {IMAGE}")
    print("To publish, run:")
    print(f"  python {os.path.basename(sys.argv[0])} --push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
