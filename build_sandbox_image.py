"""Build (and optionally push) the toolbot sandbox image to Docker Hub."""
import argparse
import os
import subprocess
import sys

IMAGE = "elpechos/toolbot-sandbox:latest"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true",
                        help="push the image to Docker Hub after building")
    args = parser.parse_args()

    context_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "sandbox_image")
    build = subprocess.run(["docker", "build", "-t", IMAGE, context_dir])
    if build.returncode != 0:
        return build.returncode
    print(f"Built {IMAGE}")
    if args.push:
        push = subprocess.run(["docker", "push", IMAGE])
        return push.returncode
    print("To publish, run:")
    print("  docker login")
    print(f"  docker push {IMAGE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
