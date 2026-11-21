FROM python:3.13-slim

# The bash tool advertises git and npm (tools/sandbox/shell.py). They and
# the other apt packages install at whatever the pinned base image's Debian
# release (currently trixie) ships, so apt revisions track that release's
# security updates instead of freezing on stale ones.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates file ffmpeg git npm ripgrep wget && \
    rm -rf /var/lib/apt/lists/*

# Pinned: apply-patch-py's output is part of the patch tool's contract, and
# the scientific stack gets compatible-release ranges so builds stay
# reproducible without freezing out wheel updates.
RUN pip install --no-cache-dir \
    "apply-patch-py==0.4.1" "numpy~=2.5" "pillow~=13.0"

COPY helpers/ /opt/toolbot/

WORKDIR /workspace
