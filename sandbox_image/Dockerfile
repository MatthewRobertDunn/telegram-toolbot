FROM python:3.13-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ripgrep ca-certificates wget file && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir apply-patch-py numpy pillow

COPY helpers/ /opt/toolbot/

RUN mkdir -p /workspace

WORKDIR /workspace
