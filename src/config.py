"""Environment-driven configuration.

Everything is read once at import, from the process environment with the
project-root .env loaded first. Numeric reads go through the _*_env helpers
so a bad value fails with the variable named instead of a bare ValueError
from the middle of a cast.
"""

import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _int_env(name: str, default: str) -> int:
    """Read an integer from the environment, naming the variable on error."""
    raw = os.environ.get(name, default)
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _float_env(name: str, default: str) -> float:
    """Read a float from the environment, naming the variable on error."""
    raw = os.environ.get(name, default)
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


def _int_list_env(name: str, default: str) -> list[int]:
    """Read a comma-separated integer list, naming the variable on error."""
    raw = os.environ.get(name, default)
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            raise ValueError(
                f"{name} must be a comma-separated list of integers, "
                f"got {raw!r}") from None
    return values


def _find_env_file() -> str | None:
    """The project root's .env, or None when there is none.

    Only the project root (the parent of src/) is checked: dotenv's own
    discovery starts inside src/ and would silently prefer a stale src/.env
    over the .env that `pake publish_env` deploys to the project root, and
    walking further up could load an unrelated .env from an ancestor
    directory.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(root, ".env")
    return candidate if os.path.isfile(candidate) else None


_env_file = _find_env_file()
if _env_file is not None:
    load_dotenv(_env_file)

_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

API_TOKEN = os.environ.get("TELEGRAM_API_TOKEN", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
# NVIDIA-hosted image generation (IMAGE_GEN_URL / NVIDIA_FLUX2_GEN_URL)
# falls back to the main key so the README's single-key deployment keeps
# working. The fallback is announced, not silent: OPENAI_API_KEY only
# works against api.nvidia.com when it actually holds an nvapi-... key,
# so set NVIDIA_API_KEY explicitly whenever the two keys differ.
NVIDIA_API_KEY = os.environ.get(
    "NVIDIA_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
if not os.environ.get("NVIDIA_API_KEY") and NVIDIA_API_KEY:
    logger.warning(
        "NVIDIA_API_KEY is not set; falling back to OPENAI_API_KEY for "
        "NVIDIA-hosted image generation. Set NVIDIA_API_KEY if those "
        "keys differ.")
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_IMAGE_MODEL = os.environ.get(
    "CLOUDFLARE_IMAGE_MODEL", "@cf/leonardo/phoenix-1.0")
# OpenRouter image generation backend (provider "openrouter").
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_IMAGE_MODEL = os.environ.get(
    "OPENROUTER_IMAGE_MODEL", "recraft/recraft-v4.1-flash-v2:free")
LLM_MODEL = os.environ.get("LLM_MODEL", "auto:smart")
# Optional operator override for the model used by the `subagent` tool.
# Unset means use the user's current model (their /model setting).
SUBAGENT_MODEL = os.environ.get("SUBAGENT_MODEL") or None
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://192.168.0.2:3001/v1")
IMAGE_GEN_URL = os.environ.get("IMAGE_GEN_URL", f"https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux-3-image-klein")
NVIDIA_FLUX2_GEN_URL = os.environ.get(
    "NVIDIA_FLUX2_GEN_URL",
    "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.2-klein-4b")

# Sampling parameters and the LLM retry budget, in one convention.
TEMPERATURE = _float_env("TEMPERATURE", "1.0")
TOP_P = _float_env("TOP_P", "0.95")
API_RETRY_TIME = _int_env("API_RETRY_TIME", "36000")
# The longest single backoff sleep; the 1.2x growth is clamped to this so
# one sleep cannot swallow hours of the retry budget.
API_RETRY_MAX_DELAY = _float_env("API_RETRY_MAX_DELAY", "300")
# Legacy lowercase names, still imported elsewhere (user_settings,
# handlers.run_llm); they are the same values, just spelled the old way.
temperature = TEMPERATURE
top_p = TOP_P
api_retry_time = API_RETRY_TIME
api_retry_max_delay = API_RETRY_MAX_DELAY

STREAM_DRAFTS = os.environ.get("STREAM_DRAFTS", "1").strip().lower() not in ("0", "false", "off")

# Models that inline chain-of-thought in the content field ending in a
# think close tag; the final message's thinking is split out only for
# these models, so well-behaved models can output the tag freely.
THINK_SPLIT_MODELS = [
    m.strip() for m in
    os.environ.get("THINK_SPLIT_MODELS", "big-pickle").split(",")
    if m.strip()
]

ALLOWED_IDS_STR = os.environ.get("ALLOWED_IDS", "")

ALLOWED_IDS = _int_list_env("ALLOWED_IDS", "")

# SQLite database shared by the agent's memory, user settings, and journal.
# STORE_DB_PATH is the env var that feeds it (see tests/test_config.py);
# there is deliberately no STORE_DB_PATH module attribute: a one-time copy
# would not track this value, and the stores read sql_db_path at call time.
sql_db_path = os.environ.get(
    "STORE_DB_PATH", os.path.join(os.path.dirname(os.path.realpath(__file__)), '../store.db'))

SANDBOX_IMAGE = os.environ.get("SANDBOX_IMAGE", "elpechos/toolbot-sandbox:latest")
# Seconds of inactivity after which a sandbox is stopped. Stopping keeps the
# container and its files; it is restarted on the next access.
SANDBOX_IDLE_STOP_TIMEOUT = _float_env("SANDBOX_IDLE_STOP_TIMEOUT", "3600")
# Seconds since the last access after which a sandbox is removed, losing all
# changes. Measured from the last access, so it also covers the stopped period.
SANDBOX_IDLE_DESTROY_TIMEOUT = _float_env("SANDBOX_IDLE_DESTROY_TIMEOUT", "86400")
SANDBOX_EXEC_TIMEOUT = _float_env("SANDBOX_EXEC_TIMEOUT", "300")
SANDBOX_START_TIMEOUT = _float_env("SANDBOX_START_TIMEOUT", "300")
SANDBOX_REAP_INTERVAL = _float_env("SANDBOX_REAP_INTERVAL", "60")
# Docker size/limit strings ("1g", "8.0"): passed to docker run verbatim.
SANDBOX_MEMORY = os.environ.get("SANDBOX_MEMORY", "1g")
SANDBOX_CPUS = os.environ.get("SANDBOX_CPUS", "8.0")
SANDBOX_PIDS_LIMIT = _int_env("SANDBOX_PIDS_LIMIT", "256")

# Host directory mounted read-only into each sandbox at SANDBOX_SKILLS_DIR.
SKILLS_DIR = os.environ.get(
    "SKILLS_DIR", os.path.join(_SOURCE_DIR, "data", "skills"))
# Container path the skills directory is mounted at (see sandbox.py).
SANDBOX_SKILLS_DIR = os.environ.get("SANDBOX_SKILLS_DIR", "/skills")
