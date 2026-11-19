import os
from dotenv import load_dotenv

# Load the project-root .env explicitly: dotenv's default discovery starts
# from this file's directory (src/) and would silently prefer a stale
# src/.env over the .env that `pake publish_env` deploys to the project root.
_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_FILE = os.path.join(os.path.dirname(_SOURCE_DIR), ".env")
load_dotenv(_ENV_FILE if os.path.isfile(_ENV_FILE) else None)

API_TOKEN = os.environ.get("TELEGRAM_API_TOKEN", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
# AI Horde image generation backend. Accepts the documented AI_HORDE_KEY as
# well as the AI_HOARD_KEY spelling already present in some .env files.
AI_HORDE_KEY = os.environ.get("AI_HORDE_KEY", os.environ.get("AI_HOARD_KEY", ""))
AI_HORDE_API_URL = os.environ.get("AI_HORDE_API_URL", "https://aihorde.net/api/v2")
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_IMAGE_MODEL = os.environ.get(
    "CLOUDFLARE_IMAGE_MODEL", "@cf/leonardo/phoenix-1.0")
LLM_MODEL = os.environ.get("LLM_MODEL", "auto:smart")
# Optional operator override for the model used by the `subagent` tool.
# Unset means use the user's current model (their /model setting).
SUBAGENT_MODEL = os.environ.get("SUBAGENT_MODEL") or None
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://192.168.0.2:3001/v1")
IMAGE_GEN_URL = os.environ.get("IMAGE_GEN_URL", f"https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux-3-image-klein")
NVIDIA_FLUX2_GEN_URL = os.environ.get(
    "NVIDIA_FLUX2_GEN_URL",
    "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.2-klein-4b")
temperature = 1.0
top_p = 0.95
api_retry_time = 36000

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

ALLOWED_IDS = [int(x.strip()) for x in ALLOWED_IDS_STR.split(",") if x.strip()]

# SQLite database shared by the agent's memory, user settings, and journal.
sql_db_path = os.environ.get(
    "STORE_DB_PATH", os.path.join(os.path.dirname(os.path.realpath(__file__)), '../store.db'))

SANDBOX_IMAGE = os.environ.get("SANDBOX_IMAGE", "elpechos/toolbot-sandbox:latest")
# Seconds of inactivity after which a sandbox is stopped. Stopping keeps the
# container and its files; it is restarted on the next access.
SANDBOX_IDLE_STOP_TIMEOUT = float(os.environ.get("SANDBOX_IDLE_STOP_TIMEOUT", "3600"))
# Seconds since the last access after which a sandbox is removed, losing all
# changes. Measured from the last access, so it also covers the stopped period.
SANDBOX_IDLE_DESTROY_TIMEOUT = float(os.environ.get("SANDBOX_IDLE_DESTROY_TIMEOUT", "86400"))
SANDBOX_EXEC_TIMEOUT = float(os.environ.get("SANDBOX_EXEC_TIMEOUT", "300"))
SANDBOX_START_TIMEOUT = float(os.environ.get("SANDBOX_START_TIMEOUT", "300"))
SANDBOX_REAP_INTERVAL = float(os.environ.get("SANDBOX_REAP_INTERVAL", "60"))
SANDBOX_MEMORY = os.environ.get("SANDBOX_MEMORY", "1g")
SANDBOX_CPUS = os.environ.get("SANDBOX_CPUS", "8.0")
SANDBOX_PIDS_LIMIT = int(os.environ.get("SANDBOX_PIDS_LIMIT", "256"))

# Host directory mounted read-only into each sandbox at SANDBOX_SKILLS_DIR.
SKILLS_DIR = os.environ.get(
    "SKILLS_DIR", os.path.join(_SOURCE_DIR, "data", "skills"))
# Container path the skills directory is mounted at (see sandbox.py).
SANDBOX_SKILLS_DIR = os.environ.get("SANDBOX_SKILLS_DIR", "/skills")


