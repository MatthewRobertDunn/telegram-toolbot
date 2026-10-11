import os
from dotenv import load_dotenv
load_dotenv()
API_TOKEN = os.environ.get("TELEGRAM_API_TOKEN", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
LLM_MODEL = os.environ.get("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
STRONG_LLM_MODEL = os.environ.get("STRONG_LLM_MODEL", "z-ai/glm-5.3-flash")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
IMAGE_GEN_URL = os.environ.get("IMAGE_GEN_URL", f"https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.2-klein-4b")
temperature = 1.0
top_p = 0.95
api_retry_time = 60

ALLOWED_IDS_STR = os.environ.get("ALLOWED_IDS", "")

ALLOWED_IDS = [int(x.strip()) for x in ALLOWED_IDS_STR.split(",") if x.strip()]

store_db_path = os.environ.get("STORE_DB_PATH", os.path.join(os.path.dirname(os.path.realpath(__file__)), '../store.db'))

SANDBOX_IMAGE = os.environ.get("SANDBOX_IMAGE", "elpechos/toolbot-sandbox:latest")
SANDBOX_IDLE_TIMEOUT = float(os.environ.get("SANDBOX_IDLE_TIMEOUT", "3600"))
SANDBOX_EXEC_TIMEOUT = float(os.environ.get("SANDBOX_EXEC_TIMEOUT", "300"))
SANDBOX_START_TIMEOUT = float(os.environ.get("SANDBOX_START_TIMEOUT", "300"))
SANDBOX_REAP_INTERVAL = float(os.environ.get("SANDBOX_REAP_INTERVAL", "60"))
SANDBOX_MEMORY = os.environ.get("SANDBOX_MEMORY", "1g")
SANDBOX_CPUS = os.environ.get("SANDBOX_CPUS", "1.0")
SANDBOX_PIDS_LIMIT = int(os.environ.get("SANDBOX_PIDS_LIMIT", "256"))


