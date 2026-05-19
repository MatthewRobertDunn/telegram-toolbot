import os
from dotenv import load_dotenv
load_dotenv()
API_TOKEN = os.environ.get("TELEGRAM_API_TOKEN", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
LLM_MODEL = os.environ.get("LLM_MODEL", "minimaxai/minimax-m2.7")
STRONG_LLM_MODEL = os.environ.get("STRONG_LLM_MODEL", "deepseek-ai/deepseek-v4-pro")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
IMAGE_GEN_URL = os.environ.get("IMAGE_GEN_URL", f"https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.2-klein-4b")
temperature = 1.0
top_p = 0.95
api_retry_time = 60

MAX_TOOL_ROUNDS = 40
ALLOWED_IDS_STR = os.environ.get("ALLOWED_IDS", "")
ALLOWED_IDS = [int(x.strip()) for x in ALLOWED_IDS_STR.split(",") if x.strip()]

store_db_path = os.environ.get("STORE_DB_PATH", os.path.join(os.path.dirname(os.path.realpath(__file__)), '../store.db'))


