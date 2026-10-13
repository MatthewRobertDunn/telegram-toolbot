# Telegram AI Chatbot

A Telegram chatbot that connects to any OpenAI-compatible API, with tool calling, per-user sandbox environments, user preferences, and multi-modal support.

This project is based on [openai-telegram-chatbot](https://github.com/ogfirstofhisname/openai-telegram-chatbot).

Tested with Python 3.14 on Windows and Linux. Suitable for running on a Raspberry Pi, cloud server, or any machine that can stay online.

If you find this project useful, please [buy me a coffee](https://ko-fi.com/siliconjunction)

Check out my electronics blog [Silicon Junction](https://siliconjunction.top/)

### Recommended provider: NVIDIA NIM

This bot works with any OpenAI-compatible API. We recommend **NVIDIA NIM** (NVIDIA Inference Microservices), which provides high-performance serverless inference with a generous free tier — no GPU hardware required. 

The default configuration points at `https://integrate.api.nvidia.com/v1`, but you can switch to any provider (Ollama, vLLM, Groq, etc.) by changing the `OPENAI_BASE_URL` environment variable.

---

## New features (vs. the original)

- **User preferences** — each user can set their own model, temperature, top_p, and system prompt via slash commands
- **Rich command set** — 16+ slash commands for controlling model, parameters, profile, and persistent storage
- **Tool calling** — the bot can search the web, run code, fetch pages, generate images, and more using OpenAI-compatible function calling
- **Hierarchical LLM** — the primary model can delegate complex tasks to a stronger reasoning model via the `ask_strong_llm` tool
- **Persistent key-value store** — both users (via commands) and the AI (via tools) can store and retrieve data across conversations
- **Per-user sandboxes** — the AI gets a personal ephemeral Ubuntu container per user with a full coding environment (bash, read, write, edit, grep, glob, patch), where it can run commands, install software, and work with files. Sandboxes are stopped one hour after the last command but their files are kept for 24 hours, after which the sandbox is removed and all changes are lost
- **File uploads** — files users send to the bot (any file type) are placed in the sandbox at `/workspace/uploads/`, and the AI is notified so it can inspect them with the `read` tool

---

## Setup

You need an API key for an OpenAI-compatible provider and a Telegram bot token.

### 1. Set up the project

```bash
git clone <repo-url>
cd telegrambot
```

Create a `.env` file in the project root:

```
TELEGRAM_API_TOKEN=your-bot-token
OPENAI_KEY=nvapi-...
```

Then install dependencies and activate the virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or: venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Choose your provider

**NVIDIA NIM** (recommended): Get a free API key at [build.nvidia.com](https://build.nvidia.com) and set it in your `.env`:

```
OPENAI_KEY=nvapi-...
```

**Other providers**: Set `OPENAI_BASE_URL` to your provider's endpoint. For example, with Ollama:

```
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_KEY=ollama
LLM_MODEL=llama3
```

### 3. Get your Telegram bot token

Follow [this guide](https://core.telegram.org/bots/features#botfather) to create a bot, then add the token to your `.env`:

```
TELEGRAM_API_TOKEN=your-bot-token
```

### 4. Configure access control (recommended)

To restrict who can use your bot, add `ALLOWED_USER_IDS` to your `.env`:

```
ALLOWED_USER_IDS=123456789,987654321
```

Without this, anyone who discovers your bot's username can use it.

### 5. Docker (required for the sandbox and Python tools)

[Docker](https://docs.docker.com/get-docker/) is required for the per-user sandboxes (and the `python` tool).

The sandbox uses the custom image [`elpechos/toolbot-sandbox:latest`](https://hub.docker.com/r/elpechos/toolbot-sandbox) (Ubuntu with Python 3.13, ripgrep, and `apply-patch-py` preinstalled). It is pulled automatically from Docker Hub on first use, so no extra setup is needed.

To build and publish the image yourself (e.g. after modifying `sandbox_image/`):

```bash
python build_sandbox_image.py           # build only
python build_sandbox_image.py --push    # build and push to Docker Hub
```

If you don't need the sandbox, you can disable its tools by commenting out the `register_sandbox_tool` line in `src/tool_registration.py`.

### Configuration reference

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_API_TOKEN` | *(required)* | Your Telegram bot token |
| `OPENAI_KEY` | *(required)* | API key for chat completions |
| `OPENAI_BASE_URL` | `https://integrate.api.nvidia.com/v1` | OpenAI-compatible API endpoint |
| `LLM_MODEL` | `z-ai/glm-5.3-flash` | Default chat model |
| `STRONG_LLM_MODEL` | `z-ai/glm-5.3-flash` | Model used by `ask_strong_llm` |
| `NVIDIA_API_KEY` | *(falls back to `OPENAI_KEY`)* | Key for NVIDIA Flux image generation |
| `ALLOWED_USER_IDS` | *(none — open to all)* | Comma-separated Telegram user IDs |
| `SANDBOX_IMAGE` | `elpechos/toolbot-sandbox:latest` | Docker image used for sandboxes |
| `SANDBOX_IDLE_STOP_TIMEOUT` | `3600` | Seconds of inactivity before a sandbox is stopped (its files are kept and it is restarted on the next access) |
| `SANDBOX_IDLE_DESTROY_TIMEOUT` | `86400` | Seconds since the last access before a sandbox is removed (loses all changes) |
| `SANDBOX_EXEC_TIMEOUT` | `300` | Max seconds per sandbox command |
| `SANDBOX_START_TIMEOUT` | `300` | Max seconds to start a sandbox (includes image pull) |
| `SANDBOX_REAP_INTERVAL` | `60` | How often idle sandboxes are checked, stopped and destroyed |
| `SANDBOX_MEMORY` | `1g` | Memory limit per sandbox container |
| `SANDBOX_CPUS` | `1.0` | CPU limit per sandbox container |
| `SANDBOX_PIDS_LIMIT` | `256` | Process limit per sandbox container |

---

## Running the bot

```powershell
# Windows PowerShell
python src\chatbot.py
```

```bash
# Linux / macOS
python src/chatbot.py
```

The bot runs indefinitely. Use Ctrl+C to stop.

---

## Running tests

```powershell
# All unit tests (pure, no network required)
python -m unittest discover -s tests -p "test_*.py" -v

# All integration tests (requires OPENAI_API_KEY in integration_tests/.env, Docker, and the sandbox image)
python -m unittest discover -s integration_tests -p "test_*.py" -v
```

## Commands

| Command | Description |
|---|---|
| `/start` | Start a new conversation |
| `/stop` | Stop the AI's current response |
| `/systemprompt <prompt>` | Set a custom system prompt |
| `/systemprompt_reset` | Reset system prompt to default |
| `/model <model_id>` | Set the model to use |
| `/model_list [filter]` | List available models (click any to select) |
| `/model_reset` | Reset model to default |
| `/strong_model <model_id>` | Set the strong model used by `ask_strong_llm` |
| `/strong_model_reset` | Reset strong model to default |
| `/temperature <value>` | Set temperature (0.0–2.0) |
| `/temperature_reset` | Reset temperature to default |
| `/top_p <value>` | Set top_p (0.0–1.0) |
| `/top_p_reset` | Reset top_p to default |
| `/tool_output` | Toggle visibility of tool calls in chat |
| `/user_profile <text>` | Set a personal profile the AI can reference |
| `/user_profile_reset` | Clear your profile |
| `/store <key> [value]` | Get or set a stored value |
| `/store_list` | List all your stored keys |
| `/store_delete <key>` | Delete a stored key |
| `/store_delete_all` | Delete all stored keys |
| `/settings_reset` | Reset all settings to defaults |

---

## Tools available to the AI

The bot uses OpenAI-compatible function calling. These tools are available to the model:

| Tool | Description |
|---|---|
| `web_search` | Search the web via DuckDuckGo (no API key needed) |
| `web_fetch` | Fetch and extract content from a URL |
| `get_current_datetime` | Get the current UTC date and time |
| `python` | Run Python code in the sandbox (working dir `/workspace`, network enabled) |
| `bash` | Run shell commands in the user's sandbox (network + root, software installs) |
| `read` | Read files or directories in the sandbox (line numbers, offset/limit, binary detection) |
| `write` | Write files in the sandbox (creates parent directories) |
| `edit` | Exact string replacement in sandbox files (with fallback matching for whitespace drift) |
| `grep` | Regex content search in the sandbox, powered by ripgrep |
| `glob` | Find sandbox files by glob pattern, sorted by modification time |
| `patch` | Apply Codex-style patch blocks (add/update/delete/move files) in the sandbox |
| `send_file` | Send a file from the sandbox to the user as a Telegram document |
| `store_list` / `store_get` / `store_set` / `store_delete` | Persistent per-user key-value storage |
| `display_status` | Send a progress message to the chat |
| `ask_strong_llm` | Delegate a complex task to a more powerful reasoning model |
| `generate_image` | Generate an image using NVIDIA Flux |

---

## Security & privacy

Messages are held as plaintext in the server's memory. There is no end-to-end encryption.

Anyone who knows your bot's username can send messages to it, potentially incurring API costs. Use `ALLOWED_USER_IDS` to restrict access, and ensure your API key has a spending limit. Avoid using this project with sensitive information or in production environments.

The sandbox gives the AI root access inside an ephemeral, resource-limited Docker container (network enabled so it can install software). Sandbox state — including uploaded files — is destroyed when the sandbox expires, but while it runs the AI can execute arbitrary commands on it, so treat the sandbox and anything you upload to it as non-sensitive.