# Telegram AI Chatbot

A Telegram chatbot that connects to any OpenAI-compatible API, with tool calling, per-user sandbox environments, user preferences, and multi-modal support.

This project is based on [openai-telegram-chatbot](https://github.com/ogfirstofhisname/openai-telegram-chatbot).

Tested with Python 3.14 on Windows and Linux. Suitable for running on a Raspberry Pi, cloud server, or any machine that can stay online.

If you find this project useful, please [buy me a coffee](https://ko-fi.com/siliconjunction)

Check out my electronics blog [Silicon Junction](https://siliconjunction.top/)

### Recommended provider: NVIDIA NIM

This bot works with any OpenAI-compatible API. We recommend **NVIDIA NIM** (NVIDIA Inference Microservices), which provides high-performance serverless inference with a generous free tier — no GPU hardware required. 

The default configuration points `OPENAI_BASE_URL` at a private LAN endpoint (`http://192.168.0.2:3001/v1`), so for NVIDIA NIM you must set `OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1` in your `.env` (see the setup steps below). You can point it at any OpenAI-compatible provider (Ollama, vLLM, Groq, etc.) the same way.

---

## New features (vs. the original)

- **User preferences** — each user can set their own model, temperature, top_p, and system prompt via slash commands
- **Rich command set** — 30+ slash commands for controlling model, parameters, timezone, profile, and persistent storage
- **Tool calling** — the bot can search the web, run code, fetch pages, generate images, and more using OpenAI-compatible function calling
- **Subagents** — the primary model can delegate research and multi-step tasks with `subagent_start`, then follow along with `subagent_list` / `subagent_steer` / `subagent_stop`, keeping bulky tool output and intermediate work out of the main conversation
- **Agent memory** — both users (via `/memory` commands) and the AI (via memory tools) can remember and recall facts across conversations
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
OPENAI_API_KEY=nvapi-...
```

Then install dependencies and activate the virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or: venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

See `.env.example` for every variable the bot reads.

### 2. Choose your provider

**NVIDIA NIM** (recommended): Get a free API key at [build.nvidia.com](https://build.nvidia.com) and set it in your `.env`:

```
OPENAI_API_KEY=nvapi-...
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
```

**Other providers**: Point `OPENAI_BASE_URL` at your provider's endpoint. For example, with Ollama:

```
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
LLM_MODEL=llama3
```

### 3. Get your Telegram bot token

Follow [this guide](https://core.telegram.org/bots/features#botfather) to create a bot, then add the token to your `.env`:

```
TELEGRAM_API_TOKEN=your-bot-token
```

### 4. Configure access control (recommended)

To restrict who can use your bot, add `ALLOWED_IDS` to your `.env`:

```
ALLOWED_IDS=123456789,987654321
```

⚠️ **This variable is the allowlist, and a misspelled one fails open.** If
the name is wrong (e.g. `ALLOWED_USER_IDS`) or the list is empty, the bot
does not error — it simply accepts messages from anyone (see the check in
`src/handlers/middleware.py`).

### 5. Docker (required for the sandbox and Python tools)

[Docker](https://docs.docker.com/get-docker/) is required for the per-user sandboxes (and the `python` tool).

The sandbox uses the custom image [`elpechos/toolbot-sandbox:latest`](https://hub.docker.com/r/elpechos/toolbot-sandbox) (Debian slim with Python 3.13, ripgrep, `apply-patch-py`, numpy and pillow preinstalled). It is pulled automatically from Docker Hub on first use, so no extra setup is needed.

To build and publish the image yourself (e.g. after modifying `sandbox_image/`):

```bash
pake build_sandbox_image             # build for the local platform
pake push_sandbox_image              # build + push multi-arch (amd64 + arm64)
```

If you don't need the sandbox, you can disable its tools by removing the `tools.sandbox.*` modules from `TOOL_MODULES` in `src/tool_registration.py`.

### Configuration reference

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_API_TOKEN` | *(required)* | Your Telegram bot token |
| `OPENAI_API_KEY` | *(required)* | API key for the OpenAI-compatible chat completions endpoint |
| `OPENAI_BASE_URL` | `http://192.168.0.2:3001/v1` | OpenAI-compatible API endpoint. For NVIDIA NIM use `https://integrate.api.nvidia.com/v1` |
| `LLM_MODEL` | `auto:smart` | Default chat model |
| `SUBAGENT_MODEL` | *(falls back to the current `/model` setting)* | Optional override for the model used by the `subagent_start` tool (a call that passes its own `model` wins) |
| `STREAM_DRAFTS` | `1` | Stream partial replies into the chat as they are generated (`0`/`false`/`off` to disable) |
| `THINK_SPLIT_MODELS` | `big-pickle` | Comma-separated model IDs whose inline chain-of-thought is split out of the reply text |
| `TEMPERATURE` | `1.0` | Sampling temperature for chat completions |
| `TOP_P` | `0.95` | Nucleus sampling probability for chat completions |
| `API_RETRY_TIME` | `36000` | Wall-clock seconds to keep retrying a failed completion (the router may reroute to another model); `0` disables retries |
| `API_RETRY_MAX_DELAY` | `300` | Cap, in seconds, on a single retry backoff sleep |
| `ALLOWED_IDS` | *(none — open to all)* | ⚠️ Comma-separated Telegram user IDs allowed to use the bot; a misspelled/empty value fails open |
| `STORE_DB_PATH` | `store.db` in the project root | SQLite database shared by memory, user settings and journal |
| `NVIDIA_API_KEY` | *(falls back to `OPENAI_API_KEY`)* | Key for NVIDIA Flux image generation |
| `IMAGE_GEN_URL` | `https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux-3-image-klein` | Endpoint for the default `nvidia-flux3` image provider |
| `NVIDIA_FLUX2_GEN_URL` | `https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.2-klein-4b` | NVIDIA FLUX.2-Klein endpoint (`provider: "nvidia-flux2"`) |
| `AI_HORDE_KEY` | *(none — `horde` provider disabled)* | Key for AI Horde image generation (`provider: "horde"`); the `AI_HOARD_KEY` spelling is accepted as an alias. Anonymous key `0000000000` works with limits |
| `AI_HORDE_API_URL` | `https://aihorde.net/api/v2` | AI Horde API endpoint |
| `CLOUDFLARE_ACCOUNT_ID` | *(none — `cloudflare` provider disabled)* | Cloudflare account ID for Workers AI image generation (`provider: "cloudflare"`) |
| `CLOUDFLARE_API_TOKEN` | *(none — `cloudflare` provider disabled)* | Cloudflare API token for Workers AI image generation |
| `CLOUDFLARE_IMAGE_MODEL` | `@cf/leonardo/phoenix-1.0` | Workers AI image model for the `cloudflare` provider |
| `SANDBOX_IMAGE` | `elpechos/toolbot-sandbox:latest` | Docker image used for sandboxes |
| `SANDBOX_IDLE_STOP_TIMEOUT` | `3600` | Seconds of inactivity before a sandbox is stopped (its files are kept and it is restarted on the next access) |
| `SANDBOX_IDLE_DESTROY_TIMEOUT` | `86400` | Seconds since the last access before a sandbox is removed (loses all changes) |
| `SANDBOX_EXEC_TIMEOUT` | `300` | Max seconds per sandbox command |
| `SANDBOX_START_TIMEOUT` | `300` | Max seconds to start a sandbox (includes image pull) |
| `SANDBOX_REAP_INTERVAL` | `60` | How often idle sandboxes are checked, stopped and destroyed |
| `SANDBOX_MEMORY` | `1g` | Memory limit per sandbox container |
| `SANDBOX_CPUS` | `8.0` | CPU limit per sandbox container |
| `SANDBOX_PIDS_LIMIT` | `256` | Process limit per sandbox container |
| `SKILLS_DIR` | `src/data/skills` | Host directory scanned for skill definitions, mounted read-only into sandboxes |
| `SANDBOX_SKILLS_DIR` | `/skills` | Path the skills directory is mounted at inside sandboxes |

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
python tests/run.py -v

# All integration tests (requires OPENAI_API_KEY in integration_tests/.env, Docker, and the sandbox image)
python tests/run.py integration_tests -v
```

`pake unittest` / `pake integration_test` run the same commands. Both
tiers run against an isolated per-run SQLite database: the runner
(`tests/run.py`) exports `STORE_DB_PATH` to a temporary directory before
any application module is imported, so the repo's `store.db` is never
read or created by a test run. A guard also fails immediately if a test
tries to open the real database, and the runner self-checks the
isolation before discovery. Plain `python -m unittest discover -s
tests -t .` works too — the `-t .` makes unittest import the `tests`
package, whose `__init__.py` applies the same isolation (without
`-t .` the modules load top-level, the package never runs, and the real
database gets touched).

## Commands

| Command | Description |
|---|---|
| `/start` | Start a new conversation |
| `/stop` | Stop the current AI response |
| `/regenerate` | Regenerate the last response |
| `/retry` | Retry the conversation |
| `/continue` | Alias of `/retry` |
| `/undo` | Undo the last message |
| `/impersonate` | Impersonate your next message |
| `/compact` | Compact the conversation context |
| `/window_new` | Open a new thread in this chat |
| `/systemprompt <prompt>` | Set a custom system prompt |
| `/systemprompt_reset` | Reset system prompt to default |
| `/model <model_id>` | Set the model to use |
| `/model_list [filter]` | List available models (click any to select) |
| `/model_reset` | Reset model to default |
| `/subagent_model <model_id>` | Set the model used by the `subagent_start` tool |
| `/subagent_model_reset` | Reset subagent model to default |
| `/subagent_list` | List running background subagents |
| `/temperature <value>` | Set temperature (0.0–2.0) |
| `/temperature_reset` | Reset temperature to default |
| `/top_p <value>` | Set top_p (0.0–1.0) |
| `/top_p_reset` | Reset top_p to default |
| `/timezone <zone>` | Set your timezone (e.g. Australia/Sydney) |
| `/thinking_keep` | Toggle thinking output on/off |
| `/tool_output` | Toggle visibility of tool calls in chat |
| `/user_profile <text>` | Set a personal profile the AI can reference |
| `/user_profile_reset` | Clear your profile |
| `/memory <key> [value]` | Recall or remember a memory |
| `/memory_list` | List everything I remember about you |
| `/memory_delete <key>` | Forget a remembered key |
| `/memory_delete_all` | Forget everything I remember |
| `/settings_reset` | Reset all settings to defaults |
| `/sandbox_reset` | Destroy your sandbox and all its files |

---

## Tools available to the AI

The bot uses OpenAI-compatible function calling. These tools are available to the model:

| Tool | Description |
|---|---|
| `web_search` | Search the web via DuckDuckGo (no API key needed) |
| `web_fetch` | Fetch and extract content from a URL |
| `get_current_datetime` | Get the current date and time in the user's timezone (`/timezone`, default UTC), as ISO 8601 with UTC offset and weekday |
| `python` | Run Python code in the sandbox (working dir `/workspace`, network enabled) |
| `bash` | Run shell commands in the user's sandbox (network + root, software installs) |
| `read` | Read files or directories in the sandbox (line numbers, offset/limit, binary detection) |
| `write` | Write files in the sandbox (creates parent directories) |
| `edit` | Exact string replacement in sandbox files (with fallback matching for whitespace drift) |
| `patch` | Apply Codex-style patch blocks (add/update/delete/move files) in the sandbox |
| `grep` | Regex content search in the sandbox, powered by ripgrep |
| `glob` | Find sandbox files by glob pattern, sorted by modification time |
| `read_image` | Read an image from the sandbox and attach it for the model to see (needs vision); skips re-sending files the user just uploaded |
| `send_file` | Send a file from the sandbox to the user as a Telegram document |
| `generate_image` | Generate an image (default `provider: "nvidia-flux3"`; also `"nvidia-flux2"`, `"cloudflare"`, `"horde"`) |
| `memory_list` / `memory_get` / `memory_set` / `memory_delete` | The agent's persistent per-user memory |
| `journal_add` / `journal_get` / `journal_edit` / `journal_delete` / `journal_topic_edit` / `journal_query` / `journal_list_topics` | The agent's persistent per-user journal of longer-term notes, searched by text and topic |
| `subagent_start` | Spawn a subagent in a fresh conversation to work on a task (with the same tools and sandbox) and report back. Takes an optional `model` to run that subagent on a different model than the default |
| `subagent_list` / `subagent_steer` / `subagent_stop` | List running subagents, message them mid-run, or stop them |

Subagents get a restricted tool surface. Tools that write to the chat (`send_file` and the `subagent_start`/`subagent_list`/`subagent_stop`/`subagent_steer` controls) are never offered to them; `subagent_send_to_parent` is their channel back. `read_image` and `generate_image` keep the same name for both agents but swap implementation: the subagent's variant reads or generates without posting and reports the file path for the parent to share. Each subagent runs against its own isolated context copy with no bot, inbox, or generation queue it could touch. Tools declare themselves with `@agent_tool` (see `src/tool_registry.py`), which carries the schema, derives `required` from the signature, and defaults to parent-only.

---

## Security & privacy

Messages are held as plaintext in the server's memory. There is no end-to-end encryption.

Anyone who knows your bot's username can send messages to it, potentially incurring API costs. Use `ALLOWED_IDS` to restrict access (and remember: a misspelled variable name fails open), and ensure your API key has a spending limit. Avoid using this project with sensitive information or in production environments.

The sandbox gives the AI root access inside an ephemeral, resource-limited Docker container (network enabled so it can install software). Sandbox state — including uploaded files — is destroyed when the sandbox expires or when you use `/sandbox_reset`, but while it runs the AI can execute arbitrary commands on it, so treat the sandbox and anything you upload to it as non-sensitive.