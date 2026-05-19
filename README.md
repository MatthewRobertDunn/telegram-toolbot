# Telegram AI Chatbot

A Telegram chatbot that connects to any OpenAI-compatible API, with tool calling, user preferences, and multi-modal support.

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
- **Multi-modal** — supports text messages, images (vision), and file attachments

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

### 5. Docker (optional, for Python code execution tool)

The `python` tool executes code in an isolated Docker container. Install [Docker](https://docs.docker.com/get-docker/) to use this feature. If you don't need it, you can disable the tool by commenting out the `register_python_runner` line in `src/tool_registration.py:14`.

### Configuration reference

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_API_TOKEN` | *(required)* | Your Telegram bot token |
| `OPENAI_KEY` | *(required)* | API key for chat completions |
| `OPENAI_BASE_URL` | `https://integrate.api.nvidia.com/v1` | OpenAI-compatible API endpoint |
| `LLM_MODEL` | `minimaxai/minimax-m2.7` | Default chat model |
| `STRONG_LLM_MODEL` | `deepseek-ai/deepseek-v4-pro` | Model used by `ask_strong_llm` |
| `NVIDIA_API_KEY` | *(falls back to `OPENAI_KEY`)* | Key for NVIDIA Flux image generation |
| `ALLOWED_USER_IDS` | *(none — open to all)* | Comma-separated Telegram user IDs |

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

## Commands

| Command | Description |
|---|---|
| `/start` | Start a new conversation |
| `/systemprompt <prompt>` | Set a custom system prompt |
| `/systemprompt_reset` | Reset system prompt to default |
| `/model <model_id>` | Set the model to use |
| `/model_list [filter]` | List available models (click any to select) |
| `/model_reset` | Reset model to default |
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
| `python` | Run Python code in an isolated Docker container |
| `store_list` / `store_get` / `store_set` / `store_delete` | Persistent per-user key-value storage |
| `display_status` | Send a progress message to the chat |
| `ask_strong_llm` | Delegate a complex task to a more powerful reasoning model |
| `generate_image` | Generate an image using NVIDIA Flux |

---

## Security & privacy

Messages are held as plaintext in the server's memory. There is no end-to-end encryption.

Anyone who knows your bot's username can send messages to it, potentially incurring API costs. Use `ALLOWED_USER_IDS` to restrict access, and ensure your API key has a spending limit. Avoid using this project with sensitive information or in production environments.