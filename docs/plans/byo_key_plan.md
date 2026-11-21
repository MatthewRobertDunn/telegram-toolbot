# Plan: per-user OpenAI-compatible endpoint and key

> Status: **unimplemented** — this is a design record only; none of it is
> built. Written before the env-var rename: it says `OPENAI_KEY` where the
> environment variable is now `OPENAI_API_KEY` (config still names the
> Python variable `OPENAI_KEY`), and `config.api_retry_time` is now
> 36000 s, not the 600 s quoted below. Treat the concrete values as stale
> and re-check them against `src/config.py` before implementing.

Status: **design agreed, not implemented**

## Goal

Let each Telegram user point the bot's chat completions at their own
OpenAI-compatible endpoint and API key, while the operator's `.env` values
remain the default for everyone else.

New commands:

| Command | Behaviour |
|---|---|
| `/openai_uri [url]` | Show or set the user's base URL |
| `/openai_uri_reset` | Clear the override, fall back to `OPENAI_BASE_URL` |
| `/openai_key [key]` | Show (masked) or set the user's API key |
| `/openai_key_reset` | Clear the override, fall back to `OPENAI_KEY` |

## Non-goals

- Image generation stays on the operator's `IMAGE_GEN_URL` /
  `NVIDIA_API_KEY`. Only chat completions are per-user.
- No encryption of the stored key, no delete-on-write, no special
  handling of the key message in Telegram. Keys live in `store.db` and in
  chat history in plaintext. This is accepted risk: the DB is already on
  the same host as the plaintext `.env`, and Telegram private chats are
  treated as suitably private.
- No per-user model list caching, no proxy/allow-list of endpoints.
- No change to `_validate_config()`: `OPENAI_KEY` in `.env` stays
  mandatory. Per-user overrides are strictly optional.
- No database migration. Settings are a JSON blob already.

## Decisions already made

1. Reset uses separate `_reset` commands, matching `/model_reset`,
   `/top_p_reset`, etc.
2. `/openai_key` does **not** delete the user's command message and does
   not treat the key as radioactive. It does still mask the key when
   *displaying* the current value, purely to keep the chat readable.
3. The bot still refuses to start without `OPENAI_KEY` in `.env`.
4. Authentication failures fail fast instead of retrying for the full
   `config.api_retry_time` (600 s).

## Where the state lives

`UserSettingsSerialized` is serialised as JSON under the `user_settings`
key, partitioned by `user_id`, in `store.db` (see `src/settings_store.py`).
Adding two fields needs no schema change. `from_json` already drops
`None` values, so a reset is just "set the field to `None` and save".

## File-by-file changes

### 1. `src/user_settings.py`

Add to `UserSettingsSerialized`:

```python
openai_base_url: str | None = None
openai_api_key: str | None = None
```

Add to `UserSettings` (import `urllib.parse` at the top):

```python
@property
def openai_base_url(self) -> str:
    return self._settings.openai_base_url or config.OPENAI_BASE_URL

@openai_base_url.setter
def openai_base_url(self, value: str | None):
    if value is not None:
        value = value.strip().rstrip('/')
        parsed = urllib.parse.urlparse(value)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc:
            raise ValueError(
                "Base URL must be a full http(s) URL including the scheme, "
                "e.g. 'https://api.openai.com/v1'.")
    self._settings.openai_base_url = value or None

@property
def openai_api_key(self) -> str:
    return self._settings.openai_api_key or config.OPENAI_KEY

@openai_api_key.setter
def openai_api_key(self, value: str | None):
    if value is not None:
        value = value.strip()
        if not value:
            raise ValueError("API key cannot be empty. "
                             "Use /openai_key_reset to clear it.")
    self._settings.openai_api_key = value or None

@property
def has_openai_base_url_override(self) -> bool:
    return self._settings.openai_base_url is not None

@property
def has_openai_api_key_override(self) -> bool:
    return self._settings.openai_api_key is not None
```

Notes:

- Trailing slashes are stripped because the OpenAI client appends paths
  itself, and `.../v1/` + `/chat/completions` otherwise yields a double
  slash that some gateways reject.
- `rstrip('/')` applied *after* `urlparse` would be safer conceptually,
  but the netloc check above already rejects a bare `/`.
- Empty string is rejected rather than silently clearing; clearing is
  explicit through the `_reset` commands.
- The two `has_*` flags let the command handlers say "inherited from
  `.env`" without poking at the private `_settings` object.

`UserSettings.default()` and `/settings_reset` clear both automatically
because they construct a fresh `UserSettingsSerialized`.

### 2. `src/handlers/run_llm.py`

Keep the module-level default client exactly as it is:

```python
client = AsyncOpenAI(api_key=config.OPENAI_KEY,
                     base_url=config.OPENAI_BASE_URL)
```

Rationale: `tests/test_run_llm.py` patches `run_llm_mod.client`, and a
factory that reads that module global at call time keeps those tests
working untouched. Moving the client into a new module would silently
break the patches (tests would make real network calls).

Add above `run_llm`:

```python
# Clients other than the operator default, keyed by (base_url, api_key).
# Bounded in practice by how many distinct endpoints/keys users configure;
# entries are never evicted, which is fine at this scale.
_clients: dict[tuple[str, str], AsyncOpenAI] = {}


def client_for(settings) -> AsyncOpenAI:
    """The client to use for one agent's requests.

    Falls back to the shared operator client when the user has no
    overrides, so the common path stays one connection pool.
    """
    base_url = settings.openai_base_url
    api_key = settings.openai_api_key
    if (base_url, api_key) == (config.OPENAI_BASE_URL, config.OPENAI_KEY):
        return client
    cache_key = (base_url, api_key)
    llm = _clients.get(cache_key)
    if llm is None:
        llm = _clients[cache_key] = AsyncOpenAI(api_key=api_key,
                                                base_url=base_url)
    return llm
```

Also import the exception classes explicitly (`from openai import
AsyncOpenAI, AuthenticationError, PermissionDeniedError, NotFoundError`)
so typecheckers and pyright see them.

Add a constant near `NO_RESPONSE_MESSAGE`:

```python
# Returned when the API rejects the credentials or the request target.
# Not reused as NO_RESPONSE_MESSAGE: callers such as compaction compare
# against that constant to detect "model unreachable", and a config error
# deserves its own wording.
CONFIG_ERROR_MESSAGE = (
    "The API rejected the request. Check `/openai_uri`, `/openai_key`, "
    "and `/model` (error: {detail}).")
```

In `run_llm`, note that `settings` is already bound near the top
(`settings = ctx.settings`). Inside the retry loop, change

```python
response: Any = await client.chat.completions.create(**create_kwargs)
```

to

```python
llm = client_for(settings)
response: Any = await llm.chat.completions.create(**create_kwargs)
```

(`llm` can be resolved once before the outer `while True:` for clarity;
either is fine. Resolving inside the loop means a key change is picked up
on the next retry, which is nicer.)

Add a fast-fail branch to the existing `except Exception` block:

```python
except (AuthenticationError, PermissionDeniedError, NotFoundError) as e:
    # A bad key, a disabled key, a wrong base URL path, or an unknown
    # model will never succeed by retrying; don't burn 10 minutes.
    logger.warning('config error %s: %s', type(e).__name__, e)
    streamer.new_round()
    return CONFIG_ERROR_MESSAGE.format(
        detail=type(e).__name__)
except Exception as e:
    logger.warning('%s: %s, trying again...', type(e).__name__, e)
    await asyncio.sleep(time_to_wait)
```

Details that matter:

- `NotFoundError` is included because a base URL missing `/v1` (404) is
  the most common URL typo. OpenAI also returns 404 for an unknown model
  (`code: model_not_found`), which is why the message mentions `/model`
  too. If a model-specific message is wanted later, inspect
  `e.body.get('code')`.
- `APIConnectionError` stays in the retrying branch: transient network
  failures should still retry, and the existing 600 s budget is
  deliberate. A typo'd host therefore still retries for 600 s; the
  optional save-time connectivity check below mitigates that.
- `streamer.new_round()` before returning clears any partial draft so the
  error text does not get appended to half an answer. The `message is
  None` path already does this.
- Do **not** include `str(e)` wholesale: for a 404 the body is useful, but
  keep the message short. `type(e).__name__` is enough to distinguish
  "credential"/"not found".

Subagents and compaction need no changes: they call `run_llm` with a
context whose `settings` is a deep copy of the parent's
(`ChatUpdateContext.new_subagent_context`), so the overrides flow through
automatically.

### 3. `src/handlers/command_handlers.py`

Imports:

- Drop `from config import OPENAI_KEY, OPENAI_BASE_URL` (no longer used).
- Drop `import asyncio` (currently unused).
- Keep `import openai` for the error classes used by `/model_list`.
- Add `from handlers.run_llm import client_for`.

Note on the import direction: `handlers/__init__.py` is empty, and
`run_llm` does not import `command_handlers`, so this introduces no cycle.
`run_llm` stays the single owner of client construction.

Add a small helper:

```python
def _mask_key(key: str) -> str:
    """Show enough to identify a key without republishing it in chat."""
    if not key:
        return "(none)"
    if len(key) <= 8:
        return "…"
    return f"{key[:3]}…{key[-4:]}"
```

New handlers, each following the existing `@handler` + `@settings`
pattern (`@handler` loads settings + conversation, `@settings` persists
after the handler returns):

```python
@handler
@settings
async def openai_uri_command(ctx: ChatUpdateContext):
    url = ctx.command_text
    if not url:
        source = ("your override" if ctx.settings.has_openai_base_url_override
                  else "default from .env")
        await ctx.messaging.send_message(
            f"Current OpenAI-compatible base URL: `{ctx.settings.openai_base_url}` "
            f"({source})\n"
            f"Usage: `/{ctx.command_name} <url>` e.g. "
            f"`https://api.openai.com/v1`")
        return
    try:
        ctx.settings.openai_base_url = url
    except ValueError as e:
        await ctx.messaging.send_message(str(e))
        return
    await ctx.messaging.send_message(
        f"Base URL set to `{ctx.settings.openai_base_url}`.\n"
        "Models differ per provider, so you may need `/model_list` and "
        "`/model` to match.")


@handler
@settings
async def openai_uri_reset_command(ctx: ChatUpdateContext):
    ctx.settings.openai_base_url = None
    await ctx.messaging.send_message(
        f"Base URL reset to the default `{ctx.settings.openai_base_url}`.")


@handler
@settings
async def openai_key_command(ctx: ChatUpdateContext):
    key = ctx.command_text
    if not key:
        source = ("your override" if ctx.settings.has_openai_api_key_override
                  else "default from .env")
        await ctx.messaging.send_message(
            f"Current API key: `{_mask_key(ctx.settings.openai_api_key)}` "
            f"({source})\n"
            f"Usage: `/{ctx.command_name} <key>`")
        return
    try:
        ctx.settings.openai_api_key = key
    except ValueError as e:
        await ctx.messaging.send_message(str(e))
        return
    await ctx.messaging.send_message(
        f"API key set to `{_mask_key(ctx.settings.openai_api_key)}`.")


@handler
@settings
async def openai_key_reset_command(ctx: ChatUpdateContext):
    ctx.settings.openai_api_key = None
    await ctx.messaging.send_message(
        f"API key reset to the default "
        f"`{_mask_key(ctx.settings.openai_api_key)}`.")
```

Behaviour notes:

- No-argument form is the "show current" form, consistent with
  `/model`, `/temperature`, `/timezone`.
- The full key is never echoed back, only masked. The user's own command
  message still contains it, as decided.
- `/openai_uri` with an argument that has no scheme yields the setter's
  `ValueError`, which is surfaced verbatim (same pattern as
  `/temperature` and `/timezone`).
- `/openai_key` with whitespace-only text strips to empty and raises, so a
  stray space does not wipe the key.

Change `/model_list` to use the per-user client and the async API:

```python
@handler
async def model_list_command(ctx: ChatUpdateContext):
    try:
        llm = client_for(ctx.settings)
        try:
            response = await llm.models.list(
                extra_query={"available": "true"})
        except openai.BadRequestError:
            # `available` is a local-proxy extension; strict
            # OpenAI-compatible servers reject the unknown query param.
            response = await llm.models.list()
        models = sorted({model.id for model in response})
        ...
    except Exception as exc:
        await ctx.messaging.send_message(
            f"Error fetching models: {exc}")
```

This also fixes a latent bug: the current code runs a blocking
`openai.OpenAI` client inside an async handler and blocks the event loop
for the duration of the request.

`ctx.settings` is available here because `@handler` runs
`init_conversation`, which loads settings before the function body. No
`@settings` decorator is needed (nothing is persisted).

### 4. `src/chatbot.py`

Imports: add the four new handlers to the existing
`from handlers.command_handlers import (...)` block, keeping alphabetical
grouping with the other model-ish commands.

`post_init` → `set_my_commands`: add after the model entries:

```python
BotCommand("openai_uri", "Set your OpenAI-compatible base URL"),
BotCommand("openai_uri_reset", "Reset base URL to the default"),
BotCommand("openai_key", "Set your OpenAI API key"),
BotCommand("openai_key_reset", "Reset API key to the default"),
```

Handler registration in `main()`, next to the model handlers:

```python
application.add_handler(CommandHandler('openai_uri', callback_adapter(openai_uri_command)))
application.add_handler(CommandHandler('openai_uri_reset', callback_adapter(openai_uri_reset_command)))
application.add_handler(CommandHandler('openai_key', callback_adapter(openai_key_command)))
application.add_handler(CommandHandler('openai_key_reset', callback_adapter(openai_key_reset_command)))
```

Leave `_validate_config()` untouched.

### 5. `README.md`

- Commands table (after the `/model_reset` row):

```markdown
| `/openai_uri [url]` | Set your own OpenAI-compatible base URL |
| `/openai_uri_reset` | Reset base URL to the server default |
| `/openai_key [key]` | Set your own API key (shown masked) |
| `/openai_key_reset` | Reset API key to the server default |
```

- Configuration reference: extend the `OPENAI_KEY` / `OPENAI_BASE_URL`
  rows' descriptions with "(default for users who have not set
  `/openai_key` / `/openai_uri`)", and add a short paragraph after the
  table:

> Users can override the endpoint and key for their own account with
> `/openai_uri` and `/openai_key`. Overrides apply to chat completions
> only (not image generation) and are cleared by the matching `_reset`
> commands or `/settings_reset`. Everything the model sees — history,
> system prompt, memories, tool output — is sent to the endpoint the user
> configures.

- Note under Configuration that keys are stored unencrypted in
  `store.db`, same trust level as `.env`. One sentence is enough.

## Tests

### `tests/test_user_settings.py` (new)

Follow `tests/test_timezone.py` for shape: `sys.path` setup, then
`unittest.TestCase`.

| Test | Assertion |
|---|---|
| `test_openai_defaults_fall_back_to_config` | `UserSettings.default().openai_base_url == config.OPENAI_BASE_URL`, same for key |
| `test_openai_base_url_override_strips_trailing_slash` | setting `https://api.openai.com/v1/` reads back without the slash |
| `test_openai_base_url_rejects_missing_scheme` | `localhost:11434/v1` raises `ValueError()` |
| `test_openai_base_url_rejects_bare_path` | `/v1` raises |
| `test_openai_api_key_override_and_reset` | set → property returns it; set `None` → falls back to config |
| `test_openai_api_key_rejects_blank` | `"   "` raises |
| `test_openai_overrides_round_trip_through_json` | `from_json(to_json())` preserves both values |
| `test_default_clears_overrides` | overriding then `UserSettings.default()` returns config values |
| `test_override_flags` | `has_*_override` false by default, true after set, false after reset |

Use `setUp`/`tearDown` that save and restore `config.OPENAI_BASE_URL` /
`config.OPENAI_KEY` if a test needs a distinctive value; the defaults are
enough for most.

### `tests/test_run_llm.py` (existing)

The module-level `SETTINGS` `SimpleNamespace` must gain the two new
attributes, or `client_for` will `AttributeError`:

```python
SETTINGS = SimpleNamespace(
    show_tool_output=False, model='m', temperature=1.0, top_p=0.95,
    openai_base_url=config.OPENAI_BASE_URL,
    openai_api_key=config.OPENAI_KEY)
```

(Import `config` at the top.) Same for the inline namespace in
`TestToolCallAnnouncements`.

New tests:

- `test_client_for_defaults_to_the_shared_client` — build a namespace
  with the config values and assert
  `run_llm_mod.client_for(settings) is run_llm_mod.client`.
- `test_client_for_caches_per_endpoint_and_key` — patch
  `run_llm_mod.AsyncOpenAI` with a counting fake; assert one construction
  for two calls with the same overrides, two constructions for different
  keys. Reset `run_llm_mod._clients` in `setUp`/`tearDown` so tests do not
  leak clients.
- `test_auth_error_fails_fast` — fake client whose
  `chat.completions.create` raises
  `openai.AuthenticationError("bad key", response=httpx.Response(401,
  request=httpx.Request("POST", "http://x/v1")), body=None)`; use
  `unittest.mock.patch.object(run_llm_mod, 'client', fake)`; assert the
  result equals `run_llm_mod.CONFIG_ERROR_MESSAGE.format(detail=
  'AuthenticationError')` and that `create` was awaited exactly once
  (`unittest.mock.AsyncMock` + `assert_awaited_once`, and that no
  `asyncio.sleep` happened — or simply that the call returned quickly;
  asserting the call count is the cheap, robust version).
- Optional: `test_uses_client_for_the_settings_endpoint` — patch
  `run_llm_mod.client_for` returning the existing fake and assert it was
  called with `ctx.settings`. This pins the wiring without duplicating
  the cache tests.

`httpx` is already installed as an OpenAI dependency, so importing it in
tests adds no requirement.

### Integration tests

No changes. They use `integration_tests/.env` and default settings, which
still resolve to the operator client.

## Verification

```powershell
pake typecheck      # pyright over src/, tests/, integration_tests/
pake unittest
```

Manual smoke test (after `pake publish` or locally):

1. `/openai_uri` → shows the `.env` default and "default from .env".
2. `/openai_uri https://api.openai.com/v1` → confirms, mentions
   `/model_list` / `/model`.
3. `/openai_uri api.openai.com/v1` → rejected with the scheme error.
4. `/openai_key sk-...` → confirms with a masked key; message containing
   the key remains in chat (accepted).
5. `/openai_key` → shows `sk-…abcd` and "your override".
6. `/model_list` → lists models from the new endpoint.
7. `/model gpt-...`, then send a message → replies from the new endpoint.
8. `/openai_key sk-bad` → next message returns the config-error text
   promptly instead of hanging ~10 minutes.
9. `/openai_key_reset`, `/openai_uri_reset` → masked/default values,
   replies work again against `.env`.
10. `/settings_reset` → both overrides cleared.
11. Trigger the subagent tool with overrides active (the subagent inherits
    the same settings snapshot) and confirm it uses the new endpoint, e.g.
    visible in the provider's dashboard or by using a key that only one
    endpoint accepts.

## Risks and edge cases

- **Wrong host, right scheme** still retries for up to 600 s because
  `APIConnectionError` is intentionally retried. Optional mitigation
  (deferred): on setting `/openai_uri` or `/openai_key`, call
  `await client_for(ctx.settings).models.list()` once and report
  "connected, N models" or the error. Adds latency to the command and a
  failure path, so it is not in this pass.
- **Unbounded client cache.** One entry per distinct `(base_url, key)`
  ever used. With a handful of users and rare rotations this is
  negligible; no eviction is implemented. Reset commands could
  additionally drop the old entry, but that requires tracking which key
  the old client was built with — not worth it now.
- **404 ambiguity.** `NotFoundError` covers both a wrong base path and an
  unknown model, hence the message names `/openai_uri`, `/openai_key`,
  and `/model`.
- **Third-party `/models` support** varies. The `extra_query` fallback
  handles the strict-server case; an endpoint with no `/models` route
  will still show an error, which is acceptable.
- **Data exposure.** Choosing an endpoint means all conversation
  content, memories, and tool output go to that provider. Documented in
  the README.
- **Key in logs.** No new logging path includes the key:
  `client_for` logs nothing, and the fast-fail branch logs only the
  exception type/body, which does not echo the bearer token.
- **Telegram command menu length.** Four more entries; the list is
  already long, but this matches the existing one-command-per-action
  convention.
- **`/model` mismatch.** A user who changes endpoint but keeps an old
  model id gets the fast-fail message; the confirmation text nudges them
  to `/model_list`.

## Estimated effort

1.5–3 hours including tests and docs. No dependency, schema, or deploy
pipeline changes; `pake publish` is sufficient to ship.

## Checklist

- [ ] `UserSettingsSerialized` + `UserSettings` fields, validation,
      fallback properties, override flags
- [ ] `client_for` factory + cache in `run_llm.py`
- [ ] `run_llm` uses `client_for(settings)`
- [ ] Fast-fail branch + `CONFIG_ERROR_MESSAGE`
- [ ] Four command handlers
- [ ] `/model_list` uses `client_for` + async API + `extra_query` fallback
- [ ] Register handlers and `BotCommand` entries in `chatbot.py`
- [ ] New `tests/test_user_settings.py`
- [ ] `tests/test_run_llm.py`: fixture fields + client factory + fast-fail
- [ ] README commands table, config notes, plaintext-storage note
- [ ] `pake typecheck`
- [ ] `pake unittest`
- [ ] Manual smoke test against two endpoints
