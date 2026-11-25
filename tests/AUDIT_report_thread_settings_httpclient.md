# Test Suite Audit Report: Thread Settings & HTTP Client

## Assigned Files
- `tests/test_thread_settings.py` (15 tests)
- `tests/test_http_client.py` (13 tests)
- Source modules: `src/user_settings.py`, `src/http_client.py`, `src/tools/web.py`, `src/handlers/middleware.py`, `src/persistence/settings_store.py`

---

## 1. Tests to DELETE

- `tests/test_thread_settings.py:156-164` (`InitConversationThreadTest.test_unthreaded_chat_shares_template`): Redundant duplicate of `SettingsAliasingFlagTest.test_unthreaded_thread_shares_the_template_object` (lines 257-264) with identical setup and object identity checks.
- `tests/test_thread_settings.py:92-111` (`SettingsStoreThreadTest.test_thread_and_template_share_one_key`): Low value / implementation detail; tests internal database layout (`store.get_all_keys` and raw SQL query `_settings_rows`) while functional round-tripping is already tested by `test_round_trip_per_thread`.

---

## 2. Tests to CONSOLIDATE

### Group A: Settings Decorator Isolation (`tests/test_thread_settings.py`)
- **Tests**:
  - `tests/test_thread_settings.py:168-200` (`SettingsDecoratorTest.test_threaded_change_updates_thread_only`)
  - `tests/test_thread_settings.py:201-229` (`SettingsDecoratorTest.test_unthreaded_change_updates_template`)
- **Merge Strategy**: Consolidate into a single parametrised test `test_settings_decorator_isolation(thread_id, expected_template_model, expected_future_model)`. Both tests perform identical fixture setup and verification, differing only in whether the update targets a specific thread (`thread_id=2`) or the unthreaded template (`thread_id=None`).

### Group B: Initial Settings Object Aliasing (`tests/test_thread_settings.py`)
- **Tests**:
  - `tests/test_thread_settings.py:257-264` (`SettingsAliasingFlagTest.test_unthreaded_thread_shares_the_template_object`)
  - `tests/test_thread_settings.py:266-273` (`SettingsAliasingFlagTest.test_threaded_thread_owns_a_deep_copy`)
- **Merge Strategy**: Consolidate into a single parametrised test `test_thread_settings_aliasing_on_init(thread_id, expected_is_template, is_same_object)`. Both tests check initial aliasing semantics right after `init_conversation`.

### Group C: Settings Object Swap under Decorator (`tests/test_thread_settings.py`)
- **Tests**:
  - `tests/test_thread_settings.py:275-290` (`SettingsAliasingFlagTest.test_unthreaded_swap_leaves_the_thread_on_the_template`)
  - `tests/test_thread_settings.py:292-305` (`SettingsAliasingFlagTest.test_threaded_swap_keeps_the_thread_off_the_template`)
- **Merge Strategy**: Consolidate into a single parametrised test `test_settings_swap_under_decorator(thread_id, expected_is_template, is_same_object)`. Both tests execute an in-place setting object reassignment under `@middleware.settings` and check flag state and reference identity.

### Group D: HTTP Client Timeout Error Formatting (`tests/test_http_client.py`)
- **Tests**:
  - `tests/test_http_client.py:135-141` (`TestGetText.test_timeout_becomes_readable_runtime_error`)
  - `tests/test_http_client.py:148-154` (`TestRequest.test_timeout_becomes_readable_runtime_error`)
- **Merge Strategy**: Consolidate into a single parametrised test `test_timeout_becomes_readable_runtime_error(func_caller)` testing that `asyncio.TimeoutError` is translated into a readable `RuntimeError` across both `http_client.get_text` and `http_client.request`.

### Group E: HTTP Session Same-Loop Reuse & Reset (`tests/test_http_client.py`)
- **Tests**:
  - `tests/test_http_client.py:170-171` (`TestSessionLifecycle.test_session_is_reused_on_the_same_loop`)
  - `tests/test_http_client.py:173-182` (`TestSessionLifecycle.test_aclose_clears_the_session`)
- **Merge Strategy**: Merge the 1-line same-loop session reuse check (`test_session_is_reused_on_the_same_loop`) directly into `test_aclose_clears_the_session` to form a cohesive lifecycle test verifying session reuse, closure, and re-creation.

### Group F: Web Fetch Max Chars Clamping (`tests/test_http_client.py`)
- **Tests**:
  - `tests/test_http_client.py:223-230` (`TestWebFetch.test_result_never_exceeds_max_chars`)
  - `tests/test_http_client.py:232-240` (`TestWebFetch.test_zero_max_chars_is_clamped`)
- **Merge Strategy**: Consolidate into a single parametrised test `test_web_fetch_max_chars_clamping(max_chars_input, expected_max_len)` testing truncation for normal character bounds (`max_chars=10`) and zero/negative bounds (`max_chars=0`).

---

## 3. Tests to KEEP

### `tests/test_thread_settings.py`
- `tests/test_thread_settings.py:77-79` (`SettingsStoreThreadTest.test_missing_returns_none`): Essential contract test verifying `load_settings` returns `None` for nonexistent rows.
- `tests/test_thread_settings.py:81-90` (`SettingsStoreThreadTest.test_round_trip_per_thread`): Essential isolation test ensuring per-thread saves do not accidentally generate template rows.
- `tests/test_thread_settings.py:113-127` (`SettingsStoreThreadTest.test_repeated_template_save_does_not_duplicate`): High-value SQLite edge case test checking UPSERT behavior for `NULL` `thread_id` unique constraints.
- `tests/test_thread_settings.py:131-144` (`InitConversationThreadTest.test_new_thread_clones_template`): Core behavior test ensuring new threads clone the template and mutations do not leak back.
- `tests/test_thread_settings.py:145-154` (`InitConversationThreadTest.test_existing_thread_row_wins_over_template`): Core precedence test ensuring existing per-thread settings override the template during conversation init.
- `tests/test_thread_settings.py:230-246` (`SettingsDecoratorTest.test_unthreaded_change_saves_once`): Critical optimization/bug guard checking that modifying unthreaded settings saves to DB only once despite aliasing.
- `tests/test_thread_settings.py:309-318` (`SubagentSnapshotTest.test_subagent_gets_thread_settings_snapshot`): Critical subagent context isolation test ensuring child subagents get independent snapshots.

### `tests/test_http_client.py`
- `tests/test_http_client.py:106-114` (`TestGetText.test_multibyte_content_still_yields_max_chars`): Critical UTF-8 edge case test preventing premature character truncation on multi-byte characters.
- `tests/test_http_client.py:115-123` (`TestGetText.test_body_is_capped_during_the_read`): High-value memory guard verifying `content.read()` uses a byte cap before reading large bodies into RAM.
- `tests/test_http_client.py:124-133` (`TestGetText.test_non_2xx_raises_status_error_with_details`): High-value error reporting test ensuring non-2xx responses raise `HttpStatusError` with body/status details.
- `tests/test_http_client.py:156-163` (`TestRequest.test_response_is_buffered_with_status_and_headers`): Core contract test for `http_client.request` and `Response` header lookups.
- `tests/test_http_client.py:183-193` (`TestSessionLifecycle.test_session_from_a_dead_loop_is_replaced`): Essential async lifecycle test handling stale sessions when test runner event loops are closed.
- `tests/test_http_client.py:194-216` (`TestSessionLifecycle.test_session_from_a_live_loop_is_closed_not_leaked`): Essential cross-thread async cleanup test ensuring live-loop sessions are closed thread-safely via `call_soon_threadsafe`.
- `tests/test_http_client.py:242-256` (`TestWebFetch.test_fragment_is_percent_encoded_into_the_path`): Critical URL encoding edge case test ensuring fragments (`#`) are percent-encoded rather than stripped by `yarl`.

---

## 4. Summary Counts

| Category / Metric | `test_thread_settings.py` | `test_http_client.py` | Total |
| :--- | :---: | :---: | :---: |
| **Total Tests Audited** | **15** | **13** | **28** |
| **Recommended for Deletion** | 2 | 0 | **2** |
| **Recommended for Consolidation** | 6 (into 3) | 6 (into 3) | **12 (into 6)** |
| **Recommended to Keep As-Is** | 7 | 7 | **14** |
| **Net Target Test Count** | **10** | **10** | **20** |
