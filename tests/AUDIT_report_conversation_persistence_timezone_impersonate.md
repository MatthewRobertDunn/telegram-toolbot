# Test Audit: conversation persistence, timezone/settings defaults, impersonate

Scope:
- `tests/test_conversation_persistence.py` (10 tests) — source: `src/persistence/settings_store.py` (wraps `src/persistence/sqlite_store.py`)
- `tests/test_timezone.py` (10 tests) — source: `src/handlers/command_handlers.py` (`timezone_command`), `src/user_settings.py`
- `tests/test_impersonate.py` (10 tests) — source: `src/handlers/impersonate_prompt.py`, `src/handlers/conversation_handlers.py`

Note: the assigned source list says `sqlite_store.py`, but the persistence tests exercise `persistence.settings_store` (the SQL/store logic under test lives there; `sqlite_store.py` supplies `AsyncSQLiteStore`). Both were reviewed.

---

## 1. Tests to DELETE

| Test | File:line | Reason |
|---|---|---|
| `test_last_updated_is_stamped_by_the_db` | test_conversation_persistence.py:145 | Tests the SQLite `DEFAULT CURRENT_TIMESTAMP` + `NOT NULL` schema rather than application behaviour: the column cannot be NULL when the row is inserted by `settings_store.set`, so the assertion is guaranteed by the framework. The only app-specific bit (`thread_id == 2` is stored) is already covered by `test_threads_persist_independently`. |
| `test_default_timezone_is_utc` | test_timezone.py:50 | Redundant: `TestUserSettingsDefaults.test_defaults_come_from_the_module_and_config` (line 98) already asserts `settings.timezone == DEFAULT_TIMEZONE`, which is strictly stronger (checks the constant source, not a hardcoded literal). |
| `test_timezone_survives_json_round_trip` | test_timezone.py:81 | Redundant subset of `test_every_setting_survives_a_json_round_trip` (line 133), which round-trips timezone ('Australia/Sydney') alongside every other field. Duplicate of the same behaviour with one field. |

## 2. Tests to CONSOLIDATE

- **`load` failure paths → one test** (`test_load_missing_returns_none` :64 + `test_load_corrupt_returns_none` :68): both assert "absent or corrupt → `None`, no raise". Merge into a single parametrised test (`missing`, `corrupt-payload`) with one `assertIsNone` per case. Saves a setup-heavy test; the assert is one line either way.
- **Image pruning is tested twice through the same helper** (`test_save_prunes_old_inline_images` :164 + `test_load_prunes_unbounded_stored_images` :196): both exercise `_prune_inline_images` via different entry points. They are NOT duplicates (save-time mutation-safety and load-time legacy-blob handling are distinct guarantees) so keep both, but share the over-cap message-fixture builder (they hand-roll nearly identical loops). Optional.
- **`test_impersonation_is_added_as_a_user_turn` :85 + `test_impersonated_turn_is_tagged_and_scoped_locally` :106**: both drive the same scenario (impersonate onto a 1-turn history) and inspect the resulting last turn — content/telegram ids in one, `message_type` + endpoint invisibility in the other. Merge into one test with the four assertions; the second's setup/teardown is pure duplication.
- **`test_no_arg_shows_current` :78 (test_timezone.py) vs `test_command_handlers.py:238`**: cross-file overlap on the no-arg `/timezone` path (usage text). The other file asserts the usage line against default UTC, so it can't prove the *current* value is interpolated. Keep this test but trim it to its unique bit (the non-default `America/New_York` appears in the output); the usage-line assertion (`assertIn('/timezone <IANA timezone', ...)`) duplicates `test_command_handlers.py`.
- **`TestUserSettingsDefaults` is misplaced**: it belongs in a `test_user_settings.py`, not in `test_timezone.py` (only ~2 of 10 tests there concern timezone at all). Pure relocation, no deletions needed.

## 3. Tests to KEEP

### test_conversation_persistence.py
| Test | Why |
|---|---|
| `test_round_trip_preserves_id_messages_and_telegram_ids` :51 | Core persistence contract: id, messages, and `custom.telegram_message_ids` survive the round trip. |
| `test_overwrite_keeps_latest` :74 | Guards the documented UPDATE-then-INSERT upsert (a naive `INSERT OR REPLACE` *duplicates* NULL-thread rows — a real, non-obvious regression trap). |
| `test_worker_persists_conversation_after_queued_job` :84 | Proves the worker→save wiring end-to-end and verifies a real load afterwards; the arg-spy is fine because a real read follows. |
| `test_init_conversation_loads_persisted_history` :106 | Guarantees the system head is re-derived from current settings on load and the persisted id survives — easy to regress silently. |
| `test_threads_persist_independently` :122 | Behavioural half (plain vs thread conversation are distinct) is high value. Suggest dropping only the raw `sqlite3` row-introspection block (:135-143): it asserts implementation (one key, `thread_id` column usage) already covered more thoroughly by `test_thread_settings.py`. |
| `test_save_prunes_old_inline_images` :164 | Prune-on-save + proof the live conversation is never mutated. Keeps DB payloads bounded. |
| `test_load_prunes_unbounded_stored_images` :196 | Prune-on-load path that save never covers (legacy/oversized stored blobs); checks message shapes stay intact. |

### test_timezone.py
| Test | Why |
|---|---|
| `test_set_valid_timezone` :53 | Core command behaviour incl. save being awaited. |
| `test_invalid_timezone_is_rejected` :61 | Error path, settings unchanged. Worth strengthening: also assert `save` was *not* awaited (currently untested — a failed set that still persists would slip through). |
| `test_no_arg_shows_current` :68 | See consolidation note above — trim the duplicated usage assertion, keep the current-value interpolation. |
| `test_defaults_come_from_the_module_and_config` :91 | Single source-of-truth check for every default (config-derived and module constants). |
| `test_default_prompt_is_read_at_call_time` :105 | Genuinely subtle late-binding behaviour (startup replacement of `DEFAULT_SYSTEM_PROMPT`); not guessable. |
| `test_blank_model_is_rejected` :115 | Validation edge cases including the whitespace-only trap and the `None`-resets semantics. |
| `test_blank_system_prompt_is_rejected` :125 | Same, different field with its own reset semantics. |
| `test_every_setting_survives_a_json_round_trip` :133 | Full serialization contract for `UserSettings`; the one place every field is exercised together. |

### test_impersonate.py — nothing to delete
| Test | Why |
|---|---|
| `test_empty_conversation_says_nothing_to_impersonate` :76 | Boundary: empty-history guard message. |
| `test_impersonation_is_added_as_a_user_turn` :85 | Core flow: fake turn added, posted as rich markdown, posted id attached to the message. (Merge with :106.) |
| `test_impersonated_turn_is_tagged_and_scoped_locally` :106 | Guarantees the `IMPERSONATION` tag lives in `custom` metadata and the endpoint never sees it — a privacy/correctness invariant. (Merge with :85.) |
| `test_clone_is_isolated_from_the_real_history` :119 | Critical isolation guarantee (clone mutation must not pollute real history, new session id). |
| `test_impersonation_prompt_added_as_final_user_turn` :145 | Verifies the prompt is appended as the last user turn of the clone. Minor: `assertIn('impersonating the user', ...)` couples to prompt wording — `from handlers.impersonate_prompt import IMPERSONATE_PROMPT` and compare against the constant instead (less brittle, same assurance). No delete. |
| `test_regenerate_of_impersonation_redoes_the_impersonation` :163 | Integration: regenerate detects the impersonation target, deletes the right Telegram message, re-runs the impersonation flow, retags. |
| `test_undo_after_impersonation_preserves_thinking_exchange` :188 | Already parametrised via `subTest`; protects the undo-walkback invariant against kept thinking blocks — nasty edge, must stay. |
| `test_stop_during_post_commits_impersonation_for_undo` :218 | The only test of the `/stop`-vs-atomic-commit race (post accepted, response not yet delivered). Complex but irreproducible elsewhere. |
| `test_impersonation_rechecks_content_when_its_queue_item_runs` :265 | TOCTOU recheck: history emptied between command arrival and queue run. Real concurrency guard. |
| `test_no_model_output_reports_failure` :302 | Error path: unusable model output → failure message, history untouched, also implicitly covers `NO_RESPONSE_MESSAGE`/`NO_CONTENT_MESSAGE` filtering. |

---

## 4. Summary

| Area | Tests | Delete | Consolidate |
|---|---|---|---|
| test_conversation_persistence.py | 10 | 1 (`last_updated...`) | 1 merge (fail-paths), 1 fixture share (pruning), 1 trim (SQL introspection in threads test) |
| test_timezone.py | 10 | 2 (`default_timezone_is_utc`, `timezone_survives_json_round_trip`) | 1 trim (`no_arg` usage assertion), 1 relocation (`TestUserSettingsDefaults`) |
| test_impersonate.py | 10 | 0 | 1 merge (:85 + :106), 1 wording → constant |
| **Total** | **30** | **3** | **4 groupings** |

Net: 30 → 27 tests, with no loss of behavioural coverage. The suite in this area is healthy overall: `test_impersonate.py` in particular is high-quality behavioural testing (concurrency, isolation, undo semantics) with no anemic tests. The main wins are removing two trivial/redundant settings tests, one SQLite-default test, and tightening a few cross-file overlaps.
