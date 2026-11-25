# Test Audit: tests/test_regenerate.py + tests/test_config.py

Scope: `tests/test_regenerate.py` (19 tests), `tests/test_config.py` (18 tests).
Source under test: `src/handlers/conversation_handlers.py` (/regenerate, /undo, /retry,
`message_preview`), `src/config.py`.

Method note: I also cross-checked `src/conversation.py` (where the actual trim logic
`prepare_regeneration` / `undo_last_exchange` lives and which has its own unit tests in
`tests/test_conversation.py:397-426`), `tests/mock_context.py`, and `pakefile.py` /
`pyrightconfig.json` to judge redundancy against neighbouring suites and static analysis.

---

## 1. Tests to DELETE

| Test | Location | Reason |
|---|---|---|
| `test_reply_without_stored_id_deletes_nothing` | tests/test_regenerate.py:137 | Redundant: the "assistant turn without a telegram id → nothing deleted" path is already asserted in `test_trailing_tool_results_are_skipped_to_find_the_reply` (line 135), and the no-id delete loop is also covered on the undo side (192, 208, 297). Its remaining assertions (`fresh reply` content, roles) duplicate lines 92-97/108. |
| `test_undo_without_telegram_id_deletes_nothing` | tests/test_regenerate.py:186 | Redundant: `bot.deleted == []` for id-less removals is asserted three times already (135, 208, 297); the pop of a bare user turn is exercised by 154 (with ids) and 275 — same code path, only the ids dict differs. |
| `test_every_referenced_name_exists` (ConfigReferenceTest) | tests/test_config.py:198 | Redundant with static analysis: I verified that pyright — which `pake typecheck` runs over `src/` — already errors on `config.UNKNOWN_ATTR` (`reportAttributeAccessIssue`). The test's own docstring premise ("Nothing imports config with a type checker in the loop") is outdated. Also brittle: regex-scans comments/strings (hence the `FILE_NAME_WORDS = {'py'}` special case) and false-positives on any docstring mentioning `config.<name>`. *Caveat:* keep it if typecheck is not enforced in the PR/test pipeline; it is otherwise dead weight. |

Total: 3 deletes (2 clear-cut, 1 conditional on typecheck being enforced).

## 2. Tests to CONSOLIDATE

### tests/test_config.py

1. **The five "bad X names the variable and value" tests (lines 34, 41, 48, 55, 62) → one
   parametrised test.** They are the identical 5-line pattern with only the env var and bad
   value varying (int / float / float / float / int-list helpers). A single
   `subTest`-driven test over `(env, expected var name, expected value)` keeps coverage of
   all three `_env` helpers while removing 4 copies of the reload/assert machinery. 5 → 1.
2. **The four NVIDIA-key tests (lines 110, 115, 122, 128) → one parametrised test.** All
   load the same config path with one env combination differing. Cases:
   (OPENAI only → fallback value **and** warning, lines 110+115 are the same code path),
   (both keys → explicit wins, no warning), (no keys → `''`, silent). One
   `(env, expected_key, expect_warning)` table replaces four reloads. 4 → 1.
   At minimum, merge 110 + 115, which assert value and log of the *same* load.
3. **FindEnvFileTest (lines 148, 157, 167, 178) → one table-driven test** over directory
   layouts (`.env` at root only / root + stale src / ancestor only / none). Each is the same
   8-line tempdir scaffold with a different layout; 167 and 178 in particular assert the
   identical outcome (`None`). 4 → 1 (or 4 → 2).
4. **`test_store_db_path_is_not_a_stale_copy` (line 104) → fold into
   `test_store_db_path_env_var_feeds_sql_db_path` (line 100)** as one extra
   `assertFalse(hasattr(...))` line with the existing comment. It is a one-line structural
   guard, not a behavioural test on its own.

### tests/test_regenerate.py

5. **Retry: `test_retry_appends_a_reply_without_touching_the_history` (302) +
   `test_retry_answers_an_unanswered_user_turn` (321) → one test with two subTest
   histories.** Both assert the same behaviour (history untouched, fresh reply appended,
   nothing deleted) with marginally different tails (assistant vs user at the end).
6. **Undo walk-back: `test_undo_removes_everything_from_the_last_user_prompt_inclusively`
   (260) + `test_undo_removes_only_the_last_user_turn_when_it_is_the_tail` (275) → one
   parametrised test.** They are the two branches of the same rule (walk back to the last
   user turn inclusively) differing only in where the tail sits.
7. **Undo report format: `test_undoes_last_message_and_deletes_it_from_telegram` (154) +
   `test_undo_of_assistant_turn_names_the_role` (173) → one parametrised test** covering the
   single-message ("Removed message (Role): ...") and multi-message
   ("Removed N message(s).") reply formats. Note 173's name is misleading — it does not
   test the assistant-turn role naming at all (its history has both a user and an assistant
   turn, so it hits the multi-message branch); a rename is needed either way.
8. *(Optional)* **Regenerate: `test_last_assistant_is_popped_deleted_and_regenerated` (82)
   + `test_last_user_turns_into_a_reply_without_appending` (99)** could be parametrised over
   (initial tail, expected deletions); they are the two ANSWER branches. Lower priority —
   they each read clearly as-is.
9. **Trim within `test_defaults_are_unchanged` (line 83):** lines 88-89 re-assert the
   `API_RETRY_MAX_DELAY` alias that `test_numeric_settings_are_env_tunable` (77-81) already
   covers; drop the duplicate. More broadly, this is a 14-assertion characterisation test —
   consider trimming to the operationally meaningful defaults (sandbox timeouts/limits,
   retry budget) so deliberate default tuning doesn't require blanket test edits.

**Brittleness note (fix while consolidating):** line 273 asserts deletion order
`[(5, 12), (5, 11)]`, but `undo_last_exchange`'s docstring states removed order "is not
significant". Assert as a set/sorted list to avoid pinning pop order.

## 3. Tests to KEEP (high value)

tests/test_regenerate.py — the suite is genuinely behavioural (drives the real
`@handler`-wrapped commands through the generation queue with `run_llm` stubbed):

- **82** — the core /regenerate flow: pop, Telegram deletion, fresh reply, new message id recorded. The one end-to-end assertion of id re-linking (`CUSTOM['telegram_message_ids'] == [555]`).
- **99** — regenerate on a user-tail must answer *without duplicating* the user turn (a real business rule).
- **112, 289, 332** — the three "Nothing to X" guards; each exercises a *different* code path (`prepare_regeneration` NOTHING vs `undo_last_exchange` empty vs `_has_content`), so keep all three.
- **120** — abandoned trailing tool round: real replayability hazard, and the only test that tool metadata doesn't leak into the regenerated turn.
- **194, 210, 227, 244** — the tool-call cascade family: whole-round removal, earlier rounds untouched, parallel sibling results, unanswered mid-call pop. All distinct edge cases protecting LLM-history replayability; do not delete any. (10 and 227 could share a parametrisation, but both inputs are worth keeping.)
- **165** — `message_preview` truncation boundary (over-limit vs exactly-at-limit); cheap off-by-one guard for a user-facing message.
- **154, 173, 260, 275** — kept (consolidated per §2): core undo semantics and user-visible report format.

tests/test_config.py:

- **34-62 (the behaviour, consolidated per §2)** — a malformed env value must fail at import *naming the variable*; this is the module's stated contract and error-path behaviour.
- **100** — `STORE_DB_PATH` → `sql_db_path`; deployment-critical.
- **110-128 (the behaviour, consolidated per §2)** — NVIDIA key fallback semantics and that the fallback is *announced* (silent fallback is a footgun); all three resolutions (fallback / explicit / none) must stay covered.
- **148, 157, 167** — the `.env` discovery precedence rules (root wins over stale `src/.env`; ancestors ignored) protect a documented deployment hazard (`pake publish_env`); keep the behaviour even if merged.
- **69** — the legacy lowercase aliases (`temperature`, `api_retry_time`, …) staying in sync is load-bearing (`user_settings`, `handlers.run_llm` import them); keep the alias assertions even if the trivial uppercase reads are trimmed.
- **83** — defaults pinning has value (accidental default drift on sandbox timeouts/limits is a real risk); trim, don't delete.

### Coverage gaps noticed while auditing (for the report reader, not low-value tests)

- The `RegenerationAction.IMPERSONATE` branch of `regeneration_turn`
  (src/handlers/conversation_handlers.py:223-231) — /regenerate on an impersonation turn
  re-running the impersonation flow — has **no test at all** in either file.
- `_has_content`'s exclusion of thinking messages (retry/impersonate "Nothing to X" when the
  history holds only a stopped thinking block) is untested; only the system-head case (332) is.
- `ALLOWED_IDS` positive parsing (a valid list) is never asserted — only the failure case (62).

## 4. Summary

| | tests/test_regenerate.py | tests/test_config.py | Total |
|---|---|---|---|
| Tests audited | 19 | 18 | **37** |
| Recommended for deletion | 2 | 1 (conditional on typecheck being enforced) | **3** |
| Recommended for consolidation (tests merged away) | 4 (two retry; two undo walk-back; plus two into the report-format pair) | 10 (4 bad-env → 1, 4 nvidia → 1, 4 findenv → 1-2, 1 folded) | **~14** |
| Tests to keep as-is | 13 | 7 | 20 |

Net effect if all recommendations are applied: 37 tests → ~20, with no behavioural coverage
lost (three consolidations fold duplicate machinery; the deletions remove paths asserted
elsewhere in the same files or by `pake typecheck`). The two files are otherwise healthy:
both are behavioural, well-commented, and protect real error paths — the waste is almost
entirely copy-paste of the same assertion pattern with different inputs.
