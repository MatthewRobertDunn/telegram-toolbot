# Test Suite Audit — `tests/test_journal.py`

Area: journal persistence (`src/persistence/journal_store.py`, `src/persistence/sqlite_store.py`) and the `journal_*` tool layer (`src/tools/journal_tool.py`).

**Overall verdict:** This is an unusually strong suite. Most tests target real edge cases (keyset paging across backdated rows, same-millisecond ties, FTS trigger consistency, epoch-vs-lexical ordering) with failure modes spelled out in comments. Very little is genuinely low-value. The problems found are duplication (the same validation asserted in 2–3 tests) and a handful of assertions that test nothing or test private state.

---

## 1. Tests to DELETE

No test is entirely worthless except one tool-level duplicate. Several contain dead or implementation-detail assertions worth deleting *within* otherwise-valuable tests.

| Location | Item | Reason |
|---|---|---|
| `tests/test_journal.py:1105` | `test_query_tool_cursor_resumes_inside_tied_milliseconds` (whole test) | **Redundant.** The identical tie scenario (ids 1–3 sharing one millisecond, paging both directions with the id cursor) is already covered at the store level by `test_search_page_pages_inside_tied_milliseconds` (line 610). The tool layer adds no tie-specific logic — it just passes `cursor` through, and cursor resumption formatting is already covered by `test_query_tool_paging` (line 1041). ~30 lines of duplicated fixture for zero incremental coverage. |
| `tests/test_journal.py:71–80` | Inside `test_schema_script_runs_once_per_database` | **Implementation-detail.** The `mock.patch.object(second, "create_schema", wraps=...)` spy, `spy.await_count == 0`, and `second._schema_ready` assertions test private state and call patterns. The observable proof of the shared memo is already in the test: `counter["calls"] == 1` plus `second` successfully reading `first`'s data. Delete the spy block, keep the counter and read-through assertions. |
| `tests/test_journal.py:755–760` | Inside `test_list_topics` ("Topics vanish once unused" block) | **Dead assertion.** The deleted entry (`first`, topic "Mood", id=2) is duplicated by entry id=3 (topic "mood"), so `list_topics` *cannot* change after the delete — the before and after assertions are identical and the block would pass even if vanishing were broken. Either delete the block, or fix it: add a topic with exactly one entry, delete that entry, and assert the topic disappears. |
| `tests/test_journal.py:1097–1103` | Inside `test_query_tool_paging` (limit=0 / limit=51 checks) | **Redundant.** The exact same two limit-validation assertions appear in `test_search_tool_validation_and_format` (lines 1000–1004). Keep once (see consolidation E). |
| `tests/test_journal.py:1178` | Inside `test_all_seven_tools_are_declared` (`self.assertEqual(tool.func.__name__, name)`) | **Trivial.** The registry wires `func` and `name` from the same declaration; this asserts the language, not behaviour. The name-set equality already catches a missing/renamed tool. |

---

## 2. Tests to CONSOLIDATE

**A. Topic/entry/limit validation — 3 tests re-cover the same branches.**
- `test_validation_raises_typed_error` (line 105): empty topic, empty `old_string`, limit=0.
- `test_add_rejects_empty_and_long` (line 161): empty topic (again), empty entry, 128/129 boundary.
- `test_search_start_window_and_order` (lines 438–442): limit=0 and limit=51 (again, third time).

`test_validation_raises_typed_error`'s unique value is *only* the typed-error contract (message equals user-facing text, is a `ValueError` subclass). Everything else duplicates. Merge: keep one parametrised validation test for the branches (empty topic/entry, 128 vs 129, empty `old_string`, limit bounds) and keep a 3-line `test_validation_raises_typed_error` that only checks the `StoreValidationError`/`ValueError` + verbatim-message contract on one representative call. Lines 105–121 vs 161–184 overlap ≈ 60%.

**B. `test_replace_identical_rejected` (line 278) → fold into `test_replace` (line 226).**
One assertion, same fixture, same function under test as `test_replace` (which already asserts the empty-`old_string` ValueError at lines 232–235). A one-line addition to `test_replace`.

**C. Start-window tests share fixture and topic:**
- `test_search_start_window_and_order` (line 410) and
- `test_search_start_is_inclusive_in_both_directions` (line 444)

Both insert the identical three-entry fixture (same stamps) and test the same `start_ms` semantics. Merge into a single start-semantics test: keep the inclusivity boundary pairs from 444 (they're the sharpest assertions) plus the sort-direction/limit checks from 410. Saves a duplicated fixture and puts both halves of the "start is an inclusive bound, direction decides which side" contract in one place.

**D. Timezone-on-add tests → parametrise:**
- `test_add_uses_configured_timezone` (line 785)
- `test_add_defaults_to_utc_without_settings` (line 801)

Same shape: add via tool, read back the row, assert the `created` offset. One test parametrised over `timezone_name ∈ {"Australia/Sydney", None}` (expecting `+00:00` for `None`).

**E. Tool-level limit validation (lines 1000–1004 and 1098–1103).**
Identical assertion pairs in two tests. Keep in `test_search_tool_validation_and_format` only. (And the store-level check at lines 438–442 is the one worth keeping under consolidation A.)

**F. Registry tests → one test:**
- `test_query_tool_paging_parameters_are_declared` (line 1160)
- `test_all_seven_tools_are_declared` (line 1168)

Both are static registry-metadata checks. Merge into a single `test_journal_tools_are_declared` covering the name set, `scope`, `_needs_ctx`, and the `cursor`/`start` parameter types. Also note the type assertions are close to trivial (the registry is a static declarative table), so one compact test is enough; if consolidating, consider whether the whole pair is worth keeping at all.

**G. Tool tests that re-enumerate every store validation branch:**
- `test_replace_tool` (line 865) repeats empty / identical / not-found / text-not-found, all already asserted at store level in `test_replace` + `test_validation_raises_typed_error`.
- `test_topic_edit_tool` (line 934) repeats six validation branches already covered by `test_update_entry_topic`.

The design contract is that the tool surfaces store messages *verbatim*, which is worth one proof each — not six. Trim each to: happy path + updated-state check + 2 representative error branches (one raised by the store verbatim, one pre-validated by the tool). The full branch sweep stays at the store layer where it belongs.

---

## 3. Tests to KEEP

These are the load-bearing tests — distinctive edge cases or critical behaviour that must not be dropped:

- `test_search_page_pages_by_timestamp_with_backdated_rows` (564) — ordering by `created_epoch_ms`, not id; a later id carrying an earlier timestamp would break an id-ordered implementation.
- `test_search_page_pages_inside_tied_milliseconds` (610) — id cursor resumes inside same-millisecond ties; no skip, no repeat.
- `test_search_page_rejects_stale_cursor` (640) — deleted/foreign cursor id fails loudly instead of guessing a position.
- `test_search_page_start_positions_then_cursor_resumes` (660) — inclusive start bound + cursor hand-off interaction.
- `test_search_page_hints` (496) — `has_more` / `next_cursor` in every page state (middle, first, last, full, empty, topic-filtered).
- `test_search_sorts_by_epoch_not_text` (705) — offset-vs-Z lexical trap; sharp, cheap.
- `test_search_text_semantics` (359) — FTS AND-semantics, phrase, prefix, case, operator-injection safety.
- `test_delete_removes_fts_record` (723) — external-content FTS5 trigger consistency.
- `test_user_isolation` (202) and `test_cross_user_tools_see_nothing` (848) — the security boundary, at both layers.
- `test_ids_never_reused` (186) — AUTOINCREMENT contract.
- `test_add_and_get_round_trip` (122), `test_get_entries_batch` (143) — core read/write behaviour incl. requested-order and partial-miss semantics.
- `test_replace` (226), `test_replace_multiple_matches` (289), `test_update_entry_topic` (315) — outcome enum, first-occurrence-only, unchanged-after-ambiguity, timestamp preservation.
- `test_search_start_combines_with_query_filter` (470) — start positions the scan; the text filter still applies within it.
- `test_search_topic_filter_case_insensitive` (396) — lowercase storage + NOCASE match.
- `test_schema_script_runs_once_per_database` (49) — keep the body minus the spy block (see deletions): the executescript counter is real regression protection for the shared-memo optimisation.
- `test_db_path_resolved_at_call_time` (88) — config resolved at connect, not import.
- `test_add_uses_configured_timezone` (785) — user-timezone timestamp generation.
- `test_search_tool_validation_and_format` (976), `test_search_tool_start_filter_with_offsets` (1012), `test_query_tool_paging` (1041) — start parsing, cursor/start mutual exclusion, offset filtering, hint formatting and resumption.
- `test_get_and_delete` (828), `test_get_multiple_ids` (892), `test_edit_tool_multiple_matches` (919), `test_list_topics_tool` (1151), `test_add_validation` (811) — tool-layer contracts; cheap and distinct enough.

---

## 4. Summary

| Metric | Count |
|---|---|
| Tests in area | 42 (25 store + 17 tool) |
| Recommend deleting entirely | 1 (`test_query_tool_cursor_resumes_inside_tied_milliseconds`) |
| Recommend deleting assertions within tests | 5 blocks (spy/`_schema_ready`, dead "topics vanish" block, duplicate limit checks ×2 sets, `func.__name__` trivia) |
| Consolidation groups | 7 (A–G above), touching ~14 tests |
| Net effect | ~38–39 tests with *higher* effective coverage: the dead "topics vanish" code path actually gets tested once fixed |

No important business logic, error handling, or edge case is recommended for removal; all deletions are duplicates, dead assertions, or private-state checks.
