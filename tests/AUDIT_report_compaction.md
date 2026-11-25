# Audit Report: tests/test_compaction.py

**Auditor:** Subagent (compaction area)  
**Date:** 2026-11-25  
**Scope:** 28 tests in `tests/test_compaction.py` + source `src/compaction.py`

---

## 1. Tests to DELETE

### A. Tests of imported utility modules (not compaction logic)

These tests belong in `test_transcript.py` / `test_tokens.py`, which do not exist. They test functions that are *imported* by `compaction.py`, not the compaction logic itself. Their absence from compaction tests is fine; their presence here is misplaced.

| # | Test | Line | Reason |
|---|------|------|--------|
| 1 | `TestTokenCount.test_quarter_token_per_character` | 37 | Trivial: tests `len(text) * 0.25` arithmetic, not compaction |
| 2 | `TestTokenCount.test_estimates_text_and_images` | 41 | Anaemic: tests `tokens.estimate_conversation_tokens` (imported), not compaction |
| 3 | `TestTokenCount.test_estimates_tool_calls` | 53 | Anaemic: tests `tokens.estimate_message_tokens` (imported), not compaction |
| 4 | `TestMessageToText.test_system_messages_are_left_out` | 107 | Redundant: tests `transcript.message_to_text` (imported), not compaction |
| 5 | `TestMessageToText.test_images_become_placeholder_text` | 112 | Redundant: tests `transcript.message_to_text` (imported), not compaction |
| 6 | `TestMessageToText.test_assistant_tool_calls` | 123 | Redundant: tests `transcript.message_to_text` (imported), not compaction |
| 7 | `TestMessageToText.test_tool_output_is_truncated` | 136 | Redundant: tests `transcript.message_to_text` (imported), not compaction |
| 8 | `TestMessageToText.test_thinking_blocks_are_left_out` | 143 | Redundant: tests `transcript.message_to_text` (imported), not compaction |

> **Note:** Deleting these leaves `transcript.py` and `tokens.py` with zero direct unit tests. A separate audit should create `test_transcript.py` and `test_tokens.py` to cover them. These 8 tests should be *moved*, not discarded.

### B. Tests of constant strings / math guarantees

| # | Test | Line | Reason |
|---|------|------|--------|
| 9 | `TestSummaryPrompt.test_omission_instruction_is_one_complete_sentence` | 78 | Trivial: asserts a hardcoded string is `in` a constant |
| 10 | `TestSummaryPrompt.test_checkpoint_instruction_is_one_complete_sentence` | 81 | Trivial: asserts a hardcoded string is `in` a constant |
| 11 | `TestSummaryPrompt.test_omission_instruction_precedes_the_checkpoint_instruction` | 84 | Trivial: asserts ordering of two substrings within a constant |
| 12 | `TestSummaryPrompt.test_no_misspelled_transcript` | 91 | Trivial: asserts absence of a typo in a constant; no behavioural assurance |
| 13 | `TestThresholds.test_min_compact_tokens_exceeds_the_default_keep_budget` | 100 | Trivial: asserts `30_000 > 15_000`, guaranteed by `MIN_COMPACT_TOKENS = 2 * DEFAULT_KEEP_TOKENS` |

### C. Tests fully covered by the `test_table` parametrised cases

The `TestBoundaryRule.test_table` method (line 244) runs 6 `CASES` that already exercise these scenarios with stronger assertions (budget compliance, tool-pair integrity, summary-fraction floor, marker placement).

| # | Test | Line | Reason |
|---|------|------|--------|
| 14 | `TestSplitForCompaction.test_tool_call_batch_never_splits_across_the_boundary` | 321 | Redundant: `CASE 'tool-call batch'` (line 200) covers the same scenario with stronger assertions |
| 15 | `TestSplitForCompaction.test_single_exchange_is_summarized_with_nothing_kept` | 344 | Redundant: `CASE 'single exchange'` (line 216) covers the same scenario |
| 16 | `TestSplitForCompaction.test_newest_messages_are_kept_verbatim` | 310 | Redundant: `CASE 'user-first'` (line 191) covers the same boundary placement; the extra `tokens_before > MIN_COMPACT_TOKENS` assertion is also in `test_compacts_and_rebuilds_conversation` (line 371) |

### D. Duplicate error-path test

| # | Test | Line | Reason |
|---|------|------|--------|
| 17 | `TestCompactContext.test_keep_fraction_refusal_also_names_the_budget` | 456 | Redundant: tests the same `CompactionError` path as `test_custom_keep_budget_refusal_names_the_budget` (line 428) with only a different keep-budget number; no new branch or assertion shape |

---

## 2. Tests to CONSOLIDATE

No consolidation is required after the deletions above. The two refusal tests (#16–17 in the original file) are the only candidates for merging, and deleting the duplicate (`test_keep_fraction_refusal_also_names_the_budget`) is the cleaner fix than parametrising. If a future auditor wants to parametrise, `test_custom_keep_budget_refusal_names_the_budget` can become a `subTest` loop over `(keep_tokens, transcript)` pairs — but that is a maintenance improvement, not a necessity.

---

## 3. Tests to KEEP

These 11 tests provide meaningful, non-redundant assurance of compaction behaviour:

| # | Test | Line | Why it stays |
|---|------|------|--------------|
| 1 | `TestBoundaryRule.test_table` | 244 | **High-value.** Parametrised table over 6 conversation shapes; verifies budget compliance, tool-pair integrity, summary-fraction floor, refusal, and marker placement in one method. This is the spine of the boundary-rule test suite. |
| 2 | `TestBoundaryRule.test_something_is_always_summarized` | 282 | **High-value edge case.** Guards against the boundary landing before the first entry (empty summary request that still costs a model call). Not covered by `test_table` because no CASE exercises a keep budget that would produce an empty older side. |
| 3 | `TestSplitForCompaction.test_small_conversation_is_not_split` | 294 | Valuable: pins the `MIN_COMPACT_TOKENS` size floor (transcript < 30k → no compaction). |
| 4 | `TestSplitForCompaction.test_context_that_fits_in_keep_tokens_is_not_split` | 302 | Valuable: pins the keep-budget check (transcript ≤ keep_tokens → no compaction). |
| 5 | `TestSplitForCompaction.test_nothing_to_transcribe_is_not_split` | 306 | Valuable: pins the empty-entries guard (system-only conversation → `None`). |
| 6 | `TestCompactContext.test_compacts_and_rebuilds_conversation` | 359 | **High-value integration test.** Exercises the full `compact_context` flow: LLM call, summary capture, checkpoint format, conversation rebuild, OpenAI-contract validity, and token-savings report. |
| 7 | `TestCompactContext.test_compacts_without_recent_context` | 394 | Valuable: exercises the `split.recent_text == ''` branch and verifies the checkpoint omits `<recent-context>`. |
| 8 | `TestCompactContext.test_small_context_is_left_untouched` | 411 | Valuable: pins the `CompactionError("Context too small to compact")` path and verifies the conversation is left unchanged. |
| 9 | `TestCompactContext.test_custom_keep_budget_refusal_names_the_budget` | 428 | Valuable: pins the trivial-summary refusal path and the specific error-message format that names the keep budget. |
| 10 | `TestCompactContext.test_empty_summary_fails_and_keeps_conversation` | 483 | Valuable: pins the empty-summary error path (`CompactionError`) and verifies conversation integrity on failure. |
| 11 | `TestCompactContext.test_on_start_runs_inline_before_the_summary_call` | 497 | Valuable: pins the `on_start` callback ordering (before `run_llm`) and the inline-execution guarantee (no internal task spawn). |

---

## 4. Summary Counts

| Metric | Count |
|--------|-------|
| Total tests in file | **28** |
| Recommended for deletion | **17** |
| Recommended for consolidation | **0** (all candidates already covered) |
| Recommended to keep | **11** |

### Post-audit test inventory

After cleanup, the file would contain:

- `TestBoundaryRule` — 2 tests (`test_table`, `test_something_is_always_summarized`)
- `TestSplitForCompaction` — 3 tests (size floor, keep budget, empty entries)
- `TestCompactContext` — 6 tests (full flow, no-recent-context, small-context error, budget-refusal error, empty-summary error, on_start ordering)

The 8 `message_to_text` / `token_count` tests should be relocated to new `test_transcript.py` and `test_tokens.py` files rather than deleted outright — they are the only direct coverage those modules currently receive.