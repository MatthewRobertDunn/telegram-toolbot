# Audit Report: test_supervision.py, test_think_split_flag.py, test_system_prompt_skills.py

> **Source-mapping correction for the orchestrator.** The assigned source hints
> don't match what these files actually exercise:
> - `test_supervision.py` does **not** test `src/handlers/middleware.py`; it tests
>   `submit_task`/`_supervise` in **`src/handlers/generation_queue.py`** (the
>   `/stop` supervision lives there, not in middleware). `middleware.py` itself
>   is covered by `test_thread_settings.py`, `test_timezone.py` and
>   `test_command_handlers.py`.
> - `test_think_split_flag.py` does **not** test `src/messages.py`; it tests
>   `_split_thinking`/`consume_stream` in **`src/handlers/consume_stream.py`**
>   plus `_draft_id` rotation in **`src/handlers/draft_streamer.py`**.
> - `test_system_prompt_skills.py` tests the `ChatUpdateContext.system_prompt`
>   property in **`src/chat_update_context.py`** (lines 230–251). `src/system_prompt`
>   is a data file (the prompt text), not code under test.

## 1. Tests to DELETE

| Test | Location | Reason |
|---|---|---|
| `test_a_fresh_job_wraps_the_thunk_exactly_once` | `tests/test_supervision.py:130` | **Anaemic + cross-file duplicate.** Verbatim copy (same body, same comments, same assert) of `test_a_thunk_is_invoked_once_when_the_worker_dequeues_it` (`tests/test_generation_queue.py:174`), which the parallel `AUDIT_report_generation_queue.md` already recommends deleting. Asserting "the thunk ran once on a happy path" guards nothing the two drop-path tests (`abandon`, worker cancellation) don't already protect more meaningfully. Delete here and in `test_generation_queue.py` (coordinated). |

That is the only clear deletion. The remaining 10 tests each pin a distinct
behavioural contract.

## 2. Tests to CONSOLIDATE

| Group | Tests | Suggestion |
|---|---|---|
| Optional: `/stop` on the tracked task | `test_running_job_is_tracked_as_the_inner_task` (`test_supervision.py:61`) + `test_stop_posts_the_stopped_message` (`test_supervision.py:81`) | They differ only in `stopped_message=None` vs `'Undo stopped.'` and could be merged into one parametrised test (expected sent-messages: `[]` vs `['Undo stopped.']`). **Low priority** — test 1 additionally asserts the tracked task is not the worker (`assertIsNot(tracked, ctx.generation_worker)`), which is the core `/stop` invariant, so merging costs some clarity for 1 test saved. Keep both if in doubt. |

No consolidation opportunities in the other two files: the think-split tests
already share a `_run` helper and `test_drafts_stream_raw_regardless` is already
loop-driven; the two skills tests are complementary positive/negative cases.

## 3. Tests to KEEP

### tests/test_supervision.py (keep 4 of 5)

| Test | Line | Why it stays |
|---|---|---|
| `test_running_job_is_tracked_as_the_inner_task` | 61 | Core safety invariant: `/stop` must cancel the inner (model-facing) task, never the queue worker awaiting it; also verifies tracking is cleared and no spurious message. |
| `test_stop_posts_the_stopped_message` | 81 | The user-visible contract of a stopped `/undo`-style job: the item's `stopped_message` is posted exactly once. |
| `test_worker_cancel_takes_the_work_down_without_a_message` | 98 | Distinct cancellation path (bot shutdown): work taken down, submitter woken, **nothing** posted — the asymmetry with `/stop` is exactly what must never drift. |
| `test_job_failure_is_reraised_and_clears_tracking` | 117 | Overlaps `test_worker_survives_a_failing_job` (`test_generation_queue.py:148`) on the re-raise, but uniquely asserts failure posts no stopped message and clears `ctx.generation_task`. Of the duplicated pair (`test_generation_queue.py:165` is the weaker twin), this is the one to keep. |

### tests/test_think_split_flag.py (keep 4 of 4)

| Test | Line | Why it stays |
|---|---|---|
| `test_strip_enabled` | 47 | Chain-of-thought must never reach Telegram: verifies the inline `(ViewGroup...)` block is excised from the final content. Also implicitly verifies the `reasoning_content` field wins over streamed inline reasoning (both present in the fixture). |
| `test_strip_disabled_keeps_tag_in_content` | 55 | The other half of the flag: with stripping off, content is passed through verbatim (close tag inside `<code>` untouched) and field reasoning still survives. |
| `test_drafts_stream_raw_regardless` | 62 | Verifies live drafts receive deltas verbatim under **both** flag values — a genuinely different guarantee (drafts ≠ final message) than the two content tests. |
| `test_new_round_regenerates_draft_id` | 72 | Only coverage of `new_round()`; guards the documented regression (reusing a draft id makes clients re-render the bubble at its old position mid-run). **Note:** it asserts on the private `_draft_id`; a higher-fidelity rewrite would capture `draft_id` from fake-bot `_post` payloads, but as-is it is cheap and correct. |

### tests/test_system_prompt_skills.py (keep 2 of 2)

| Test | Line | Why it stays |
|---|---|---|
| `test_lists_each_active_skill` | 20 | Pins the exact block format (`# Available skills` / `## name` / `- Path:` / description) that the model parses, plus base-prompt prefixing. Format drift here silently degrades skill use. |
| `test_omits_section_when_no_active_skills` | 35 | Negative case: no skills → prompt is exactly the base prompt, no dangling header. |

### Coverage gaps noticed (not test defects, for the orchestrator's radar)

- `ChatUpdateContext.system_prompt`'s **`user_profile` branch** (`<user_profile>` wrapping, `chat_update_context.py:234–241`) has no test in any file I could find. A one-test addition to `test_system_prompt_skills.py` would close it cheaply.
- `consume_stream`'s tool-call reassembly (id-keyed calls, index continuation, anon calls — `consume_stream.py:88–131`) is untested in this file; if no other file covers it, that is a significant gap for a streaming bot.

## 4. Summary Counts

| Metric | Count |
|---|---|
| Total tests in area | 11 (5 + 4 + 2) |
| Recommended for deletion | 1 |
| Recommended for consolidation | 2 (→ 1, optional, net −1) |
| **Tests after changes** | **9–10** |

The area is in good shape: the supervision tests encode the cancellation
semantics matrix (stop vs shutdown vs failure) that the module docstring
declares must never drift, and the think-split/skills tests each pin a distinct
observable contract. Only the thunk-once test is dead weight — and it is
deleted best in coordination with its twin in `test_generation_queue.py`.
