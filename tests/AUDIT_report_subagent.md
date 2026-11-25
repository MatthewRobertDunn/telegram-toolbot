# Subagent Test Audit Report

## 1. Tests to DELETE

### `test_send_to_parent_is_subagent_only` — `tests/test_subagent.py:291`
**Reason: Redundant.** This test checks that `subagent_send_to_parent` is absent from the PARENT surface and present in the SUBAGENT surface. Both assertions are already covered by `test_parent_tools_exclude_send_to_parent` (line 129: `assertNotIn('subagent_send_to_parent', names)`) and `test_subagent_tools_have_no_subagent_controls` (line 133: `assertIn('subagent_send_to_parent', names)`). No additional coverage.

### `test_each_role_executes_its_own_implementation` — `tests/test_subagent.py:158`
**Reason: Anaemic / implementation-detail.** This test asserts that `parent['read_image'] is not subagent['read_image']` (and same for `generate_image`). It checks object identity of function objects rather than any observable behaviour. The actual behavioural difference (subagent tool calls stay silent, parent tool calls announce) is already covered by `test_subagent_turns_stay_silent` and `test_parent_turns_announce`. Testing `is not` on two function objects tells the reader nothing about what differs or why.

---

## 2. Tests to CONSOLIDATE

### Pair A: Inbox-empty wakeup no-op tests
**Tests:** `test_redundant_wakeup_jobs_never_send` (line 463) + `test_empty_inbox_schedules_nothing` (line 506)

Both tests exercise the same code path: `_wakeup_parent` calls `mail_pending()`, which returns `False`, so no job is submitted. The only difference is *why* `mail_pending()` is False:
- `test_redundant_wakeup_jobs_never_send`: mail was put then drained before the job ran.
- `test_empty_inbox_schedules_nothing`: inbox was never populated.

**Suggested merge:** Parametrise into one test with two scenarios (drained-then-empty vs. never-populated), asserting `run_llm_mock.assert_not_awaited()` and `submitted == []` in both. Both scenarios are meaningful edge cases of the wakeup protocol and belong together.

### Pair B: Silent-receipt tests
**Tests:** `test_silent_when_not_announced` (line 828) + `test_silent_when_show_tool_output_off` (line 837)

Both tests assert the same thing: `send.assert_not_awaited()` and `drained == 1`. They differ only in *why* the receipt is suppressed:
- `test_silent_when_not_announced`: `announce=False` (subagent context).
- `test_silent_when_show_tool_output_off`: `show_tool_output=False` (parent context with display off).

**Suggested merge:** Parametrise into one test with `(announce, show_tool_output)` combinations, asserting the receipt is suppressed in both cases. The two conditions gate different branches in `_drain_inbox` but the observable outcome is identical.

---

## 3. Tests to KEEP (high-value)

These tests protect important business logic, edge cases, or error handling and must stay:

| Test | Line | Why it matters |
|---|---|---|
| `test_inbox_messages_land_in_conversation` | 74 | End-to-end inbox → conversation pipeline; verifies structured mail is formatted and injected as a user turn. |
| `test_no_inbox_means_no_user_turn` | 92 | Negative-path: confirms no spurious user turn when no inbox is given. |
| `test_parent_tools_exclude_send_to_parent` | 120 | Tool-surface security: parent must not be able to call subagent-only tools. |
| `test_subagent_tools_have_no_subagent_controls` | 131 | Tool-surface security: subagent must not be able to call parent-only controls or chat-posting tools. |
| `test_surface_pairs_definitions_with_implementations` | 150 | Registry invariant: definition names ↔ function keys. Catches registration bugs that would cause "Unknown tool" or unexecutable tools. |
| `test_run_llm_gets_an_isolated_silent_context` | 170 | Context isolation: subagent gets an `AgentContext` (no `messaging`, no chat access), shares conversation, isolated `user_data`. |
| `test_send_to_parent_uses_the_parent_context` | 197 | Critical routing: subagent mail goes through the *parent* context, not the silent subagent context. Prevents mail being written to the wrong conversation. |
| `test_start_list_send_stop` | 222 | Full lifecycle integration: start → list → steer → stop → double-stop → unknown-id errors. |
| `test_start_rejects_empty_description` | 260 | Input validation: empty/whitespace descriptions rejected, no state pollution. |
| `test_send_to_parent_queues_for_main` | 267 | Notification path: verifies the message is queued to the main inbox with correct sender identity. |
| `test_send_to_parent_without_record_is_an_error` | 286 | Guard clause: calling `send_to_parent` outside a subagent context returns an error. |
| `test_cancel_all` | 299 | Reset behaviour: `cancel_all_subagents` clears all records. |
| `test_explicit_model_overrides_the_setting` | 339 | Model override: explicit model wins. |
| `test_omitted_model_uses_the_setting` | 344 | Model fallback: omitted model uses the configured setting. |
| `test_blank_model_is_no_override` | 348 | Edge case: blank/whitespace model is treated as "no override" (a model can pass `""`). |
| `test_idle_mail_wakes_the_parent_and_drains_it` | 400 | Wakeup protocol: idle parent receives mail, wakes, drains inbox. |
| `test_busy_mail_queues_behind_the_current_job` | 414 | Wakeup protocol: mail arriving while parent is busy is queued and drained after the current job. |
| `test_five_notifications_drain_without_loss` | 445 | Wakeup protocol: 5 concurrent notifications all drain exactly once; redundant wakeups no-op. |
| `test_stale_epoch_skips_entirely` | 486 | Epoch guard: a wakeup scheduled before a `/start` reset is skipped entirely. |
| `test_cancel_all_clears_the_inbox` | 518 | Reset behaviour: `cancel_all` drops stale inbox mail so it cannot leak into a fresh conversation. |
| `test_late_steer_to_finished_subagent_is_dropped` | 534 | Edge case: steering a finished subagent is rejected; no wakeup is scheduled; nothing reaches the parent inbox. |
| `test_mail_during_final_round_waits_for_next_turn` | 552 | Cutoff semantics: mail arriving mid-flight stays queued for the next turn (cutoff is at request time). |
| `test_max_subagents_cap` | 584 | Resource limit: `MAX_SUBAGENTS` cap is enforced. |
| `test_finished_records_pruned` | 602 | Memory hygiene: finished records are pruned when `MAX_RECORDS` is reached. |
| `test_failed_subagent_notifies` | 625 | Error handling: a failed subagent notifies the parent with the error detail; nothing goes straight to Telegram. |
| `test_format_duration` | 652 | Pure-function correctness for duration formatting used in `subagent_list` output. |
| `test_running_shows_elapsed` | 658 | Display: running subagent shows elapsed time. |
| `test_finished_shows_total` | 671 | Display: finished subagent shows total run time. |
| `test_subagent_turns_stay_silent` | 733 | Announce gate: subagent turns (`announce=False`) produce no Telegram posts. |
| `test_parent_turns_announce` | 738 | Announce gate: parent turns (`announce=True`) post tool-call announcements. |
| `test_receipt_posted_and_linked` | 767 | Receipt correctness: summary names sender + state, body is full result, Telegram IDs linked to the injected turn for `/undo`. |
| `test_receipt_carries_full_result` | 785 | No truncation: 2000-char result is preserved in both the Telegram post and the conversation turn. |
| `test_receipt_empty_body_falls_back_to_state` | 796 | Edge case: empty result body falls back to the state name in parentheses. |
| `test_receipt_parent_note_summary` | 805 | Display: parent-originated notes get a "Parent note" summary, not a subagent one. |
| `test_receipt_running_omits_state_suffix` | 814 | Display: running-state receipts omit the terminal-state suffix (sender is still alive). |
| `test_rejects_unstructured_puts` | 823 | Type safety: `SubagentInbox.put_nowait` rejects non-`SubagentMessage` values. |
| `test_undo_skips_agent_turns` | 846 | `/undo` behaviour: agent-message turns are not deletable; only human turns are removed. |
| `test_empty` (handlers) | 29 | Handler: empty subagent list returns "No subagents". |
| `test_lists_running_subagent` (handlers) | 34 | Handler: running subagent shows id, description, and status in the `/subagent_list` output. |

---

## 4. Borderline Tests (keep, but note low value)

These tests are kept because they do verify something meaningful, but they are lower-value than the tests above and could be dropped in a future pass if coverage elsewhere expands:

| Test | Line | Note |
|---|---|---|
| `test_no_tools_by_default` | 102 | Verifies that a default `run_llm` call (no tools) does not send a `tools` key in the request. Thin but confirms end-to-end default behaviour. |
| `test_model_is_declared_optional` | 354 | Tests the tool schema metadata (`model` not in `required`). The behavioural consequence (omitted model → setting) is already covered by `test_omitted_model_uses_the_setting`. |
| `test_rejects_unstructured_puts` | 823 | Tests the type guard on `SubagentInbox.put_nowait`. Simple but protects a real invariant. |

---

## 5. Summary Counts

| Metric | Count |
|---|---|
| Total tests in `test_subagent.py` | 45 |
| Total tests in `test_subagent_handlers.py` | 2 |
| **Grand total** | **47** |
| Recommended for **deletion** | 2 |
| Recommended for **consolidation** (4 tests → 2) | 4 |
| **Net reduction** | **4 tests** |
| Recommended final count | **43** |

### Deletion rationale summary
- **2 tests** are redundant (`test_send_to_parent_is_subagent_only`) or anaemic/implementation-detail (`test_each_role_executes_its_own_implementation`). Neither protects business logic, edge cases, or error handling — their coverage is fully supplied by other tests in the file.

### Consolidation rationale summary
- **2 pairs** (4 tests) test the same observable outcome with marginally different setups. Merging them into 2 parametrised tests removes duplication without losing coverage.