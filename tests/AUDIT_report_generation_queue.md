# Audit Report: test_generation_queue.py

## 1. Tests to DELETE

### `test_submit_task_reraises_the_job_error` — `tests/test_generation_queue.py:165`
**Redundant.** `test_worker_survives_a_failing_job` (line 148) already verifies that a failing job's error is re-raised to the submitter, while also testing the more important property that the worker survives and processes the next queued job. The only difference is the exception type (`ValueError` vs `RuntimeError`), which adds no coverage. Delete this test.

### `test_a_thunk_is_invoked_once_when_the_worker_dequeues_it` — `tests/test_generation_queue.py:174`
**Anaemic / implementation-detail.** It verifies that a thunk (zero-arg callable) is called once, which is trivially guaranteed by the source: `_run_item` → `_supervise` → `item.job()`. The important correctness property of the thunk pattern — that a *dropped* item's thunk never runs, leaving no un-awaited coroutine — is already covered by `test_abandon_releases_a_dropped_item_without_running_it` (line 191) and `test_worker_cancellation_releases_still_queued_items` (line 208). This test adds nothing new. Delete it.

## 2. Tests to CONSOLIDATE

### Group: `TestMessagingReturnShapes` (3 tests → 1)
**Tests affected:**
- `test_preformatted_returns_ids` — line 373
- `test_image_returns_id` — line 378
- `test_document_returns_id` — line 382

All three test the same contract: chat-posting helpers return `list[int]` of message IDs so callers can link them to conversation turns (`/undo`, `/retry`). They differ only in which method is exercised and which mock return value is checked. Merge into a single parametrized test:

```python
async def test_posting_helpers_return_message_ids(self):
    bot = self._bot()
    ctx = mock_ctx(user_id=5, bot=bot)
    cases = [
        (ctx.messaging.send_preformatted_message, ('hello',), [7]),
        (ctx.messaging.send_image, (b'png',), [11]),
        (ctx.messaging.send_document, (b'pdf', 'f.pdf'), [12]),
    ]
    for method, args, expected in cases:
        self.assertEqual(await method(*args), expected)
```

**Net reduction:** 2 tests.

### Soft candidate (not requiring consolidation): `TestSplitAtLinebreak` (5 tests)
The five tests (`test_degenerate_limit_makes_progress`, `test_leading_newline_produces_no_empty_chunk`, `test_splits_without_newlines`, `test_prefers_line_breaks`, `test_short_and_empty_text`) could be merged into one table-driven test. However, their descriptive names add readability for a pure-function test suite, and each tests a distinct edge case. Recommend keeping as-is unless the file grows substantially further.

## 3. Tests to KEEP

All remaining 22 tests protect important business logic, edge cases, or behavioural contracts:

| Test | Line | Why it stays |
|---|---|---|
| `test_jobs_run_one_at_a_time_in_order` | 52 | Serialization ordering + worker lifecycle (transient worker cleanup) |
| `test_submit_generation_queues_a_reply_unit` | 69 | Full reply pipeline: user+assistant turns, CUSTOM metadata with telegram_message_ids, generation_task cleared |
| `test_a_finished_generation_is_not_dangling_thinking` | 84 | Critical: a completed reply's thinking must survive (it has its answer after it) |
| `test_no_content_appends_no_user_turn` | 112 | content=None guard: bare generation (/regenerate) doesn't add a user turn |
| `test_stopped_generation_records_no_reply` | 122 | Stop behaviour: no reply sent, generation_task cleared, user turn committed |
| `test_worker_survives_a_failing_job` | 148 | Worker resilience: failing job doesn't kill the worker; error re-raised to submitter |
| `test_abandon_releases_a_dropped_item_without_running_it` | 191 | Drop path: `WorkItem.abandon()` releases without running the job thunk |
| `test_worker_cancellation_releases_still_queued_items` | 208 | Worker cancellation cleanup: queued items released, thunks never run, queue drained |
| `test_stop_deletes_orphan_thinking` | 248 | Dangling thinking cleanup on aborted reply: history entry and Telegram post removed |
| `test_stop_drops_queued_items_and_reports` | 285 | Stop command with queued items: all dropped, done events set, user notified with count |
| `test_stop_with_idle_bot_says_nothing_running` | 300 | Stop command on idle bot: user told nothing is running |
| `test_returns_the_coroutine_result` | 316 | `run_with_typing` returns wrapped coroutine's result; typing action sent |
| `test_wrapped_exception_propagates` | 330 | Exception propagation through `run_with_typing`; no leftover tasks |
| `test_cancellation_propagates` | 343 | Cancellation propagation through `run_with_typing`; no leftover tasks |
| `test_degenerate_limit_makes_progress` | 390 | `_split_at_linebreak` edge case: limit ≤ 0 returns text whole (prevents infinite loop) |
| `test_leading_newline_produces_no_empty_chunk` | 395 | `_split_at_linebreak`: leading newline doesn't produce empty chunk |
| `test_splits_without_newlines` | 399 | `_split_at_linebreak`: split at char limit when no newline exists |
| `test_prefers_line_breaks` | 403 | `_split_at_linebreak`: prefers line breaks over char limit |
| `test_short_and_empty_text` | 405 | `_split_at_linebreak`: short text and empty text boundary cases |
| `test_stop_after_drain_keeps_the_turn_answerable` | 420 | Inbox drain ordering: turn committed before stop, next request answers it |
| `test_stop_before_drain_keeps_mail_for_the_next_run` | 443 | Inbox drain ordering: mail not lost if stop lands before drain; next run processes it |
| `test_stop_mid_run_after_drain_keeps_the_turn` | 474 | Inbox drain ordering: cancellation mid-run after drain keeps the committed turn |

## 4. Summary Counts

| Metric | Count |
|---|---|
| Total tests in area | 27 |
| Recommended for deletion | 2 |
| Recommended for consolidation | 3 (→ 1, net −2) |
| **Tests after changes** | **23** |

The two deletions are low-value redundancy. The consolidation is a maintainability win with no coverage loss. All 22 remaining tests provide meaningful assurance of business logic, error handling, or edge-case behaviour and should be kept.