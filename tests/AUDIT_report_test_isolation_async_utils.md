# Audit Report: test_test_isolation.py & test_async_utils.py

## 1. Tests to DELETE

### tests/test_test_isolation.py

| File:Line | Test | Reason |
|-----------|------|--------|
| `tests/test_test_isolation.py:93-97` | `TestOneIsolatedDatabasePerRun.test_temp_dir_is_removed_at_exit` | **Implementation-detail.** Mocks `shutil.rmtree` and calls the private `bootstrap._cleanup()` directly. It does not test that the temp dir is actually removed at interpreter exit — it tests that `_cleanup` calls `rmtree`. The atexit registration is already implicitly covered by the suite running without leftover dirs. |
| `tests/test_test_isolation.py:82-91` | `TestOneIsolatedDatabasePerRun.test_bootstrap_db_is_a_single_file_in_its_own_temp_dir` | **Brittle / low-value.** The `extra == []` assertion (lines 89-91) checks that the temp dir contains only `store.db` plus SQLite sidecars. This is an implementation-detail check on directory contents — if SQLite adds a new sidecar file type (e.g., a new WAL/journal variant), this test breaks. The meaningful parts (file exists, dir name prefix) are already implicitly covered by `test_test_db_itself_still_opens` (which opens the DB and checks tables exist). |

### tests/test_async_utils.py

**None.** All 8 tests provide meaningful assurance.

---

## 2. Tests to CONSOLIDATE

### Group A: Guard tests — connect API parametrisation (test_test_isolation.py)

**Tests to merge:**
- `TestRealDatabaseGuard.test_aiosqlite_open_of_repo_db_raises` (lines 102-110)
- `TestRealDatabaseGuard.test_sqlite3_open_of_repo_db_raises` (lines 112-114)
- `TestRealDatabaseGuard.test_file_uri_spelling_of_repo_db_raises` (lines 127-133)
- `TestRealDatabaseGuard.test_file_uri_spelling_of_repo_db_raises_for_aiosqlite` (lines 135-146)

**Suggested consolidation:** Parametrise over the connect callable (`sqlite3.connect` vs `aiosqlite.connect`) and the path spelling (direct path vs `file:` URI). This reduces 4 tests to 1 parametrised test with ~4-6 cases. The two URI tests already share identical structure (lines 129-133 vs 140-143) — they only differ in which connect function they call.

**Why not delete instead:** These test genuinely different connect APIs and URI spellings. The parametrisation preserves coverage while removing duplication.

### Group B: "Non-forbidden paths still open" (test_test_isolation.py)

**Tests to merge:**
- `TestRealDatabaseGuard.test_test_db_itself_still_opens` (lines 153-166)
- `TestRealDatabaseGuard.test_unrelated_paths_still_open` (lines 168-175)

**Suggested consolidation:** Both verify the guard doesn't over-block. `test_test_db_itself_still_opens` is the stronger test (it checks that tables exist in the isolated DB). `test_unrelated_paths_still_open` is a weaker duplicate (it just creates a table in an arbitrary temp file). Merge into a single test that covers both scenarios — the isolated DB (with table checks) and an unrelated temp path.

**Why not delete instead:** The "unrelated path" case does add a tiny bit of extra coverage (a path completely outside the bootstrap), but it's marginal. Merging is cleaner than deleting.

---

## 3. Tests to KEEP

### tests/test_test_isolation.py

| Test | Why KEEP |
|------|----------|
| `TestOneIsolatedDatabasePerRun.test_schema_created_once_for_the_whole_run` (lines 23-79) | **High-value.** Pins the core invariant of the bootstrap: DDL runs exactly once per table per process, and subsequent connects reuse it. This is the most behaviour-rich test in the file. It's complex and implementation-detail heavy, but the behaviour it guards is critical — without it, every store operation would re-run DDL. |
| `TestRealDatabaseGuard.test_relative_path_that_resolves_to_repo_db_raises` (lines 116-125) | **Important edge case.** Verifies the guard uses `realpath` (not string comparison) so a relative `'store.db'` from the repo root is caught. The CWD manipulation makes this a genuine test of path resolution, not just a hardcoded path check. |
| `TestRealDatabaseGuard.test_old_integration_test_store_db_raises` (lines 148-151) | **Important edge case.** Verifies the old integration-tier `test_store.db` path is also guarded. Without this, a test using the old path would silently read dev data. |

### tests/test_async_utils.py

| Test | Why KEEP |
|------|----------|
| `TestAtomicDecorator.test_decorator_runs_the_transaction_and_returns_its_result` (lines 16-25) | **Baseline contract.** Verifies the decorator runs the function and propagates its return value. Basic but necessary — the return-value contract is the primary API guarantee. |
| `TestAtomicDecorator.test_decorator_cancellation_finishes_the_transaction` (lines 27-45) | **Critical behaviour.** This is THE core guarantee of the `@atomic` decorator: cancellation waits for the shielded transaction to complete, then re-raises. The whole module exists for this behaviour. |
| `TestAtomicDecorator.test_decorator_transaction_failure_propagates` (lines 47-53) | **Error contract.** Verifies exceptions from the decorated function propagate unchanged. The decorator adds no error handling, so this confirms the wrapper is transparent to failures. |
| `TestAtomicCall.test_runs_action_and_callback_in_order` (lines 58-69) | **Baseline contract.** Verifies action-then-callback ordering. Basic but establishes the call sequence contract. |
| `TestAtomicCall.test_sync_callback_commits_the_result` (lines 71-79) | **Branch coverage.** Tests the synchronous callback path (`inspect.isawaitable` returns `False`). This is a distinct code branch from the async callback path; without this test, the sync path is untested. |
| `TestAtomicCall.test_cancellation_during_action_finishes_the_commit` (lines 81-104) | **Critical behaviour.** Cancellation during the action phase — the shielded transaction (action + callback) must complete before CancelledError propagates. Distinct phase from the callback-cancellation test. |
| `TestAtomicCall.test_cancellation_during_callback_finishes_the_commit` (lines 106-126) | **Critical behaviour.** Cancellation during the callback phase — same guarantee as the action-cancellation test but for a different phase. The structural similarity to the action test is intentional: both are needed because the cancellation can land in either phase. |
| `TestAtomicCall.test_commit_failure_is_not_hidden_by_cancellation` (lines 128-147) | **Error-priority edge case.** When the callback itself fails while cancellation is unwinding, the concrete failure (ValueError) must surface, not a bare CancelledError. This tests the interaction between `asyncio.shield` and exception priority — a subtle and important correctness detail. |

---

## 4. Summary Counts

| File | Total Tests | Delete | Consolidate (net reduction) | Keep |
|------|-------------|--------|-----------------------------|------|
| `tests/test_test_isolation.py` | 11 | 2 | 3 (6→3) | 6 |
| `tests/test_async_utils.py` | 8 | 0 | 0 | 8 |
| **Combined** | **19** | **2** | **3** | **16** |

After recommended changes:
- `test_test_isolation.py` would go from 11 → **6 tests** (2 deleted, 3 pairs consolidated into 1 parametrised test each)
- `test_async_utils.py` stays at **8 tests** (all high-value, no changes needed)

### Rationale for asymmetric treatment

The two files test fundamentally different kinds of behaviour:

- **`test_test_isolation.py`** tests infrastructure/glue code (bootstrap harness, connect guards). Many of its tests are variations on "does this path spelling get caught?" — naturally repetitive and amenable to parametrisation. The guard mechanism itself is a pure function (`is_forbidden_path` → `_realpath` comparison), so testing it is inherently about enumerating input spellings, which is what parametrisation is for.

- **`test_async_utils.py`** tests a concurrency primitive (`atomic`/`atomic_call`) where each test exercises a distinct state-space quadrant: normal flow, action-phase cancellation, callback-phase cancellation, callback-failure-during-cancellation, sync vs async callbacks. The structural similarity between the cancellation tests is a feature — they pin the same invariant across different event-loop phases. Merging them would obscure the phase distinction.