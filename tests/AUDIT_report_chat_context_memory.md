# Audit Report: chat_context & memory test suites

**Scope:** `tests/test_chat_context.py` (12 tests), `tests/test_memory.py` (11 tests)
**Source modules:** `src/chat_update_context.py`, `src/persistence/memory_store.py`, `src/persistence/sqlite_store.py`

---

## 1. Tests to DELETE

### test_chat_context.py

| Test | Line | Reason |
|------|------|--------|
| `test_real_user_settings_are_copied` | 113–121 | **Redundant.** Duplicates `test_settings_are_a_snapshot` (line 105). Both verify that `new_subagent_context` deep-copies the parent's effective settings and that mutating the child's settings does not affect the parent. The only difference is that this one routes settings through `user_data.settings` (the `effective is None` branch) instead of `thread_data.settings`, but that's an implementation detail of the `settings` property, not a distinct behavioural scenario. The deepcopy isolation is already proven. |
| `test_parent_coordination_state_is_unreachable` | 85–91 | **Anaemic.** The test name claims to verify the parent's coordination state is unreachable from the child, but it never sets any state on the parent. It simply asserts that a fresh `ThreadData` has `main_inbox=None`, `conversation_epoch=0`, `generation_queue=None`, `generation_task=None`, and `subagents={}` — i.e., it tests the dataclass field defaults. The actual isolation (child.thread_data is not parent.thread_data) is already covered by `test_user_data_is_its_own_object` (line 77). |
| `test_sandbox_read_tracking_starts_empty` | 134–138 | **Trivial.** Checks that `child.user_data.sandbox_read_files` is an empty set. The real isolation guarantee — that `child.user_data is not parent.user_data` — is already tested in `test_user_data_is_its_own_object` (line 80). This test merely verifies that a freshly constructed `UserData` has an empty `set` for that field, which is a `field(default_factory=set)` guarantee. No meaningful behaviour is asserted. |

### test_memory.py

| Test | Line | Reason |
|------|------|--------|
| `test_empty_keys` | 77–81 | **Trivial.** Tests two early-return guards: `get_values(1, [])` returns `{}`, and `delete_values(1, [])` is a no-op. These are language-level guard clauses (`if not keys: return {}` / `return`). Neither asserts any behaviour that would fail if the guard were removed — they just confirm the guard exists. Could be deleted without losing meaningful coverage. |
| `test_second_store_adopts_shared_schema_readiness` | 174–190 | **Redundant + implementation-detail.** The observable assertion — that a second store on the same file can read data written by the first — is already covered by `test_schema_runs_once_per_database_across_stores` (lines 127–132). The only new assertion is `self.assertTrue(second._schema_ready)` (line 188), which tests a private instance flag. That's an implementation detail, not observable behaviour. |

---

## 2. Tests to CONSOLIDATE

### Group A: `test_conversation_is_the_subagents_own` → `test_user_data_is_its_own_object`

**File:** `test_chat_context.py`

**Tests involved:**
- `test_user_data_is_its_own_object` (line 77)
- `test_conversation_is_the_subagents_own` (line 93)

**Why merge:** Both verify that a subagent's context does not share mutable state with the parent. `test_user_data_is_its_own_object` already asserts `child.thread_data is not parent.thread_data` and that mutating `child.thread_data.conversation_epoch` does not affect the parent. `test_conversation_is_the_subagents_own` adds three more assertions:
1. `child.conversation is own` — the passed-in Conversation is stored (trivial parameter passthrough).
2. Adding a message to the child's conversation does not appear in the parent's conversation — this is the real isolation check, and it's already implied by `child.thread_data is not parent.thread_data` (different ThreadData → different Conversation).
3. Reassigning `child.conversation` does not affect the parent — again a consequence of separate ThreadData.

The only non-redundant contribution is assertion #2, which can be folded into `test_user_data_is_its_own_object` as an additional mutation check on the conversation object itself.

**Suggested merged test:**
```python
def test_subagent_state_is_independent(self):
    parent = _parent()
    child = _child(parent)
    self.assertIsNot(child.user_data, parent.user_data)
    self.assertIsNot(child.thread_data, parent.thread_data)

    # Mutating child's thread_data does not affect parent
    child.thread_data.conversation_epoch = 99
    self.assertNotEqual(99, parent.thread_data.conversation_epoch)

    # The child's conversation is its own: writes stay local
    own = Conversation()
    child2 = parent.new_subagent_context(own)
    self.assertIs(child2.conversation, own)
    own.add_user('subagent thought')
    self.assertNotIn('subagent thought',
                      [m.get('content')
                       for m in parent.conversation.messages])
```

---

### Group B: `test_memory_databases_never_share_schema_readiness` → `test_schema_runs_once_per_database_across_stores`

**File:** `test_memory.py`

**Tests involved:**
- `test_schema_runs_once_per_database_across_stores` (line 110)
- `test_memory_databases_never_share_schema_readiness` (line 135)

**Why merge:** Both tests mock `aiosqlite.Connection.executescript` with a counting wrapper and assert the number of calls. The only difference is the database path: one uses a temp file (`:memory:`-less), the other uses `:memory:`. The behavioural distinction — file-based paths share schema readiness via `_SCHEMA_READY`, in-memory paths do not — is a single branch in `_schema_already_applied()`. Parametrising on the path makes one test cover both cases and removes the structural duplication.

**Suggested parametrised test:**
```python
async def test_schema_runs_once_per_database(self):
    cases = [
        ("file path shares readiness", tempfile.mkstemp(suffix='.db'), 1),
        (":memory: never shares readiness", ':memory:', 2),
    ]
    for label, path, expected_calls in cases:
        # ... set up counter, create stores, assert counter == expected_calls
```

(Note: the `:memory:` case needs two separate stores in a loop, while the file case needs two sequential stores. The parametrisation handles this with a small helper.)

---

## 3. Tests to KEEP

### test_chat_context.py

| Test | Line | Why keep |
|------|------|----------|
| `test_interactive_context_is_a_chat_context` | 55 | Documents the type hierarchy contract: tools annotated with `AgentContext` must accept `ChatUpdateContext`. A regression here (e.g., removing the inheritance) would break tool dispatch silently. Borderline trivial (it's a class declaration check), but it pins a design invariant that's easy to overlook. |
| `test_parent_has_messaging_and_child_does_not` | 60 | Core security/isolation invariant. The child lacking `messaging` is what prevents a subagent from posting to the user's chat. This is the single most important assertion in the file. |
| `test_no_way_to_reach_telegram` | 68 | Core security/isolation invariant. Verifies the child context has no `bot`, `chat`, `chat_id`, `message`, `update`, or `context` — the capability surface that would let a subagent write to Telegram. This is the most important assertion in the file. |
| `test_user_data_is_its_own_object` | 77 | After consolidation (see Group A), this becomes the primary mutable-state isolation test. It proves `child.user_data` and `child.thread_data` are distinct objects and that mutations on the child do not leak to the parent. |
| `test_settings_are_a_snapshot` | 105 | Verifies the deepcopy of settings: child gets its own `UserSettings` object with the same values, and mutating the child's settings does not affect the parent. This is the key isolation guarantee for LLM configuration. |
| `test_silent_surface_is_minimal` | 123 | Contract test for the `AgentContext` public API. Uses `dir()` to assert the public surface is exactly `{user_id, user_data, thread_data, conversation, settings}`. Any new public attribute on `AgentContext` will fail this test and force a deliberate review. Valuable despite being somewhat implementation-detail-adjacent. |
| `test_same_thread_returns_shared_state` | 143 | Tests thread-keyed storage: two contexts for the same thread (same `message_thread_id`) share `thread_data` and `user_data`. This is the core behaviour of `_thread_data_for` and `_ensure_user_data`. |
| `test_different_threads_are_isolated_but_share_the_user` | 150 | Tests thread isolation: different `message_thread_id` values get separate `thread_data` but share `user_data`. The final assertion (setting conversation on one thread doesn't affect another) is a meaningful isolation check. |

### test_memory.py

| Test | Line | Why keep |
|------|------|----------|
| `test_set_and_get` | 40 | Fundamental CRUD test. Sets values across multiple users, verifies `get_values` returns correct values with `None` for missing keys, and `get_all_keys` returns the right key sets per user. This is the baseline behaviour contract for the store. |
| `test_delete` | 57 | Tests `delete_values`: deleted keys disappear from `get_all_keys` and return `None` from `get_values`, while undeleted keys remain. Distinct behaviour from `set`/`get`. |
| `test_overwrite` | 70 | Tests `INSERT OR REPLACE` semantics: setting the same key twice yields the second value. Small but distinct behaviour. |
| `test_schema_ensured_without_startup_call` | 83 | Tests lazy schema creation: a store that never calls `create_schema` explicitly still works on its first operation. This is the ad-hoc/test usage path and is distinct from the explicit startup path. |
| `test_db_path_resolved_at_call_time` | 95 | Tests that `config.sql_db_path` is read at connection time, not import time. Patching the config after store creation redirects the store. Important configuration behaviour. |
| `test_schema_runs_once_per_database_across_stores` | 110 | After consolidation (see Group B), this becomes the primary schema-lifecycle test, parametrised on path type. It proves the DDL script runs exactly once per (path, table) for file-based databases and once per store for in-memory databases. |
| `test_concurrent_first_uses_run_schema_once` | 150 | Tests the schema lock under concurrency: 4 stores on the same file, started concurrently, run the DDL exactly once. This is the only concurrency test in the suite and validates the `_schema_lock` mutual exclusion. |
| `test_reset_schema_cache_allows_recreated_file` | 192 | Tests the `_reset_schema_cache()` recovery path: a deleted-and-recreated file at the same path is skipped by the stale memo (writes fail), and resetting the cache restores self-healing. This documents an important edge case and the test hook itself. |

---

## 4. Summary Counts

| Metric | test_chat_context.py | test_memory.py | Total |
|--------|---------------------|----------------|-------|
| Tests in area | 12 | 11 | **23** |
| Recommended for deletion | 3 | 2 | **5** |
| Recommended for consolidation | 2 tests → 1 merged | 2 tests → 1 parametrised | **4 tests → 2** |
| Recommended to keep | 8 | 8 | **16** |

After consolidation and deletion, the suite would have **16 tests** (down from 23), with the remaining tests covering the same behavioural ground more efficiently.

### What was preserved

The audit deliberately did **not** touch:
- The core subagent isolation guarantees (`test_no_way_to_reach_telegram`, `test_parent_has_messaging_and_child_does_not`, `test_user_data_is_its_own_object`, `test_settings_are_a_snapshot`).
- The thread-keyed storage behaviour (`test_same_thread_returns_shared_state`, `test_different_threads_are_isolated_but_share_the_user`).
- The schema lifecycle and concurrency behaviour (`test_schema_runs_once_per_database_across_stores`, `test_concurrent_first_uses_run_schema_once`, `test_reset_schema_cache_allows_recreated_file`).
- The contract/interface tests (`test_silent_surface_is_minimal`, `test_interactive_context_is_a_chat_context`).

These are the tests that protect important business logic, edge cases, and critical application behaviour.