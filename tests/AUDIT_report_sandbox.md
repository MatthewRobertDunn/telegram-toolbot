# Audit Report: tests/test_sandbox.py

**Total tests:** 35
**Recommended for deletion:** 3
**Recommended for consolidation:** 3 groups (5 tests involved)
**Recommended to keep:** 27

---

## 1. Tests to DELETE

### `test_one_sandbox_per_user_are_distinct` (tests/test_sandbox.py:195–200)
**Reason:** Trivial. Tests that `f"toolbot-sandbox-{user_id}"` produces different strings for different ints — guaranteed by the language. The name format is already verified by `test_ensure_starts_container_once` (line 183), and the two-user scenario adds no new assurance. Could be absorbed into that test as an extra assertion if desired, but offers zero independent value on its own.

### `test_reap_does_not_count_a_failed_stop` (tests/test_sandbox.py:464–472)
**Reason:** Redundant with `test_stop_failure_keeps_sandbox_running_for_retry` (line 448). The latter is strictly more comprehensive: it verifies `running` stays `True`, the container stays `'true'`, and that a subsequent successful reap *does* stop it. This test only checks that `reap_idle()` returns `([], [])` and the sandbox is still tracked — a strict subset of the coverage already provided.

### `test_store_file_script_is_atomic` (tests/test_sandbox.py:583–593)
**Reason:** Anaemic + implementation-detail. Despite its name, it does **not** test atomicity (that claim lives in `test_store_file_writes_once_and_returns_path`, which verifies a single exec roundtrip). This test only checks that the private `_store_script()` output, when shlex-quoted, appears as a substring of the exec command. It asserts on internal script text (`mkdir -p`, `cat > "$p"`, `/workspace/uploads`) rather than observable behaviour. The substring checks are also brittle — they would break on any refactor of `_STORE_SCRIPT` even if the real atomicity guarantee is unchanged.

---

## 2. Tests to CONSOLIDATE

### Group A: reap() boundary tests → parametrise
**Tests:** `test_reap_stops_only_idle_sandboxes` (350), `test_reap_leaves_sandbox_just_under_stop_timeout` (366), `test_reap_destroys_sandboxes_past_destroy_timeout` (385), `test_reap_ignores_already_stopped_sandbox` (396)

All four exercise `reap_idle()` with different idle-offset inputs and assert on the `(stopped, destroyed)` result plus container state. They could be a single parametrised test with a table of `(user_id, idle_offset, expected_running, expected_stopped, expected_destroyed, expected_tracked)` rows. The individual tests are clear and each covers a distinct boundary, so this is a maintainability refactor rather than a deletion — but consolidating would remove the repetitive setup (`_ensure` ×2, manual `last_activity` assignment) shared across all four.

### Group B: stop-failure tests → merge
**Tests:** `test_stop_failure_keeps_sandbox_running_for_retry` (448), `test_reap_does_not_count_a_failed_stop` (464)

`test_reap_does_not_count_a_failed_stop` should be deleted (see above); its assertions belong as an extra block inside `test_stop_failure_keeps_sandbox_running_for_retry`, which already sets up the failed-stop scenario and then verifies a successful retry.

### Group C: ensure() distinctness → absorb
**Test:** `test_one_sandbox_per_user_are_distinct` (195)

Absorb its assertion (`second.container_name == 'toolbot-sandbox-2'`, two `run` calls) into `test_ensure_starts_container_once` as a second-user block, or delete outright (see above).

### Group D (optional): exec timeout-classification tests → parametrise
**Tests:** `test_exec_container_timeout_preserves_partial_output` (260), `test_exec_kill_escalation_counts_as_timeout` (269), `test_exec_oom_kill_is_not_a_timeout` (282), `test_exec_backstop_reports_lost_output` (296)

These four test the four distinct timeout/outcome classifications a caller can see. They share setup (ensure a sandbox, possibly set a FakeClock) and assert on the same `result` keys (`stdout`, `stderr`, `returncode`, `timeout`). A parametrised test with a table of `(command, clock_elapsed, expected_timeout, expected_rc, expected_stdout_substring)` rows would reduce duplication. **However**, these tests are high-value and the differences in setup (FakeClock, exec_elapsed) make a clean parametrisation non-trivial — this is a "nice to have" refactor, not a priority. I recommend keeping them as-is unless maintainability becomes a concern.

---

## 3. Tests to KEEP

All remaining 27 tests protect important behaviour and should stay:

| Test | Why it must stay |
|---|---|
| `test_ensure_starts_container_once` (179) | Core `ensure()` path; verifies run args (label, `--init`, memory, image, `sleep infinity`) and identity caching. |
| `test_ensure_restarts_stopped_container` (202) | Distinguishes restart (via `start`) from recreate — filesystem preservation depends on this. |
| `test_ensure_recreates_when_restart_fails` (217) | Error fallback: `start` fails → `rm -f` + `run`. |
| `test_ensure_failure_raises` (225) | `SandboxStartError` classification; verifies raw docker stderr is **not** surfaced to the caller. |
| `test_ensure_does_not_refresh_timer` (331) | Already-running sandboxes must not reset their idle timer — guards a subtle correctness rule. |
| `test_exec_returns_output_and_updates_timer` (236) | Baseline exec: output, returncode, timeout flag, timer refresh. |
| `test_exec_deadline_runs_inside_the_container` (249) | Verifies the `timeout -k 5 N bash -c …` wrapping and client timeout = deadline + grace. |
| `test_exec_container_timeout_preserves_partial_output` (260) | Exit 124 path: partial output survives, `timeout=True`. |
| `test_exec_kill_escalation_counts_as_timeout` (269) | 137 **after** deadline + kill grace → `timeout=True`. |
| `test_exec_oom_kill_is_not_a_timeout` (282) | 137 **within** deadline → `timeout=False` (in-container kill, not escalation). **This is the most subtle and important test in the file** — it validates the business rule that distinguishes SIGKILL escalation from OOM/ordinary kill. |
| `test_exec_backstop_reports_lost_output` (296) | Client-side hang: `returncode=None`, empty stdout, `timeout=True`. |
| `test_exec_translates_gone_container` (307) | Container vanished mid-command → plain message, `returncode=None`, `timeout=False`. |
| `test_exec_passes_command_failures_through` (321) | Guards against over-eager `_container_gone` matching: a command's own stderr must pass through untouched. |
| `test_destroy_removes_container` (337) | Core destroy: `rm -f`, tracking removal, no-op for missing user. |
| `test_reap_stops_only_idle_sandboxes` (350) | Baseline reap: stops idle, keeps active, keeps stopped tracked. |
| `test_reap_leaves_sandbox_just_under_stop_timeout` (366) | Boundary: just-under-stop stays running; just-under-destroy (but past stop) is stopped and kept. |
| `test_reap_destroys_sandboxes_past_destroy_timeout` (385) | Destroy path of reap (distinct code branch from the stop path). |
| `test_reap_ignores_already_stopped_sandbox` (396) | Guards the `sandbox.running` check — prevents double-stopping. |
| `test_recover_adopts_stopped_leftover` (408) | Recover: stopped leftover adoption + `last_activity` estimation from `FinishedAt`. |
| `test_recover_adopts_running_leftover` (423) | Recover: running leftover adoption with `last_activity = now`. |
| `test_recover_removes_containers_past_destroy_timeout` (436) | Recover: destroys containers past the destroy timeout. |
| `test_recover_removes_unidentifiable_leftovers` (416) | Recover: containers without a `toolbot.uid` label are removed as stale. |
| `test_stop_failure_keeps_sandbox_running_for_retry` (448) | Failed stop leaves `running=True`; reap retries next pass. |
| `test_destroy_waits_for_a_running_exec` (474) | Concurrency: destroy queues behind a running exec's lock; `rm` is the last call. |
| `test_read_marks_are_cleared_on_destroy` (489) | Successful destroy clears read marks; duplicate registration is idempotent. |
| `test_failed_destroy_keeps_read_marks` (509) | Failed `rm` must **not** clear read marks (files still exist); retry clears them. |
| `test_reaper_rechecks_activity_after_waiting_for_the_lock` (530) | Concurrency: reaper re-reads `last_activity` under the lock so a command that ran while it waited isn't wrongly stopped. |
| `test_read_marks_holder_is_not_kept_alive` (554) | Validates the weakref design: throwaway holders must not be kept alive by the registry. Borderline (tests gc behaviour) but protects a documented invariant. |
| `test_store_file_writes_once_and_returns_path` (568) | Baseline store_file: single exec, bounded timeout, stdin passthrough, `-i` flag, path returned. |
| `test_store_file_reports_write_failure` (595) | store_file error path: write failure → `(None, error)`. |
| `test_store_file_reports_timeout` (602) | store_file error path: timeout → `(None, "timed out")`. |
| `test_store_file_reports_empty_path` (614) | store_file error path: empty stdout → "did not report the stored path". |

---

## 4. Summary Counts

| Metric | Count |
|---|---|
| Total tests in file | 35 |
| Recommended for deletion | 3 |
| Recommended for consolidation (groups) | 3 (involving 5 tests) |
| Recommended to keep | 27 |

The test suite is, on the whole, strong. The exec timeout-classification tests (especially `test_exec_oom_kill_is_not_a_timeout`) and the concurrency tests (`test_destroy_waits_for_a_running_exec`, `test_reaper_rechecks_activity_after_waiting_for_the_lock`) are high-value and model real business rules. The three tests flagged for deletion each offer zero independent assurance relative to their maintenance cost.