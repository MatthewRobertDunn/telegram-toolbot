# Test Audit Report: upload_cache / settings_cleanup / reasoning_filter

**Scope:** `tests/test_upload_cache.py` (7), `tests/test_settings_cleanup.py` (6),
`tests/test_reasoning_filter.py` (5) — 18 tests total.
**Sources:** `src/upload_cache.py`, `src/persistence/cleanup.py` (+ `SettingsStore.purge_stale_threads`),
`src/handlers/consume_stream.py::_split_thinking` (the reasoning filter lives in `handlers/consume_stream.py`,
not `src/messages.py` as stated in the assignment).

---

## 1. Tests to DELETE

| Test | Location | Reason |
|---|---|---|
| `test_rerecording_updates_the_entry` | tests/test_upload_cache.py:50 | Trivial/anaemic: recording a key twice then checking membership is guaranteed by dict semantics. The second `record_upload` call could be deleted and the test would still pass — it verifies nothing about the "refresh the TTL entry" behaviour its name claims. |

No other deletions recommended.

## 2. Tests to CONSOLIDATE

### upload_cache — four "different key → miss" tests are one scenario with different inputs
- `test_same_name_in_different_directories_do_not_collide` (tests/test_upload_cache.py:18)
- `test_relative_path_is_a_different_file` (tests/test_upload_cache.py:28)
- `test_other_user_misses` (tests/test_upload_cache.py:40)
- `test_other_file_misses` (tests/test_upload_cache.py:45)

All four assert "record key K, query key K′ ≠ K → False". Suggest a single
parametrised test over `(record_user, record_path, query_user, query_path)`.
If you want to preserve the two *behavioural* names (per-directory keying and
per-user scoping are documented contracts), a reasonable middle ground is: keep
the directory-collision and other-user tests, and fold `test_relative_path_is_a_different_file`
and `test_other_file_misses` into one of them as extra assertions. Net: 4 → 1–2 tests.

### upload_cache — two "same key → hit" tests
- `test_same_file_path_hits` (tests/test_upload_cache.py:23)
- `test_equivalent_path_spelling_hits` (tests/test_upload_cache.py:33)

Both assert a hit after one record; the second just varies the path spelling.
Mergeable into one test (record canonical path, assert hit for canonical,
`./`-prefixed, and trailing-slash spellings). Net: 2 → 1.

### settings_cleanup — subsumed baseline
- `test_recent_group_survives` (tests/test_settings_cleanup.py:83) is fully subsumed by
  `test_group_with_one_recent_row_survives_whole` (tests/test_settings_cleanup.py:89):
  the latter also asserts the purge returns 0 and rows survive whenever *any* row in the
  group is recent (the pure-recent group is the special case where the recent row is the
  only row). Fold it in as a second inserted row, or keep as a cheap baseline — marginal either way.

**Consolidation total:** 6 tests reducible to 2–3 (net −3 to −4).

## 3. Tests to KEEP

### tests/test_upload_cache.py
- `test_same_name_in_different_directories_do_not_collide` — protects the documented contract
  that keying is on the full path (the module docstring calls this out explicitly).
- `test_other_user_misses` — per-user scoping is real business logic (one user's uploads must
  not suppress another's duplicate check).
- `test_same_file_path_hits` / `test_equivalent_path_spelling_hits` — verify `_key()` normalisation
  is actually wired into `record_upload`/`was_recently_uploaded` (would catch the normalisation
  being dropped or bypassed).
- `test_relative_path_is_a_different_file` — mild, but confirms relative and absolute paths land in
  different key spaces (relevant if the sandbox ever passes both spellings).

### tests/test_settings_cleanup.py
The strongest file of the three; keep nearly everything. Every test pins a distinct SQL-semantics
behaviour of the group purge:
- `test_stale_thread_group_is_purged` — whole-group deletion (per-group, not per-row); asserts both
  the return count and the DB state.
- `test_group_with_one_recent_row_survives_whole` — MAX(last_updated) per group keeps older siblings;
  the subtlest and most valuable case in the file.
- `test_unthreaded_rows_are_never_touched` — NULL `thread_id` exemption protects the settings
  template that new threads clone; data-loss prevention.
- `test_users_and_threads_are_judged_separately` — (user, thread) scoping; one stale user must not
  purge an active user sharing the thread id.
- `test_cutoff_days_widen_and_narrow_the_purge` — proves `max_age_days` actually reaches the SQL
  comparison in both directions; the clock-skew margin (3-day-old row vs 10/2-day cutoffs) is safe.
- `test_recent_group_survives` — see consolidation note above.

### tests/test_reasoning_filter.py
All five test distinct, behaviourally meaningful scenarios of `_split_thinking`; all cheap to run:
- `test_no_tags` — baseline passthrough.
- `test_full_span` — the core strip path (note: only the *close* tag drives the split; the open tag
  is inert — the test's name slightly oversells it, but the assertion is correct).
- `test_no_open_tag_bare_close` — covers the documented real-world backend behaviour (inline
  reasoning ending in a bare close tag) plus the stray-close-tag corner. Two sub-cases in one test
  is fine here since both share the "no open tag" condition.
- `test_unclosed_tag_left_alone` — fail-safe behaviour: unterminated reasoning must reach the chat
  rather than swallow the visible reply. Important error-handling pin.
- `test_case_sensitive` — pins that uppercase `<THINK>` is *not* stripped (visible content survives);
  would catch an accidental `re.IGNORECASE`.

## 4. Summary counts

| Area | Tests | Delete | Consolidate (tests involved → target) |
|---|---|---|---|
| test_upload_cache.py | 7 | 1 | 6 → 2–3 (miss-group + hit-group) |
| test_settings_cleanup.py | 6 | 0 | 1 → 0 (optional merge into :89) |
| test_reasoning_filter.py | 5 | 0 | 0 |
| **Total** | **18** | **1** | **7 → 2–3** |

Overall this is a healthy suite: the settings-cleanup and reasoning-filter files are almost entirely
behavioural tests over distinct edge cases, and the only clearly worthless test is the re-record test
in test_upload_cache.py.

## 5. Observation (coverage gap, not a deletion)

`upload_cache`'s only genuinely time-dependent behaviour — TTL expiry — is untested, while the one
test that *claims* to exercise entry updating (`test_rerecording_updates_the_entry`) does not. If a
replacement test is wanted, the deleted test's budget is better spent asserting that an entry with an
expired TTL (e.g. a cache patched with `ttl=0`, or a frozen-clock mock) misses.
