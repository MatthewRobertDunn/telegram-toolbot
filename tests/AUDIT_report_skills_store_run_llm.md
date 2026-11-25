# Test Audit: tests/test_skills_store.py & tests/test_run_llm.py

Auditor scope: `tests/test_skills_store.py` (23 tests) and `tests/test_run_llm.py` (23 tests)
against `src/skills_store.py` and `src/handlers/run_llm.py`.

Both files are in good shape overall: they test observable behaviour, use deterministic
clocks instead of real sleeps, and carry good regression comments. The findings below are
the genuinely low-value remainder — nothing here protects critical business logic.

---

## 1. Tests to DELETE

### tests/test_skills_store.py

| Test | Line | Reason |
|---|---|---|
| `TestSkillInfo.test_equality_compares_all_fields` | 29 | Trivial: asserts Python `@dataclass(frozen=True)` generated `__eq__`/`__hash__` behaviour. No code in `src/` uses `SkillInfo` equality or hashing (verified by grep) — the class is only read field-by-field. |
| `TestSkillInfo.test_frozen` | 37 | Trivial: asserts the language-level frozen-dataclass mutation error (`FrozenInstanceError`), not any project behaviour. A `frozen=True` flag is not something this suite needs to re-verify. |
| `TestSkillInfo.test_positional_construction_and_repr` | 42 | Trivial/anaemic: asserts the default dataclass `repr` format. Nothing consumes the repr; deleting the `@dataclass` decorator would be caught by every other test via construction failure. |

### tests/test_run_llm.py

| Test | Line | Reason |
|---|---|---|
| `TestRetryPolicy.test_budget_is_the_same_for_both_context_kinds` | 530 | Redundant/implementation-detail: `_retry_budget()` has no context-kind branch (source comment confirms the special case was removed), so the invariant cannot regress without a visible code change. The interactive half duplicates `test_run_llm_returns_no_response_when_budget_spent` almost exactly. ~40 lines of setup to test the absence of a code path. If the invariant is wanted, fold a single `AgentContext` case into the budget-spent test. |

---

## 2. Tests to CONSOLIDATE

### tests/test_skills_store.py

1. **`TestFrontmatterParsing` (6 tests, lines 188–210) → 1–2 parametrised tests.**
   `test_basic_fields`, `test_no_opening_delimiter_yields_empty`,
   `test_missing_closing_delimiter_parses_to_eof`,
   `test_quoted_values_are_unquoted`, `test_indented_continuation_appends_to_previous_key`,
   `test_empty_lines_and_whitespace_only_lines_are_skipped` all have the shape
   `assertEqual(_parse(input), expected)`. Convert to a single test looping over
   `(input, expected)` pairs with `self.subTest()` — same coverage, ~6 methods → 1.

2. **`TestSandboxPathPrefix` (3 tests, lines 224–237) → 1 test with subTests.**
   `test_default_prefix`, `test_prefix_without_trailing_slash_does_not_double_it`, and
   `test_windows_separators_become_posix` differ only in the configured prefix. One test
   iterating over `('/skills', '/custom')` covers both prefix shapes; the Windows case is a
   platform-gated extra input rather than its own test. Note: the Windows test is skipped
   on every non-Windows host, so as written it provides zero assurance on Linux CI.

3. **`test_get_skill` (line 162) could merge into `TestSkillDiscovery`** as it is a thin
   lookup over the same fixture, but this is optional — it is cheap and readable as is.

### tests/test_run_llm.py

4. **`test_auth_error_fails_immediately` (line 419) + `test_permission_error_fails_immediately`
   (line 427) → one test parametrised over `(401, 403)`.** Identical body, only the status
   code differs; `subTest` matches how `test_tools_offered_by_default_and_omitted_on_request`
   already handles two cases in one test.

5. **`test_context_conversation_keeps_one_session_id` (line 99) +
   `test_nested_conversation_gets_a_fresh_session_id` (line 117) → one session-header test.**
   Both assert where `x-opencode-session` comes from (ctx conversation vs explicit
   conversation). A single test covering "header is the conversation's id, whatever
   conversation that is" removes the duplicated client-capture scaffolding. Also mildly
   anaemic as written: the first test's two-run "stability" assertion just re-derives the
   same deterministic `conversation.id`.

6. **Helper duplication in `TestRetryPolicy`:** `fake_sleep` is defined twice (line 394 in
   `_patched` and line 511 in `test_cancellation_interrupts_the_retry_sleep`) and a third
   time as module-level `_advance` (line 584). Fold into one `_advance(clock, sleeps,
   cancel_after=None)` helper.

---

## 3. Tests to KEEP

### tests/test_skills_store.py — all of `TestSkillDiscovery` (11 tests)
These are the core value of the file and each covers a distinct documented rule:
- `test_discovers_directory_flat_and_nested_forms` (61): the two discovery forms.
- `test_id_comes_from_path_not_frontmatter_name` (80): the "path is the ID" business rule.
- `test_missing_description_defaults_to_empty` (87): cheap edge case.
- `test_duplicate_ids_last_wins` (96): override ordering rule, non-obvious yield order.
- `test_skips_hidden_and_node_modules` (111), `test_depth_cutoff` (122, exact boundary at
  depth 6), `test_skill_dir_is_not_searched_deeper` (133): all walk-pruning rules.
- `test_malformed_frontmatter_is_not_fatal` (143), `test_missing_dir_yields_empty_list`
  (156): error-tolerance guarantees callers rely on.
- `test_get_active_skills_returns_a_copy` (172): protects the registry from caller mutation.

### tests/test_run_llm.py — keep nearly everything else
- `test_tools_offered_by_default_and_omitted_on_request` (49): core request-assembly
  contract (`tools`/`tool_choice` only when a surface is passed). Minor: the name says
  "by default" but only explicit values are exercised — pass no `tools` kwarg for one
  subTest case or rename.
- `test_empty_reply_retries_then_recovers` (75), `test_reasoning_only_reply_is_returned`
  (87): return-path edge cases with real fallback semantics.
- `TestReasoningContentPassback` (190, 197): documented regression guard for a real bug
  (DeepSeek/Kimi session poisoning); keep both the echo and the no-field absence case.
- `test_announcement_ids_attach_to_the_assistant_message` (204): /undo cleanup depends on it.
- `TestThinkingMessages` (249, 265, 276): three distinct gating conditions (main-loop only,
  nested silent, thinking_keep off); the `conversation is ctx.conversation` identity rule is
  subtle and worth these three.
- `test_bad_json_becomes_a_tool_result` (298), `test_unknown_tool_is_reported_not_raised`
  (326): error-handling contracts that keep the agent loop alive.
- `TestRetryPolicy` (remaining 8 after consolidation): request-construction exceptions
  propagate vs SDK/stream `TypeError` retried (a genuinely subtle distinction), zero-budget
  = no retries, budget clamp, backoff cap, cancellation during sleep — all real policy with
  deterministic fakes. Keep as-is.

---

## 4. Summary counts

| File | Tests | Delete | Consolidate (methods removed by merging) |
|---|---|---|---|
| tests/test_skills_store.py | 23 | 3 | 8 → 3 (frontmatter 6→1, sandbox prefix 3→1) |
| tests/test_run_llm.py | 23 | 1 | 7 → 4 (401+403 → 1, two session-header tests → 1) |
| **Total** | **46** | **4** | **~11 methods removed** |

Net effect: ~46 → ~35 tests with no loss of behavioural coverage. The deleted tests verify
Python dataclass semantics and a removed code path; the consolidated ones collapse
same-shape assertions without dropping any input case.
