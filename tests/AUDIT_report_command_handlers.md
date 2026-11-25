# Audit Report: `tests/test_command_handlers.py`

**Total tests in area:** 26  
**Recommended for deletion:** 5  
**Recommended for consolidation:** 3 groups (absorbing 6 tests into 3, net −3)

---

## 1. Tests to DELETE

### `CommandTableTest.test_every_callback_is_callable` (line 88–91)
**Reason:** Trivial — `Command.callback` is typed `Callable[[Any], Awaitable[object]]` and the field is populated from handler functions at module-import time. The type system and the dataclass guarantee callability; this test cannot fail unless someone deliberately assigns a non-callable, which `pyright` would catch immediately. No behavioural assurance is gained.

### `CommandTableTest.test_lookup_misses_are_none` (line 102–105)
**Reason:** Trivial — `lookup_command` is a thin wrapper around `_BY_NAME.get(name.lower())`. Testing that `dict.get()` returns `None` for a missing key tests the language, not the code. The empty-string case (`lookup_command('')`) is an edge case of the same triviality.

### `RegistrationAgreementTest.test_registration_needs_no_handler_outside_the_table` (line 165–168)
**Reason:** Fully redundant — `test_every_name_and_alias_is_registered` already asserts `set(registered) == expected`, which is set equality. Set equality implies both "every expected key is present" **and** "no unexpected keys are present", making the count assertion (`len(app.handlers) == len(COMMANDS)`) a strict subset of what the other test checks.

### `CommandParsingTest.test_parse_helpers` (line 218–225)
**Reason:** Implementation-detail — tests the private helpers `_parse_command_name` and `_parse_command_text` directly. Their behaviour is already exercised through the public `create_chat_update_context` factory by `test_mentioned_command_resolves_to_the_bare_name`, `test_mixed_case_invocations_resolve_to_their_rows`, and `test_non_command_text_has_no_command`. Testing private functions in isolation adds no coverage the public-path tests don't already provide, and couples the test to internal names that may legitimately change.

### `CommandParsingTest.test_usage_line_renders_the_bare_name` (line 227–231)
Reason: Two problems. (1) The first assertion (`usage_line('top_p', ...) == 'Usage: ...'`) is a unit test for `usage_line`, a trivial f-string formatter — it tests the language, not application behaviour. (2) The second assertion (`assertNotIn('@MyBot', usage_line('model', 'x'))`) is guaranteed by the implementation: `usage_line` returns `f'Usage: `/{command_name} {usage}`'` and neither argument contains `@MyBot`. The `@MyBot` concern is already covered end-to-end by `test_mixed_case_timezone_without_args_renders_usage` (line 238), which asserts `assertNotIn('@MyBot', bot.sent[0])` on a real handler output.

---

## 2. Tests to CONSOLIDATE

### Group A: `test_compact_usage_mirrors_its_handler_text` → `test_usage_matches_the_handler_module_constants`

`test_compact_usage_mirrors_its_handler_text` (line 130–137) checks that `compact`'s table row usage matches the string `'[tokens of recent history to keep]'` and that this string appears in `usage_line('compact', ...)`. The first check is a hardcoded-string implementation detail; the second is trivially true by `usage_line`'s definition. The real invariant — that the table row's `usage` field agrees with the handler module's `COMPACT_USAGE` constant — is exactly what `test_usage_matches_the_handler_module_constants` (line 110) enforces for all other commands.

**Suggested merge:** Add `'compact': (conversation_handlers, 'COMPACT_USAGE')` to the `expected` dict in `test_usage_matches_the_handler_module_constants`, then delete `test_compact_usage_mirrors_its_handler_text`. This absorbs the drift-check into the existing parametric-style loop with zero loss of coverage.

### Group B: `test_mentioned_command_resolves_to_the_bare_name` + `test_mixed_case_invocations_resolve_to_their_rows`

Both tests exercise the same behaviour: mixed-case invocations and `@Bot` mentions resolve to the correct table row via `create_chat_update_context`. The first also checks `command_text` extraction; the second checks three additional case/mention permutations but not `command_text`.

**Suggested merge:** Parametrise into a single test with `(text, expected_name, expected_text)` tuples:

```python
@pytest.mark.parametrize("text,name,text_after", [
    ('/model@MyBot sonnet-4', 'model', 'sonnet-4'),
    ('/MODEL', 'model', ''),
    ('/Start@MyBot', 'start', ''),
    ('/Timezone', 'timezone', ''),
])
def test_command_parsing_resolves(self, text, name, text_after):
    ctx = self._ctx(text)
    self.assertEqual(ctx.command_name, name)
    self.assertEqual(ctx.command_text, text_after)
    self.assertIs(lookup_command(ctx.command_name), lookup_command(name))
```

This covers all four scenarios in one test, removing the overlap.

### Group C: Three `test_mixed_case_*_without_args_renders_usage` tests

`test_mixed_case_timezone_without_args_renders_usage` (line 238), `test_mixed_case_memory_delete_without_args_renders_usage` (line 250), and `test_mixed_case_memory_without_args_renders_usage` (line 257) all test the identical behaviour: a mixed-case invocation with no arguments renders the handler's usage line. They differ only in which handler is invoked and what the expected output string is.

**Suggested merge:** Parametrise by `(invocation_text, handler_coro, expected_message)`:

```python
@pytest.mark.parametrize("text,handler,expected", [
    ('/Timezone@MyBot', command_handlers.timezone_command,
     'Current timezone is `UTC`'),
    ('/MEMORY_DELETE', memory_handlers.memory_delete_command,
     'Usage: `/memory_delete <key>`'),
    ('/Memory', memory_handlers.memory_command,
     'Usage: `/memory <key> [value]`'),
])
async def test_mixed_case_without_args_renders_usage(self, text, handler, expected):
    bot = _Bot()
    ctx = CommandParsingTest._ctx(text, bot=bot)
    with _isolated_stores():
        await handler(ctx)
    self.assertEqual(len(bot.sent), 1)
    self.assertIn(expected, bot.sent[0])
    self.assertNotIn('@MyBot', bot.sent[0])
```

The timezone test additionally checks the "Current timezone is `UTC`" prefix and the full usage string; those can stay as extra assertions in the parametrised row or be folded in with a more flexible assertion. The key win is removing three near-identical test bodies.

---

## 3. Tests to KEEP

All remaining tests provide meaningful assurance and should stay:

| Test | Why |
|---|---|
| `test_names_and_aliases_are_unique` (line 72) | Guards against duplicate command declarations — a real startup-error invariant. |
| `test_every_command_has_non_empty_help` (line 83) | Prevents empty menu entries; cheap and user-facing. |
| `test_continue_is_an_alias_of_retry_not_a_command` (line 93) | Protects a specific historical business rule (`/continue` → `/retry`). |
| `test_lookup_is_case_insensitive` (line 106) | Verifies real case-folding behaviour in `lookup_command`. |
| `test_usage_matches_the_handler_module_constants` (line 110) | Drift-detection between the command table and handler usage constants — high value. |
| `test_menu_lists_primary_names_only` (line 139) | Ensures aliases don't leak into the Telegram menu. |
| `test_every_name_and_alias_is_registered` (line 147) | Set-equality check between the table and PTB registration. |
| `test_registered_callbacks_come_from_the_table` (line 155) | Prevents foreign callbacks being wired into registrations. |
| `test_mentioned_command_resolves_to_the_bare_name` (line 195) | (Consolidate, not delete) — checks `command_text` extraction, the `@Bot` strip, and case folding together. |
| `test_mixed_case_invocations_resolve_to_their_rows` (line 202) | (Consolidate, not delete) — additional case/mention permutations. |
| `test_non_command_text_has_no_command` (line 213) | Genuine edge case: non-command text yields empty `command_name`. |
| `test_mixed_case_timezone_without_args_renders_usage` (line 238) | (Consolidate, not delete) — end-to-end usage rendering for timezone. |
| `test_mixed_case_memory_delete_without_args_renders_usage` (line 250) | (Consolidate, not delete) — end-to-end usage rendering for memory_delete. |
| `test_mixed_case_memory_without_args_renders_usage` (line 257) | (Consolidate, not delete) — end-to-end usage rendering for memory. |
| `test_crash_is_logged_and_answered_not_raised` (line 267) | Critical error-boundary behaviour: crashes are caught, logged, and answered. |
| `test_cancelled_error_is_not_swallowed` (line 280) | Critical control-flow: `CancelledError` propagates through the boundary. |
| `test_persists_before_announcing` (line 294) | Ordering guarantee: save before announce; subagent-cancel failure is logged. |
| `test_cancel_failure_stills_bumps_the_epoch` (line 320) | Resilience: epoch bump is independent of cancel success. |
| `test_bump_failure_is_logged_on_its_own` (line 334) | Error handling: bump failure is logged and epoch stays at 0. |
| `test_drain_releases_queued_items_so_the_submitter_wakes` (line 348) | Queue-drain behaviour: queued items are released without running; submitter wakes. |

---

## 4. Summary Counts

| Metric | Count |
|---|---|
| Total tests in `test_command_handlers.py` | 26 |
| Recommended for **deletion** | 5 |
| Recommended for **consolidation** (groups) | 3 (absorbing 6 tests → 3) |
| Tests remaining after recommended changes | 18 |

The 5 deletions remove only trivial, redundant, or implementation-detail tests — no test that protects important business logic, edge cases, error handling, or critical application behaviour is recommended for removal. The 3 consolidations reduce near-duplicate test bodies without losing coverage.