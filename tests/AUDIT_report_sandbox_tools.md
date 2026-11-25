# Audit Report: `tests/test_sandbox_tools.py`

**Total tests in area:** 70
**Recommended for deletion:** 2
**Recommended for consolidation:** 4 groups (covering 10 tests → 4)

---

## 1. Tests to DELETE

### `TestUploadNaming.test_fallback_name_from_kind_and_path` (line 483–505)
**Reason:** Implementation-detail test of a private function (`upload_handler._fallback_name`) outside the sandbox tools module. The test fabricates empty marker classes (`Document`, `PhotoSize`, `Voice`) purely to exercise `isinstance`-based type dispatch — this is testing *how* the function decides a name, not *what name the user sees*. The observable behaviour (filename on a sent document) is already covered by `TestSendFile.test_send_file_success`, which asserts `bot.sent[0]['filename']`. The `_fallback_name` path only triggers when Telegram fails to provide a `file_path`, an edge case not covered elsewhere, but the test as written adds little value relative to its maintenance cost (fragile coupling to internal type markers).

### `TestSandboxToolHelpers.test_resolve_path` (line 546–552)
**Reason:** Tests the private `_resolve_path` helper, whose logic is largely a thin wrapper around `posixpath.normpath` (a stdlib guarantee). The three assertions cover: relative→`/workspace` prefix, absolute passthrough, and `..` normalization. While path normalisation is security-relevant (traversal prevention), this test provides minimal coverage for the effort — it does not test edge cases like trailing slashes, `./` prefixes, or repeated `..` that would escape the workspace root. The function's behaviour is implicitly exercised by every other test that calls `files.read`, `files.write`, `files.edit`, `search.grep`, etc. through the real code paths. **Borderline keep** — if the team wants explicit coverage of `_resolve_path`, expand it into a proper edge-case suite rather than a three-assertion smoke test.

---

## 2. Tests to CONSOLIDATE

### Group A: Edit-helper payload rejection (4 tests → 1–2)
**Tests:**
- `TestEditHelper.test_payload_rejects_unknown_fields` (203–209)
- `TestEditHelper.test_payload_rejects_missing_or_wrong_typed_fields` (211–224)
- `TestEditHelper.test_payload_rejects_non_json` (226–230)
- `TestEditHelper.test_payload_rejects_non_object` (232–236)

**Suggestion:** Merge into a single parametrised test (`test_payload_rejection`) covering: unknown fields, missing fields, wrong-typed fields, non-JSON input, non-object JSON. All four tests call the same `_run()` helper, assert the same shape (`code == 1`, `result['ok'] is False`, error message contains a keyword), and test the same `read_payload()` validation chain. The `subTest` already used in `test_payload_rejects_missing_or_wrong_typed_fields` shows the pattern works; just extend it to cover all cases.

### Group B: Command-output formatting (2 tests → 1)
**Tests:**
- `TestSandboxTools.test_bash_formats_output` (647–653)
- `TestSandboxTools.test_python_runner_shares_bash_formatting` (655–669)

**Suggestion:** Merge into a single parametrised test over `(tool, command, exec_response, expected_substrings)`. Both tests exercise the identical `format_command_result` code path through different entry points (`shell.bash` vs `python_runner.run_python_code`). The python-runner test adds a second case (error output), which is the only reason it isn't fully redundant — parametrisation captures both naturally.

### Group C: Grep error paths (2 tests → 1)
**Tests:**
- `TestSandboxTools.test_grep_reports_helper_error` (777–784)
- `TestSandboxTools.test_grep_reports_helper_crash` (786–795)

**Suggestion:** Merge into `test_grep_error_handling` with two subcases: (1) helper returns `ok: False` with an error string, (2) helper exits non-zero with a traceback. Both test the same `search.grep` → `_run_helper` → error-string passthrough chain. The assertions are structurally identical (`assertIn(error_text, result)`).

### Group D: Duplicate file-guard tests across send_file / read_image (2 tests → 1)
**Tests:**
- `TestSendFile.test_file_not_found` (881–890)
- `TestSendFile.test_file_too_large` (892–906)

**Suggestion:** `test_file_too_large` is already parametrised over `(tool, path, limit)`. Fold `test_file_not_found` into the same parametrised pattern. Both tests exist because `send_file` and `read_image` each independently re-implement the same `stat` → error guard (the source comment even acknowledges the duplication: *"read_image and send_file duplicate this guard; cover both"*). A single parametrised test over `(tool, path, stat_rc, stat_size, expected_keyword)` covers both tools in both scenarios without losing the cross-tool coverage intent.

---

## 3. Tests to KEEP

These tests protect important business logic, edge cases, or critical application behaviour and must stay:

| Test | Why keep |
|---|---|
| `test_exact_replacement`, `test_replace_all`, `test_unique_context_disambiguates`, `test_line_trimmed_fallback` | Core edit-tool behaviour; the replacer chain (exact → line-trimmed → block-anchor → whitespace-normalised → indentation-flexible) is the heart of the edit tool. |
| `test_disproportionate_match_rejected` | Security guard against accidental huge replacements. |
| `test_line_endings_preserved` | CRLF preservation is a real cross-platform correctness concern. |
| `test_helper_script_end_to_end` | True end-to-end test through `main()` including file I/O. |
| `test_offset_and_limit`, `test_offset_out_of_range` | Pagination and boundary validation for read. |
| `test_long_line_truncated` | Output-size guard (2000-char line limit). |
| `test_binary_file_rejected` | Input validation — prevents binary garbage reaching the model. |
| `test_not_found_with_suggestions` | Error UX — "Did you mean" suggestions. |
| `test_sorted_by_mtime_descending`, `test_truncation_cap` | Glob ordering and result cap. |
| `test_matches_are_parsed_and_made_absolute`, `test_non_json_lines_tolerated`, `test_truncation_cap_and_early_stop`, `test_long_lines_capped`, `test_rg_failure_after_clean_read` | Grep parsing, tolerance of malformed output, truncation + early kill, line capping, and rg failure handling. |
| `test_matches_by_user_and_path`, `test_entries_expire_after_ttl` | Upload-cache correctness and TTL expiry. |
| `test_valid_success_response`, `test_valid_failure_response`, `test_nonzero_exit_reports_stderr`, `test_nonzero_exit_without_stderr`, `test_success_payload_on_nonzero_exit_reports_the_run` | Core `_run_helper` protocol validation — the most important tests in the file. |
| `test_missing_version_read_as_v1` | Backward compatibility with older sandbox images — critical for deploys. |
| `test_versions_up_to_max_accepted`, `test_future_protocol_version_rejected` | Protocol version gating. |
| `test_missing_ok_flag`, `test_success_without_data`, `test_failure_without_error_message` | Payload shape validation. |
| `test_write_creates_and_marks_read`, `test_requires_read_first` | Write tool + read-before-write guard. |
| `test_edit_runs_helper`, `test_edit_identical_strings` | Edit tool delegation + no-op guard. |
| `test_read_formats_lines`, `test_read_error`, `test_read_directory` | Read tool formatting, error passthrough, directory mode + read-mark behaviour. |
| `test_grep_delegates_to_the_helper`, `test_grep_truncates`, `test_glob_formats_files` | Tool-level delegation, truncation messaging, empty-result contract. |
| `test_patch_success`, `test_patch_error` | Patch tool success + error passthrough. |
| `test_send_file_success` (incl. binary round-trip), `test_send_file_base64_failure` | Send-file happy path, binary safety, base64 failure. |
| `test_unsupported_extension`, `test_success_sends_photo_and_returns_image`, `test_images_not_shown_are_not_sent`, `test_recent_upload_is_not_resent_to_the_chat`, `test_send_photo_failure_still_attaches` | Read-image: extension validation, photo send + attach, `send_to_chat` flag, recent-upload dedup (per-user!), send-photo resilience. |
| `test_execute_tool_call_passes_results_through`, `test_image_tool_call_appends_user_message` | ToolCallResult contract + LLM integration with image messages. |

---

## 4. Summary Counts

| Metric | Count |
|---|---|
| Total tests in file | 70 |
| Recommended for deletion | 2 |
| Recommended for consolidation | 4 groups (10 tests → 4) |
| Recommended to keep | 68 (after consolidation: 64) |

### After consolidation + deletion: 64 tests

The consolidated file would have **fewer but higher-density** tests: each test covers more scenarios via parametrisation, and the two deleted tests removed genuine low-value coverage (private-function testing with marker classes, and a thin smoke test of a stdlib wrapper).