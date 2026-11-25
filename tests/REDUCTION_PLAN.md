# Test Reduction Plan (consolidated from 16 audit reports)

Baseline: 602 tests, all passing, 85% coverage (`baseline_coverage.txt`).

## Worker assignments (disjoint file ownership)

### W1 — tests/test_sandbox_tools.py (70 tests)
- DELETE: `TestUploadNaming.test_fallback_name_from_kind_and_path` (~L483, implementation-detail, private `_fallback_name` with fabricated marker classes); `TestSandboxToolHelpers.test_resolve_path` (~L546, thin posixpath wrapper smoke test).
- CONSOLIDATE:
  - `TestEditHelper` payload rejection: `test_payload_rejects_unknown_fields`, `test_payload_rejects_missing_or_wrong_typed_fields`, `test_payload_rejects_non_json`, `test_payload_rejects_non_object` → 1 parametrised test.
  - `test_bash_formats_output` + `test_python_runner_shares_bash_formatting` → 1 parametrised over (tool, command, expected substrings).
  - `test_grep_reports_helper_error` + `test_grep_reports_helper_crash` → 1 test, two subcases.
  - `TestSendFile.test_file_not_found` + `test_file_too_large` → 1 parametrised over (tool, path, stat behaviour, expected keyword).

### W2 — tests/test_subagent.py + tests/test_subagent_handlers.py (47)
- DELETE: `test_send_to_parent_is_subagent_only` (~L291, fully redundant with tool-surface tests L120/L131); `test_each_role_executes_its_own_implementation` (~L158, function-identity assertion).
- CONSOLIDATE: `test_redundant_wakeup_jobs_never_send` (~L463) + `test_empty_inbox_schedules_nothing` (~L506) → 1 parametrised; `test_silent_when_not_announced` (~L828) + `test_silent_when_show_tool_output_off` (~L837) → 1 parametrised.

### W3 — tests/test_image_gen.py (43)
- DELETE: `test_default_provider_is_a_known_provider` (~L262), `test_default_is_plain_string` (~L187), `test_no_resolution_still_works_for_every_provider` (~L505).
- CONSOLIDATE: NVIDIA dispatch 3→1 (`test_nvidia_flux3_url_and_default_steps`, `test_nvidia_flux2_url_and_default_steps`, `test_nvidia_explicit_steps_are_sent`); Phoenix 2→1 (`test_phoenix_defaults_to_25_steps`, `test_phoenix_json_base64_shape`); store-error 2→1 (`test_store_error_reports_delivery`, `test_store_error_without_sending`); OpenRouter response errors 2→1 (`test_no_images_raises`, `test_missing_b64_raises`); OpenRouter param validation 2→1 (`test_resolution_is_not_a_parameter`, `test_invalid_aspect_ratio_is_reported`); aspect-ratio passthrough 2→1 (`test_explicit_aspect_ratio_is_sent`, `test_valid_aspect_ratio_reaches_the_render`).

### W4 — tests/test_conversation.py (42)
- DELETE: `test_default_is_none` (~L131), `test_max_images_defaults_to_MAX_IMAGES` (~L190).
- CONSOLIDATE: `test_constructor_wraps_plain_dicts` + `test_set_messages_wraps_plain_dicts_too` → 1 parametrised (keep `test_wrapping_keeps_your_messages` separate); `test_add_image_appends_a_caption_and_base64_image` + `test_add_image_without_caption_has_just_the_image` → 1; `test_max_images_is_a_constructor_parameter` + `test_only_the_most_recent_max_images_are_kept` → 1 parametrised on max_images; `test_equals_a_message_list_with_the_same_contents` → fold into `test_two_conversations_with_same_messages_are_equal_but_distinct_ids`; `test_prepare_regeneration_drops_thinking_and_answer` + `test_prepare_regeneration_drops_orphan_thinking` → 1 parametrised.

### W5 — tests/test_journal.py (42)
- DELETE whole test: `test_query_tool_cursor_resumes_inside_tied_milliseconds` (~L1105; store-level `test_search_page_pages_inside_tied_milliseconds` L610 covers the tie logic).
- TRIM within tests: spy/`_schema_ready` block in `test_schema_script_runs_once_per_database` (L71–80, keep the executescript counter + read-through); in `test_list_topics` (L755–760) the "topics vanish" block is dead — FIX it so the deleted entry's topic is unique, then assert the topic disappears; drop duplicate limit=0/limit=51 checks in `test_query_tool_paging` (L1097–1103); drop `tool.func.__name__` assertion in `test_all_seven_tools_are_declared` (~L1178).
- CONSOLIDATE: `test_validation_raises_typed_error` (L105) shrinks to typed-error contract only; merge its duplicated branches into `test_add_rejects_empty_and_long` (L161, parametrised); `test_replace_identical_rejected` (L278) → fold into `test_replace` (L226); `test_search_start_window_and_order` (L410) + `test_search_start_is_inclusive_in_both_directions` (L444) → 1 test; `test_add_uses_configured_timezone` (L785) + `test_add_defaults_to_utc_without_settings` (L801) → 1 parametrised; `test_query_tool_paging_parameters_are_declared` (L1160) + `test_all_seven_tools_are_declared` (L1168) → 1 registry test; trim `test_replace_tool` (L865) and `test_topic_edit_tool` (L934) to happy path + 2 representative error branches each (store layer keeps the full branch sweep).

### W6 — tests/test_sandbox.py (35)
- DELETE: `test_one_sandbox_per_user_are_distinct` (~L195, absorb two-user assertion into `test_ensure_starts_container_once`); `test_reap_does_not_count_a_failed_stop` (~L464, absorb into `test_stop_failure_keeps_sandbox_running_for_retry`); `test_store_file_script_is_atomic` (~L583, asserts internal script text).
- CONSOLIDATE: the four `reap_idle()` boundary tests (`test_reap_stops_only_idle_sandboxes` L350, `test_reap_leaves_sandbox_just_under_stop_timeout` L366, `test_reap_destroys_sandboxes_past_destroy_timeout` L385, `test_reap_ignores_already_stopped_sandbox` L396) → 1 parametrised with an idle-offset table.
- Do NOT touch the exec timeout-classification tests (high value).

### W7 — tests/test_tool_registry.py (30)
- DELETE: `test_needs_ctx_is_recorded_on_the_tool` (~L200), `test_successful_load_is_sticky` (~L347), `test_unknown_role_is_rejected` (~L402).
- CONSOLIDATE: `test_definition_shape` (L104) + `test_no_required_key_when_everything_is_optional` (L124) → 1 parametrised (bash: has required; journal_query: all-optional omits `required` key).
- Groups B–D are optional; only consolidate if it stays clean.

### W8 — tests/test_compaction.py (28)
- MOVE 8 tests (do not delete) into new files `tests/test_transcript.py` (5 `TestMessageToText` tests) and `tests/test_tokens.py` (3 `TestTokenCount` tests) — the only direct coverage of `src/transcript.py` / `src/tokens.py`.
- DELETE: all 4 `TestSummaryPrompt` tests (L78–91, substring checks on a hardcoded constant), `TestThresholds.test_min_compact_tokens_exceeds_the_default_keep_budget` (L100), `TestSplitForCompaction.test_newest_messages_are_kept_verbatim` (L310), `test_tool_call_batch_never_splits_across_the_boundary` (L321), `test_single_exchange_is_summarized_with_nothing_kept` (L344) — all covered by `TestBoundaryRule.test_table` CASES, and `TestCompactContext.test_keep_fraction_refusal_also_names_the_budget` (L456).
- KEEP: `test_table`, `test_something_is_always_summarized`, the 3 no-split guards, and all 6 `TestCompactContext` tests (minus the deleted refusal duplicate).

### W9 — tests/test_generation_queue.py + tests/test_supervision.py (32)
- DELETE: `test_submit_task_reraises_the_job_error` (gq ~L165), `test_a_thunk_is_invoked_once_when_the_worker_dequeues_it` (gq ~L174), `test_a_fresh_job_wraps_the_thunk_exactly_once` (sup ~L130 — verbatim duplicate of the gq thunk test).
- CONSOLIDATE: `test_preformatted_returns_ids` + `test_image_returns_id` + `test_document_returns_id` (gq ~L373–382) → 1 parametrised "posting helpers return message ids" test.
- KEEP: `test_job_failure_is_reraised_and_clears_tracking` (sup ~L117) — the stronger failure twin.

### W10 — tests/test_command_handlers.py (26)
- DELETE: `test_every_callback_is_callable` (L88), `test_lookup_misses_are_none` (L102), `test_registration_needs_no_handler_outside_the_table` (L165), `test_parse_helpers` (L218), `test_usage_line_renders_the_bare_name` (L227).
- CONSOLIDATE: `test_compact_usage_mirrors_its_handler_text` (L130) → add `compact` to the expected dict in `test_usage_matches_the_handler_module_constants` (L110) and delete it; `test_mentioned_command_resolves_to_the_bare_name` (L195) + `test_mixed_case_invocations_resolve_to_their_rows` (L202) → 1 parametrised; the three `test_mixed_case_*_without_args_renders_usage` tests (L238, L250, L257) → 1 parametrised.

### W11 — tests/test_skills_store.py + tests/test_run_llm.py (46)
- DELETE: all three `TestSkillInfo` tests (skills_store L29, L37, L42 — dataclass semantics); `test_budget_is_the_same_for_both_context_kinds` (run_llm ~L530).
- CONSOLIDATE: `TestFrontmatterParsing` 6 tests (L188–210) → 1 subTest loop; `TestSandboxPathPrefix` 3 tests (L224–237) → 1; `test_auth_error_fails_immediately` (L419) + `test_permission_error_fails_immediately` (L427) → 1 parametrised (401, 403); `test_context_conversation_keeps_one_session_id` (L99) + `test_nested_conversation_gets_a_fresh_session_id` (L117) → 1 session-header test.

### W12 — tests/test_regenerate.py + tests/test_config.py (37)
- DELETE: nothing.
- CONSOLIDATE: five `test_bad_*_names_the_variable_and_value` tests (config L34–67) → 1 table-driven parametrised test; `test_store_db_path_is_not_a_stale_copy` (config L104) → fold `assertFalse(hasattr(config, 'STORE_DB_PATH'))` into `test_store_db_path_env_var_feeds_sql_db_path` (L100); four NVIDIA key fallback tests (L110–128) → 1 table-driven; four `_find_env_file` tests (L148–178) → 1 table-driven; `test_undo_preview_is_truncated_to_the_limit` (regenerate L165) → parametrise the two cases with subTest (keep the test).
- KEEP: `ConfigReferenceTest.test_every_referenced_name_exists`, the tool-round undo tests, and the three "Nothing to X" guards.

### W13 — tests/test_thread_settings.py + tests/test_http_client.py (28)
- DELETE: `test_unthreaded_chat_shares_template` (ts ~L156), `test_thread_and_template_share_one_key` (ts ~L92, raw-SQL implementation detail).
- CONSOLIDATE: `SettingsDecoratorTest.test_threaded_change_updates_thread_only` (L168) + `test_unthreaded_change_updates_template` (L201) → 1 parametrised, and fold `test_unthreaded_change_saves_once` (L230) into the unthreaded case; `SettingsAliasingFlagTest` init pair (L257, L266) → 1 parametrised; swap pair (L275, L292) → 1 parametrised; http timeout pair (hc L135 + L148) → 1 parametrised; fold `test_session_is_reused_on_the_same_loop` (hc L170) into `test_aclose_clears_the_session` (L173); `test_result_never_exceeds_max_chars` (L223) + `test_zero_max_chars_is_clamped` (L232) → 1 parametrised.
- KEEP `test_body_is_capped_during_the_read` and `test_round_trip_per_thread`.

### W14 — tests/test_chat_context.py + tests/test_memory.py (23)
- DELETE: `test_real_user_settings_are_copied` (cc L113), `test_parent_coordination_state_is_unreachable` (cc L85), `test_sandbox_read_tracking_starts_empty` (cc L134), `test_empty_keys` (mem L77), `test_second_store_adopts_shared_schema_readiness` (mem L174).
- CONSOLIDATE: `test_conversation_is_the_subagents_own` (cc L93) → fold its real assertion (child conversation writes stay local) into `test_user_data_is_its_own_object` (L77); `test_memory_databases_never_share_schema_readiness` (mem L135) → parametrise into `test_schema_runs_once_per_database_across_stores` (L110) over file vs `:memory:` paths.

### W15 — tests/test_conversation_persistence.py + tests/test_timezone.py + tests/test_impersonate.py (30)
- DELETE: `test_last_updated_is_stamped_by_the_db` (cp ~L145), `test_default_timezone_is_utc` (tz ~L50), `test_timezone_survives_json_round_trip` (tz ~L81).
- CONSOLIDATE: `test_load_missing_returns_none` (cp L64) + `test_load_corrupt_returns_none` (cp L68) → 1 parametrised; `test_impersonation_is_added_as_a_user_turn` (imp L85) + `test_impersonated_turn_is_tagged_and_scoped_locally` (imp L106) → 1 test; trim the raw sqlite3 row-introspection block from `test_threads_persist_independently` (cp ~L135–143); trim the duplicated usage-line assertion in `test_no_arg_shows_current` (tz ~L78, keep the current-value interpolation).

### W16 — tests/test_test_isolation.py + tests/test_upload_cache.py + tests/test_settings_cleanup.py + tests/test_reasoning_filter.py (29)
- DELETE: `test_temp_dir_is_removed_at_exit` (iso L93), `test_bootstrap_db_is_a_single_file_in_its_own_temp_dir` (iso L82), `test_rerecording_updates_the_entry` (uc L50).
- CONSOLIDATE: the four guard tests (`test_aiosqlite_open_of_repo_db_raises` L102, `test_sqlite3_open_of_repo_db_raises` L112, `test_file_uri_spelling_of_repo_db_raises` L127, `test_file_uri_spelling_of_repo_db_raises_for_aiosqlite` L135) → 1 parametrised; `test_test_db_itself_still_opens` (L153) + `test_unrelated_paths_still_open` (L168) → 1; upload_cache hit tests (`test_same_file_path_hits` + `test_equivalent_path_spelling_hits`) → 1, and fold `test_relative_path_is_a_different_file` + `test_other_file_misses` into the kept miss tests; `test_recent_group_survives` (sc L83) → fold into `test_group_with_one_recent_row_survives_whole` (L89); reasoning passthrough (`test_no_tags`, `test_unclosed_tag_left_alone`, `test_case_sensitive`) → 1 parametrised.
- REPLACE the deleted rerecord test with a TTL-expiry test (patch the TTLCache with ttl=0 or freeze time, assert a recorded upload misses after expiry).
- KEEP: all settings_cleanup tests except the folded one; `test_full_span`, `test_no_open_tag_bare_close`.

## Rules for all workers
1. Only touch the assigned test files. Never edit `src/` or shared test infra (`tests/run.py`, `tests/_bootstrap.py`, `tests/mock_context.py`, `tests/__init__.py`).
2. Locate tests by name (line numbers shift while editing).
3. When consolidating, preserve every input case via parametrisation/subTest. Every removed test's unique assertion must survive somewhere.
4. Match existing style: unittest, descriptive test names, comments explaining the failure mode.
5. Verify with `python tests/run.py -v` (full suite, ~10s) — must be green.
6. Commit on the worktree branch.
