# Audit Report: tests/test_image_gen.py

**Total tests:** 43  
**Recommended for deletion:** 3  
**Recommended for consolidation:** 13 tests across 6 groups (net savings: 7)  
**Recommended to keep:** 33 tests (after consolidation)

---

## 1. Tests to DELETE

### `test_default_provider_is_a_known_provider` (line 262–263)
**Reason:** Trivial. Tests that `DEFAULT_PROVIDER` is in `IMAGE_PROVIDERS`, but the source defines `DEFAULT_PROVIDER = PROVIDER_NVIDIA_FLUX3` and `IMAGE_PROVIDERS = tuple(PROVIDERS)` where `PROVIDERS` contains that same key. This is guaranteed by the language — no test value.

### `test_default_is_plain_string` (line 187–189)
**Reason:** Anaemic. Asserts only `isinstance(result, str)` with no behavioural content. The default `add_to_context=False` path is already exercised meaningfully by `test_send_to_chat_branches` (which checks the returned string content) and `test_attach_returns_image_result` (which checks the non-default path). This test adds zero additional assurance.

### `test_no_resolution_still_works_for_every_provider` (line 505–519)
**Reason:** Misleading + redundant. The name claims "for every provider" but only tests the default `nvidia-flux3`. The actual assertion — that the payload contains `width: 1024, height: 1024, steps: 50` — is already covered by `test_nvidia_flux3_url_and_default_steps` (line 311), which calls `generate_image` with no resolution and checks the same payload fields. The test adds nothing new and its name is factually wrong.

---

## 2. Tests to CONSOLIDATE

### Group 1 — NVIDIA provider dispatch (3 tests → 1)
**Tests:** `test_nvidia_flux3_url_and_default_steps` (line 311), `test_nvidia_flux2_url_and_default_steps` (line 324), `test_nvidia_explicit_steps_are_sent` (line 337)

All three verify the same thing: the NVIDIA render function posts to the right URL with the right `steps` value. They differ only in provider (flux3 vs flux2) and steps source (default vs explicit). Parametrise over `(provider, url_attr, expected_steps)` and assert the URL + payload in one test.

### Group 2 — Phoenix response shapes (2 tests → 1)
**Tests:** `test_phoenix_defaults_to_25_steps` (line 350), `test_phoenix_json_base64_shape` (line 361)

Both test the Phoenix provider's response parsing. One covers raw JPEG bytes, the other covers the `{"result": {"image": ...}}` JSON shape. Merge into a single parametrised test with two response fixtures, asserting both the decoded image bytes and the payload (`num_steps: 25`).

### Group 3 — Store-error delivery branches (2 tests → 1)
**Tests:** `test_store_error_reports_delivery` (line 156), `test_store_error_without_sending` (line 160)

Both test the same scenario (store_file fails) with the only difference being `send_to_chat=True` vs `False`, which changes the expected result string. Parametrise over `(send_to_chat, expected_substring)`.

### Group 4 — OpenRouter response errors (2 tests → 1)
**Tests:** `test_no_images_raises` (line 457), `test_missing_b64_raises` (line 462)

Both test that malformed OpenRouter responses raise `RuntimeError` with a specific message. Parametrise over `(response_fixture, expected_substring)`.

### Group 5 — OpenRouter parameter validation (2 tests → 1)
**Tests:** `test_resolution_is_not_a_parameter` (line 467), `test_invalid_aspect_ratio_is_reported` (line 476)

Both test that invalid parameters are reported to the model via `generate_and_send_image` rather than silently dropped. Parametrise over `(param, value, expected_substring)`.

### Group 6 — Aspect ratio passthrough (2 tests → 1)
**Tests:** `test_explicit_aspect_ratio_is_sent` (line 434), `test_valid_aspect_ratio_reaches_the_render` (line 482)

Both verify that a user-supplied `aspect_ratio` reaches the backend. The first tests it at the `generate_image` layer; the second at the `generate_and_send_image` layer. The first is sufficient — the end-to-end path is already covered by the fact that `generate_and_send_image` calls `generate_image`. Merge into the Group 1-style payload assertion or keep one clean test that asserts the aspect_ratio appears in the POST body. (Note: `test_explicit_aspect_ratio_is_sent` also has a structural smell — `_render_post` is defined at line 439, *after* the test that calls it at line 435. While Python resolves this at runtime, it hurts readability.)

---

## 3. Tests to KEEP

These tests provide meaningful, non-redundant coverage of behaviour, edge cases, or error handling:

| Test | Line | Why keep |
|---|---|---|
| `test_http_error_body_is_returned` | 61 | Distinct branch: HTTP error with body echoed to model |
| `test_http_error_without_a_body_names_the_status` | 68 | Distinct branch: HTTP error without body falls back to status+URL |
| `test_generic_error_message_is_returned` | 74 | Distinct branch: non-HTTP errors use `str(e)` |
| `test_long_error_text_is_truncated` | 80 | Truncation at `MAX_ERROR_CHARS` — not covered by other error tests |
| `test_valid_dimensions` | 94 | Parser handles whitespace, mixed separators (`x`, `X`, `*`) |
| `test_invalid_dimensions` | 101 | 5 distinct invalid inputs (non-numeric, single-part, out-of-range, zero) |
| `test_invalid_resolution_reported_to_the_model` | 106 | End-to-end: invalid resolution reported, generation never starts |
| `test_send_to_chat_branches` | 139 | Both `send_to_chat=True` and `False` in one test — core branching |
| `test_store_error_reports_delivery` / `test_store_error_without_sending` | 156, 160 | (Consolidate into Group 3, but keep the coverage) |
| `test_attach_returns_image_result` | 191 | `add_to_context=True` returns `ToolCallResult` with image attachment |
| `test_schema_has_no_send_switch` | 224 | Subagent schema differs from parent — observable contract |
| `test_both_roles_see_the_same_name` | 231 | Both roles register same tool name, different functions |
| `test_never_posts_and_reports_the_path` | 240 | Subagent variant never sends to chat, reports sandbox path |
| `test_schema_exposes_provider_for_both_roles` | 252 | Provider enum identical for both roles |
| `test_unknown_provider_reported_to_the_model` | 265 | Unknown provider rejected at tool layer |
| `test_provider_reaches_the_backend` | 271 | Provider parameter passes through to `generate_image` |
| `test_nvidia_flux3_url_and_default_steps` | 311 | (Consolidate into Group 1, keep coverage) |
| `test_nvidia_flux2_url_and_default_steps` | 324 | (Consolidate into Group 1, keep coverage) |
| `test_nvidia_explicit_steps_are_sent` | 337 | (Consolidate into Group 1, keep coverage) |
| `test_content_filter_is_rejected` | 343 | `CONTENT_FILTERED` finishReason → RuntimeError |
| `test_phoenix_defaults_to_25_steps` | 350 | (Consolidate into Group 2, keep coverage) |
| `test_phoenix_json_base64_shape` | 361 | (Consolidate into Group 2, keep coverage) |
| `test_steps_out_of_range_is_rejected` | 369 | Steps validation across all providers (0 and 51 rejected) |
| `test_unknown_provider_raises` | 377 | Direct API test: `generate_image` raises `ValueError` for unknown provider (different layer from line 265) |
| `test_missing_setting_is_reported` | 386 | Credential checks for all 4 providers (5 cases) |
| `test_payload_and_default_aspect_ratio` | 421 | OpenRouter payload shape, default `aspect_ratio='auto'`, `n=6` |
| `test_first_of_several_images_is_kept` | 450 | First image kept when OpenRouter returns multiple |
| `test_no_images_raises` | 457 | (Consolidate into Group 4, keep coverage) |
| `test_missing_b64_raises` | 462 | (Consolidate into Group 4, keep coverage) |
| `test_resolution_is_not_a_parameter` | 467 | (Consolidate into Group 5, keep coverage) |
| `test_invalid_aspect_ratio_is_reported` | 476 | (Consolidate into Group 5, keep coverage) |
| `test_valid_aspect_ratio_reaches_the_render` | 482 | (Consolidate into Group 6, keep coverage) |
| `test_explicit_aspect_ratio_is_sent` | 434 | (Consolidate into Group 6, keep coverage) |
| `test_render_overrunning_the_deadline_is_stopped` | 526 | Deadline enforcement — critical safety behaviour |
| `test_known_magics` | 545 | PNG/JPEG/WEBP magic byte detection |
| `test_unknown_container_raises` | 555 | Unknown container rejected (not silently guessed as PNG) |
| `test_unknown_container_reported_to_the_model` | 562 | End-to-end: unsupported format reported to model |
| `test_webp_stored_with_webp_extension_and_mime` | 573 | WEBP stored with correct extension and MIME |
| `test_webp_sent_as_document_not_photo` | 598 | WEBP delivered as document, not photo (Telegram constraint) |

---

## 4. Summary Counts

| Metric | Count |
|---|---|
| Total tests in file | **43** |
| Recommended for deletion | **3** |
| Tests involved in consolidation | **13** (becoming **6** after merge) |
| Net tests after changes | **33** (43 − 3 − 13 + 6) |
| Tests with no change needed | **27** |

### Highest-value tests that must never be deleted:
- `test_render_overrunning_the_deadline_is_stopped` — deadline enforcement is a safety guarantee
- `test_webp_sent_as_document_not_photo` — Telegram delivery constraint
- `test_unknown_container_raises` — prevents undeliverable files from reaching the user
- `test_long_error_text_is_truncated` — context-flood protection
- `test_steps_out_of_range_is_rejected` — input validation across all providers
- `test_content_filter_is_rejected` — content policy enforcement