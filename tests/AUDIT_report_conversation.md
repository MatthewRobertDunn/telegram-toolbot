# Audit Report: tests/test_conversation.py

**Total tests in area:** 42
**Recommended for deletion:** 2
**Recommended for consolidation:** 5 pairs (reducing 10 tests → ~5)

---

## 1. Tests to DELETE

### `test_default_is_none` — `tests/test_conversation.py:131`
**Reason:** Trivial. Tests that `message.message_type` returns `None` when no `custom` metadata exists. This is guaranteed by `custom_of(message).get('message_type')` returning `None` for a missing key — a plain `dict.get()` behaviour, not application logic. No meaningful assurance.

### `test_max_images_defaults_to_MAX_IMAGES` — `tests/test_conversation.py:190`
**Reason:** Trivial. Tests that `Conversation().max_images == MAX_IMAGES`. The constructor signature is literally `max_images: int = MAX_IMAGES`; this test verifies Python's default-parameter mechanism, not any behaviour of the class. The constant `MAX_IMAGES` is also already exercised indirectly by `test_only_the_most_recent_max_images_are_kept` (line 224), which uses the default and verifies pruning works.

---

## 2. Tests to CONSOLIDATE

### Pair 1: Wrapping behaviour (3 tests → 1–2)
| Test | Line |
|---|---|
| `test_constructor_wraps_plain_dicts` | 19 |
| `test_wrapping_keeps_your_messages` | 27 |
| `test_set_messages_wraps_plain_dicts_too` | 32 |

**Suggested merge:** Keep `test_wrapping_keeps_your_messages` (it tests identity preservation — a distinct contract: already-wrapped messages are not re-wrapped). Merge the other two into a single parametrised test covering both the constructor and `set_messages` entry points, since they test the exact same wrapping logic (`isinstance(message, ConversationMessage)` check).

### Pair 2: add_image output format (2 tests → 1)
| Test | Line |
|---|---|
| `test_add_image_appends_a_caption_and_base64_image` | 202 |
| `test_add_image_without_caption_has_just_the_image` | 215 |

**Suggested merge:** Parametrise on the caption argument. Both tests verify the same `add_image` output structure; the only difference is whether a text caption part is prepended. One parametrised test with `caption='look at this'` and `caption=None` covers both.

### Pair 3: Pruning with different max_images values (2 tests → 1)
| Test | Line |
|---|---|
| `test_max_images_is_a_constructor_parameter` | 172 |
| `test_only_the_most_recent_max_images_are_kept` | 224 |

**Suggested merge:** Parametrise on `max_images` (e.g. 2 and the default `MAX_IMAGES`). Both tests verify the same pruning invariant: only the N most recent images survive, and older ones merge their placeholder into the caption. The first test also implicitly covers the constructor-parameter path, which the second does not, but the pruning assertion is identical.

### Pair 4: Conversation equality (2 tests → 1)
| Test | Line |
|---|---|
| `test_equals_a_message_list_with_the_same_contents` | 343 |
| `test_two_conversations_with_same_messages_are_equal_but_distinct_ids` | 349 |

**Suggested merge:** One test covering both `Conversation == list` and `Conversation == Conversation` comparisons, plus the distinct-ids assertion. The second test adds the important assertion that equal conversations have different ids (equality is content-based, not identity-based), so don't delete it — just fold the first into it.

### Pair 5: prepare_regeneration scenarios (2 tests → 1)
| Test | Line |
|---|---|
| `test_prepare_regeneration_drops_thinking_and_answer` | 406 |
| `test_prepare_regeneration_drops_orphan_thinking` | 418 |

**Suggested merge:** Parametrise on the trailing message sequence (thinking+answer vs orphan thinking alone). Both verify the same `prepare_regeneration` contract: thinking blocks are dropped, the user prompt remains, and the action is `ANSWER`. The second test's orphan-thinking edge case is worth keeping but can be a parametrised variant rather than a standalone test.

---

## 3. Tests to KEEP (high-value)

These tests protect important business logic, edge cases, or error handling and must stay:

| Test | Line | Why keep |
|---|---|---|
| `test_id_survives_mutations` | 44 | Identity stability is a core contract — the session UUID must not change across mutations. |
| `test_add_helpers_store_telegram_message_ids_in_custom` | 55 | Verifies both `add_user` and `add_assistant` store Telegram ids, covering single-int and list inputs. |
| `test_no_telegram_message_id_leaves_no_custom_key` | 63 | Important edge case: no ids → no `custom` key at all (not an empty container). |
| `test_add_telegram_message_id_collects_every_chunk` | 70 | Verifies accumulation across multiple `add_telegram_message_id` calls. |
| `test_to_openai_strips_custom_metadata` | 78 | Critical: verifies `custom` is stripped from OpenAI payload but retained in stored messages. |
| `test_set_system_head_creates_head_when_missing` | 106 | Core system-head management: creation, in-place replacement, no stacking. |
| `test_roundtrip_sets_the_tag` | 135 | Basic get/set round-trip for `message_type`. |
| `test_setting_none_clears_the_tag` | 142 | Important edge case: untagging removes the key and cleans up empty `custom` container. |
| `test_add_user_accepts_a_message_type` | 150 | Verifies `message_type` flows through `add_user` and is stripped by `to_openai`. |
| `test_unknown_persisted_tag_is_not_a_known_type` | 159 | Backward-compatibility guard: unknown tags read as `None`, don't raise. |
| `test_max_images_below_one_is_rejected` | 193 | Input validation: nonsensical caps raise `ValueError`. |
| `test_only_the_most_recent_max_images_are_kept` | 224 | Core pruning invariant. |
| `test_pruning_leaves_messages_without_images_untouched` | 239 | Important: non-image messages are not affected by pruning. |
| `test_text_sharing_a_message_with_a_pruned_image_is_kept` | 251 | Edge case: text in the same message as a pruned image survives via placeholder merge. |
| `test_pruned_image_without_caption_becomes_the_placeholder` | 264 | Edge case: pruned image with no preceding text becomes a standalone placeholder. |
| All 6 `TestOpenAiContract` tests (278–338) | — | These are the highest-value tests in the file. They verify the endpoint contract: well-formed detection, multiple-system detection, off-head system detection, unanswered tool calls, orphan tool results, and thinking-block exclusion. None are redundant. |
| `test_add_thinking_tags_an_assistant_message` | 360 | Verifies thinking block tagging and Telegram id attachment. |
| `test_to_openai_skips_thinking` | 368 | Critical: thinking blocks are excluded from the OpenAI payload. |
| `test_unknown_tag_is_treated_as_not_thinking` | 377 | Backward-compatibility: unknown tags are not treated as thinking. |
| `test_token_estimate_skips_thinking` | 389 | Verifies token estimation excludes thinking blocks. |
| `test_undo_removes_thinking_with_the_exchange` | 397 | Verifies undo removes thinking blocks along with the exchange. |
| `test_clone_for_impersonation_has_a_fresh_id` | 428 | Verifies clone gets a fresh id and appends the impersonation prompt without touching the original. |

---

## 4. Summary Counts

| Metric | Count |
|---|---|
| Total tests in `test_conversation.py` | 42 |
| Recommended for deletion | 2 |
| Recommended for consolidation (pairs) | 5 (affecting 10 tests → ~5 after merge) |
| Recommended for keep (high-value) | 32 |

### Low-value / marginal tests (not deleted, but flagged)
These don't meet the deletion threshold but offer thin assurance relative to their cost:

- `test_add_returns_the_message` (line 92) — tests that `add()` returns the message and appends it. Basic contract check, but the return value is exercised by virtually every other test that calls `add_user`/`add_assistant`.
- `test_add_tool_carries_the_tool_call_id` (line 98) — tests that a kwarg becomes a dict key. Trivial, but `add_tool` is used by `validate_openai_contract` tests so the format is indirectly verified.
- `test_head_is_system` (line 119) — tests a simple boolean property. The empty-conversation and non-system-head cases are the only non-trivial assertions; the `True` case is covered by `test_set_system_head_creates_head_when_missing`.
- `test_max_images_defaults_to_MAX_IMAGES` (line 190) — already listed for deletion.