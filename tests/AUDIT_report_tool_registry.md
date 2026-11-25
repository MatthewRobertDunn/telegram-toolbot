# Audit Report: tests/test_tool_registry.py

**Total tests:** 30  
**Recommended for deletion:** 3  
**Recommended for consolidation:** 4 groups

---

## 1. Tests to DELETE

### `TestDeclaration::test_needs_ctx_is_recorded_on_the_tool` (line 200–214)
**Reason:** Redundant + implementation-detail. The `Tool.needs_ctx` half is already covered by `TestToolSurface::test_subagent_tools_accept_a_silent_context` (line 87–95), which iterates every subagent tool and asserts its first parameter is `AgentContext` when `tool.needs_ctx` is true. The `getattr(func, '_needs_ctx')` half tests a private function attribute consumed by `run_llm.execute_tool_call` — this is cross-module coupling, not the tool registry's observable behaviour. Nothing is lost by removing it.

### `TestRegistryLifecycle::test_successful_load_is_sticky` (line 347–352)
**Reason:** Redundant with `test_loader_runs_once` (line 280–288). Both verify the loader runs exactly once after a successful load. `test_loader_runs_once` is the stronger test — it calls three different lookup functions (`tools_for(PARENT)`, `tools_for(SUBAGENT)`, `all_tools()`) and still asserts one call. `test_successful_load_is_sticky` only repeats `tools_for(PARENT)` twice, adding nothing new.

### `TestRegistryLifecycle::test_unknown_role_is_rejected` (line 402–405)
**Reason:** Trivial. Tests the `if role not in ROLES: raise ValueError` guard in `ToolRegistry.tools_for`. The guard is a two-line bounds check on a tuple literal; every other test in the file exercises the valid-role path heavily. The test provides no additional assurance beyond what a glance at the source confirms.

---

## 2. Tests to CONSOLIDATE

### Group A: Definition-shape checks
**Tests:** `TestDefinitions::test_definition_shape` (line 104–110) + `TestDefinitions::test_no_required_key_when_everything_is_optional` (line 124–127)  
**Suggestion:** Merge into a single parametrised test that checks definition shape across multiple tools — `bash` (has required, strict=False, additionalProperties=False) and `journal_query` (no required key when all optional). The second case is a meaningful edge case of the `definition()` method (it omits the `required` key entirely when the tuple is empty), but testing it in isolation with one tool is anaemic. Parametrising makes the coverage explicit and removes the duplication.

### Group B: ChatUpdateContext boundary
**Tests:** `TestDeclaration::test_interactive_context_cannot_be_offered_to_subagents` (line 185–191) + `TestDeclaration::test_interactive_context_is_fine_for_the_parent` (line 193–198)  
**Suggestion:** These two test the same security boundary (ChatUpdateContext must not reach subagents) from opposite directions — rejection and acceptance. They could be a single parametrised test with cases `(scope=BOTH, should_raise=True)` and `(scope=PARENT, should_raise=False)`. The current separation is clear and readable, so this is a lower-priority consolidation; do it only if the file grows further.

### Group C: Declaration clash detection
**Tests:** `TestDuplicateDeclarations::test_same_scope_is_rejected_with_both_modules` (line 236–246) + `TestDuplicateDeclarations::test_overlapping_audiences_are_rejected` (line 256–265)  
**Suggestion:** Both test `_reject_clash` raising `ValueError` for overlapping audiences. They could be parametrised with scenarios `(PARENT+PARENT, clash)`, `(PARENT+BOTH, clash)`, `(PARENT+SUBAGENT, no clash)`. The third scenario is already covered by `test_same_name_for_disjoint_audiences_is_allowed` (line 248–254), so a parametrised test could cover all three in one place. Again, the current separation is readable; consolidate only if maintenance burden becomes an issue.

### Group D: Failed-load retry path
**Tests:** `TestRegistryLifecycle::test_failed_load_is_not_sticky` (line 290–301) + `TestRegistryLifecycle::test_retry_after_failed_load_succeeds` (line 323–345)  
**Suggestion:** Both test the failed-load → retry path. The first verifies the loader is re-invoked (not sticky), the second verifies the retry actually succeeds and the tool joins the surface. They test different assertions and could be kept separate, but they share the same `flaky_loader` pattern and setup. A single test with two phases (assert retry happens, then assert retry succeeds) would reduce boilerplate. Lower priority — the current split is defensible.

---

## 3. Tests to KEEP

These tests protect important business logic, edge cases, or critical application behaviour and must stay:

| Test | Line | Why it matters |
|---|---|---|
| `test_every_tool_is_offered_somewhere` | 53 | Core invariant: no orphaned tools. Union of audiences == all tools. |
| `test_subagent_surface_has_no_chat_writing_tools` | 57 | **The primary failure this file guards against.** Chat-writing tools must never reach subagents. |
| `test_shared_tools_are_offered_to_both` | 63 | Positive case for shared tools. Complements the above. |
| `test_dual_implementation_keeps_one_name` | 70 | Protects the shared-vocabulary convention (no `_subagent` suffix). |
| `test_dual_implementations_differ` | 80 | Verifies dual-implementation tools actually have different code, not just different entries. |
| `test_subagent_tools_accept_a_silent_context` | 87 | Structural half of the silent-context guarantee: subagent tools take `AgentContext`, not `ChatUpdateContext`. |
| `test_required_is_derived_from_the_signature` | 112 | Schema/signature drift detection across every tool. High-value invariant. |
| `test_every_parameter_is_declared_and_callable` | 129 | Forward-direction schema/signature check. Complements `_check`'s reverse validation. |
| `test_unknown_scope_is_rejected` | 153 | Basic validation of the scope parameter. (Keep despite being simple — it documents the contract.) |
| `test_parameter_not_in_signature_is_rejected` | 159 | Schema→signature validation. |
| `test_signature_parameter_not_in_schema_is_rejected` | 166 | Signature→schema validation, plus error message content check. |
| `test_required_without_declaration_is_rejected` | 179 | Required→schema validation. |
| `test_interactive_context_cannot_be_offered_to_subagents` | 185 | **Security boundary:** ChatUpdateContext + BOTH scope must be rejected. |
| `test_interactive_context_is_fine_for_the_parent` | 193 | Positive case for the same boundary. |
| `test_same_scope_is_rejected_with_both_modules` | 236 | Clash detection: same name, same scope, different modules. |
| `test_same_name_for_disjoint_audiences_is_allowed` | 248 | Positive case: PARENT + SUBAGENT for the same name is the dual-implementation pattern. |
| `test_overlapping_audiences_are_rejected` | 256 | Clash detection: PARENT + BOTH overlap. |
| `test_loader_runs_once` | 280 | Lazy loading: loader runs exactly once. |
| `test_failed_load_is_not_sticky` | 290 | Failed loads are retried, not cached. |
| `test_failed_load_reports_the_real_error_not_a_clash` | 303 | Rollback: partial declarations are rolled back on failure. **Critical resilience test.** |
| `test_retry_after_failed_load_succeeds` | 323 | Retry succeeds after failure; tool joins the surface. |
| `test_declaration_after_load_joins_the_surface` | 354 | Index invalidation: late declarations are visible on next lookup. |
| `test_reload_of_a_tool_module_is_rejected_not_duplicated` | 373 | **Concrete reload failure mode:** `importlib.reload` raises instead of doubling the surface. |
| `test_default_loader_is_bound_to_its_registry` | 407 | Fresh `ToolRegistry` binds its loader to itself, not the module global. |
| `test_default_loader_fills_a_fresh_registry` | 421 | End-to-end: fresh registry loads real tool modules. |

---

## 4. Summary Counts

| Metric | Count |
|---|---|
| Total tests in file | 30 |
| Recommended for deletion | 3 |
| Recommended for consolidation (groups) | 4 |
| Tests to keep (including consolidated forms) | 27 |

### Deletion impact
After removing the 3 tests, the file still has 27 tests covering all major behaviours: audience surface invariants, schema/signature conformance, declaration validation, clash detection, and registry lifecycle (lazy load, rollback, retry, reload). No business-logic gap is introduced.

### Consolidation impact
The 4 consolidation groups are lower-priority refactors. Group A (definition shape) is the most worthwhile — it removes a single-case anaemic test while preserving its edge-case coverage through parametrisation. Groups B–D are readability improvements that can be pursued when the file grows further or if maintenance burden becomes a concern.