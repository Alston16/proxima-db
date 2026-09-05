# Spec Quality Checklist: CentroidTable

**Purpose**: Requirements quality review of spec.md — validates completeness, clarity, consistency, measurability, and scenario coverage. Serves as both author pre-flight and PR review gate.
**Created**: 2026-06-07
**Feature**: [spec.md](../spec.md)
**Audience**: Author (pre-implementation) + Peer Reviewer (PR gate)
**Primary risk depth**: Concurrency requirements (FR-010)
**Scope**: spec.md only

---

## Requirement Completeness

- [ ] CHK001 — Does FR-003 specify the behavior when the input iterator yields duplicate centroid IDs — is last-write-wins documented as the rule, or is it undefined? [Completeness, Spec §FR-003, Gap]
- [ ] CHK002 — Are the exact error return types for `from_json` and `from_bincode` specified in requirements, or only described as "descriptive error"? [Completeness, Spec §FR-006, FR-007]
- [ ] CHK003 — Is there a requirement for `is_empty` — does the spec state what it must return for a zero-entry table, or is it implied by FR-008 (`len`)? [Completeness, Gap]
- [ ] CHK004 — Does the spec state whether `len()` (FR-008) counts centroid IDs or distinct shard addresses? The distinction matters when multiple centroids map to the same address. [Completeness, Spec §FR-008]
- [ ] CHK005 — Is the "no schema versioning" decision (Clarification Q5) reflected as an explicit functional requirement (e.g., FR-012) or only documented in Clarifications and Assumptions? [Completeness, Spec §Clarifications]
- [ ] CHK006 — Are requirements defined for what happens when the coordinator distributes an empty CentroidTable (K=0) to the staging shard — can bulk-assignment proceed, error, or is this case excluded? [Completeness, Spec §User Story 2, Gap]

---

## Requirement Clarity

- [ ] CHK007 — Is "constant time" in FR-002 qualified as amortized O(1) or worst-case O(1)? HashMap lookup is amortized; the distinction matters when defining the SC-003 benchmark methodology. [Clarity, Spec §FR-002]
- [ ] CHK008 — Is "well-defined 'not found' result" in FR-009 specific enough to distinguish between `Option::None`, a sentinel return value, or a typed error variant? [Clarity, Spec §FR-009]
- [ ] CHK009 — Does the spec define what "100% fidelity" means for the JSON round-trip (SC-001) — specifically, does key ordering need to be preserved, or only that all K mappings survive? [Clarity, Spec §SC-001]
- [ ] CHK010 — Is "byte-for-byte equivalent to the original" (User Story 2, Acceptance Scenario 2) referring to the deserialized table's logical contents or to the raw serialized bytes? If bytes, is this achievable given non-deterministic HashMap iteration order at serialization time? [Clarity, Conflict, Spec §User Story 2]
- [ ] CHK011 — Does "valid JSON" in FR-004 mean RFC 8259 compliance only, or also human-parseable by standard tooling (e.g., `jq`, `python -m json.tool`)? [Clarity, Spec §FR-004]
- [ ] CHK012 — Is "human-readable" in User Story 3 and Acceptance Scenario 1 defined with measurable criteria (e.g., field names match the entity names in Key Entities), or is it subjective? [Clarity, Spec §User Story 3]

---

## Concurrency Requirements (Primary Risk)

- [ ] CHK013 — Does FR-010 specify which Rust safety bound is required — `Sync` (safe to share `&CentroidTable` across threads), `Send` (safe to transfer ownership), or both? The bound determines what wrapper types are valid. [Clarity, Spec §FR-010]
- [ ] CHK014 — Is there a requirement prescribing the caller-side sharing mechanism for concurrent access — e.g., must callers use `Arc<CentroidTable>`, or is any safe wrapper acceptable? [Completeness, Spec §FR-010]
- [ ] CHK015 — Does the spec define whether FR-010 covers concurrent reads during the construction phase — i.e., is it valid for one thread to call `get()` while another thread is still calling `CentroidTable::new()`? [Completeness, Spec §FR-010, Edge Case]
- [ ] CHK016 — Is there a requirement addressing what happens if a lookup is attempted on a partially constructed table — is this ruled out by assumption or must the API prevent it by design? [Completeness, Spec §FR-010, Spec §Assumptions]
- [ ] CHK017 — Does FR-010 apply to concurrent serialization — i.e., are two threads calling `to_bincode()` simultaneously on the same table in scope for the concurrency guarantee? [Completeness, Spec §FR-010, Gap]
- [ ] CHK018 — Does the spec state whether FR-010's concurrency requirement applies within a single process only, or also across processes sharing memory (e.g., via shared-memory IPC)? [Clarity, Spec §FR-010]
- [ ] CHK019 — Is there a requirement for memory visibility — e.g., does a thread that receives a `CentroidTable` via `Arc::clone` always observe the fully-populated table contents, or is this left to Rust's ownership model implicitly? [Completeness, Spec §FR-010, Gap]
- [ ] CHK020 — Is the concurrency requirement in FR-010 consistent with the Assumptions section ("read-only during operational lifetime") — does the spec clarify that "concurrent writes never occur after init" is an assumption, not a runtime-enforced invariant? [Consistency, Spec §FR-010, Spec §Assumptions]

---

## Serialization Requirements

- [ ] CHK021 — Is the JSON key format (centroid IDs as decimal strings vs. JSON numeric literals) specified in the requirements? `serde_json` serializes integer-keyed maps as string keys; is this documented? [Completeness, Spec §FR-004, Gap]
- [ ] CHK022 — Are maximum serialized size bounds defined — e.g., maximum acceptable JSON or bincode payload size at K=512? [Completeness, Gap]
- [ ] CHK023 — Does the spec define behavior when `from_bincode` receives bytes that are valid bincode but represent a different type entirely (type confusion)? [Edge Case, Spec §FR-007]
- [ ] CHK024 — Is the bincode round-trip fidelity requirement (User Story 2, Scenario 2) consistent with the unordered HashMap clarification (Clarification Q2)? Two separately deserialized tables with the same mappings may differ in internal hash state, producing different bincode bytes on re-serialization. [Conflict, Spec §User Story 2, Spec §Clarifications]
- [ ] CHK025 — Is the requirement that both JSON and bincode "must coexist" (Assumptions) stated as a functional requirement, or only as an assumption? Should it be promoted to an explicit FR? [Completeness, Spec §Assumptions, Gap]

---

## Acceptance Criteria Quality

- [ ] CHK026 — Are the 50 ms (JSON) and 5 ms (bincode) time bounds in SC-001 and SC-002 defined with a measurement methodology — e.g., single run, median of N runs, p95, cold vs. warm? [Measurability, Spec §SC-001, SC-002]
- [ ] CHK027 — Is "development hardware" in SC-001 and SC-002 defined with a minimum hardware specification, or is the bound open to interpretation across different machines? [Clarity, Spec §SC-001, SC-002]
- [ ] CHK028 — Can SC-003 ("lookup time does not grow linearly") be objectively verified — are the benchmark K values, iteration count, and statistical method specified? [Measurability, Spec §SC-003]
- [ ] CHK029 — Is SC-004 ("all unit tests pass") a measurable success criterion, or is it circular in that passing depends on what tests are written? Should it reference a specific test list (e.g., quickstart checklist)? [Measurability, Spec §SC-004]
- [ ] CHK030 — Is SC-005 ("100% of test cases, no panics") anchored to a defined test corpus, or is the set unbounded? [Measurability, Spec §SC-005]

---

## Scenario Coverage

- [ ] CHK031 — Are requirements defined for the scenario where `get()` is called immediately after constructing an empty table (K=0 via empty iterator)? [Edge Case, Spec §Edge Cases, Spec §FR-009]
- [ ] CHK032 — Does the spec address the scenario where two centroid IDs map to the same shard address and a bulk-assignment is in progress — are there any ordering or consistency requirements? [Coverage, Spec §Edge Cases]
- [ ] CHK033 — Is there a requirement for re-requesting the CentroidTable from the coordinator if the staging shard drops it prematurely (e.g., crash before bulk-assignment completes) — or is this explicitly deferred to Stage 3/Stage 5? [Coverage, Spec §Assumptions, Gap]
- [ ] CHK034 — Does User Story 1 (coordinator query routing) cover the scenario where `get()` is called with a centroid ID that was never registered — and is the "not found" path's behavior in that routing context specified? [Coverage, Spec §User Story 1, Spec §FR-009]

---

## Dependencies & Assumptions

- [ ] CHK035 — Is the assumption that k-means output contains no duplicate centroid IDs stated as a validated precondition or as an unchecked assumption? If unchecked, does the spec define behavior when duplicates occur? [Assumption, Spec §Assumptions, Spec §FR-003]
- [ ] CHK036 — Is the dependency on Stage 2 Step 1 (k-means) traceable to a specific output interface, or described only in prose? If the k-means output format changes, would FR-003 still hold? [Dependency, Spec §Assumptions]
- [ ] CHK037 — Is the assumption that the staging shard receives the serialized table as a single atomic payload (not in chunks) stated explicitly — and does the spec define behavior for partial receipts? [Assumption, Gap]

---

## Ambiguities & Conflicts

- [ ] CHK038 — Does FR-001 ("exactly one shard address per centroid ID") risk confusion with the edge case "two centroids registered to the same shard address"? These are different relationships but the phrasing is easy to misread. [Clarity, Spec §FR-001, Spec §Edge Cases]
- [ ] CHK039 — Is there a conflict between the Staging Shard lifecycle assumption ("drops table on deactivation") and FR-011 ("coordinator distributes to staging shard only") — if a second cold-start cycle begins, does FR-011 require re-distribution, and is that covered? [Conflict, Spec §FR-011, Spec §Assumptions]
- [ ] CHK040 — Does the spec use "centroid ID" and "centroid" interchangeably? If so, is there a risk of conflating the `CentroidTable`'s key type (`u32`) with the `Centroid` struct in `common` which also has an `id: u32` field? [Ambiguity, Spec §Key Entities]

---

## Notes

- Check items off as completed: `[x]`
- Add findings inline (e.g., "— confirmed: FR-009 uses Option<&str> per contracts/public-api.md")
- Items marked `[Gap]` indicate requirements absent from the spec — resolve by adding an FR or documenting as intentionally out of scope
- Items marked `[Conflict]` indicate potential inconsistencies between spec sections — resolve by updating one or both sections
- Items marked `[Assumption]` indicate unchecked assumptions that may need promotion to explicit requirements
- CHK010, CHK024 (byte-for-byte + HashMap ordering) are the highest-priority items — resolve before implementation starts
