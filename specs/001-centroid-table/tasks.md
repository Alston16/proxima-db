# Tasks: CentroidTable

**Input**: Design documents from `specs/001-centroid-table/`

**Prerequisites**: plan.md âœ… | spec.md âœ… | research.md âœ… | data-model.md âœ… | contracts/ âœ…

**Tests**: Included â€” required by Constitution Principle III and Success Criterion SC-004.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

---

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on in-progress tasks)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Exact file paths are included in all descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add dependencies and wire up the new module â€” unblocks all subsequent work.

- [x] T001 Add `serde_json = "1"` and `bincode = "1"` to `[dependencies]` in `common/Cargo.toml`
- [x] T002 [P] Add `criterion = { version = "0.5", features = ["html_reports"] }` to `[dev-dependencies]` and `[[bench]] name = "centroid_table" harness = false` to `common/Cargo.toml`
- [x] T003 [P] Add `pub mod centroid_table;` and `pub use centroid_table::CentroidTable;` to `common/src/lib.rs`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Define the `CentroidTable` struct and its constructor â€” every user story depends on this.

**âš ï¸ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T004 Create `common/src/centroid_table.rs` â€” define `pub struct CentroidTable` with private `entries: HashMap<u32, String>` field; add `use std::collections::HashMap;` and `use serde::{Deserialize, Serialize};`; derive `#[derive(Debug, Clone, Serialize, Deserialize)]`
- [x] T005 Implement `pub fn new(entries: impl IntoIterator<Item = (u32, String)>) -> Self` in `common/src/centroid_table.rs` â€” collects into `HashMap` (last-write-wins on duplicate centroid IDs); rustdoc explains duplicate behaviour; `Panics: Never`

**Checkpoint**: `cargo build -p common` succeeds â€” foundation ready for all user stories.

---

## Phase 3: User Story 1 â€” Query Routing (Priority: P1) ðŸŽ¯ MVP

**Goal**: The coordinator can look up a shard address by centroid ID in O(1); concurrent reads are safe via `Arc<CentroidTable>`.

**Independent Test**: `cargo test -p common test_construction_and_lookup test_lookup_missing_id test_len_and_is_empty test_concurrent_reads test_same_address_multiple_centroids`

### Implementation for User Story 1

- [x] T006 [US1] Implement `pub fn get(&self, centroid_id: u32) -> Option<&str>` in `common/src/centroid_table.rs` â€” delegates to `self.entries.get(&centroid_id).map(String::as_str)`; rustdoc with `Panics: Never` (FR-002, FR-009)
- [x] T007 [US1] Implement `pub fn len(&self) -> usize` and `pub fn is_empty(&self) -> bool` in `common/src/centroid_table.rs` â€” delegate to `self.entries.len()` / `self.entries.is_empty()`; add rustdoc for each (FR-008)
- [x] T008 [US1] Add struct-level rustdoc on `CentroidTable` in `common/src/centroid_table.rs` documenting `Arc<CentroidTable>` as the safe sharing pattern for concurrent reads; add compile-time `Send + Sync` assertion: `const fn _assert_send_sync<T: Send + Sync>() {} const _: () = _assert_send_sync::<CentroidTable>();` (FR-010, Constitution III)

### Tests for User Story 1

- [x] T009 [P] [US1] Write `test_construction_and_lookup` in `common/tests/centroid_table.rs` â€” construct table with K=64 entries mapping centroid IDs 0â€“63 to addresses `"127.0.0.1:70XX"`; assert `get(42)` returns `Some("127.0.0.1:7042")` (FR-001, FR-002, FR-003)
- [x] T010 [P] [US1] Write `test_lookup_missing_id` in `common/tests/centroid_table.rs` â€” call `get(999)` on a table with max centroid ID 63; assert `None` is returned without panic (FR-009)
- [x] T011 [P] [US1] Write `test_len_and_is_empty` in `common/tests/centroid_table.rs` â€” assert empty table has `len() == 0` and `is_empty() == true`; assert a 3-entry table has `len() == 3` and `is_empty() == false` (FR-008)
- [x] T012 [P] [US1] Write `test_concurrent_reads` in `common/tests/centroid_table.rs` â€” wrap a K=64 table in `Arc::new`; spawn 8 threads via `std::thread::spawn`, each calling `get()` 1000 times on random centroid IDs; assert all returned values match expected addresses and no thread panics (FR-010)
- [x] T013 [P] [US1] Write `test_same_address_multiple_centroids` in `common/tests/centroid_table.rs` â€” map centroid IDs 0, 1, 2 to the same address `"127.0.0.1:7001"`; assert all three `get()` calls return `Some("127.0.0.1:7001")` (Edge case â€” one shard hosts multiple partitions)

**Checkpoint**: `cargo test -p common` passes â€” US1 fully functional and independently testable. MVP deliverable.

---

## Phase 4: User Story 2 â€” Distribution via Serialization (Priority: P2)

**Goal**: `CentroidTable` serializes to JSON and bincode and deserializes back with 100% fidelity; malformed payloads return descriptive errors without panicking.

**Independent Test**: `cargo test -p common test_bincode_round_trip test_bincode_malformed test_json_round_trip test_json_malformed`

### Implementation for User Story 2

- [x] T014 [US2] Implement `pub fn to_bincode(&self) -> Result<Vec<u8>, bincode::Error>` and `pub fn from_bincode(bytes: &[u8]) -> Result<Self, bincode::Error>` in `common/src/centroid_table.rs`; add `# Errors` rustdoc on both; use `bincode::serialize` / `bincode::deserialize` (FR-005, FR-007)
- [x] T015 [US2] Implement `pub fn to_json(&self) -> Result<String, serde_json::Error>` and `pub fn from_json(json: &str) -> Result<Self, serde_json::Error>` in `common/src/centroid_table.rs`; add `# Errors` rustdoc on both; document JSON shape `{ "entries": { "<centroid_id>": "<host:port>" } }` and that key order is non-deterministic (FR-004, FR-006)

### Tests for User Story 2

- [x] T016 [P] [US2] Write `test_bincode_round_trip` in `common/tests/centroid_table.rs` â€” construct K=64 table, call `to_bincode()` (assert `Ok`), call `from_bincode()` on the bytes (assert `Ok`), assert all 64 centroid-to-address mappings match the original (FR-005, FR-007)
- [x] T017 [P] [US2] Write `test_bincode_malformed` in `common/tests/centroid_table.rs` â€” call `from_bincode(b"not valid bincode garbage")`, assert `Err` is returned and no panic occurs; assert no side effects on existing table instances (FR-007 error path)
- [x] T018 [P] [US2] Write `test_json_round_trip` in `common/tests/centroid_table.rs` â€” construct K=64 table, call `to_json()` (assert `Ok` and non-empty string), call `from_json()` on the output (assert `Ok`), assert all 64 mappings survive (FR-004, FR-006)
- [x] T019 [P] [US2] Write `test_json_malformed` in `common/tests/centroid_table.rs` â€” call `from_json("{not: valid")`, assert `Err` returned and no panic; also call `from_json("{\"entries\": \"wrong_type\"}")` and assert `Err` (FR-006 error path)

**Checkpoint**: `cargo test -p common` passes â€” US1 and US2 both fully functional.

---

## Phase 5: User Story 3 â€” Inspection and Debugging (Priority: P3)

**Goal**: A developer can export `CentroidTable` to JSON, read human-readable output with labelled fields, and re-import without loss; an empty table serializes without error.

**Independent Test**: `cargo test -p common test_json_human_readable test_empty_table_serialize`

**Note**: No new implementation required â€” US3 reuses `to_json()` and `from_json()` from Phase 4.

### Tests for User Story 3

- [x] T020 [P] [US3] Write `test_json_human_readable` in `common/tests/centroid_table.rs` â€” serialize a 3-entry table to JSON; parse the raw JSON string and assert it contains a top-level `"entries"` key; assert centroid IDs appear as decimal string keys (e.g., `"0"`, `"1"`, `"2"`); assert shard addresses appear as string values (US3 Acceptance Scenario 1, FR-004)
- [x] T021 [P] [US3] Write `test_empty_table_serialize` in `common/tests/centroid_table.rs` â€” construct `CentroidTable::new(std::iter::empty())`; call `to_json()` and assert `Ok` with a non-empty string; call `to_bincode()` and assert `Ok`; call `from_json()` and `from_bincode()` on the outputs and assert both produce a table where `len() == 0` and `is_empty() == true` (US3 Acceptance Scenario 2, FR-004, FR-005)

**Checkpoint**: `cargo test -p common` passes â€” all three user stories fully functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Benchmarks, lint, doc, and knowledge-base update.

- [x] T022 [P] Create `common/benches/centroid_table.rs` with `bench_lookup_constant_time` â€” use `criterion::Criterion`; benchmark `CentroidTable::get()` at K=64, K=128, K=256, K=512; confirm throughput does not degrade linearly with K (SC-003)
- [x] T023 [P] Add `bench_json_round_trip` and `bench_bincode_round_trip` to `common/benches/centroid_table.rs` â€” K=512 entries; measure median round-trip time; targets: JSON â‰¤ 50 ms, bincode â‰¤ 5 ms (SC-001, SC-002)
- [x] T024 Run `cargo bench -p common` and record results; confirm SC-001 (â‰¤ 50 ms JSON), SC-002 (â‰¤ 5 ms bincode), SC-003 (constant-time lookup) are met; paste criterion output summary as a comment in `specs/001-centroid-table/plan.md` under a `## Benchmark Results` heading
- [x] T025 Run `cargo test -p common` and confirm all tests pass with zero failures (SC-004)
- [x] T026 [P] Run `cargo clippy -p common -- -D warnings` and fix all lint warnings
- [x] T027 [P] Run `cargo doc -p common --no-deps` and confirm no missing-doc warnings; verify `# Errors` section present on `to_json`, `from_json`, `to_bincode`, `from_bincode` (Constitution III)
- [x] T028 Add a `centroid-table` entry to `.github/` decision log (or create `centroid-table.md` in the relevant `.github/` subfolder) documenting: HashMap chosen for O(1) lookup, bincode 1.x for serde compatibility, `Arc<CentroidTable>` as the concurrency model, no schema versioning in V1 (Constitution V)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies â€” start immediately; T002 and T003 are parallel to each other
- **Foundational (Phase 2)**: Depends on T001 (serde_json/bincode in Cargo.toml) and T003 (module wired up); T004 before T005
- **US1 (Phase 3)**: Depends on T005 (new() complete); T006 before T007, T008; T009â€“T013 are independent of each other [P]
- **US2 (Phase 4)**: Depends on Phase 3 complete; T014 and T015 are independent [P]; T016â€“T019 depend on T014/T015 respectively but are independent of each other [P]
- **US3 (Phase 5)**: Depends on T015 (to_json/from_json); T020 and T021 are independent [P]
- **Polish (Phase 6)**: Depends on all user story phases complete; T022/T023 parallel; T026/T027 parallel

### User Story Dependencies

- **US1 (P1)**: Starts after Phase 2 â€” no dependency on US2 or US3
- **US2 (P2)**: Starts after Phase 3 complete â€” builds on same struct, adds serialization methods
- **US3 (P3)**: Starts after T015 (to_json/from_json implemented) â€” no new implementation; tests only

---

## Parallel Opportunities

### Phase 1 (can all run together after T001)

```
T002  â†’  add criterion bench config to common/Cargo.toml
T003  â†’  wire up pub mod in common/src/lib.rs
```

### Phase 3 â€” US1 tests (all independent once T006/T007/T008 done)

```
T009  â†’  test_construction_and_lookup
T010  â†’  test_lookup_missing_id
T011  â†’  test_len_and_is_empty
T012  â†’  test_concurrent_reads
T013  â†’  test_same_address_multiple_centroids
```

### Phase 4 â€” US2 (T014 and T015 in parallel; then tests in parallel)

```
T014  â†’  to_bincode / from_bincode
T015  â†’  to_json / from_json

T016  â†’  test_bincode_round_trip     (after T014)
T017  â†’  test_bincode_malformed      (after T014)
T018  â†’  test_json_round_trip        (after T015)
T019  â†’  test_json_malformed         (after T015)
```

### Phase 5 â€” US3 tests (both independent, after T015)

```
T020  â†’  test_json_human_readable
T021  â†’  test_empty_table_serialize
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001â€“T003)
2. Complete Phase 2: Foundational (T004â€“T005)
3. Complete Phase 3: User Story 1 (T006â€“T013)
4. **STOP and VALIDATE**: `cargo test -p common` â€” US1 tests pass
5. Coordinator has a working `CentroidTable` for query routing

### Incremental Delivery

1. Setup + Foundational â†’ struct compiles âœ…
2. US1 complete â†’ lookup works, concurrent reads safe â†’ **MVP**
3. US2 complete â†’ JSON + bincode round-trips work â†’ distribution ready
4. US3 complete â†’ human-readable JSON verified â†’ inspection/debugging ready
5. Polish â†’ benchmarks green, clippy clean, docs complete

---

## Notes

- `[P]` tasks touch different files â€” safe to parallelize
- `[Story]` labels map each task to a user story for traceability
- Each user story phase is independently testable via the listed `cargo test` filter
- Commit after each phase checkpoint at minimum
- Constitution Principle III: all public `Result`-returning methods must have `# Errors` rustdoc before merge
- Constitution Principle V: `.github/` decision log update (T028) is required â€” not optional

