# Implementation Plan: CentroidTable

**Branch**: `001-centroid-table` | **Date**: 2026-06-07 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-centroid-table/spec.md`

---

## Summary

Build `CentroidTable` in the `common` crate: an immutable, in-memory `HashMap<u32, String>` mapping centroid ID to shard network address, with JSON and bincode serialization. The table is created by the coordinator after k-means clustering and distributed (via bincode) to the staging shard, which uses it for bulk-assignment during cold-start. Normal partition shards never hold the table.

---

## Technical Context

**Language/Version**: Rust 1.94.0, edition 2024

**Primary Dependencies**:
- `serde 1.x` with `derive` feature — already in `common/Cargo.toml`
- `serde_json 1.x` — to be added to `common/Cargo.toml`
- `bincode 1.x` — to be added to `common/Cargo.toml`
- `std::collections::HashMap` — no new dep

**Storage**: In-memory only (`HashMap<u32, String>`); no persistence in V1

**Testing**: `cargo test` (unit + integration), `cargo bench` (criterion, already in `shard` dev-deps; added to `common`)

**Target Platform**: Linux server, single-machine multi-process for V1

**Project Type**: Library crate within Cargo workspace (`common`)

**Performance Goals**: O(1) lookup (amortized); JSON round-trip ≤ 50 ms at K=512; bincode round-trip ≤ 5 ms at K=512

**Constraints**: K ≤ 512 entries; read-only after construction; no schema versioning; no persistence; `ShardAddress` is a plain `String`

**Scale/Scope**: ≤ 512 centroid entries; single `common` module; no network I/O in this feature

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Proximity-Aware Architecture First | ✅ Pass | `CentroidTable` IS the routing primitive — enables centroid-based shard lookup. No broadcast fan-out introduced. |
| II. Deterministic Distance and Routing | ✅ Pass | Lookup is a pure hash-map get; deterministic for identical inputs. No distance computation in this feature. JSON key order is non-deterministic but consumers use key lookup, not positional access. |
| III. Testable Changes as a Merge Gate | ✅ Pass | All 11 FRs have mapped acceptance tests (see quickstart.md). Public API documented with `# Errors` on all fallible methods. Integration tests under `common/tests/`. |
| IV. Measured Performance and Recall | ✅ Pass | SC-001–SC-003 define concrete time bounds and constant-time benchmarks. No search math changes; no recall measurement required. |
| V. Knowledge Base Synchronization | ✅ Pass | Design decisions logged in research.md and data-model.md. CentroidTable key decisions (HashMap, bincode 1.x, Arc-only thread safety) to be reflected in `.github/` decision log as part of tasks. |

**No violations. Complexity Tracking section omitted.**

---

## Project Structure

### Documentation (this feature)

```
specs/001-centroid-table/
├── plan.md              ← this file
├── research.md          ← Phase 0 — decisions on HashMap, bincode, thread-safety, API shape
├── data-model.md        ← Phase 1 — entities, wire formats, state transitions
├── quickstart.md        ← Phase 1 — Cargo changes, files, verification steps, test checklist
├── contracts/
│   └── public-api.md   ← Phase 1 — stable API contract for coordinator + shard consumers
└── tasks.md             ← Phase 2 (/speckit-tasks — not created by /speckit-plan)
```

### Source Code (repository root)

```
common/
├── Cargo.toml                     ← add serde_json, bincode
├── src/
│   ├── lib.rs                     ← add: pub mod centroid_table; pub use centroid_table::CentroidTable;
│   └── centroid_table.rs          ← NEW: CentroidTable struct, impl, serde derives
└── tests/
    └── centroid_table.rs          ← NEW: integration tests (FR-001 through FR-010 + benchmarks)
```

**Structure decision**: Single `common` crate module. No new crates, no new binaries. Follows the existing workspace boundary: shared types in `common`, consumed by `coordinator` and `shard`.

---

## Phase 0: Research

**Status**: Complete. See [research.md](research.md).

Key decisions resolved:

| Unknown | Decision |
|---------|----------|
| Map type | `HashMap<u32, String>` — O(1) lookup, no ordering needed |
| Serialization libs | `serde_json 1.x` + `bincode 1.x` — both use serde traits, single derive |
| Thread-safety | `Arc<CentroidTable>` (no lock) — immutable after construction |
| ShardAddress type | `String` — no structured parsing in V1 |
| API shape | Named methods (`to_json`, `from_json`, `to_bincode`, `from_bincode`) |
| Existing type alignment | `Centroid` unchanged — complementary, not overlapping |

---

## Phase 1: Design & Contracts

**Status**: Complete.

| Artifact | Path | Status |
|----------|------|--------|
| Data model | [data-model.md](data-model.md) | Done |
| Public API contract | [contracts/public-api.md](contracts/public-api.md) | Done |
| Quickstart | [quickstart.md](quickstart.md) | Done |

**Post-design Constitution re-check**: All five principles hold. No new violations introduced by the design.

---

## Implementation Sequence (for /speckit-tasks)

The following ordered steps are handed to `/speckit-tasks` for decomposition into discrete tasks:

1. Add `serde_json` and `bincode` to `common/Cargo.toml`.
2. Create `common/src/centroid_table.rs` — `CentroidTable` struct with `HashMap<u32, String>`, serde derives, and all public methods (`new`, `get`, `len`, `is_empty`, `to_json`, `from_json`, `to_bincode`, `from_bincode`) with `# Errors` doc on all fallible methods.
3. Expose `CentroidTable` in `common/src/lib.rs` (`pub mod centroid_table; pub use ...`).
4. Create `common/tests/centroid_table.rs` — all 11 integration tests from the quickstart checklist (FR-001 through FR-010 + edge cases).
5. Add criterion benchmarks to `common` (`bench_lookup_constant_time`, `bench_json_round_trip`, `bench_bincode_round_trip`) — verify SC-001, SC-002, SC-003.
6. Run `cargo test -p common` and `cargo clippy -p common -- -D warnings` — all must pass.
7. Update `.github/` decision log with CentroidTable design decisions (Constitution Principle V).
