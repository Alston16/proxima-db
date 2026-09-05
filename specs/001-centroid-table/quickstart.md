# Quickstart: CentroidTable

**Branch**: `001-centroid-table` | **Date**: 2026-06-07

---

## What ships

A new module `common::centroid_table` exposing `CentroidTable` — an in-memory hash map from centroid ID (`u32`) to shard network address (`String`), with JSON and bincode serialization.

---

## Cargo changes required

Add to `common/Cargo.toml`:

```toml
[dependencies]
serde      = { version = "1", features = ["derive"] }   # already present
serde_json = "1"
bincode    = "1"
```

No changes needed to `coordinator/Cargo.toml` or `shard/Cargo.toml` — they already depend on `common`.

---

## Files to create

```
common/src/centroid_table.rs   ← new module (struct + impl + serde derives)
common/tests/centroid_table.rs ← integration tests (FR-001 through FR-010)
```

Add to `common/src/lib.rs`:

```rust
pub mod centroid_table;
pub use centroid_table::CentroidTable;
```

---

## Verification steps

```bash
# All unit + integration tests pass
cargo test -p common

# Benchmark: lookup is O(1)
cargo bench -p common

# No new clippy warnings
cargo clippy -p common -- -D warnings

# Doc check (public API has # Errors documented)
cargo doc -p common --no-deps
```

---

## Test coverage checklist (maps to FRs)

| Test | FR |
|------|----|
| `test_construction_and_lookup` | FR-001, FR-002, FR-003 |
| `test_lookup_missing_id` | FR-009 |
| `test_len_and_is_empty` | FR-008 |
| `test_json_round_trip` | FR-004, FR-006 |
| `test_bincode_round_trip` | FR-005, FR-007 |
| `test_json_malformed` | FR-006 (error path) |
| `test_bincode_malformed` | FR-007 (error path) |
| `test_empty_table_serialize` | FR-004, FR-005 |
| `test_concurrent_reads` | FR-010 |
| `test_same_address_multiple_centroids` | Edge case |
| `bench_lookup_constant_time` | SC-003 |
| `bench_json_round_trip` | SC-001 |
| `bench_bincode_round_trip` | SC-002 |
