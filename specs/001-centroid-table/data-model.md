# Data Model: CentroidTable

**Branch**: `001-centroid-table` | **Date**: 2026-06-07

---

## Entities

### `CentroidTable`

Primary artifact. An immutable in-memory routing table, initialized once after k-means and then shared read-only across threads.

| Field | Type | Description |
|-------|------|-------------|
| `entries` | `HashMap<u32, String>` | Maps each centroid ID to the shard's network address |

**Invariants**:
- Each centroid ID maps to exactly one shard address (FR-001).
- The table is populated at construction time and never mutated thereafter.
- Multiple centroid IDs may map to the same shard address (one shard hosts multiple partitions).

**Lifecycle**:
1. Created by the coordinator after k-means produces `(centroid_id, shard_address)` pairs.
2. Serialized to bincode and sent to the staging shard.
3. Staging shard deserializes, uses table for bulk-assignment, then drops it when transitioning to inactive.
4. Coordinator retains its copy for query routing (Stage 5).

---

### `CentroidId` (type alias)

```
type CentroidId = u32
```

- Identifies one Voronoi cell / IVF partition.
- Range: 0 to K−1 (K ≤ 512 for V1).
- Matches existing `Centroid.id: u32` in `common`.

---

### `ShardAddress` (type alias, deferred)

```
type ShardAddress = String
```

- Network address of a shard node (e.g., `"127.0.0.1:7001"`).
- Plain `String` for V1; a typed newtype may be introduced in Stage 5.

---

## Serialization Wire Formats

### JSON (human-readable)

```json
{
  "entries": {
    "0": "127.0.0.1:7001",
    "1": "127.0.0.1:7002",
    "2": "127.0.0.1:7001"
  }
}
```

- Keys are centroid IDs serialized as JSON strings (serde_json requirement for map keys).
- Key ordering is non-deterministic (HashMap); consumers must use key lookup, not positional access.
- Used for human inspection and debugging (User Story 3).

### Bincode (compact binary)

- serde-derived layout; no version field (V1 schema is stable — Clarification Q5).
- Used for coordinator → staging shard distribution (User Story 2).
- Approximate size: ≤ 512 entries × ~30 bytes/entry ≈ 15 KB at maximum K.

---

## Public API Surface

Defined in `common::centroid_table`.

```
CentroidTable::new(entries: impl IntoIterator<Item = (u32, String)>) -> Self
CentroidTable::get(&self, centroid_id: u32) -> Option<&str>
CentroidTable::len(&self) -> usize
CentroidTable::is_empty(&self) -> bool
CentroidTable::to_json(&self) -> Result<String, serde_json::Error>
CentroidTable::from_json(json: &str) -> Result<Self, serde_json::Error>
CentroidTable::to_bincode(&self) -> Result<Vec<u8>, bincode::Error>
CentroidTable::from_bincode(bytes: &[u8]) -> Result<Self, bincode::Error>
```

All fallible methods document `# Errors` per Constitution Principle III.

---

## State Transitions

```
[k-means output]
      │
      ▼
CentroidTable::new(pairs)    ← coordinator constructs table
      │
      ├─── coordinator retains for query routing (Stage 5)
      │
      └─── serialize to bincode → send to staging shard
                │
                ▼
         staging shard: CentroidTable::from_bincode(bytes)
                │
                ▼
         bulk-assign staged vectors → shard writes (Stage 3)
                │
                ▼
         staging shard transitions to inactive → table dropped
```

---

## Relationship to Existing Types

| Existing type | Relationship |
|---------------|-------------|
| `common::Centroid` | Complementary — holds geometric data and abstract `shard_id`; `CentroidTable` holds network routing data. No changes needed. |
| `common::ShardId` (= `u32`) | `CentroidTable` keys are also `u32` centroid IDs (same width, different semantic). Not aliased to `ShardId` to avoid confusion between shard identity and centroid identity. |
| `common::Vector` | Not directly related — `CentroidTable` does not store vectors. |
