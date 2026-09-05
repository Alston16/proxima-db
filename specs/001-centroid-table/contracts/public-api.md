# Public API Contract: `common::centroid_table`

**Branch**: `001-centroid-table` | **Date**: 2026-06-07
**Crate**: `common` | **Module**: `centroid_table`

This contract defines the stable public interface of `CentroidTable`. Both `coordinator` and `shard` crates compile against this interface. Changes to signatures here require coordinated updates across all consumers.

---

## Type: `CentroidTable`

```rust
pub struct CentroidTable { /* private fields */ }
```

Immutable after construction. `Send + Sync` — safe to share across threads via `Arc<CentroidTable>`.

---

## Constructors

### `new`

```rust
pub fn new(entries: impl IntoIterator<Item = (u32, String)>) -> Self
```

Constructs a table from an iterator of `(centroid_id, shard_address)` pairs. Duplicate centroid IDs overwrite earlier entries (last write wins). An empty iterator produces a valid empty table.

**Panics**: Never.

---

## Read Methods

### `get`

```rust
pub fn get(&self, centroid_id: u32) -> Option<&str>
```

Returns the shard address for the given centroid ID, or `None` if not registered.

**Panics**: Never.

### `len`

```rust
pub fn len(&self) -> usize
```

Returns the number of centroid entries in the table.

### `is_empty`

```rust
pub fn is_empty(&self) -> bool
```

Returns `true` if the table has no entries.

---

## Serialization

### `to_json`

```rust
pub fn to_json(&self) -> Result<String, serde_json::Error>
```

Serializes the table to a UTF-8 JSON string.

**JSON shape**:
```json
{ "entries": { "<centroid_id>": "<host:port>", ... } }
```

Keys are centroid IDs rendered as decimal strings. Key order is non-deterministic.

**Errors**: Returns `serde_json::Error` if serialization fails (in practice, infallible for this type).

### `from_json`

```rust
pub fn from_json(json: &str) -> Result<Self, serde_json::Error>
```

Deserializes a table from a JSON string produced by `to_json`.

**Errors**: Returns `serde_json::Error` if the input is not valid JSON or does not match the expected shape.

### `to_bincode`

```rust
pub fn to_bincode(&self) -> Result<Vec<u8>, bincode::Error>
```

Serializes the table to a compact binary encoding.

**Errors**: Returns `bincode::Error` if serialization fails (in practice, infallible for this type).

### `from_bincode`

```rust
pub fn from_bincode(bytes: &[u8]) -> Result<Self, bincode::Error>
```

Deserializes a table from a bincode byte slice produced by `to_bincode`.

**Errors**: Returns `bincode::Error` if the bytes are malformed, truncated, or do not represent a valid `CentroidTable`.

---

## Stability Contract

- All signatures above are stable for the V1 prototype lifetime.
- No schema versioning is embedded in the serialized formats (Clarification Q5). If the schema changes, all nodes are redeployed together.
- The private field layout is not part of the contract.

---

## Usage Pattern (coordinator)

```rust
use common::centroid_table::CentroidTable;
use std::sync::Arc;

// After k-means:
let table = CentroidTable::new(kmeans_output); // Vec<(u32, String)>
let table = Arc::new(table);

// Query routing (Stage 5):
if let Some(addr) = table.get(centroid_id) {
    // connect to addr via gRPC
}

// Send to staging shard (Stage 3):
let bytes = table.to_bincode()?;
// ... ship bytes over transport
```

## Usage Pattern (staging shard, Stage 3)

```rust
use common::centroid_table::CentroidTable;

// Receive bytes from coordinator:
let table = CentroidTable::from_bincode(&received_bytes)?;

// Bulk-assign buffered vectors:
for (vector_id, vector) in staged_vectors {
    let centroid_id = nearest_centroid(&vector, &centroids);
    let addr = table.get(centroid_id).expect("centroid must be registered");
    // write vector to shard at addr
}

// Drop table when staging shard deactivates:
drop(table);
```
