# Pattern: Flat Vector Storage (Stage 1)

## Intent
Provide a simple, deterministic, shard-local storage format that supports fast sequential scans and safe reopen behavior.

## Use When
- Dataset is stored per-shard.
- Vectors are fixed dimension.
- You need a robust baseline before ANN indexing.
- Search path is brute-force scan over all records.

## Structure
- File starts with a fixed-size header.
- Header contains:
  - magic bytes
  - format version
  - vector dimension
  - record count (`len`)
  - record capacity (`capacity`)
- Record format is fixed-width:
  - `u64` id
  - `dimension * f32` payload

## API Shape
- `open_or_create(path, dimension, initial_capacity)`
- `insert(vector)`
- `insert_batch(vectors)`
- `len()`, `capacity()`, `dimension()`
- `record_ref(index)` for borrowed access
- `iter()` for sequential read

## Invariants
- Dimension must match the file dimension for all inserts.
- Header magic and version must match expected values.
- `len <= capacity` always.
- File length must match header-derived layout.

## Growth Strategy
- If required length exceeds capacity, grow capacity geometrically (doubling).
- Resize file, then remap.
- Persist updated capacity in header.

## Concurrency Boundary
- Keep storage object single-writer/simple.
- Apply synchronization in higher-level shard state (`tokio::sync::Mutex`).

## Why This Pattern
- Simple to reason about and test.
- Good baseline for later Stage 1 brute-force search.
- Layout is compatible with efficient linear scan and future SIMD distance paths.
