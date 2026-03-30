# Current Project State (March 2026)

## Summary
- Stage 0 is complete.
- Stage 1 is in progress.
- Flat vector storage foundation is implemented in the shard crate.
- Brute-force search (L2/cosine), SIMD optimization, and benchmarks are not implemented yet.

## Implemented Components
- Shared data model in `common`:
  - `VectorId`, `ShardId`, `Vector`, `Centroid`, `SearchResult`
- gRPC skeleton in `shard` and `coordinator`:
  - `ClusterService` with `Ping` RPC
- Shard storage subsystem:
  - `FlatVectorStore` in `shard/src/storage.rs`
  - `ShardState` in `shard/src/state.rs`
  - shard library surface in `shard/src/lib.rs`

## Flat Storage Capabilities
- Memory-mapped file-backed storage using `memmap2`
- Fixed-dimension vectors
- Append-only inserts
- Batch insert support
- Sequential iteration support for future brute-force scan
- Header validation on reopen (magic/version/dimension/len/capacity)
- Capacity growth with remap

## Shard Runtime State
- `shard/src/main.rs` initializes storage at process startup.
- Current runtime config is environment-variable based:
  - `SHARD_ID`
  - `SHARD_ADDR`
  - `SHARD_STORE_PATH`
  - `SHARD_DIMENSION`
  - `SHARD_INITIAL_CAPACITY`
- `Ping` behavior remains unchanged.

## Code Quality Conventions

These conventions are now enforced across the codebase and must be followed in
all future contributions (see `.github/instructions/code-quality.md`):

- **Documentation** — every public item has a `///` doc comment; all
  `Result`-returning public functions include a `# Errors` section.
- **Tests** — tests live in `<crate>/tests/<module>.rs` integration test files;
  inline `#[cfg(test)]` blocks inside source files are not used.

## Test Coverage Snapshot

Tests live under `shard/tests/` (integration test files):
- `shard/tests/storage.rs` covers:
  - create and reopen
  - capacity growth
  - dimension mismatch validation
  - batch insert order
- `shard/tests/state.rs` covers persistence through state-owned store.

## Open Work (Stage 1)
- Implement brute-force k-NN search
- Add L2 and cosine distance functions
- Add SIMD acceleration path
- Add query-vs-reference correctness tests for top-k
- Add single-node QPS/latency benchmark suite
