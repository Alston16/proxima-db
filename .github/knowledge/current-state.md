# Current Project State (April 2026)

## Summary
- **Stage 0:** ✅ Complete
- **Stage 1:** ✅ Complete — single-node flat storage, brute-force search, benchmarks
- **Stage 2 Step 1:** ✅ Complete — k-means clustering (fit_kmeans, assign_to_shards, full error handling)
- **Stage 2 Step 2–6:** ⏳ In progress — centroid table, distributed routing, HNSW, benchmarks

## Implemented Components

### Common Crate (Shared Types & Utilities)
- **Types:** `VectorId`, `ShardId`, `Vector`, `Centroid`, `SearchResult`, `DistanceMetric`
- **Distance metrics** in `distance.rs`:
  - `l2_distance(q: &[f32], v: &[f32]) -> f32`
  - `cosine_distance(q: &[f32], v: &[f32]) -> f32`
  - Support for both normalized and raw vectors
- **Top-k selection** in `topk.rs`:
  - `select_topk_by_vector(query, candidates, k, metric) -> Vec<(ShardId, f32)>`
  - Used by clustering assignment and search routing
- **Tests:** `common/tests/distance.rs` (correctness), `common/tests/topk.rs` (selection)

### Coordinator Crate (Query Router & Centroid Manager)
- **K-Means clustering** in `clustering.rs` (274 lines, Stage 2 Step 1 ✅ COMPLETE):
  - `fit_kmeans(vectors: &[Vector], config: &KMeansConfig) -> Result<Vec<Centroid>>`
    - Full-batch k-means via linfa with k-means++ initialization
    - Supports L2 and cosine metrics
    - Cosine: L2-normalizes vectors before training, stores normalized centroids
    - Seeded RNG (SmallRng) for reproducibility
    - Comprehensive error handling (EmptyInput, InvalidK, DimensionMismatch, ZeroNormVector, etc.)
  - `assign_to_shards(query: &[f32], centroids: &[Centroid], nprobe, metric) -> Result<Vec<ShardId>>`
    - Finds top-nprobe nearest centroids for a query
    - Returns shard IDs sorted by distance (nearest first)
    - Normalizes cosine queries before lookup
    - Uses `select_topk_by_vector()` from common
  - `KMeansConfig` struct: k, metric, seed, max_iterations, tolerance
  - `ClusteringError` enum: 8 distinct error types
- **Tests:** `coordinator/tests/clustering.rs` (409 lines, ~15 test functions)
  - Empty input validation
  - Dimension consistency checks
  - K-means convergence correctness
  - Cosine zero-norm rejection
  - Centroid assignment topk ordering
  - Reproducibility (same seed → same centroids)
- **Next:** CentroidTable (persistence, shard assignment, distribution)

### Proto (Shared RPC Schema)
- `proto/cluster.proto`: `ClusterService` with `Ping` RPC (connectivity smoke test)
- **Next:** Add InsertRequest, QueryRequest, SearchLocal messages

### Shard Crate (Data Plane)
- **Stage 1 baseline:** Flat mmap storage, brute-force search over vectors
- **Components:**
  - `storage.rs`: `FlatVectorStore` (mmap-backed append-only, 32-byte header) with search using `common::distance` and `common::topk`
  - `state.rs`: `ShardState` async wrapper (tokio::sync::Mutex)
  - `main.rs`: tonic server, environment-based config (SHARD_ID, SHARD_ADDR, SHARD_STORE_PATH, SHARD_DIMENSION, SHARD_INITIAL_CAPACITY)
- **Tests:** shard storage/search/state integration tests; shared distance/top-k tests live in `common/tests`
- **Benchmarks:** Criterion search_topk @ 1k/10k/100k vectors, d=128/d=512

### Client Crate
- Placeholder binary (no RPC implementation yet)

## Stage 1 Baseline Capabilities
- Memory-mapped file-backed storage using `memmap2`
- Fixed-dimension append-only record layout with a 32-byte header
- Safe reopen and header validation
- Geometric capacity growth with file remap
- Borrowed record access via `record_ref()` and sequential zero-copy iteration via `iter()`
- Brute-force nearest-neighbour search over all records using:
  - `DistanceMetric::L2`
  - `DistanceMetric::Cosine`
- Distance backend auto-selection in `distance.rs`:
  - scalar reference path for correctness and short vectors
  - `wide`-based SIMD path for lane-chunked accumulation with scalar tail handling
- Deterministic top-k ordering:
  - ascending distance
  - ascending `VectorId` when distances tie
- Async shard wrapper through `ShardState::search_topk`

## Runtime State
- `shard/src/main.rs` configures the shard process from environment variables:
  - `SHARD_ID`
  - `SHARD_ADDR`
  - `SHARD_STORE_PATH`
  - `SHARD_DIMENSION`
  - `SHARD_INITIAL_CAPACITY`
- The shard boots its local store before starting tonic.
- The exposed RPC surface is still only `Ping`; insert and search RPCs are not wired yet.
- `coordinator/src/main.rs` is still a connectivity smoke test, not a routing service.
- `client/src/main.rs` remains a placeholder.

## Testing And Benchmark Coverage
- `shard/tests/distance.rs` covers scalar correctness, SIMD-vs-scalar parity on odd-tail dimensions, randomized auto-backend parity, and zero-norm cosine handling.
- `shard/tests/storage.rs` covers create/reopen, growth, dimension validation, and batch insert ordering.
- `shard/tests/search.rs` covers:
  - empty-store and `k == 0` behavior
  - `k > len` behavior
  - dimension mismatch errors
  - exact nearest-neighbour ranking for L2 and cosine
  - deterministic tie-breaking by ascending `VectorId`
  - async parity through `ShardState`
  - randomized parity against a naive reference implementation
- `shard/tests/state.rs` covers persistence through the state-owned store.
- `shard/benches/search_bench.rs` provides Criterion benchmarks for:
  - L2: `1k_d128`, `10k_d128`, `100k_d128`, `10k_d512`
  - cosine: `1k_d128`, `10k_d128`

## Important Nuance
- SIMD is now implemented in `shard/src/distance.rs` through a crate abstraction (`wide`) rather than manual intrinsics.
- The scalar path remains available as the correctness reference and explicit backend option.
- Brute-force search orchestration and deterministic top-k ordering in `FlatVectorStore::search_topk()` are unchanged.

## Code Quality Conventions
- Public items are expected to carry `///` docs.
- Public `Result`-returning functions are expected to document `# Errors`.
- Tests live in crate-level integration test files under `tests/` rather than inline `#[cfg(test)]` modules.

## Open Work

### Stage 2 Step 2 (Centroid Table & Routing)
- [ ] Implement `CentroidTable` (persist centroids, reload on startup, assign real shard IDs)
- [ ] Proto: Define `InsertRequest`, `InsertResponse`, `QueryRequest`, `QueryResponse`, `SearchLocalRequest`, `SearchLocalResponse`
- [ ] Coordinator main loop: Accept write/read requests, route via `assign_to_shards()`, fan-out to shards
- [ ] Soft assignment write path: Route vector to top-2 shards, wait for both to ACK
- [ ] Read path: Route query to top-nprobe shards, merge results, re-rank

### Stage 3 (Staging Shard & Cold Start)
- [ ] Flat staging shard (buffer vectors until k-means threshold)
- [ ] Cold-start orchestration: k-means trigger, bulk assignment, partition activation

### Stage 4 (Per-Shard HNSW)
- [ ] HNSW index integration (instant-distance or hnsw_rs)
- [ ] Shard decision logic: Use flat for < ~5k vectors, HNSW for >= ~5k
- [ ] Index rebuild/incremental updates

### Stage 5 (Distributed Deployment & Integration Testing)
- [ ] 3-shard local cluster test: write + read workflows
- [ ] Verify routing decisions (correct shards touched per query)
- [ ] Connection pooling in client

### Stage 6 (Benchmarking)
- [ ] ANN benchmark datasets (SIFT-128, synthetic Gaussian clusters)
- [ ] Recall@k vs nprobe measurements
- [ ] Soft (top-2) vs hard (top-1) assignment comparison
- [ ] Shards-touched per query analysis

### Quality & Ops
- [ ] GitHub Actions CI (build + test on push)
- [ ] Decision logs for architectural choices (2 of 3 in progress)
- [ ] Pattern library (flat-store, topk-search, k-means done; more to add)
- [ ] Integration test coverage for routing paths
