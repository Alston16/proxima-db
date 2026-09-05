# Decision: Full-Batch K-Means with Seeded RNG (Stage 2 Step 1)

**Date:** 2026-04-18  
**Status:** DECIDED & IMPLEMENTED  
**Commit:** 043ec130414e30c3afd4a08150be2751fa655a82  
**References:** `coordinator/src/clustering.rs`, `coordinator/tests/clustering.rs`

---

## Context

**Problem:** ProximaDB needs to partition a training corpus into k Voronoi cells (clusters) to create a centroid table for proximity-aware routing. We must decide:

1. **K-Means variant:** Full-batch vs. mini-batch vs. streaming
2. **Distance metric handling:** How to support both L2 and cosine
3. **Reproducibility:** How to ensure deterministic centroid computation across restarts
4. **Error handling:** Which validation errors to expose to callers

---

## Decision

### 1. Full-Batch K-Means (No Mini-Batch)

**Choice:** Use linfa's full-batch `KMeans` with k-means++ initialization.

**Rationale:**
- **Correctness first:** Full-batch EM guarantees convergence to a local optimum; easier to validate against reference implementations
- **Simplicity:** linfa's API is straightforward; no need for custom EM implementation
- **Test repeatability:** Deterministic convergence makes test fixtures reliable
- **Stage 2 scope:** Training on ~10k vectors during cold start; speed not critical yet
- **Baseline:** Establish correctness baseline before optimizing to mini-batch

**Tradeoff:**
- Slower on datasets > 100k vectors
- Memory overhead (all vectors in memory during EM)
- Not incremental (must re-fit entire corpus when new vectors arrive)

**Mitigation (Post-V1):**
- Mini-batch k-means for streaming updates
- Incremental centroid adjustment (online learning)
- Re-clustering thresholds (e.g., every 100k new vectors)

---

### 2. Cosine Normalization Before Training

**Choice:** For cosine metric, L2-normalize vectors before passing to k-means; store centroids normalized; normalize queries before lookup.

**Implementation:**
```rust
if metric == DistanceMetric::Cosine {
    for each vector v:
        let norm = ||v||_2
        if norm == 0.0: return ZeroNormVector error
        normalized[v] = v / norm
}
// Train k-means on normalized vectors
// Store centroids normalized
```

**Rationale:**
- **Mathematical:** In normalized space, Euclidean distance = cosine distance (angular proximity)
- **Consistency:** Ensures training and assignment operate in the same space
- **Validation:** Zero-norm vectors explicitly rejected (cosine undefined for zero vector)
- **Efficiency:** Single normalization step; linfa uses L2Dist internally

**Invariant:**
- Cosine centroids always stored normalized
- Cosine queries always normalized before distance computation
- L2 vectors/centroids stored raw (not normalized)

**Error case:** Zero-norm vector with cosine metric → `ZeroNormVector { index }` error

---

### 3. Seeded RNG for Reproducibility

**Choice:** Require `KMeansConfig.seed: u64`; use `SmallRng::seed_from_u64()` for k-means++ initialization.

**Implementation:**
```rust
let rng = SmallRng::seed_from_u64(config.seed);
let model = KMeans::params_with(config.k, rng, L2Dist)
    .fit(&dataset)?;
```

**Rationale:**
- **Reproducibility:** Same seed + same input vectors = identical centroids (crucial for restart safety)
- **Testing:** Deterministic fixtures; can validate against known outputs
- **Distributed restarts:** Multiple coordinator instances can re-sync centroids without coordination
- **Audit trail:** Seed + input vectors fully determine centroid positions

**Design:**
- Seed is `u64`, not `Option<u64>` (always required)
- Caller passes seed; coordinator typically uses fixed seed (e.g., 0) or hash of input corpus
- Different seeds produce different (but valid) clusterings

**Future option:** Seed corpus hash dynamically to ensure deterministic clustering without explicit seed management.

---

### 4. Placeholder Shard IDs

**Choice:** Centroids emitted with `shard_id = 0` (placeholder); real shard assignment deferred to Stage 2 Step 2 (`CentroidTable`).

**Rationale:**
- **Separation of concerns:** Clustering module knows nothing about shard topology
- **Flexibility:** CentroidTable can assign centroids to shards via various strategies (round-robin, load-aware, etc.)
- **Modularity:** Easier to test clustering independently of routing logic
- **Scalability:** Future: support remapping centroids to different shards without re-clustering

**Invariant:**
- `Centroid.id` is stable (0..k) and set by `fit_kmeans()`
- `Centroid.shard_id` is filled in by `CentroidTable` (Stage 2 Step 2)

**Impact:** `assign_to_shards()` uses `centroid.shard_id` to return shard routing decisions.

---

### 5. Explicit Error Validation

**Choice:** Use comprehensive `ClusteringError` enum with explicit validation for invalid inputs.

**Errors:**
- `EmptyInput` — no vectors provided
- `InvalidK` — k outside [1, n_samples]
- `DimensionMismatch` — vectors have different dimensions
- `ZeroDimension` — vector dimension is 0
- `ZeroNormVector` — zero-norm vector with cosine metric
- `NprobeTooLarge` — nprobe > number of centroids
- `QueryDimensionMismatch` — query dim ≠ centroid dim
- `Linfa(String)` — internal linfa error

**Rationale:**
- **API contract:** Callers expect specific error types for recovery logic
- **Validation boundary:** Clustering module validates all inputs upfront
- **Testability:** Can test each error condition independently
- **Debugging:** Clear error messages with context (index, dimensions, counts)

---

## Alternatives Considered

### Mini-Batch K-Means
- **Pros:** Faster on large datasets; streaming-friendly
- **Cons:** Non-deterministic convergence; harder to validate; requires custom implementation
- **Decision:** Defer to post-V1; implement mini-batch when dataset size demands it

### Cosine Without Normalization
- **Pros:** Avoid normalization overhead
- **Cons:** Inconsistent semantics; hard to reason about; requires custom distance metric in linfa
- **Decision:** Normalization is cheap; consistency outweighs micro-optimization

### Optional Seeding
- **Pros:** Simpler API if determinism not needed
- **Cons:** Lose reproducibility; breaks testing, distributed restarts
- **Decision:** Require seed in API; clarity > false simplicity

### Deferred Shard Assignment
- **Pros:** All in one step; simpler flow
- **Cons:** Violates separation of concerns; clustering module leaks routing knowledge
- **Decision:** Keep clustering pure; assign shards in CentroidTable

---

## Implementation Details

- **File:** `coordinator/src/clustering.rs` (274 lines)
- **Public functions:**
  - `fit_kmeans(vectors: &[Vector], config: &KMeansConfig) -> Result<Vec<Centroid>>`
  - `assign_to_shards(query: &[f32], centroids: &[Centroid], nprobe: usize, metric: DistanceMetric) -> Result<Vec<ShardId>>`
- **Dependencies:** linfa, linfa-clustering, linfa-nn, ndarray, rand, thiserror
- **Tests:** 409 lines in `coordinator/tests/clustering.rs`

---

## Next Steps

1. **Stage 2 Step 2:** Implement `CentroidTable` (persist, reload, distribute centroids; assign shard IDs)
2. **Proto definitions:** Add gRPC messages for Insert, Query, SearchLocal
3. **Coordinator main loop:** Accept routing requests, call `assign_to_shards()`
4. **Integration:** Connect k-means to distributed write/read paths

---

## Revisit Criteria

- **If dataset > 1M vectors:** Consider mini-batch k-means for memory efficiency
- **If clustering time > 5s:** Profile; may need incremental updates
- **If recall < 90% @ nprobe=2:** Explore k-means alternatives (k-center, spectral, etc.)
