# K-Means Clustering Pattern

## Purpose

Partition a training corpus of vectors into k clusters by computing centroid positions via iterative k-means++ initialization and expectation-maximization. This pattern produces a static centroid table used for routing both write and read requests to subsets of shards.

## Location

- **Implementation:** `coordinator/src/clustering.rs`
- **Tests:** `coordinator/tests/clustering.rs` (409 lines, comprehensive coverage)
- **Shared deps:** `common/src/distance.rs`, `common/src/topk.rs`

## Design Choices

### Full-Batch K-Means (No Mini-Batch)
```rust
let model = KMeans::params_with(config.k, rng, L2Dist)
    .max_n_iterations(config.max_iterations)
    .fit(&dataset)?;
```
- **Rationale:** Correctness-first; deterministic convergence; easier to validate against reference implementations
- **Tradeoff:** Slower on very large datasets; acceptable for Stage 2 (training on ~10k vectors during cold start)
- **Future:** Mini-batch k-means post-V1 for streaming updates

### Cosine Normalization (L2-normalize before training)
```rust
let norm = v.data.iter().map(|x| x * x).sum::<f32>().sqrt();
let normalized = v.data.iter().map(|x| x / norm).collect::<Vec<f32>>();
```
- **Rationale:** Euclidean distance in normalized space = angular proximity (cosine distance)
- **Invariant:** Centroids stored in normalized space; queries normalized before lookup
- **Validation:** Zero-norm vectors rejected as invalid
- **Impact:** Ensures consistency between training and assignment

### Seeded RNG for Reproducibility
```rust
let rng = SmallRng::seed_from_u64(config.seed);
let mut builder = KMeans::params_with(config.k, rng, L2Dist);
```
- **Rationale:** Same seed + same input = identical centroids (crucial for testing, distributed restarts)
- **Implementation:** `rand::rngs::SmallRng` initialized from `u64` seed
- **API:** `KMeansConfig.seed` passed by caller

### Placeholder Shard IDs
```rust
Centroid {
    id: id as u32,
    data: row.iter().map(|&x| x as f32).collect(),
    shard_id: 0,  // ← Placeholder; assigned by CentroidTable in Stage 2 Step 2
}
```
- **Rationale:** Centroid IDs are stable (0..k); shard assignment deferred to routing layer
- **Separation of concerns:** Clustering knows nothing about shard topology; CentroidTable handles assignment
- **Future:** Stage 2 Step 2 fills in real shard IDs based on load balancing strategy

## Configuration Structure

```rust
pub struct KMeansConfig {
    pub k: usize,                   // Number of clusters (e.g., 64–512)
    pub metric: DistanceMetric,     // L2 or Cosine
    pub seed: u64,                  // RNG seed for reproducibility
    pub max_iterations: u64,        // Default 300 (linfa default)
    pub tolerance: Option<f64>,     // Convergence tolerance; None uses linfa default
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 64,
            metric: DistanceMetric::L2,
            seed: 0,
            max_iterations: 300,
            tolerance: None,
        }
    }
}
```

## Error Handling

```rust
#[derive(Debug, Error, PartialEq)]
pub enum ClusteringError {
    EmptyInput,                           // No vectors
    InvalidK { k, n_samples },            // k ∉ [1, n_samples]
    DimensionMismatch { index, ... },     // Vector dimension inconsistency
    ZeroDimension,                        // dim = 0
    ZeroNormVector { index },             // Zero-norm with cosine metric
    NprobeTooLarge { nprobe, ... },       // nprobe > num_centroids
    QueryDimensionMismatch { ... },       // Query dim ≠ centroid dim
    Linfa(String),                        // Internal linfa error
}
```

**Validation Strategy:**
1. Check corpus non-empty
2. Check vector dimension consistency
3. Check k validity (1 ≤ k ≤ n_samples)
4. For cosine metric: reject zero-norm vectors explicitly
5. Wrap linfa errors as `ClusteringError::Linfa`

## Assignment (Routing)

```rust
pub fn assign_to_shards(
    query: &[f32],
    centroids: &[Centroid],
    nprobe: usize,
    metric: DistanceMetric,
) -> Result<Vec<ShardId>, ClusteringError>
```

- **Input:** Query vector (unnormalized), centroid list, nprobe, metric
- **Output:** Top-nprobe shard IDs (nearest first)
- **Cosine handling:** Query is normalized inside function (matches centroid space)
- **Mechanism:** Uses `select_topk_by_vector()` from common crate for distance computation
- **Usage:** Write path (nprobe=2), Read path (nprobe=config), Routing validation

## Testing Pattern

```rust
#[cfg(test)]
mod tests {
    // Empty input & dimension validation
    // Cosine zero-norm rejection
    // K-means convergence correctness (small synthetic datasets)
    // Assignment topk ordering
    // Reproducibility (same seed → same centroids)
    // Dimension mismatch handling
}
```

Test coverage: 409 lines, ~15 test functions, comprehensive edge case coverage

## When to Use This Pattern

1. **Cold start:** Fit k-means on accumulated staging shard vectors (Stage 3)
2. **Coordinator initialization:** Load/compute centroid table on startup (Stage 2 Step 2)
3. **Routing:** Call `assign_to_shards()` for every write and read (Stage 5)
4. **Benchmarking:** Reproducible clustering for recall comparisons (Stage 6)

## Related Patterns

- **Flat Vector Storage** (shard/src/storage.rs) — storage target for assigned vectors
- **Top-K Selection** (common/src/topk.rs) — used by `assign_to_shards()`
- **Distance Metrics** (common/src/distance.rs) — L2/cosine computation

## Future Considerations

- **Mini-batch k-means:** Stream updates without full re-clustering
- **Online centroid updates:** Adjust centroids as new vectors arrive
- **Partition rebalancing:** Split/merge clusters to address skew
- **Alternative initializations:** k-means|| (parallel k-means++) for distributed training
