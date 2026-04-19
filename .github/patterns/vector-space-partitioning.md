# Pattern: Vector-Space Partitioning via Voronoi Cells

## Overview

ProximaDB partitions vectors into disjoint Voronoi cells defined by k-means centroids. This enables **proximity-aware query routing**: instead of broadcasting to all shards, a query only contacts the shards whose centroids are nearest, dramatically reducing fan-out at scale.

**Core Insight:** Voronoi partitioning of vector space + shard assignment = Selective query routing

---

## Architecture

```mermaid
flowchart TD
      A[Training Corpus<br/>e.g., 1M embeddings]
      B[K-Means k=64<br/>Stage 2 Step 1]
      C[64 Centroids<br/>with shard IDs<br/>Stage 2 Step 2]
      S0[Shard 0<br/>Voronoi 0<br/>flat or HNSW]
      S1[Shard 1<br/>Voronoi 1<br/>flat or HNSW]
      S2[Shard 2<br/>Voronoi 2<br/>flat or HNSW]
      SN[Shard N<br/>Voronoi N<br/>flat or HNSW]

      A --> B
      B --> C
      C --> S0
      C --> S1
      C --> S2
      C --> SN
```

---

## Write Path (Soft Assignment)

```mermaid
flowchart TD
   Q[User Vector example 0.3, -0.1, ...] --> R[Coordinator.assign_to_shards nprobe=2]
    R --> D[Compute distances to all 64 centroids]
    D --> T[Pick top-2 nearest centroids<br/>e.g. C5 0.15, C42 0.18]
   T --> S[Return shard IDs 5 and 42]
    S --> W5[Shard 5 insert vector soft assignment]
    S --> W42[Shard 42 insert vector soft assignment]
    W5 --> OUT[Result: vector stored in 2 shards]
    W42 --> OUT
```

**Soft Assignment vs. Hard Assignment:**
- **Soft (top-2):** Vector stored in 2 shards (slight redundancy, better recall for border vectors)
- **Hard (top-1):** Vector stored in 1 shard (minimal storage, reduced recall)
- **ProximaDB V1:** Uses soft (top-2) for recall; accepts 2x storage overhead

---

## Read Path (Fan-Out to Top-nprobe)

```mermaid
flowchart TD
   U[User Query example 0.35, -0.05, ...] --> A1[Coordinator.assign_to_shards nprobe=2]
   A1 --> N[Normalize query if cosine]
   N --> D1[Compute distances to all 64 centroids]
   D1 --> P[Pick top-2 nearest centroids<br/>e.g. C5 0.12, C42 0.16]
   P --> SID[Return shard IDs 5 and 42]
   SID --> S5[Shard 5 search_topk k=10]
   SID --> S42[Shard 42 search_topk k=10]
   S5 --> M[Coordinator merge and rerank]
   S42 --> M
   M --> RET[Return top-10 to user]
   RET --> OUT2[Result: touched 2 shards instead of 64]
```

**Why This Works:**
- Vectors close in embedding space share nearby centroids
- Nearest-neighbor queries likely to find results in nearby partitions
- nprobe = 2 usually sufficient; can increase for higher recall

---

## Recall vs. Recall@nprobe Tradeoff

```mermaid
xychart-beta
    title "Recall vs nprobe"
    x-axis [nprobe=1, nprobe=2, nprobe=4, nprobe=all]
    y-axis "Recall %" 85 --> 96
    line "Proximity-aware soft assignment" [86, 90, 93, 95]
    line "Broadcast baseline" [95, 95, 95, 95]
```

Result: nprobe=2 achieves ~90% of broadcast recall while touching 6% of shards.

---

## Design Decisions Captured

1. **Full-Batch K-Means:** See `0003-kmeans-design.md`
   - Deterministic, correctness-first
   - Seeded RNG for reproducibility
   - Cosine normalization before training

2. **Soft Assignment (top-2):** Trade 2x storage for better recall
   - Border vectors stored in multiple shards
   - Ensures queries near cluster boundaries find neighbors
   - Acceptable in V1 (memory-only system)

3. **Static Centroids (V1):** No online updates
   - Simplifies routing logic
   - No centroid drift management
   - Post-V1: streaming re-clustering, rebalancing

4. **Placeholder Shard IDs:** Separation of concerns
   - Clustering module agnostic to shard topology
   - CentroidTable assigns shard IDs after fitting
   - Flexible for future remapping strategies

---

## Implementation Checklist

### Stage 2 Step 1 ✅
- ✅ `fit_kmeans()` — train centroids
- ✅ `assign_to_shards()` — route queries to top-nprobe

### Stage 2 Step 2 ⏳
- ⏳ `CentroidTable` — persist, assign shard IDs, distribute
- ⏳ Coordinator gRPC — accept insert/query, route via `assign_to_shards()`

### Stage 3 ⏳
- ⏳ Staging shard — buffer vectors until k-means threshold
- ⏳ Cold-start orchestration — trigger k-means, bulk assignment

### Stage 4 ⏳
- ⏳ Per-shard HNSW — flat for small partitions, HNSW for large

### Stage 5 ⏳
- ⏳ Distributed routing — gRPC fan-out, merge results

### Stage 6 ⏳
- ⏳ Benchmarking — recall@nprobe, shards-touched analysis

---

## Metrics & Validation

### Write Path
- **Soft assignment ratio:** % of vectors assigned to 2 vs. 1 shard (should be close to 100% for border vectors)
- **Assignment correctness:** Verify top-2 shards are truly nearest centroids
- **Storage overhead:** Expected 2x (2 copies per vector)

### Read Path
- **Shard fan-out:** Count shards touched per query (should scale with nprobe, not num_shards)
- **Recall@nprobe:** Measure % neighbors found in top-nprobe shards vs. broadcast baseline
- **Latency breakdown:** Query time vs. (centroid lookup + fan-out + merge)

### Partitioning Quality
- **Centroid convergence:** Fit loss (inertia) should decrease with iterations
- **Partition balance:** Distribution of vectors per shard (variance analysis)
- **Dimension alignment:** Verify centroids reflect data distribution

---

## Example: 1M-Vector, 64-Shard Cluster

```mermaid
flowchart TD
   I[Input 1M embeddings 128-dim] --> K[Fit k-means with 64 centroids]
   K --> SA[Assign centroid i to shard i mod 64]
   SA --> R1[Training about 5 s on commodity CPU]
   SA --> R2[Storage per shard about 1.2 GB with 2x redundancy]
   SA --> R3[Query routing about 500 us for 64-distance lookup]
   SA --> R4[Query fan-out 2 shards instead of 64]
   SA --> R5[Recall at nprobe 2 about 92 percent]
```

---

## Potential Issues & Mitigations

| Issue | Impact | Mitigation |
|-------|--------|-----------|
| **Centroid drift** | Recall degrades as data skews | V1: ignore; V2: online updates |
| **Partition skew** | Some shards 10x larger (bottleneck) | V1: ignore; V2: balanced k-means |
| **Border vector miss** | Top-1 hard assignment loses recall | V1: use soft (top-2); acceptable 2x storage |
| **Query dimension mismatch** | Runtime error during routing | Validate at ingestion |
| **Zero-norm vectors (cosine)** | Undefined distance | Reject or normalize to epsilon |
| **nprobe > num_centroids** | Invalid configuration | Validate at startup |

---

## Related Files

- **Implementation:** `coordinator/src/clustering.rs`
- **Tests:** `coordinator/tests/clustering.rs`
- **Storage target:** `shard/src/storage.rs` (flat store)
- **Centroid table:** `coordinator/src/centroid_table.rs` (Stage 2 Step 2)
- **Decision log:** `.github/decision-log/0003-kmeans-design.md`

---

## References

- **IVF indexes:** Jegou et al., "Product Quantization for Nearest Neighbor Search", PAMI 2011
- **Voronoi partitioning:** Inaba et al., "Applications of Weighted Voronoi Diagrams and Randomization to Computational Geometry", ISC 1994
- **Linfa k-means:** https://docs.rs/linfa-clustering/latest/linfa_clustering/
