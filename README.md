# ProximaDB

A research-oriented vector database written in Rust, exploring **proximity-aware distributed sharding** — where shards are IVF (Inverted File Index) partitions, meaning vectors closer in embedding space live on the same node. Queries are routed only to the shards most likely to contain nearest neighbors, avoiding the broadcast fan-out that plagues conventional distributed vector databases.

---

## Motivation

Most distributed vector databases (Milvus, Qdrant, Weaviate) shard data by ID range or user-defined partition keys. This means a query must fan out to every shard, collect candidates, and merge — scaling linearly with node count. The core insight ProximaDB explores is:

> **If shard boundaries follow vector-space geometry (Voronoi cells), a query only needs to contact the 1–2 shards whose centroids are nearest to the query vector.**

This is the same partitioning strategy used internally by IVF indexes (Milvus's IVF_FLAT, Pinecone's geometric sub-indices), but applied at the distributed shard level rather than within a single node's index. No open-source vector database currently ships this as a first-class distributed sharding model.

---

## Overall Goals

- **Understand the tradeoffs** of vector-space partitioning as a sharding primitive — recall degradation, partition skew, centroid drift, and border vector problems.
- **Build a working distributed prototype** in Rust that demonstrates query routing to a subset of shards based on centroid proximity.
- **Benchmark recall and latency** against a broadcast-fan-out baseline to quantify the benefit of proximity-aware routing at various `nprobe` values and dataset sizes.
- **Document design decisions and failure modes** as a research artifact — particularly around cold start, soft assignment recall gains, and the boundary between local HNSW and flat scan.

---

## Architecture Overview

```
                        ┌─────────────────────┐
                        │     Coordinator      │
                        │  (centroid table +   │
                        │   query router)      │
                        └────────┬────────────┘
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
           ┌─────────┐     ┌─────────┐     ┌─────────┐
           │ Shard 0  │     │ Shard 1  │     │ Shard N  │
           │Voronoi 0 │     │Voronoi 1 │     │Voronoi N │
           │local HNSW│     │local HNSW│     │local HNSW│
           └─────────┘     └─────────┘     └─────────┘
                 ▲
           ┌─────────┐
           │ Staging  │  ← cold start buffer (pre-centroid)
           │  Shard   │
           └─────────┘
```

**Write path:** coordinator assigns each vector to its top-2 nearest centroids → writes to 2 shards (soft assignment).

**Read path:** coordinator finds top-`nprobe` centroids for query → fans out to only those shards → merges and re-ranks candidates.

---

## V1 Scope

### In Scope
- **Static centroids** — centroids are computed once (k-means on a training sample) and fixed for the lifetime of the cluster. No online rebalancing.
- **Soft assignment (top-2)** — each vector is written to the 2 nearest shards. Significantly improves recall for border vectors at the cost of ~2x storage.
- **Staging shard for cold start** — incoming vectors buffer in a flat staging shard until enough data exists to run k-means and initialize the centroid table. On threshold, bulk-assign staged vectors to their partitions.
- **Per-shard HNSW index** — once a shard's partition reaches a configurable size threshold, build a local HNSW index on it. Below the threshold, use flat (brute-force) scan.

### Out of Scope for V1
- Online centroid rebalancing / re-partitioning
- Partition split/merge on skew detection
- Payload filtering / metadata indexes
- Persistence durability (WAL, crash recovery)
- Authentication, TLS, multi-tenancy

---

## V1 Stage Tasks

### Stage 0 — Project Setup
- [x] Initialize Cargo workspace with crates: `coordinator`, `shard`, `common`, `client`
- [x] Define shared types in `common`: `Vector`, `VectorId`, `Centroid`, `ShardId`, `SearchResult`
- [x] Set up inter-node communication (start with gRPC via `tonic`, or raw TCP with `tokio`)
- [x] Write a basic health-check ping between coordinator and a shard node

### Stage 1 — Single-Node Vector Store
- [x] Implement flat vector storage: `Vec<(VectorId, Vec<f32>)>` backed by a memory-mapped binary file (`memmap2`)
- [ ] Implement brute-force k-NN search (L2 and cosine distance) with SIMD distance via `simsimd` or manual `std::arch` AVX2 intrinsics
- [ ] Write unit tests: insert N vectors, query top-k, verify results against a naive reference implementation
- [ ] Benchmark single-node QPS and latency baseline

### Stage 2 — Centroid Table & Partition Assignment
- [ ] Implement k-means clustering (`K` configurable, e.g. 64–512) on a training corpus using the `linfa` crate or a custom mini-batch k-means
- [ ] Build the `CentroidTable`: an in-memory struct mapping centroid ID → shard address, serializable to JSON/bincode for distribution to all nodes
- [ ] Implement `assign_to_shards(vector, nprobe=2) -> Vec<ShardId>` — returns top-2 nearest centroid shards
- [ ] Unit test: verify that vectors cluster correctly and that border vectors are assigned to 2 shards

### Stage 3 — Staging Shard & Cold Start
- [ ] Implement the staging shard as a plain flat store with no partition assignment
- [ ] Add a configurable `staging_threshold` (e.g. 10,000 vectors)
- [ ] On threshold: run k-means on staged vectors, generate centroid table, bulk-assign staged vectors to their partitions, swap staging shard to inactive
- [ ] Write integration test: insert vectors incrementally, verify handoff from staging to partitioned shards

### Stage 4 — Per-Shard HNSW
- [ ] Integrate the `instant-distance` or `hnsw_rs` Rust crate for HNSW indexing
- [ ] Add a configurable `hnsw_threshold` per shard (e.g. 5,000 vectors)
- [ ] Below threshold: flat scan. Above threshold: build HNSW index, switch search to graph traversal
- [ ] Ensure HNSW index is rebuilt correctly when new vectors are soft-assigned into an existing shard
- [ ] Benchmark recall@10 and QPS for flat vs HNSW at 1k, 10k, 100k vectors per shard

### Stage 5 — Coordinator & Distributed Query Routing
- [ ] Implement coordinator node: holds centroid table, routes writes and reads to appropriate shards
- [ ] Write path: receive vector → `assign_to_shards` → parallel write to 2 shards via gRPC
- [ ] Read path: receive query → find top-`nprobe` centroids → parallel fan-out to those shards → merge top-k results
- [ ] Run a 3-shard cluster locally (3 processes), insert vectors, run queries, verify routing hits the correct shards and not all shards

### Stage 6 — Recall & Routing Benchmarks
- [ ] Use the ANN Benchmarks dataset (e.g. `sift-128-euclidean`) or generate synthetic Gaussian clusters
- [ ] Measure **recall@10** vs `nprobe` (1, 2, 4, 8) to quantify the recall-cost of proximity routing
- [ ] Measure **shards contacted per query** vs `nprobe` to show fan-out reduction vs broadcast baseline
- [ ] Compare soft assignment (top-2 shards) vs hard assignment (top-1) recall at `nprobe=1`
- [ ] Document findings: at what `nprobe` does recall match a full broadcast, and what is the latency gain at that point?

---

## Key Design Decisions to Revisit Post-V1

| Decision | V1 Choice | Future Options |
|---|---|---|
| Centroid updates | Static (fixed at init) | Streaming k-means, background re-clustering |
| Shard assignment | Top-2 soft assignment | Top-k soft, learned routing |
| Partition skew | Ignored | Balanced k-means, split/merge on size trigger |
| Border vector handling | Soft assignment covers it | Explicit replication radius |
| Storage durability | In-memory only | WAL + memmap segments (Qdrant-style) |
| Distribution | Single-machine multi-process | True distributed with Raft metadata |

---

## References

- Qdrant Architecture Docs — segment model, WAL, filterable HNSW
- Milvus Architecture Docs — IVF partitioning, separated compute/storage
- Pinecone Engineering Blog — geometric sub-indices, freshness layer
- HARMONY (SIGMOD 2025) — hybrid vector+dimension partition-aware sharding
- arXiv 2310.11703 — Survey on Vector Database Storage and Retrieval
