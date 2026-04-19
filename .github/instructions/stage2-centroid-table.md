# Instructions: Stage 2 Step 2 — CentroidTable & Query Router

## Objective

Implement a `CentroidTable` that:
1. Holds a static centroid list (fitted via k-means in Stage 2 Step 1)
2. Assigns real shard IDs to each centroid
3. Provides `assign_to_shards(query)` routing for writes and reads
4. Persists centroids to disk for coordinator restarts
5. Distributes centroids to all shards (read-only for now)

Then wire the coordinator's main loop to accept and route incoming write/read requests.

---

## Files to Create/Modify

```
coordinator/
├── src/
│   ├── centroid_table.rs     (NEW)    — CentroidTable struct + methods
│   ├── clustering.rs         (DONE)   — k-means fitting & assignment
│   ├── lib.rs                (MODIFY) — export centroid_table module
│   └── main.rs               (MODIFY) — load CentroidTable, start gRPC server
├── tests/
│   ├── clustering.rs         (DONE)   — k-means tests
│   └── centroid_table.rs     (NEW)    — CentroidTable tests
└── Cargo.toml                (MODIFY) — add dependencies (serde, bincode)

proto/
└── cluster.proto             (MODIFY) — add Insert, Query, SearchLocal RPC messages

shard/
├── src/
│   ├── lib.rs                (MODIFY) — export centroid_table module (read-only reference)
│   └── main.rs               (MODIFY) — load centroids at startup
└── Cargo.toml                (MODIFY) — add serde dependency
```

---

## Implementation Checklist

### Step 1: Define Proto Messages

**File:** `proto/cluster.proto`

Add to `ClusterService`:

```protobuf
service ClusterService {
  rpc Ping(PingRequest) returns (PingResponse);
  
  // NEW: Vector insertion with soft assignment
  rpc Insert(InsertRequest) returns (InsertResponse);
  
  // NEW: Query routing to top-nprobe shards
  rpc Query(QueryRequest) returns (QueryResponse);
  
  // NEW: Search in local partition (called by coordinator for fan-out)
  rpc SearchLocal(SearchLocalRequest) returns (SearchLocalResponse);
}

message InsertRequest {
  repeated Vector vectors = 1;
  uint32 nprobe = 2;  // top-nprobe assignment (typically 2)
}

message InsertResponse {
  bool ok = 1;
  string error = 2;
  uint64 inserted = 3;  // count of vectors inserted
}

message QueryRequest {
  repeated float query = 1;
  uint32 topk = 2;
  uint32 nprobe = 3;
}

message QueryResponse {
  repeated SearchResult results = 1;
  string error = 2;
}

message SearchLocalRequest {
  repeated float query = 1;
  uint32 topk = 2;
  uint32 shard_id = 3;  // which shard to search (validate locally)
}

message SearchLocalResponse {
  repeated SearchResult results = 1;
  string error = 2;
}

message Vector {
  uint64 id = 1;
  repeated float data = 2;
}

message SearchResult {
  uint64 vector_id = 1;
  float distance = 2;
}
```

Run `cargo build` to regenerate proto bindings.

---

### Step 2: Implement CentroidTable

**File:** `coordinator/src/centroid_table.rs`

```rust
use common::{Centroid, DistanceMetric, ShardId};
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CentroidTableError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("serde error: {0}")]
    Serde(#[from] bincode::Error),
    
    #[error("no centroids loaded")]
    Empty,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentroidTable {
    /// Centroids with assigned shard_id (0..num_shards)
    pub centroids: Vec<Centroid>,
    
    /// Distance metric (must match training metric)
    pub metric: DistanceMetric,
    
    /// Number of shards in cluster
    pub num_shards: usize,
}

impl CentroidTable {
    /// Build centroid table from fitted centroids, assigning shard IDs round-robin.
    pub fn from_centroids(
        centroids: Vec<Centroid>,
        metric: DistanceMetric,
        num_shards: usize,
    ) -> Self {
        let mut table = CentroidTable {
            centroids,
            metric,
            num_shards,
        };
        // Assign shard IDs round-robin: centroid i → shard (i % num_shards)
        for (i, centroid) in table.centroids.iter_mut().enumerate() {
            centroid.shard_id = (i % num_shards) as ShardId;
        }
        table
    }
    
    /// Persist to disk (bincode format for simplicity)
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), CentroidTableError> {
        let data = bincode::serialize(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }
    
    /// Load from disk
    pub fn load(path: impl AsRef<Path>) -> Result<Self, CentroidTableError> {
        let data = std::fs::read(path)?;
        Ok(bincode::deserialize(&data)?)
    }
    
    /// Get top-nprobe shard IDs for a query
    pub fn route_query(&self, query: &[f32], nprobe: usize) -> Result<Vec<ShardId>, String> {
        if self.centroids.is_empty() {
            return Err("no centroids loaded".to_string());
        }
        
        // Use assign_to_shards from coordinator::clustering
        // (Call from here or inline distance computation)
        // For now, sketch the logic:
        let mut distances = Vec::new();
        for (i, centroid) in self.centroids.iter().enumerate() {
            let dist = match self.metric {
                DistanceMetric::L2 => {
                    // Compute L2 distance
                    let mut sum = 0.0;
                    for (q, c) in query.iter().zip(centroid.data.iter()) {
                        let diff = q - c;
                        sum += diff * diff;
                    }
                    sum.sqrt()
                }
                DistanceMetric::Cosine => {
                    // Compute cosine distance (assuming normalized vectors)
                    let mut dot = 0.0;
                    for (q, c) in query.iter().zip(centroid.data.iter()) {
                        dot += q * c;
                    }
                    1.0 - dot
                }
            };
            distances.push((i, centroid.shard_id, dist));
        }
        
        // Sort by distance, take top-nprobe, return unique shard IDs
        distances.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        let mut shards = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (_, shard_id, _) in distances.iter().take(nprobe) {
            if seen.insert(shard_id) {
                shards.push(*shard_id);
            }
        }
        
        Ok(shards)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_roundrobin_assignment() {
        let centroids = vec![
            Centroid { id: 0, data: vec![0.0, 0.0], shard_id: 0 },
            Centroid { id: 1, data: vec![1.0, 0.0], shard_id: 0 },
            Centroid { id: 2, data: vec![0.0, 1.0], shard_id: 0 },
        ];
        let table = CentroidTable::from_centroids(centroids, DistanceMetric::L2, 3);
        
        // Expect: centroid 0 → shard 0, 1 → shard 1, 2 → shard 2
        assert_eq!(table.centroids[0].shard_id, 0);
        assert_eq!(table.centroids[1].shard_id, 1);
        assert_eq!(table.centroids[2].shard_id, 2);
    }
    
    #[test]
    fn test_save_load() {
        let centroids = vec![Centroid { id: 0, data: vec![0.5, 0.5], shard_id: 0 }];
        let table = CentroidTable::from_centroids(centroids, DistanceMetric::Cosine, 1);
        
        let path = "/tmp/test_centroid_table.bin";
        table.save(path).unwrap();
        let loaded = CentroidTable::load(path).unwrap();
        
        assert_eq!(table.metric, loaded.metric);
        assert_eq!(table.num_shards, loaded.num_shards);
        assert_eq!(table.centroids.len(), loaded.centroids.len());
    }
}
```

---

### Step 3: Update Coordinator Main Loop

**File:** `coordinator/src/main.rs`

```rust
use coordinator::{CentroidTable, KMeansConfig, fit_kmeans};
use tonic::{transport::Server, Request, Response, Status};
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load or create centroid table
    let centroid_table = if let Ok(table) = CentroidTable::load("centroids.bin") {
        println!("Loaded centroid table from disk");
        table
    } else {
        println!("No centroid table found; starting in cold-start mode");
        // Cold-start: empty table, will be populated on first staging threshold
        CentroidTable::from_centroids(vec![], common::DistanceMetric::L2, 2)
    };
    
    let table = Arc::new(RwLock::new(centroid_table));
    
    // 2. Start gRPC server
    let addr = "[::1]:50051".parse()?;
    let coordinator_service = CoordinatorService { centroid_table: table };
    
    println!("Coordinator listening on {}", addr);
    Server::builder()
        .add_service(cluster_service_server::ClusterServiceServer::new(coordinator_service))
        .serve(addr)
        .await?;
    
    Ok(())
}

struct CoordinatorService {
    centroid_table: Arc<RwLock<CentroidTable>>,
}

#[tonic::async_trait]
impl cluster_service_server::ClusterService for CoordinatorService {
    async fn insert(
        &self,
        request: Request<InsertRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        let req = request.into_inner();
        let table = self.centroid_table.read().await;
        
        // Assign vectors to top-nprobe shards
        // For each vector, call table.route_query() to get shard IDs
        // Then send to those shards (WIP: not yet implemented)
        
        Ok(Response::new(InsertResponse {
            ok: true,
            error: String::new(),
            inserted: req.vectors.len() as u64,
        }))
    }
    
    async fn query(
        &self,
        request: Request<QueryRequest>,
    ) -> Result<Response<QueryResponse>, Status> {
        let req = request.into_inner();
        let table = self.centroid_table.read().await;
        
        // Route query to top-nprobe shards
        // Merge results from all shards
        // Re-rank by distance
        // Return top-k
        
        let results = vec![];  // TODO: implement
        Ok(Response::new(QueryResponse {
            results,
            error: String::new(),
        }))
    }
    
    async fn search_local(
        &self,
        request: Request<SearchLocalRequest>,
    ) -> Result<Response<SearchLocalResponse>, Status> {
        // Forward to local shard (same process or RPC to shard)
        Err(Status::unimplemented("search_local not yet implemented"))
    }
    
    async fn ping(
        &self,
        request: Request<PingRequest>,
    ) -> Result<Response<PingResponse>, Status> {
        // Existing ping implementation
        Ok(Response::new(PingResponse {
            sender: request.into_inner().sender,
            shard_id: 0,
            ok: true,
        }))
    }
}
```

---

### Step 4: Add Dependencies

**File:** `coordinator/Cargo.toml`

**File:** `shard/Cargo.toml`

---

### Step 5: Write Tests

**File:** `coordinator/tests/centroid_table.rs`

- Test round-robin shard assignment
- Test query routing (top-nprobe distinct shards)
- Test persistence (save/load)
- Test with empty centroid table

---

## Testing Checklist

```bash
# Rebuild proto bindings
cargo build --all

# Test k-means clustering (already passing)
cargo test coordinator::clustering

# Test centroid table (new)
cargo test coordinator::centroid_table
cargo test -p coordinator --test centroid_table

# Integration: coordinator + shard
# (WIP: requires gRPC wire-up)
```

---

## Success Criteria

1. ✅ Proto: Insert, Query, SearchLocal messages defined and compiled
2. ✅ CentroidTable: Loads/saves centroids, assigns shard IDs, routes queries
3. ✅ Coordinator: Starts gRPC server, loads centroid table on startup
4. ✅ Tests: CentroidTable persistence and routing logic validated
5. ⏳ Integration: Write and read paths wired to coordinator main loop (Stage 3)

---

## Related Patterns & Decisions

- **Pattern:** `k-means-clustering.md` — design & implementation
- **Decision:** `0003-kmeans-design.md` — seeded RNG, cosine normalization, shard ID assignment
- **Pattern:** `flat-vector-storage.md` — storage target for assigned vectors

---

## Next Steps (After Step 2 Complete)

1. **Staging Shard & Cold Start** (Stage 3): Buffer vectors, trigger k-means at threshold
2. **Per-Shard HNSW** (Stage 4): Add HNSW index for larger partitions
3. **End-to-End Routing** (Stage 5): Wire write/read to all shards, merge results
