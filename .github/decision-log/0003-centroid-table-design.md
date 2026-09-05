# Decision 0003: CentroidTable Design (Stage 2 Step 2)

## Context

Stage 2 requires a routing table that maps each centroid ID to the shard network
address that owns that Voronoi partition. The table is produced once after k-means
clustering, distributed to the staging shard, and used for O(1) centroid-to-shard
lookup during query routing (coordinator) and bulk-assignment (staging shard).

## Decision

`CentroidTable` is implemented in the `common` crate as a newtype wrapping
`HashMap<u32, String>` with JSON and bincode serialization derived via `serde`.

## Rationale

**Map type — `HashMap<u32, String>`**: FR-002 requires O(1) lookup. `HashMap` gives
amortized O(1) get with no extra dependencies. `BTreeMap` (O(log n)) and `IndexMap`
(insertion-ordered O(1)) were rejected as unnecessary; no iteration-order guarantee
was required (Clarification Q2).

**`u32` centroid ID**: K ≤ 512 by V1 design; `u32` provides comfortable headroom
(max ~4 billion) at 4 bytes per entry. `u64` doubles per-entry size for no benefit
at this scale (Clarification Q1).

**`String` shard address**: Addresses are plain `"host:port"` strings consumed
directly by tonic as gRPC endpoints. Structured parsing (`SocketAddr`) is deferred
to Stage 5 (Clarification Q4).

**bincode 1.x + serde_json**: Both formats use the same `#[derive(Serialize,
Deserialize)]` — no second set of trait impls needed. bincode 2.x requires separate
`Encode`/`Decode` derives; rejected for added complexity with no benefit at K ≤ 512
(research decision 2).

**Thread-safety — `Arc<CentroidTable>` only, no `RwLock`**: The table is read-only
after construction (Assumptions). An immutable value behind `Arc` is `Send + Sync`
in Rust with zero runtime cost. `RwLock` would add overhead on every lookup for no
benefit (research decision 3).

**No schema versioning**: V1 schema is stable; all nodes redeploy together if the
schema changes. No version field is embedded in JSON or bincode output (Clarification
Q5).

**Staging shard only**: Normal partition shards never hold a `CentroidTable` at
runtime. Only the coordinator (query routing) and the staging shard (bulk-assignment)
hold populated instances. The `common` crate exposes the type to all crates for
compilation but not for runtime use (FR-011).

## Tradeoffs

- **Iteration order non-deterministic**: JSON key order varies between serializations
  of the same table. Consumers must use key lookup, not positional access. Acceptable
  for inspection use (User Story 3) and correct for deserialization (serde_json is
  order-independent).
- **Duplicate centroid IDs at construction**: Last-write-wins. k-means output is
  assumed to be duplicate-free; no validation is performed at construction time.
- **No persistence**: Table lives in memory only; rebuilt from k-means on each cold
  start. WAL/durability is explicitly out of V1 scope.
