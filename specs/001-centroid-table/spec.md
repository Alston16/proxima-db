# Feature Specification: CentroidTable

**Feature Branch**: `001-centroid-table`

**Created**: 2026-06-07

**Status**: Draft

**Input**: User description: "Build the CentroidTable: an in-memory struct mapping centroid ID → shard address, serializable to JSON/bincode for distribution to all nodes. Stage 2 step 2 as mentioned in the README file."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Query Routing via CentroidTable (Priority: P1)

A coordinator node receives a vector search query and must determine which shards to contact. It looks up the top-`nprobe` centroids nearest to the query vector, then resolves each centroid ID to its shard address using the CentroidTable, and fans out the query only to those shards.

**Why this priority**: Without this lookup, every query must broadcast to all shards. This is the core routing primitive that makes proximity-aware sharding possible.

**Independent Test**: Can be fully tested by constructing a CentroidTable with K entries, performing a centroid-ID-to-shard-address lookup, and verifying the returned address matches what was inserted — delivers the core routing contract independently.

**Acceptance Scenarios**:

1. **Given** a CentroidTable with K=64 entries mapping centroid IDs 0–63 to distinct shard addresses, **When** the coordinator looks up centroid ID 42, **Then** the exact shard address registered for centroid 42 is returned.
2. **Given** a populated CentroidTable, **When** the coordinator looks up a centroid ID that does not exist in the table, **Then** the lookup returns a clear "not found" signal without panicking.
3. **Given** a CentroidTable, **When** multiple concurrent lookups occur simultaneously, **Then** all lookups return correct results with no data races.

---

### User Story 2 - CentroidTable Distribution via Serialization (Priority: P2)

After k-means clustering produces the centroid-to-shard mapping, the coordinator must distribute the CentroidTable to the staging shard. The staging shard is the only shard that needs a copy of the table — it uses it to bulk-assign its buffered vectors to the correct partition shards when the staging threshold is reached. Normal (partition) shards do not use the table; they only receive and store the vectors routed to them. The table is serialized (to bincode for efficiency or JSON for readability) and sent over the wire from coordinator to staging shard.

**Why this priority**: The staging shard cannot perform bulk-assignment without knowing which shard owns each centroid. Asking the coordinator for every vector's destination during a bulk-assign of thousands of vectors would be prohibitively slow; a local copy of the table lets the staging shard resolve all destinations in memory. Reliable serialization is what makes this hand-off possible.

**Independent Test**: Can be fully tested by serializing a CentroidTable to both bincode and JSON, deserializing each back, and asserting the round-tripped table is identical to the original — delivers the distribution contract without requiring a live cluster.

**Acceptance Scenarios**:

1. **Given** a fully populated CentroidTable with K entries, **When** it is serialized to JSON, **Then** the JSON output is valid, human-readable, and contains all K centroid-to-shard mappings.
2. **Given** a fully populated CentroidTable, **When** it is serialized to bincode and then deserialized, **Then** the resulting table is byte-for-byte equivalent to the original.
3. **Given** a JSON-serialized CentroidTable, **When** it is deserialized by any node in the cluster, **Then** the reconstructed table's lookups return the same addresses as the original.
4. **Given** a malformed or truncated bincode payload, **When** deserialization is attempted, **Then** the system returns a descriptive error rather than corrupting state or panicking.

---

### User Story 3 - CentroidTable Inspection and Debugging (Priority: P3)

A developer or operator wants to inspect the current partition layout — which centroid maps to which shard — to diagnose routing issues or verify that k-means output was applied correctly. They export the table to JSON and read it directly.

**Why this priority**: Observability is important for a research prototype, but is not on the critical data path. Human-readable export is a convenience that does not block the core routing or distribution stories.

**Independent Test**: Can be fully tested by exporting a CentroidTable to JSON and verifying that the output contains legible centroid IDs and shard addresses, is valid JSON, and can be re-imported without loss.

**Acceptance Scenarios**:

1. **Given** a CentroidTable, **When** exported to JSON, **Then** each entry is human-readable with clearly labeled centroid ID and shard address fields.
2. **Given** an empty CentroidTable (K=0), **When** serialized to JSON or bincode, **Then** the output represents an empty mapping without errors and can be deserialized back to an empty table.

---

### Edge Cases

- What happens when the CentroidTable is looked up with a centroid ID larger than the maximum registered centroid?
- How does the table behave when two centroids are registered to the same shard address (one shard hosts multiple partitions)?
- What happens when the CentroidTable is initialized with zero entries and the system attempts a lookup immediately?
- How does deserialization handle a JSON payload produced by a different version of the table schema? → V1 decision: no schema versioning; schema is considered stable and all nodes redeploy together if it changes. A future version field is out of scope for V1.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST maintain an in-memory, unordered (hash-based) mapping from each centroid ID (`u32`) to exactly one shard address. No iteration-order guarantee is required.
- **FR-002**: The system MUST support O(1) lookup of a shard address given a valid centroid ID.
- **FR-003**: The system MUST allow bulk construction of the CentroidTable from a list of (centroid ID, shard address) pairs produced by k-means clustering.
- **FR-004**: The system MUST serialize the CentroidTable to valid JSON, preserving all centroid-to-address mappings without loss.
- **FR-005**: The system MUST serialize the CentroidTable to bincode (compact binary format), preserving all mappings without loss.
- **FR-006**: The system MUST deserialize a CentroidTable from a valid JSON payload, reconstructing a fully functional table.
- **FR-007**: The system MUST deserialize a CentroidTable from a valid bincode payload, reconstructing a fully functional table.
- **FR-008**: The system MUST expose the total number of centroids registered in the table.
- **FR-009**: The system MUST return a well-defined "not found" result (not a panic) when a lookup is performed for an unregistered centroid ID.
- **FR-010**: The system MUST support concurrent read access to the CentroidTable without data corruption.
- **FR-011**: The CentroidTable MUST be distributed by the coordinator to the staging shard only. Normal partition shards MUST NOT receive or hold a copy of the table — they only store and serve vectors that are routed to them.

### Key Entities

- **CentroidTable**: The primary artifact — an in-memory mapping from centroid ID to shard address. Created once after k-means, then treated as read-only during normal cluster operation. Must support full round-trip serialization.
- **CentroidId**: A `u32` value uniquely identifying one Voronoi cell / IVF partition. The range is determined by K (the number of centroids, e.g. 64–512); `u32` provides comfortable headroom well beyond the K ≤ 512 V1 limit.
- **ShardAddress**: A network location string (e.g., `host:port`) identifying the shard node responsible for a given partition. Multiple centroid IDs may map to the same shard address.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A CentroidTable with K=512 entries serializes to JSON and deserializes back within 50 ms on development hardware, with 100% fidelity (all mappings preserved).
- **SC-002**: A CentroidTable with K=512 entries serializes to bincode and deserializes back within 5 ms on development hardware, with 100% fidelity.
- **SC-003**: Single centroid-ID lookup completes in constant time regardless of K, verified by benchmark showing lookup time does not grow linearly with K=64, 128, 256, 512.
- **SC-004**: All unit tests for construction, lookup, JSON round-trip, and bincode round-trip pass with zero failures.
- **SC-005**: The system handles an attempted lookup of an unregistered centroid ID without panicking in 100% of test cases.

## Assumptions

- The CentroidTable is initialized once (after k-means) and is read-only during the cluster's operational lifetime — no concurrent writes after initialization.
- The table fits comfortably in memory for all supported K values (K ≤ 512 by V1 design), so no paging or lazy-loading is needed.
- JSON format serves human inspection and debugging; bincode serves efficient network distribution. Both must coexist.
- A shard address is represented as a plain string (e.g., `"127.0.0.1:7001"`) — structured parsing of addresses is out of scope for this feature.
- The CentroidTable type lives in the `common` crate so both the `coordinator` and `shard` crates can compile against it without a circular dependency — but only the coordinator and the staging shard ever hold a populated instance at runtime.
- Normal partition shards never hold or consult the CentroidTable. Their role is purely to store and serve the vectors assigned to them; all routing decisions happen upstream (coordinator) or during bulk-assignment (staging shard).
- The staging shard drops its CentroidTable copy when it transitions to inactive (after bulk-assignment completes). If a second cold-start cycle is needed, the coordinator distributes a fresh table at that time.
- Stage 2 Step 1 (k-means clustering) will produce the raw centroid-to-shard pairs that are fed into CentroidTable construction; this feature receives those pairs and does not compute them.
- Reliable delivery of the serialized CentroidTable from coordinator to staging shard (acknowledgment, retry on failure) is out of scope for this feature. That concern belongs to the transport layer built in Stage 5 (gRPC). This feature only guarantees correct serialization and deserialization.
- No schema versioning is included in V1. The serialized format has no embedded version field; schema is considered stable for the lifetime of the V1 prototype. If the schema changes, all nodes are redeployed together.

## Clarifications

### Session 2026-06-07

- Q: What is the concrete type of `CentroidId`? → A: `u32`
- Q: Does the CentroidTable require a guaranteed iteration order? → A: No — unordered (hash-based); O(1) lookup, no iteration-order guarantee
- Q: What happens to the staging shard's CentroidTable copy after bulk-assignment completes? → A: Drop it on deactivation; coordinator distributes a fresh copy if a new cold-start cycle begins
- Q: Who owns distribution failure handling (ack/retry) when coordinator sends the table to the staging shard? → A: Out of scope — transport-layer concern owned by Stage 5 (gRPC); this feature only guarantees correct serialization
- Q: Should the serialized CentroidTable include a schema version field? → A: No — V1 schema is stable; all nodes redeploy together if schema changes; no version field needed
