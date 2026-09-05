# Research: CentroidTable

**Branch**: `001-centroid-table` | **Date**: 2026-06-07

---

## Decision 1: Map type — `HashMap<u32, String>`

**Decision**: Use `std::collections::HashMap<u32, String>` as the underlying store.

**Rationale**: FR-002 requires O(1) lookup; FR-001 confirms no iteration-order guarantee is needed. `HashMap` is the standard Rust choice — it provides amortized O(1) get, is part of std (no extra dep), and serializes correctly via serde. `BTreeMap` would give O(log n) lookup with sorted iteration, which contradicts FR-002. `IndexMap` (insertion-ordered) is unnecessary: no ordering guarantee was requested (Clarification Q2).

**Alternatives considered**:
- `BTreeMap<u32, String>` — O(log n) lookup, deterministic JSON key order; rejected because FR-002 requires O(1).
- `IndexMap<u32, String>` — O(1) + insertion order; rejected — overkill, extra dependency, ordering not required.
- `Vec<(u32, String)>` — O(n) lookup; rejected — does not satisfy FR-002.

---

## Decision 2: Serialization libraries — `serde_json` + `bincode` 1.x

**Decision**: Add `serde_json = "1"` and `bincode = "1"` to `common/Cargo.toml`. Derive `serde::Serialize` / `serde::Deserialize` on `CentroidTable`; call `serde_json::to_string` / `bincode::serialize` as thin wrappers.

**Rationale**: `serde` is already a dependency of `common`. `bincode` 1.x uses serde traits directly — no second derive macro needed. Both formats work on the same derived `#[derive(Serialize, Deserialize)]`. Bincode 2.x uses its own `Encode`/`Decode` traits, requiring a second set of derives; the added complexity is not justified for K ≤ 512 entries.

**JSON key ordering**: `HashMap` serializes with non-deterministic key order. This is acceptable — JSON is used for human inspection (User Story 3), not for byte-level comparison. Deserialization is order-independent.

**Alternatives considered**:
- `bincode` 2.x — different `Encode`/`Decode` traits; requires additional derives alongside serde; rejected for unnecessary complexity.
- `postcard` — compact binary, no-std friendly; rejected — project already leans on serde ecosystem; bincode is simpler for the scope.
- `messagepack` (rmp-serde) — comparable to bincode; rejected — no advantage over bincode for this use case.

---

## Decision 3: Thread-safety model — `Arc<CentroidTable>` (no RwLock)

**Decision**: `CentroidTable` itself carries no synchronization. Callers wrap it in `Arc<CentroidTable>` for shared ownership. No `RwLock` or `Mutex` is needed.

**Rationale**: The spec (Assumptions) states the table is initialized once and is read-only thereafter. A shared immutable reference behind `Arc` is fully thread-safe in Rust without any locking — `Arc<T>` is `Send + Sync` when `T: Send + Sync`, and an immutable `HashMap` satisfies this. Adding `RwLock` would impose unnecessary overhead on every lookup and complicate the API for no benefit.

**Alternatives considered**:
- `Arc<RwLock<CentroidTable>>` — supports writes post-init; rejected because writes after init are out of scope (Assumptions).
- `Mutex<CentroidTable>` — serializes all access; rejected — worse than RwLock and still unnecessary.

---

## Decision 4: `ShardAddress` representation — `String`

**Decision**: Represent shard addresses as plain `String` (e.g., `"127.0.0.1:7001"`). No parsed `SocketAddr` or structured type.

**Rationale**: The spec (Assumptions) explicitly states "structured parsing of addresses is out of scope." Keeping it as `String` avoids parsing errors at table construction time and stays consistent with how Stage 5 (gRPC) will consume addresses — tonic accepts `&str` / `String` endpoints directly.

**Alternatives considered**:
- `std::net::SocketAddr` — parsed, validated; rejected — out of scope, breaks string-only gRPC endpoint consumption.
- Newtype `ShardAddress(String)` — adds type safety; deferred — low value for V1, easy to introduce in Stage 5.

---

## Decision 5: Public API shape — methods vs bare serde calls

**Decision**: Expose named methods (`to_json`, `from_json`, `to_bincode`, `from_bincode`) on `CentroidTable` rather than requiring callers to invoke `serde_json` / `bincode` directly.

**Rationale**: Wrapping serialization in methods hides the library choice, satisfies Constitution Principle III (documented public API), and makes the contract clear in the `# Errors` doc. Callers in `coordinator` and the staging shard do not need to depend on `serde_json`/`bincode` directly — they use the `common` crate's public API.

**Alternatives considered**:
- Bare `#[derive(Serialize, Deserialize)]` only — callers do their own serde; rejected — leaks library choice, harder to document errors.
- Separate `CentroidTableCodec` struct — unnecessary indirection for two formats.

---

## Decision 6: Existing `Centroid` type alignment

The existing `common::Centroid` struct holds `id: u32`, `shard_id: ShardId` (u32), and `data: Vec<f32>`. `CentroidTable` maps `centroid_id: u32` → `shard_address: String`. These are complementary:
- `Centroid` carries geometric data and the abstract shard ID.
- `CentroidTable` carries routing data: which network address serves that shard.

No changes to the existing `Centroid` type are required or planned. The coordinator will use both: `Centroid.data` for distance computation, `CentroidTable` for address resolution.
