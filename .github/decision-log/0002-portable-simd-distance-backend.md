# Decision 0002: Portable SIMD Distance Backend for Stage 1 Search

## Context
Stage 1 brute-force search was correct but used scalar-only distance kernels. The project needed a performance-oriented implementation step that improves local distance computation without changing search semantics or introducing architecture-specific unsafe intrinsics.

## Decision
Use a portable SIMD crate abstraction (`wide`) in `shard/src/distance.rs` for L2 and cosine kernels, while retaining explicit scalar reference implementations.

Implementation shape:
- keep public `l2_distance` and `cosine_distance` APIs stable
- add backend-selectable helpers (`Auto`, `Scalar`, `Simd`) for validation and testing
- use SIMD lane chunks with scalar tail handling for non-multiple dimensions
- keep `FlatVectorStore::search_topk` orchestration unchanged

## Rationale
- Improves local search math throughput with limited code churn.
- Avoids per-architecture intrinsics and target feature management in Stage 1.
- Keeps scalar behavior available as a correctness oracle for parity testing.
- Preserves deterministic top-k ordering because candidate maintenance/sorting logic is unchanged.

## Tradeoffs
- Portable abstraction may leave peak performance on the table versus hand-tuned AVX2/NEON intrinsics.
- Additional API surface in `distance.rs` (backend-selectable helpers) must be maintained.
- SIMD speedup magnitude depends on dimension size and memory bandwidth characteristics.

## Consequences
- Distance tests now include scalar-vs-SIMD parity checks across odd-tail dimensions and randomized inputs.
- Benchmark analysis must compare post-change results against pre-change baseline runs using the same Criterion labels.
- Future optimization can still introduce architecture-specific kernels behind the same backend model if justified.

## Status
Accepted (April 2026)
