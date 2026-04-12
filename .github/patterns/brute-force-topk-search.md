# Pattern: Brute-Force Top-K Search

## Intent
Provide a deterministic Stage 1 nearest-neighbour baseline over the flat memory-mapped store before introducing ANN indexing.

## Use When
- Search is still local to one shard.
- The codebase needs a correctness-first baseline for later HNSW comparison.
- Dataset sizes are small enough that full scan cost is acceptable.
- Future routing and ANN work needs a reference implementation for parity tests.

## Structure
- Keep distance math in `shard/src/distance.rs`.
- Keep scan orchestration in `FlatVectorStore::search_topk()`.
- Iterate records through `FlatVectorStore::iter()` so scans stay zero-copy.
- Track candidates in a fixed-size `BinaryHeap<Candidate>`.
- Sort the final heap contents before returning `Vec<SearchResult>`.

## Ordering Rules
- Primary ordering is ascending distance.
- Ties are resolved by ascending `VectorId`.
- Heap ordering is intentionally inverted so the current worst candidate is easy to evict.

## Guardrails
- Validate query dimension before scanning.
- Return an empty vector for `k == 0` or an empty store.
- Keep distance functions side-effect free and preserve scalar reference behavior when SIMD is enabled.
- Preserve deterministic ordering so benchmark and parity tests stay stable.

## SIMD Optimization Notes
- Keep scan orchestration in `FlatVectorStore::search_topk()` unchanged; optimize only metric kernels in `distance.rs`.
- Use backend-selectable distance entrypoints so tests can compare scalar and SIMD outputs directly.
- SIMD kernels must support non-lane dimensions via scalar tail handling.
- Preserve cosine edge semantics (`1.0` for zero-norm vectors) and clamp behavior to avoid ranking regressions.
- Benchmark comparison should reuse existing Criterion labels and compare against recorded pre-change baseline runs.

## Why This Pattern
- Gives the project a trustworthy reference path for search correctness.
- Matches the current flat-store file layout and zero-copy iteration model.
- Creates a stable baseline for future comparisons against SIMD and HNSW.