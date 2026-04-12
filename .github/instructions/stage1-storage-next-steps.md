# Stage 1 Storage Next Steps

This file now tracks the work that follows the landed Stage 1 local-storage and brute-force-search baseline.

## Completed Baseline
1. Distance functions exist for L2 and cosine with scalar reference and SIMD-accelerated paths.
2. `FlatVectorStore::search_topk()` performs brute-force top-k search.
3. Integration tests cover search correctness, edge cases, tie handling, and randomized parity.
4. Criterion benchmarks exist for representative L2 and cosine dataset sizes.

## Next Technical Work
1. Keep Stage 1 stable while beginning Stage 2 centroid-table and shard-assignment work.
2. Expand the proto and RPC surface from `Ping` to insert and search operations.
3. Move coordinator from connectivity smoke test to actual routing logic.
4. Add shared configuration and error types so coordinator and shard stop relying on ad hoc setup.
5. Benchmark and tune the landed SIMD implementation using unchanged Criterion labels and pre-change baseline artifacts.

## Guardrails
- Do not mix ANN indexing work into the current routing and RPC foundation work.
- Keep RPC schema changes separate from local search semantics unless both ends are updated together.
- Preserve the fixed-dimension invariant and header compatibility guarantees.
- Keep top-k ordering deterministic so tests and benchmarks remain comparable across optimizations.

## Validation Checklist
- `cargo test -p shard` stays green after Stage 2 prep changes.
- Search parity tests remain unchanged unless behavior intentionally changes.
- Benchmark labels remain stable so historical Criterion output stays comparable.
- Any SIMD tuning must be measured against the recorded scalar baseline, not replace correctness guardrails.
