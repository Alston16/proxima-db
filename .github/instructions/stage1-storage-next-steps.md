# Stage 1 Storage Next Steps

This file defines immediate implementation guidance after landing flat storage.

## Next Technical Work
1. Add distance functions (L2 and cosine) in a reusable module.
2. Implement brute-force top-k search over `FlatVectorStore::iter()`.
3. Add correctness tests comparing top-k results against a naive reference.
4. Add benchmark harness for single-node QPS and latency baseline.

## Guardrails
- Do not add HNSW in this phase.
- Keep RPC schema expansion separate from local search implementation.
- Keep fixed-dimension invariant in place.
- Preserve header compatibility checks during all read/write paths.

## Validation Checklist
- Unit tests for distance metrics.
- Unit tests for top-k ordering and tie handling.
- Property-style test coverage for random vector sets.
- Bench output includes median/p95 latency and QPS.
