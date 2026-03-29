# Decision 0001: Fixed-Dimension Append-Only Mmap Store

## Context
Stage 1 requires flat vector storage backed by memory-mapped files. The project needed a practical baseline that is easy to validate and supports upcoming brute-force search.

## Decision
Use a shard-local memory-mapped binary format with:
- fixed vector dimension per file
- append-only records
- fixed-width record layout (`u64` id + contiguous `f32` values)
- explicit file header metadata (magic/version/dimension/len/capacity)

## Rationale
- Enables deterministic offsets and simple scan logic.
- Minimizes implementation complexity for Stage 1.
- Provides clear corruption/compatibility checks during reopen.
- Avoids early complexity around compaction/deletes and variable-length records.

## Tradeoffs
- No delete/compaction yet.
- Single format per dimension (cannot mix dimensions in one file).
- No WAL/crash-recovery semantics yet.

## Consequences
- Future Stage 1 search can iterate records directly from mmap-backed storage.
- Higher-level shard state handles synchronization around storage access.
- Future durability features may require format extension or sidecar metadata.

## Status
Accepted (March 2026)
