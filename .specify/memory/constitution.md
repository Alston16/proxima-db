<!--
Sync Impact Report
- Version change: N/A (template) -> 1.0.0
- Modified principles:
	- Placeholder Principle 1 -> I. Proximity-Aware Architecture First
	- Placeholder Principle 2 -> II. Deterministic Distance and Routing
	- Placeholder Principle 3 -> III. Testable Changes as a Merge Gate
	- Placeholder Principle 4 -> IV. Measured Performance and Recall
	- Placeholder Principle 5 -> V. Knowledge Base Synchronization
- Added sections:
	- Technical Standards
	- Development Workflow & Quality Gates
- Removed sections:
	- None
- Templates requiring updates:
	- ✅ updated: .specify/templates/plan-template.md
	- ✅ updated: .specify/templates/spec-template.md
	- ✅ updated: .specify/templates/tasks-template.md
	- ⚠ pending: .specify/templates/commands/*.md (directory not present in repository)
- Follow-up TODOs:
	- None
-->

# ProximaDB Constitution

## Core Principles

### I. Proximity-Aware Architecture First
All feature work MUST preserve the core design thesis of ProximaDB: vectors are
partitioned and routed by vector-space proximity rather than broadcast fan-out.
Changes that dilute this model (for example, unconditional all-shard query fan-out)
MUST include an explicit temporary exception plan and rollback criteria.
Rationale: this repository exists to validate proximity-aware distributed routing.

### II. Deterministic Distance and Routing
Distance computation, top-k ordering, and shard assignment MUST be deterministic for
identical inputs, configuration, and seed values. Tie-breaking MUST remain explicit
and stable. Metric semantics MUST be consistent end-to-end (for example, cosine uses
normalized vectors for clustering and query assignment).
Rationale: deterministic behavior is required for reproducible research and debugging.

### III. Testable Changes as a Merge Gate
Behavioral changes MUST include tests at the appropriate level before merge:
unit/integration tests for crate behavior, and cross-crate integration tests when
routing or RPC contracts change. Public APIs MUST include documentation, and public
`Result` functions MUST document `# Errors`.
Rationale: this project advances in staged increments and cannot regress correctness.

### IV. Measured Performance and Recall
Any change to search math, routing strategy, or index selection MUST define a
measurement plan and report impact using stable benchmark labels and task-relevant
metrics (at minimum latency and recall for affected paths). Optimizations MUST be
validated against a known-correct baseline path.
Rationale: ProximaDB is a research prototype where claims require empirical evidence.

### V. Knowledge Base Synchronization
Any architecture, workflow, or convention change MUST be reflected in the
repository's knowledge system under `.github/` (instructions, patterns,
decision-log, knowledge). Major technical decisions MUST include explicit context,
decision, rationale, and tradeoffs.
Rationale: the project intentionally treats `.github/` as persistent engineering memory.

## Technical Standards

- Rust workspace structure and crate boundaries MUST remain explicit (`common`,
	`coordinator`, `shard`, `client`) unless a change proposal documents why a
	boundary change is necessary.
- Fixed-dimension vector invariants and on-disk compatibility guarantees MUST be
	preserved unless accompanied by migration and validation steps.
- Integration tests MUST live under crate-level `tests/` directories and exercise
	public APIs rather than private internals.
- RPC and schema changes MUST be coordinated across `proto`, coordinator handlers,
	and shard implementations in the same feature scope or behind guarded rollout steps.

## Development Workflow & Quality Gates

1. Feature specs MUST state measurable outcomes and a validation approach tied to
	recall, latency, correctness, or routing fan-out.
2. Implementation plans MUST pass the Constitution Check before design and again
	before task generation.
3. Tasks MUST identify required tests, benchmark work, and documentation/decision-log
	synchronization when applicable.
4. Pull requests MUST state which principle(s) are impacted and provide evidence for
	compliance (tests, benchmark output, or explicit justified exception).

## Governance

This constitution is the highest-priority process contract for project delivery.
When lower-level guidance conflicts with this document, this constitution takes
precedence.

Amendment process:
1. Propose changes in a pull request that includes rationale and downstream impact.
2. Update dependent templates and guidance files in the same change set, or record
	explicit follow-up items in the Sync Impact Report.
3. Obtain maintainer approval before merging.

Versioning policy:
- MAJOR: Backward-incompatible governance changes or principle removals/redefinitions.
- MINOR: New principle/section or materially expanded mandatory guidance.
- PATCH: Clarifications, wording improvements, typo fixes, and non-semantic edits.

Compliance review expectations:
- Every plan and task set MUST include a constitution compliance check.
- Every merge request MUST provide evidence for affected principles.
- Periodic review SHOULD occur at each stage transition to ensure continued alignment.

**Version**: 1.0.0 | **Ratified**: 2026-05-21 | **Last Amended**: 2026-05-21
