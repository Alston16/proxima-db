# Code Quality: Documentation and Testing Conventions

This file defines the mandatory code quality conventions for ProximaDB.
All new code and all modifications to existing code must follow these rules.

---

## Documentation

Every public item — type alias, struct, enum, enum variant, trait, function, and
method — **must** have a `///` doc comment.

### Rules

- Use `///` (outer doc comments) for all public items.
- Use `//!` (inner doc comments) only at the top of a module file to describe
  the module as a whole, if the module warrants a summary.
- Begin each comment with a short, imperative sentence describing what the item
  **is** or **does**.
- Add a blank `///` line between the summary sentence and any extended
  description, `# Examples`, `# Errors`, or `# Panics` sections.
- Always include a `# Errors` section on any function that returns `Result`.
- For structs and enums, document every public field and every variant in
  addition to the type itself.
- Cross-link related types with `[TypeName]` or `[module::TypeName]` where
  helpful.

### Example

```rust
/// Unique identifier for a vector within the database.
pub type VectorId = u64;

/// Errors that can occur when reading from or writing to a [`FlatVectorStore`].
#[derive(Debug)]
pub enum StorageError {
    /// An underlying I/O error.
    Io(io::Error),
    /// The file header is missing, truncated, or contains unexpected values.
    InvalidHeader(&'static str),
    /// The dimension of a vector does not match the store's configured dimension.
    DimensionMismatch { expected: usize, actual: usize },
}

impl FlatVectorStore {
    /// Opens an existing store file or creates a new one at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] when the file cannot be opened or the header
    /// is invalid.
    pub fn open_or_create(path: impl AsRef<Path>, dimension: usize) -> Result<Self, StorageError> {
        // ...
    }
}
```

---

## Testing

Tests must live in **separate integration test files** under the crate's
`tests/` directory. Do **not** use inline `#[cfg(test)]` modules inside
source files.

### Rules

- Place all tests under `<crate-root>/tests/`.
- Name each test file after the source module it exercises:
  - `src/storage.rs` → `tests/storage.rs`
  - `src/state.rs`   → `tests/state.rs`
- Each integration test file uses the crate's public API (no `use super::*`).
- Import types from the crate and from `common` by their crate paths:
  ```rust
  use shard::storage::{FlatVectorStore, StorageError};
  use common::Vector;
  ```
- Async tests must use `#[tokio::test]`.
- Test names must be descriptive snake_case sentences, e.g.
  `creates_and_reopens_store`, `rejects_dimension_mismatch`.

### Example file layout

```
shard/
  src/
    storage.rs   ← implementation only, no #[cfg(test)] blocks
    state.rs     ← implementation only, no #[cfg(test)] blocks
  tests/
    storage.rs   ← integration tests for storage
    state.rs     ← integration tests for state
```

### Example test file

```rust
// shard/tests/storage.rs
use common::Vector;
use shard::storage::{FlatVectorStore, StorageError};
use tempfile::tempdir;

#[test]
fn creates_and_reopens_store() {
    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("vectors.bin");
    // ...
}
```

---

## Checklist for every PR

- [ ] Every new or modified public item has a `///` doc comment.
- [ ] All `Result`-returning public functions have a `# Errors` section.
- [ ] New tests are placed in `tests/<module>.rs`, not in `#[cfg(test)]` blocks.
- [ ] Existing inline `#[cfg(test)]` blocks are migrated to `tests/` when the
      surrounding source file is touched.
