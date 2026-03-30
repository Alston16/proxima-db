use std::path::Path;

use tokio::sync::Mutex;

use crate::storage::{FlatVectorStore, StorageError};

/// Thread-safe state for a single shard node.
///
/// Wraps a [`FlatVectorStore`] in a [`Mutex`] so that the gRPC service handler
/// and any background tasks can share mutable access to the vector store
/// without data races.
#[derive(Debug)]
pub struct ShardState {
    /// The underlying vector store, protected by an async mutex.
    pub store: Mutex<FlatVectorStore>,
}

impl ShardState {
    /// Opens or creates the vector store at `path` and wraps it in a
    /// [`ShardState`].
    ///
    /// # Errors
    ///
    /// Propagates any [`StorageError`] returned by
    /// [`FlatVectorStore::open_or_create`].
    pub fn open_or_create(
        path: impl AsRef<Path>,
        dimension: usize,
        initial_capacity: usize,
    ) -> Result<Self, StorageError> {
        let store = FlatVectorStore::open_or_create(path, dimension, initial_capacity)?;
        Ok(Self {
            store: Mutex::new(store),
        })
    }
}