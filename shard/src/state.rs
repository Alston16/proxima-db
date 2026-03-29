use std::path::Path;

use tokio::sync::Mutex;

use crate::storage::{FlatVectorStore, StorageError};

#[derive(Debug)]
pub struct ShardState {
    pub store: Mutex<FlatVectorStore>,
}

impl ShardState {
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

#[cfg(test)]
mod tests {
    use common::Vector;
    use tempfile::tempdir;

    use crate::storage::FlatVectorStore;

    use super::ShardState;

    #[tokio::test]
    async fn shard_state_opens_store_and_persists_inserted_vectors() {
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().join("vectors.bin");

        let state = ShardState::open_or_create(&path, 2, 1).unwrap();
        {
            let mut store = state.store.lock().await;
            store
                .insert(&Vector {
                    id: 99,
                    data: vec![3.0, 4.0],
                })
                .unwrap();
            store.flush().unwrap();
        }

        let reopened = FlatVectorStore::open_or_create(&path, 2, 1).unwrap();
        let vector = reopened.get(0).unwrap().unwrap();
        assert_eq!(vector.id, 99);
        assert_eq!(vector.data, vec![3.0, 4.0]);
    }
}