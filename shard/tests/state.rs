use common::Vector;
use shard::state::ShardState;
use shard::storage::FlatVectorStore;
use tempfile::tempdir;

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
