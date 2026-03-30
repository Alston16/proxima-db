use common::Vector;
use shard::storage::{FlatVectorStore, StorageError};
use tempfile::tempdir;

#[test]
fn creates_and_reopens_store() {
    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("vectors.bin");

    {
        let mut store = FlatVectorStore::open_or_create(&path, 3, 2).unwrap();
        store
            .insert(&Vector {
                id: 42,
                data: vec![1.0, 2.0, 3.0],
            })
            .unwrap();
        store.flush().unwrap();
    }

    let reopened = FlatVectorStore::open_or_create(&path, 3, 1).unwrap();
    assert_eq!(reopened.len(), 1);
    let vector = reopened.get(0).unwrap().unwrap();
    assert_eq!(vector.id, 42);
    assert_eq!(vector.data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn grows_when_capacity_is_exhausted() {
    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("vectors.bin");
    let mut store = FlatVectorStore::open_or_create(&path, 2, 1).unwrap();

    store
        .insert(&Vector {
            id: 1,
            data: vec![1.0, 1.0],
        })
        .unwrap();
    store
        .insert(&Vector {
            id: 2,
            data: vec![2.0, 2.0],
        })
        .unwrap();

    assert_eq!(store.len(), 2);
    assert!(store.capacity() >= 2);
    let values: Vec<_> = store.iter().map(|record| (record.id, record.data.to_vec())).collect();
    assert_eq!(values.len(), 2);
    assert_eq!(values[0], (1, vec![1.0, 1.0]));
    assert_eq!(values[1], (2, vec![2.0, 2.0]));
}

#[test]
fn rejects_dimension_mismatch() {
    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("vectors.bin");
    let mut store = FlatVectorStore::open_or_create(&path, 3, 1).unwrap();

    let error = store
        .insert(&Vector {
            id: 7,
            data: vec![1.0, 2.0],
        })
        .unwrap_err();

    match error {
        StorageError::DimensionMismatch { expected, actual } => {
            assert_eq!(expected, 3);
            assert_eq!(actual, 2);
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn inserts_batches_without_losing_order() {
    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("vectors.bin");
    let mut store = FlatVectorStore::open_or_create(&path, 2, 1).unwrap();
    let vectors = [
        Vector {
            id: 10,
            data: vec![1.0, 0.0],
        },
        Vector {
            id: 11,
            data: vec![0.0, 1.0],
        },
        Vector {
            id: 12,
            data: vec![1.0, 1.0],
        },
    ];

    store.insert_batch(vectors.iter()).unwrap();

    let ids: Vec<_> = store.iter().map(|record| record.id).collect();
    assert_eq!(ids, vec![10, 11, 12]);
}
