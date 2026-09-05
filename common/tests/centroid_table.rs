use common::CentroidTable;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a table with K entries: centroid ID i → "127.0.0.1:70{i:02}".
fn make_table(k: u32) -> CentroidTable {
    CentroidTable::new((0..k).map(|i| (i, format!("127.0.0.1:{}", 7000 + i))))
}

// ---------------------------------------------------------------------------
// US1 — Query Routing (T009–T013)
// ---------------------------------------------------------------------------

#[test]
fn test_construction_and_lookup() {
    let table = make_table(64);
    assert_eq!(table.get(0), Some("127.0.0.1:7000"));
    assert_eq!(table.get(42), Some("127.0.0.1:7042"));
    assert_eq!(table.get(63), Some("127.0.0.1:7063"));
}

#[test]
fn test_lookup_missing_id() {
    let table = make_table(64);
    assert_eq!(table.get(64), None);
    assert_eq!(table.get(999), None);
    assert_eq!(table.get(u32::MAX), None);
}

#[test]
fn test_len_and_is_empty() {
    let empty = CentroidTable::new(std::iter::empty());
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());

    let table = make_table(3);
    assert_eq!(table.len(), 3);
    assert!(!table.is_empty());
}

#[test]
fn test_concurrent_reads() {
    use std::sync::Arc;
    use std::thread;

    let table = Arc::new(make_table(64));
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let t = Arc::clone(&table);
            thread::spawn(move || {
                for i in 0..64_u32 {
                    let expected = format!("127.0.0.1:{}", 7000 + i);
                    assert_eq!(t.get(i), Some(expected.as_str()));
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("thread panicked");
    }
}

#[test]
fn test_same_address_multiple_centroids() {
    let table = CentroidTable::new([
        (0, "127.0.0.1:7001".into()),
        (1, "127.0.0.1:7001".into()),
        (2, "127.0.0.1:7001".into()),
    ]);
    assert_eq!(table.get(0), Some("127.0.0.1:7001"));
    assert_eq!(table.get(1), Some("127.0.0.1:7001"));
    assert_eq!(table.get(2), Some("127.0.0.1:7001"));
    assert_eq!(table.len(), 3);
}

// ---------------------------------------------------------------------------
// US2 — Distribution via Serialization (T016–T019)
// ---------------------------------------------------------------------------

#[test]
fn test_bincode_round_trip() {
    let original = make_table(64);
    let bytes = original.to_bincode().expect("to_bincode failed");
    let restored = CentroidTable::from_bincode(&bytes).expect("from_bincode failed");
    for i in 0..64_u32 {
        assert_eq!(
            restored.get(i),
            original.get(i),
            "mismatch at centroid {i}"
        );
    }
    assert_eq!(restored.len(), original.len());
}

#[test]
fn test_bincode_malformed() {
    let result = CentroidTable::from_bincode(b"not valid bincode garbage");
    assert!(result.is_err(), "expected Err for malformed bincode");
}

#[test]
fn test_json_round_trip() {
    let original = make_table(64);
    let json = original.to_json().expect("to_json failed");
    assert!(!json.is_empty());
    let restored = CentroidTable::from_json(&json).expect("from_json failed");
    for i in 0..64_u32 {
        assert_eq!(
            restored.get(i),
            original.get(i),
            "mismatch at centroid {i}"
        );
    }
    assert_eq!(restored.len(), original.len());
}

#[test]
fn test_json_malformed() {
    assert!(CentroidTable::from_json("{not: valid").is_err());
    assert!(CentroidTable::from_json("{\"entries\": \"wrong_type\"}").is_err());
}

// ---------------------------------------------------------------------------
// US3 — Inspection and Debugging (T020–T021)
// ---------------------------------------------------------------------------

#[test]
fn test_json_human_readable() {
    let table = CentroidTable::new([
        (0, "127.0.0.1:7000".into()),
        (1, "127.0.0.1:7001".into()),
        (2, "127.0.0.1:7002".into()),
    ]);
    let json = table.to_json().expect("to_json failed");

    // Must contain the "entries" wrapper key.
    assert!(json.contains("\"entries\""), "JSON missing 'entries' key: {json}");

    // Centroid IDs appear as string keys ("0", "1", "2").
    assert!(json.contains("\"0\""), "JSON missing centroid key '0': {json}");
    assert!(json.contains("\"1\""), "JSON missing centroid key '1': {json}");
    assert!(json.contains("\"2\""), "JSON missing centroid key '2': {json}");

    // Shard addresses appear as string values.
    assert!(json.contains("127.0.0.1:7000"), "JSON missing address: {json}");
}

#[test]
fn test_empty_table_serialize() {
    let empty: CentroidTable = CentroidTable::new(std::iter::empty());

    let json = empty.to_json().expect("to_json on empty table failed");
    assert!(!json.is_empty());
    let from_json = CentroidTable::from_json(&json).expect("from_json on empty table failed");
    assert_eq!(from_json.len(), 0);
    assert!(from_json.is_empty());

    let bytes = empty.to_bincode().expect("to_bincode on empty table failed");
    let from_bc = CentroidTable::from_bincode(&bytes).expect("from_bincode on empty table failed");
    assert_eq!(from_bc.len(), 0);
    assert!(from_bc.is_empty());
}
