/// Integration tests for brute-force k-NN search over FlatVectorStore and ShardState.
///
/// These tests validate:
///   - Correct nearest-neighbour ranking for L2 and cosine metrics.
///   - Edge cases: k=0, k>len, empty store, dimension mismatch.
///   - Tie-breaking determinism (ascending VectorId).
///   - ShardState async wrapper correctness.
///   - Randomised parity against a naive reference implementation.
use common::{SearchResult, Vector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use shard::distance::DistanceMetric;
use shard::state::ShardState;
use shard::storage::FlatVectorStore;
use tempfile::tempdir;

// ── helpers ──────────────────────────────────────────────────────────────────

fn make_store(dim: usize, vectors: &[(u64, Vec<f32>)]) -> (FlatVectorStore, tempfile::TempDir) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.bin");
    let mut store = FlatVectorStore::open_or_create(&path, dim, vectors.len().max(1)).unwrap();
    for (id, data) in vectors {
        store.insert(&Vector { id: *id, data: data.clone() }).unwrap();
    }
    (store, dir)
}

/// Naive reference: compute all distances, sort, return top-k.
fn naive_search(
    vectors: &[(u64, Vec<f32>)],
    query: &[f32],
    k: usize,
    metric: DistanceMetric,
) -> Vec<SearchResult> {
    let mut scored: Vec<SearchResult> = vectors
        .iter()
        .map(|(id, data)| {
            let distance = match metric {
                DistanceMetric::L2 => {
                    data.iter()
                        .zip(query.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>()
                        .sqrt()
                }
                DistanceMetric::Cosine => {
                    let dot: f32 = data.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
                    let na = data.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let nb = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if na == 0.0 || nb == 0.0 {
                        1.0
                    } else {
                        1.0 - (dot / (na * nb)).clamp(-1.0, 1.0)
                    }
                }
            };
            SearchResult { id: *id, distance }
        })
        .collect();
    scored.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
    scored.truncate(k);
    scored
}

// ── empty store / k=0 edge cases ─────────────────────────────────────────────

#[test]
fn search_returns_empty_for_k_zero() {
    let (store, _dir) = make_store(2, &[(1, vec![1.0, 0.0]), (2, vec![0.0, 1.0])]);
    let results = store.search_topk(&[1.0, 0.0], 0, DistanceMetric::L2).unwrap();
    assert!(results.is_empty());
}

#[test]
fn search_returns_empty_for_empty_store() {
    let (store, _dir) = make_store(2, &[]);
    let results = store.search_topk(&[1.0, 0.0], 5, DistanceMetric::L2).unwrap();
    assert!(results.is_empty());
}

#[test]
fn search_returns_all_when_k_exceeds_len() {
    let data = &[(1, vec![1.0, 0.0]), (2, vec![0.0, 1.0]), (3, vec![1.0, 1.0])];
    let (store, _dir) = make_store(2, data);
    let results = store.search_topk(&[0.5, 0.5], 10, DistanceMetric::L2).unwrap();
    assert_eq!(results.len(), 3);
}

// ── dimension mismatch ────────────────────────────────────────────────────────

#[test]
fn search_errors_on_dimension_mismatch() {
    let (store, _dir) = make_store(3, &[(1, vec![1.0, 0.0, 0.0])]);
    let err = store.search_topk(&[1.0, 0.0], 1, DistanceMetric::L2).unwrap_err();
    match err {
        shard::storage::StorageError::DimensionMismatch { expected, actual } => {
            assert_eq!(expected, 3);
            assert_eq!(actual, 2);
        }
        other => panic!("unexpected error: {other}"),
    }
}

// ── L2 nearest-neighbour ordering ────────────────────────────────────────────

#[test]
fn l2_returns_exact_nearest_neighbour() {
    let data = &[
        (1, vec![0.0_f32, 0.0]),
        (2, vec![1.0, 0.0]),
        (3, vec![0.0, 1.0]),
        (4, vec![10.0, 10.0]),
    ];
    let (store, _dir) = make_store(2, data);

    // query is the origin — nearest should be id=1 (distance 0)
    let results = store.search_topk(&[0.0, 0.0], 1, DistanceMetric::L2).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 1);
    assert_eq!(results[0].distance, 0.0);
}

#[test]
fn l2_top3_are_sorted_ascending() {
    let data = &[
        (10, vec![0.0_f32, 0.0]),
        (20, vec![1.0, 0.0]),
        (30, vec![2.0, 0.0]),
        (40, vec![5.0, 0.0]),
        (50, vec![9.0, 0.0]),
    ];
    let (store, _dir) = make_store(2, data);
    let query = [0.0_f32, 0.0];
    let results = store.search_topk(&query, 3, DistanceMetric::L2).unwrap();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].id, 10); // distance 0
    assert_eq!(results[1].id, 20); // distance 1
    assert_eq!(results[2].id, 30); // distance 2
    // verify ascending order
    for w in results.windows(2) {
        assert!(w[0].distance <= w[1].distance, "not sorted: {:?}", results);
    }
}

// ── cosine nearest-neighbour ordering ────────────────────────────────────────

#[test]
fn cosine_finds_same_direction_as_nearest() {
    let data = &[
        (1, vec![1.0_f32, 0.0]),   // same direction as query
        (2, vec![0.0_f32, 1.0]),   // orthogonal
        (3, vec![-1.0_f32, 0.0]),  // opposite
    ];
    let (store, _dir) = make_store(2, data);
    let query = [2.0_f32, 0.0]; // scaled version of id=1

    let results = store.search_topk(&query, 3, DistanceMetric::Cosine).unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].id, 1, "nearest cosine should be same direction");
    assert!(results[0].distance < 1e-6, "distance should be ~0");
    assert_eq!(results[2].id, 3, "furthest cosine should be opposite direction");
}

// ── tie-breaking determinism ──────────────────────────────────────────────────

#[test]
fn ties_broken_by_ascending_vector_id() {
    // All vectors are identical — all distances are 0. IDs must come out ascending.
    let data = &[
        (5, vec![1.0_f32, 1.0]),
        (2, vec![1.0_f32, 1.0]),
        (8, vec![1.0_f32, 1.0]),
        (1, vec![1.0_f32, 1.0]),
    ];
    let (store, _dir) = make_store(2, data);
    let results = store.search_topk(&[1.0, 1.0], 3, DistanceMetric::L2).unwrap();

    assert_eq!(results.len(), 3);
    let ids: Vec<u64> = results.iter().map(|r| r.id).collect();
    assert_eq!(ids, vec![1, 2, 5], "expected ascending IDs on tie; got {ids:?}");
}

// ── ShardState async wrapper ──────────────────────────────────────────────────

#[tokio::test]
async fn shard_state_search_topk_returns_correct_result() {
    let dir = tempdir().unwrap();
    let state = ShardState::open_or_create(dir.path().join("v.bin"), 2, 4).unwrap();
    {
        let mut store = state.store.lock().await;
        for (id, x, y) in [(1u64, 0.0f32, 0.0), (2, 1.0, 0.0), (3, 0.0, 1.0), (4, 5.0, 5.0)] {
            store.insert(&Vector { id, data: vec![x, y] }).unwrap();
        }
    }

    let results = state.search_topk(&[0.0, 0.0], 2, DistanceMetric::L2).await.unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, 1);
    assert_eq!(results[1].id, 2);
}

// ── randomised parity against naive reference ─────────────────────────────────

fn assert_parity(results: &[SearchResult], reference: &[SearchResult]) {
    assert_eq!(
        results.len(),
        reference.len(),
        "result count differs: got {} vs ref {}",
        results.len(),
        reference.len()
    );
    for (i, (got, exp)) in results.iter().zip(reference.iter()).enumerate() {
        assert_eq!(
            got.id, exp.id,
            "id mismatch at rank {i}: got {} exp {}",
            got.id, exp.id
        );
        assert!(
            (got.distance - exp.distance).abs() < 1e-4,
            "distance mismatch at rank {i}: got {} exp {}",
            got.distance,
            exp.distance
        );
    }
}

#[test]
fn l2_parity_random_vectors() {
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
    let dim = 8;
    let n = 200;
    let k = 10;

    let data: Vec<(u64, Vec<f32>)> = (0..n)
        .map(|i| (i as u64, (0..dim).map(|_| rng.gen_range(0.0_f32..1.0)).collect()))
        .collect();

    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(0.0_f32..1.0)).collect();

    let (store, _dir) = make_store(dim, &data);
    let got = store.search_topk(&query, k, DistanceMetric::L2).unwrap();
    let reference = naive_search(&data, &query, k, DistanceMetric::L2);

    assert_parity(&got, &reference);
}

#[test]
fn cosine_parity_random_vectors() {
    let mut rng = StdRng::seed_from_u64(0xCAFE_BABE);
    let dim = 16;
    let n = 150;
    let k = 7;

    let data: Vec<(u64, Vec<f32>)> = (0..n)
        .map(|i| (i as u64, (0..dim).map(|_| rng.gen_range(-0.5_f32..0.5)).collect()))
        .collect();

    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-0.5_f32..0.5)).collect();

    let (store, _dir) = make_store(dim, &data);
    let got = store.search_topk(&query, k, DistanceMetric::Cosine).unwrap();
    let reference = naive_search(&data, &query, k, DistanceMetric::Cosine);

    assert_parity(&got, &reference);
}
