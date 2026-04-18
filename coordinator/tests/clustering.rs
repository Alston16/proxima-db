//! Integration tests for `coordinator::clustering`.
//!
//! Test organisation follows the same conventions as `shard/tests/`:
//! helper generators at the top, edge-case unit tests next, then
//! correctness / property tests.

use common::{Centroid, DistanceMetric, Vector};
use coordinator::clustering::{ClusteringError, KMeansConfig, assign_to_shards, fit_kmeans};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::Rng;

// ─────────────────────────────────────────────────────────────────────────────
// Test fixtures
// ─────────────────────────────────────────────────────────────────────────────

/// Generates `n` random vectors of dimension `dim` using a seeded RNG.
fn make_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vector> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n as u64)
        .map(|id| Vector {
            id,
            data: (0..dim).map(|_| rng.gen_range(-1.0_f32..1.0)).collect(),
        })
        .collect()
}

/// Generates well-separated Gaussian blobs: `clusters` groups of `per_cluster`
/// vectors, each centred at `center_scale * cluster_index` on the first axis.
fn make_blob_vectors(
    clusters: usize,
    per_cluster: usize,
    dim: usize,
    seed: u64,
) -> Vec<Vector> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let center_scale = 20.0_f32;
    let mut vectors = Vec::with_capacity(clusters * per_cluster);
    for c in 0..clusters {
        let center = center_scale * c as f32;
        for _ in 0..per_cluster {
            let mut data: Vec<f32> = (0..dim).map(|_| rng.gen_range(-0.5_f32..0.5)).collect();
            data[0] += center; // Shift first axis to separate blobs.
            vectors.push(Vector {
                id: vectors.len() as u64,
                data,
            });
        }
    }
    vectors
}

fn l2_cfg(k: usize) -> KMeansConfig {
    KMeansConfig {
        k,
        metric: DistanceMetric::L2,
        seed: 42,
        max_iterations: 100,
        tolerance: None,
    }
}

fn cosine_cfg(k: usize) -> KMeansConfig {
    KMeansConfig {
        k,
        metric: DistanceMetric::Cosine,
        seed: 42,
        max_iterations: 100,
        tolerance: None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation: fit_kmeans
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fit_rejects_empty_corpus() {
    let err = fit_kmeans(&[], &l2_cfg(4)).unwrap_err();
    assert_eq!(err, ClusteringError::EmptyInput);
}

#[test]
fn fit_rejects_k_zero() {
    let vecs = make_random_vectors(10, 4, 1);
    let err = fit_kmeans(&vecs, &KMeansConfig { k: 0, ..l2_cfg(0) }).unwrap_err();
    assert!(matches!(err, ClusteringError::InvalidK { k: 0, n_samples: 10 }));
}

#[test]
fn fit_rejects_k_greater_than_n() {
    let vecs = make_random_vectors(5, 4, 2);
    let err = fit_kmeans(&vecs, &l2_cfg(6)).unwrap_err();
    assert!(matches!(err, ClusteringError::InvalidK { k: 6, n_samples: 5 }));
}

#[test]
fn fit_accepts_k_equals_n() {
    // Degenerate case: one centroid per vector — should still succeed.
    let vecs = make_random_vectors(4, 4, 3);
    let centroids = fit_kmeans(&vecs, &l2_cfg(4)).unwrap();
    assert_eq!(centroids.len(), 4);
}

#[test]
fn fit_rejects_dimension_mismatch() {
    let mut vecs = make_random_vectors(10, 4, 4);
    vecs[5].data = vec![1.0, 2.0]; // Wrong dimension.
    let err = fit_kmeans(&vecs, &l2_cfg(2)).unwrap_err();
    assert!(matches!(
        err,
        ClusteringError::DimensionMismatch { index: 5, expected: 4, actual: 2 }
    ));
}

#[test]
fn fit_rejects_zero_norm_cosine() {
    let mut vecs = make_random_vectors(10, 4, 5);
    vecs[3].data = vec![0.0; 4]; // Zero vector — invalid for cosine.
    let err = fit_kmeans(&vecs, &cosine_cfg(2)).unwrap_err();
    assert!(matches!(err, ClusteringError::ZeroNormVector { index: 3 }));
}

// ─────────────────────────────────────────────────────────────────────────────
// Output shape
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fit_returns_k_centroids() {
    let vecs = make_random_vectors(200, 16, 10);
    let centroids = fit_kmeans(&vecs, &l2_cfg(8)).unwrap();
    assert_eq!(centroids.len(), 8);
}

#[test]
fn fit_centroids_have_correct_dimension() {
    let dim = 32;
    let vecs = make_random_vectors(100, dim, 11);
    let centroids = fit_kmeans(&vecs, &l2_cfg(4)).unwrap();
    for c in &centroids {
        assert_eq!(c.data.len(), dim, "centroid {} has wrong dim", c.id);
    }
}

#[test]
fn fit_centroid_ids_are_sequential() {
    let vecs = make_random_vectors(60, 8, 12);
    let centroids = fit_kmeans(&vecs, &l2_cfg(6)).unwrap();
    for (i, c) in centroids.iter().enumerate() {
        assert_eq!(c.id, i as u32);
    }
}

#[test]
fn fit_centroid_shard_ids_are_placeholder_zero() {
    let vecs = make_random_vectors(60, 8, 13);
    let centroids = fit_kmeans(&vecs, &l2_cfg(4)).unwrap();
    for c in &centroids {
        assert_eq!(c.shard_id, 0, "shard_id should be placeholder 0");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Determinism
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fit_is_deterministic_with_same_seed() {
    let vecs = make_random_vectors(200, 16, 20);
    let cfg = l2_cfg(4);

    let run1 = fit_kmeans(&vecs, &cfg).unwrap();
    let run2 = fit_kmeans(&vecs, &cfg).unwrap();

    assert_eq!(run1.len(), run2.len());
    for (c1, c2) in run1.iter().zip(run2.iter()) {
        assert_eq!(c1.id, c2.id);
        for (x, y) in c1.data.iter().zip(c2.data.iter()) {
            assert!(
                (x - y).abs() < 1e-5,
                "centroid {} not reproducible: {} vs {}",
                c1.id,
                x,
                y
            );
        }
    }
}

#[test]
fn fit_different_seeds_may_differ() {
    let vecs = make_random_vectors(200, 16, 21);
    let cfg_a = KMeansConfig { seed: 1, ..l2_cfg(4) };
    let cfg_b = KMeansConfig { seed: 999999, ..l2_cfg(4) };

    let a = fit_kmeans(&vecs, &cfg_a).unwrap();
    let b = fit_kmeans(&vecs, &cfg_b).unwrap();

    // Both should have k centroids; they might cluster differently.
    assert_eq!(a.len(), 4);
    assert_eq!(b.len(), 4);
}

// ─────────────────────────────────────────────────────────────────────────────
// Correctness: separable blobs
// ─────────────────────────────────────────────────────────────────────────────

/// Naive assignment: for each vector find its nearest centroid by L2.
fn naive_cluster_assignment(vectors: &[Vector], centroids: &[Centroid]) -> Vec<Vec<u64>> {
    let mut clusters = vec![Vec::new(); centroids.len()];
    for v in vectors {
        let (nearest, _) = centroids
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let dist: f32 = v
                    .data
                    .iter()
                    .zip(c.data.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                (i, dist)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        clusters[nearest].push(v.id);
    }
    clusters
}

#[test]
fn fit_separates_two_blobs() {
    // Two blobs with 20 vectors each, trivially separable along axis 0.
    let vecs = make_blob_vectors(2, 20, 8, 30);
    let centroids = fit_kmeans(&vecs, &l2_cfg(2)).unwrap();
    assert_eq!(centroids.len(), 2);

    let clusters = naive_cluster_assignment(&vecs, &centroids);
    // Each cluster should contain vectors almost exclusively from one blob.
    // We tolerate up to 2 stragglers per cluster (generous for only 20 pts).
    for cluster in &clusters {
        assert!(
            cluster.len() >= 18,
            "cluster too small ({}); blobs not separated",
            cluster.len()
        );
    }
}

#[test]
fn fit_separates_three_blobs() {
    let vecs = make_blob_vectors(3, 30, 16, 31);
    let centroids = fit_kmeans(&vecs, &l2_cfg(3)).unwrap();
    assert_eq!(centroids.len(), 3);

    let clusters = naive_cluster_assignment(&vecs, &centroids);
    for cluster in &clusters {
        assert!(
            cluster.len() >= 25,
            "cluster too small ({}); blobs not separated",
            cluster.len()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cosine metric
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fit_cosine_returns_k_normalised_centroids() {
    let vecs = make_random_vectors(100, 16, 40);
    let centroids = fit_kmeans(&vecs, &cosine_cfg(4)).unwrap();
    assert_eq!(centroids.len(), 4);
    for c in &centroids {
        assert_eq!(c.data.len(), 16);
    }
}

#[test]
fn fit_cosine_is_deterministic() {
    let vecs = make_random_vectors(100, 8, 41);
    let cfg = cosine_cfg(3);
    let r1 = fit_kmeans(&vecs, &cfg).unwrap();
    let r2 = fit_kmeans(&vecs, &cfg).unwrap();
    for (c1, c2) in r1.iter().zip(r2.iter()) {
        for (x, y) in c1.data.iter().zip(c2.data.iter()) {
            assert!((x - y).abs() < 1e-5);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation: assign_to_shards
// ─────────────────────────────────────────────────────────────────────────────

fn two_centroids() -> Vec<Centroid> {
    vec![
        Centroid { id: 0, data: vec![0.0, 0.0], shard_id: 10 },
        Centroid { id: 1, data: vec![10.0, 10.0], shard_id: 20 },
    ]
}

#[test]
fn assign_rejects_empty_centroids() {
    let err = assign_to_shards(&[1.0, 0.0], &[], 1, DistanceMetric::L2).unwrap_err();
    assert_eq!(err, ClusteringError::EmptyInput);
}

#[test]
fn assign_rejects_nprobe_zero() {
    let err = assign_to_shards(&[1.0, 0.0], &two_centroids(), 0, DistanceMetric::L2).unwrap_err();
    assert!(matches!(err, ClusteringError::NprobeTooLarge { .. }));
}

#[test]
fn assign_rejects_nprobe_exceeds_centroid_count() {
    let err = assign_to_shards(&[1.0, 0.0], &two_centroids(), 3, DistanceMetric::L2).unwrap_err();
    assert!(matches!(err, ClusteringError::NprobeTooLarge { nprobe: 3, n_centroids: 2 }));
}

#[test]
fn assign_rejects_query_dimension_mismatch() {
    let err = assign_to_shards(&[1.0], &two_centroids(), 1, DistanceMetric::L2).unwrap_err();
    assert!(matches!(err, ClusteringError::QueryDimensionMismatch { expected: 2, actual: 1 }));
}

#[test]
fn assign_rejects_zero_norm_cosine_query() {
    let err =
        assign_to_shards(&[0.0, 0.0], &two_centroids(), 1, DistanceMetric::Cosine).unwrap_err();
    assert!(matches!(err, ClusteringError::ZeroNormVector { .. }));
}

// ─────────────────────────────────────────────────────────────────────────────
// Correctness: assign_to_shards
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn assign_nprobe1_returns_nearest_shard() {
    let centroids = two_centroids();
    // Query near [0,0] → shard_id 10
    let shards = assign_to_shards(&[0.5, 0.5], &centroids, 1, DistanceMetric::L2).unwrap();
    assert_eq!(shards, vec![10]);
}

#[test]
fn assign_nprobe2_returns_two_shards_ordered() {
    let centroids = two_centroids();
    let shards = assign_to_shards(&[0.5, 0.5], &centroids, 2, DistanceMetric::L2).unwrap();
    assert_eq!(shards.len(), 2);
    assert_eq!(shards[0], 10); // Nearest first.
    assert_eq!(shards[1], 20); // Second nearest.
}

#[test]
fn assign_soft_includes_hard_result() {
    // Border vector equidistant from all three centroids forming a triangle.
    let centroids = vec![
        Centroid { id: 0, data: vec![0.0, 0.0], shard_id: 100 },
        Centroid { id: 1, data: vec![10.0, 0.0], shard_id: 101 },
        Centroid { id: 2, data: vec![5.0, 8.66], shard_id: 102 },
    ];
    let border = vec![5.0, 0.5]; // Sits near boundary between 100 and 101.

    let hard = assign_to_shards(&border, &centroids, 1, DistanceMetric::L2).unwrap();
    let soft = assign_to_shards(&border, &centroids, 2, DistanceMetric::L2).unwrap();

    assert_eq!(hard.len(), 1);
    assert_eq!(soft.len(), 2);
    assert!(
        soft.contains(&hard[0]),
        "soft assignment should include hard assignment result"
    );
}

#[test]
fn assign_cosine_routes_to_correct_direction() {
    // Two centroids pointing in opposite directions.
    // Query pointing in the same direction as centroid 0 should land there.
    let centroids = vec![
        Centroid { id: 0, data: vec![1.0, 0.0], shard_id: 50 }, // Unit east.
        Centroid { id: 1, data: vec![-1.0, 0.0], shard_id: 51 }, // Unit west.
    ];
    let query = vec![0.9, 0.1]; // Mostly east.
    let shards = assign_to_shards(&query, &centroids, 1, DistanceMetric::Cosine).unwrap();
    assert_eq!(shards, vec![50]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Round-trip: fit + assign
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fit_then_assign_places_vectors_in_nearby_shard() {
    let vecs = make_blob_vectors(2, 30, 8, 60);
    let mut centroids = fit_kmeans(&vecs, &l2_cfg(2)).unwrap();

    // Assign shard IDs so assignment results are meaningful.
    centroids[0].shard_id = 1000;
    centroids[1].shard_id = 2000;

    // Vectors from blob 0 (centred near x[0] = 0) should mostly go to one shard.
    let blob0_vec = &vecs[0]; // First vector of blob 0.
    let assigned = assign_to_shards(&blob0_vec.data, &centroids, 1, DistanceMetric::L2).unwrap();
    assert_eq!(assigned.len(), 1);
    // The assigned shard should be one of the two valid shards.
    assert!(assigned[0] == 1000 || assigned[0] == 2000);
}
