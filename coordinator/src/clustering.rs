//! K-means training for the coordinator's centroid table.
//!
//! This module is the sole entry point for fitting cluster centroids from a
//! training corpus.  The output `Vec<Centroid>` is consumed by the centroid
//! table (Stage 2 Step 2) and the shard-assignment helper in this same file.
//!
//! # Design decisions
//!
//! * **Full-batch k-means via linfa** — deterministic, correctness-first.
//!   Mini-batch optimisation is a future task once the baseline is proven.
//! * **Cosine metric** — input vectors are L2-normalised before fitting;
//!   centroids are returned in their normalised form.  This matches the
//!   cosine-distance semantics in `common::distance` (angular proximity).
//! * **Seeded RNG** — callers pass an explicit `u64` seed so training is
//!   reproducible across restarts and test runs.
//! * **Placeholder `shard_id`** — emitted centroids carry `shard_id = 0`.
//!   Stage 2 Step 2 (`CentroidTable`) assigns real shard IDs when it builds
//!   the routing table.
//!
//! # Out of scope for this step
//!
//! Online centroid updates, mini-batch fitting, staging-threshold
//! orchestration, and distributed routing integration.

use common::{Centroid, DistanceMetric, Vector, ShardId};
use common::topk::select_topk_by_vector;
use linfa::prelude::*;
use linfa_clustering::KMeans;
use linfa_nn::distance::L2Dist;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors returned by clustering and assignment operations.
#[derive(Debug, Error, PartialEq)]
pub enum ClusteringError {
    /// Training corpus is empty.
    #[error("training corpus is empty")]
    EmptyInput,

    /// `k` must be ≥ 1 and ≤ number of training vectors.
    #[error("invalid k={k}: must be in [1, {n_samples}]")]
    InvalidK { k: usize, n_samples: usize },

    /// All training vectors must have the same dimension.
    #[error("dimension mismatch: vector {index} has {actual} dimensions, expected {expected}")]
    DimensionMismatch {
        index: usize,
        expected: usize,
        actual: usize,
    },

    /// Vector dimension must be at least 1.
    #[error("vector dimension must be ≥ 1")]
    ZeroDimension,

    /// A zero-norm vector was supplied when the cosine metric is active.
    /// Cosine distance is undefined for the zero vector.
    #[error("zero-norm vector at index {index} is invalid for cosine metric")]
    ZeroNormVector { index: usize },

    /// `nprobe` exceeds the number of available centroids.
    #[error("nprobe={nprobe} exceeds centroid count {n_centroids}")]
    NprobeTooLarge {
        nprobe: usize,
        n_centroids: usize,
    },

    /// The query vector dimension does not match the centroid dimension.
    #[error("query has {actual} dimensions; centroids have {expected}")]
    QueryDimensionMismatch { expected: usize, actual: usize },

    /// An internal linfa error that should not occur with valid inputs.
    #[error("linfa error: {0}")]
    Linfa(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a single k-means training run.
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters (partitions).  Typical values: 64–512.
    pub k: usize,
    /// Distance metric.  Cosine inputs are normalised before fitting.
    pub metric: DistanceMetric,
    /// Seed for the k-means++ initialisation RNG.  Same seed ⇒ identical
    /// centroids on identical input.
    pub seed: u64,
    /// Maximum EM iterations before forcing convergence.
    pub max_iterations: u64,
    /// Optional convergence tolerance (change in inertia).  `None` uses
    /// linfa's default.
    pub tolerance: Option<f64>,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 64,
            metric: DistanceMetric::L2,
            seed: 0,
            max_iterations: 300,
            tolerance: None,
        }
    }
}

/// Fits k-means on `vectors` using the supplied `config`.
///
/// Returns `k` centroids with stable IDs `0..k` and placeholder `shard_id = 0`.
///
/// # Errors
///
/// Returns [`ClusteringError`] on invalid input (empty corpus, dimension
/// mismatch, zero-norm cosine vector, invalid k).
pub fn fit_kmeans(
    vectors: &[Vector],
    config: &KMeansConfig,
) -> Result<Vec<Centroid>, ClusteringError> {
    // ── Input validation ──────────────────────────────────────────────────
    if vectors.is_empty() {
        return Err(ClusteringError::EmptyInput);
    }

    let dim = vectors[0].data.len();
    if dim == 0 {
        return Err(ClusteringError::ZeroDimension);
    }
    if config.k == 0 || config.k > vectors.len() {
        return Err(ClusteringError::InvalidK {
            k: config.k,
            n_samples: vectors.len(),
        });
    }

    for (i, v) in vectors.iter().enumerate() {
        if v.data.len() != dim {
            return Err(ClusteringError::DimensionMismatch {
                index: i,
                expected: dim,
                actual: v.data.len(),
            });
        }
    }

    // ── Optional cosine normalisation ─────────────────────────────────────
    // For cosine metric we project every vector onto the unit hypersphere so
    // that Euclidean distances in normalised space correspond to angular
    // proximity.  Zero-norm vectors are rejected — cosine distance is
    // undefined for them.
    let owned: Vec<Vec<f32>>;
    let rows: &[Vec<f32>] = if config.metric == DistanceMetric::Cosine {
        owned = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let norm: f32 = v.data.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm == 0.0 {
                    Err(ClusteringError::ZeroNormVector { index: i })
                } else {
                    Ok(v.data.iter().map(|x| x / norm).collect())
                }
            })
            .collect::<Result<Vec<Vec<f32>>, ClusteringError>>()?;
        &owned
    } else {
        // Keep the non-cosine path in the same owned representation as the
        // normalised cosine path, so downstream matrix construction can handle
        // both branches uniformly. This clones each input vector.
        owned = vectors.iter().map(|v| v.data.clone()).collect();
        &owned
    };

    // ── Build ndarray matrix (n_samples × dim) as f64 ────────────────────
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().map(|&x| x as f64)).collect();
    let matrix = Array2::<f64>::from_shape_vec((vectors.len(), dim), flat)
        .map_err(|e| ClusteringError::Linfa(e.to_string()))?;

    let dataset = DatasetBase::from(matrix);

    // ── Fit k-means ───────────────────────────────────────────────────────
    let rng = SmallRng::seed_from_u64(config.seed);
    let mut builder = KMeans::params_with(config.k, rng, L2Dist)
        .max_n_iterations(config.max_iterations);
    if let Some(tol) = config.tolerance {
        builder = builder.tolerance(tol);
    }

    let model = builder
        .fit(&dataset)
        .map_err(|e| ClusteringError::Linfa(e.to_string()))?;

    // ── Extract centroids ─────────────────────────────────────────────────
    // Rows of `model.centroids()` are in the same space as the fitted data
    // (normalised for cosine, raw for L2).  We keep them as-is — the
    // assignment helper uses the same space for distance comparisons.
    let centroids = model
        .centroids()
        .rows()
        .into_iter()
        .enumerate()
        .map(|(id, row)| Centroid {
            id: id as u32,
            data: row.iter().map(|&x| x as f32).collect(),
            shard_id: 0, // Assigned by CentroidTable in Stage 2 Step 2.
        })
        .collect();

    Ok(centroids)
}

/// Returns the `nprobe` nearest centroid shard IDs for `query`, sorted by
/// distance (nearest first).
///
/// This is used on the **write path** (soft assignment, `nprobe = 2`) and the
/// **read path** (fan-out to top-`nprobe` shards).
///
/// The `metric` **must** match the one used when the centroids were trained —
/// in particular, cosine centroids are stored normalised. For cosine queries,
/// this function validates that `query` has non-zero norm before computing
/// distance; it does not explicitly normalise `query` here because cosine
/// ranking is scale-invariant.
///
/// # Errors
///
/// Returns [`ClusteringError`] if `nprobe` exceeds the centroid count, the
/// centroid list is empty, the query dimension mismatches, or a zero-norm
/// query is presented with the cosine metric.
pub fn assign_to_shards(
    query: &[f32],
    centroids: &[Centroid],
    nprobe: usize,
    metric: DistanceMetric,
) -> Result<Vec<ShardId>, ClusteringError> {
    if centroids.is_empty() {
        return Err(ClusteringError::EmptyInput);
    }
    if nprobe == 0 || nprobe > centroids.len() {
        return Err(ClusteringError::NprobeTooLarge {
            nprobe,
            n_centroids: centroids.len(),
        });
    }

    let dim = centroids[0].data.len();
    if query.len() != dim {
        return Err(ClusteringError::QueryDimensionMismatch {
            expected: dim,
            actual: query.len(),
        });
    }

    // Keep explicit validation for cosine queries to preserve the API contract.
    if metric == DistanceMetric::Cosine {
        let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return Err(ClusteringError::ZeroNormVector { index: 0 });
        }
    }

    let candidates = centroids
        .iter()
        .map(|c| (c.shard_id, c.data.as_slice()));

    Ok(select_topk_by_vector(query, candidates, nprobe, metric)
        .into_iter()
        .map(|(sid, _)| sid)
        .collect())
}
