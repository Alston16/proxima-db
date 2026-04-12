use wide::f32x8;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Selects the distance metric used for nearest-neighbour search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance: `sqrt(Σ(aᵢ − bᵢ)²)`.
    L2,
    /// Cosine distance: `1.0 − cosine_similarity`.
    ///
    /// Values range from `0.0` (identical direction) to `2.0` (opposite
    /// direction). Zero-norm vectors are treated as maximally distant (`1.0`).
    Cosine,
}

/// Selects which backend a distance function uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceBackend {
    /// Lets the function choose the implementation.
    Auto,
    /// Uses the scalar reference implementation.
    Scalar,
    /// Uses the SIMD implementation.
    Simd,
}

const LANES: usize = 8;
/// Default minimum dimension used by [`DistanceBackend::Auto`] to pick SIMD.
pub const DEFAULT_AUTO_SIMD_MIN_DIM: usize = 256;

static AUTO_SIMD_MIN_DIM: AtomicUsize = AtomicUsize::new(DEFAULT_AUTO_SIMD_MIN_DIM);

/// Sets the minimum dimension used by [`DistanceBackend::Auto`] to select SIMD.
///
/// Values below `LANES` are clamped to `LANES` because SIMD kernels process
/// vectors in lane-sized chunks and already fall back to scalar for short tails.
pub fn set_auto_simd_min_dim(value: usize) {
    let clamped = value.max(LANES);
    AUTO_SIMD_MIN_DIM.store(clamped, Ordering::Relaxed);
}

/// Returns the current minimum dimension used by [`DistanceBackend::Auto`].
pub fn auto_simd_min_dim() -> usize {
    AUTO_SIMD_MIN_DIM.load(Ordering::Relaxed)
}

#[inline]
fn should_use_simd_auto(dim: usize) -> bool {
    dim >= auto_simd_min_dim()
}

#[inline]
fn load8(values: &[f32], i: usize) -> f32x8 {
    f32x8::from([
        values[i],
        values[i + 1],
        values[i + 2],
        values[i + 3],
        values[i + 4],
        values[i + 5],
        values[i + 6],
        values[i + 7],
    ])
}

#[inline]
fn reduce8(v: f32x8) -> f32 {
    let lanes: [f32; LANES] = v.into();
    lanes.into_iter().sum()
}

/// Computes Euclidean (L2) distance using the scalar reference path.
///
/// # Panics
///
/// Panics in debug builds if `a.len() != b.len()`.
pub fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "l2_distance_scalar: slice length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

/// Computes Euclidean (L2) distance using the SIMD path.
///
/// # Panics
///
/// Panics in debug builds if `a.len() != b.len()`.
pub fn l2_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "l2_distance_simd: slice length mismatch");

    if a.len() < LANES {
        return l2_distance_scalar(a, b);
    }

    let mut i = 0;
    let mut sum = f32x8::ZERO;
    while i + LANES <= a.len() {
        let va = load8(a, i);
        let vb = load8(b, i);
        let d = va - vb;
        sum += d * d;
        i += LANES;
    }

    let mut total = reduce8(sum);
    while i < a.len() {
        let d = a[i] - b[i];
        total += d * d;
        i += 1;
    }
    total.sqrt()
}

/// Computes cosine distance (`1.0 − cosine_similarity`) using the scalar reference path.
///
/// Returns `1.0` if either vector has zero norm, avoiding division by zero.
/// The similarity is clamped to `[-1.0, 1.0]` before subtraction to guard
/// against floating-point values that drift slightly outside that range.
///
/// # Panics
///
/// Panics in debug builds if `a.len() != b.len()`.
pub fn cosine_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine_distance_scalar: slice length mismatch");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    let similarity = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
    1.0 - similarity
}

/// Computes cosine distance (`1.0 − cosine_similarity`) using the SIMD path.
///
/// Returns `1.0` if either vector has zero norm, avoiding division by zero.
/// The similarity is clamped to `[-1.0, 1.0]` before subtraction to guard
/// against floating-point values that drift slightly outside that range.
///
/// # Panics
///
/// Panics in debug builds if `a.len() != b.len()`.
pub fn cosine_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine_distance_simd: slice length mismatch");

    if a.len() < LANES {
        return cosine_distance_scalar(a, b);
    }

    let mut i = 0;
    let mut dot = f32x8::ZERO;
    let mut aa = f32x8::ZERO;
    let mut bb = f32x8::ZERO;

    while i + LANES <= a.len() {
        let va = load8(a, i);
        let vb = load8(b, i);
        dot += va * vb;
        aa += va * va;
        bb += vb * vb;
        i += LANES;
    }

    let mut dot_sum = reduce8(dot);
    let mut aa_sum = reduce8(aa);
    let mut bb_sum = reduce8(bb);
    while i < a.len() {
        dot_sum += a[i] * b[i];
        aa_sum += a[i] * a[i];
        bb_sum += b[i] * b[i];
        i += 1;
    }

    let norm_a = aa_sum.sqrt();
    let norm_b = bb_sum.sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    let similarity = (dot_sum / (norm_a * norm_b)).clamp(-1.0, 1.0);
    1.0 - similarity
}

/// Computes Euclidean (L2) distance with an explicit backend preference.
///
/// # Panics
///
/// Panics in debug builds if `a.len() != b.len()`.
pub fn l2_distance_with_backend(a: &[f32], b: &[f32], backend: DistanceBackend) -> f32 {
    match backend {
        DistanceBackend::Auto => {
            if should_use_simd_auto(a.len()) {
                l2_distance_simd(a, b)
            } else {
                l2_distance_scalar(a, b)
            }
        }
        DistanceBackend::Scalar => l2_distance_scalar(a, b),
        DistanceBackend::Simd => l2_distance_simd(a, b),
    }
}

/// Computes cosine distance with an explicit backend preference.
///
/// # Panics
///
/// Panics in debug builds if `a.len() != b.len()`.
pub fn cosine_distance_with_backend(a: &[f32], b: &[f32], backend: DistanceBackend) -> f32 {
    match backend {
        DistanceBackend::Auto => {
            if should_use_simd_auto(a.len()) {
                cosine_distance_simd(a, b)
            } else {
                cosine_distance_scalar(a, b)
            }
        }
        DistanceBackend::Scalar => cosine_distance_scalar(a, b),
        DistanceBackend::Simd => cosine_distance_simd(a, b),
    }
}

/// Computes the Euclidean (L2) distance between two equal-length slices.
///
/// # Panics
///
/// Panics in debug builds if `a.len() != b.len()`.
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_distance_with_backend(a, b, DistanceBackend::Auto)
}

/// Computes cosine distance (`1.0 − cosine_similarity`) between two equal-length slices.
///
/// Returns `1.0` if either vector has zero norm, avoiding division by zero.
/// The similarity is clamped to `[-1.0, 1.0]` before subtraction to guard
/// against floating-point values that drift slightly outside that range.
///
/// # Panics
///
/// Panics in debug builds if `a.len() != b.len()`.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    cosine_distance_with_backend(a, b, DistanceBackend::Auto)
}
