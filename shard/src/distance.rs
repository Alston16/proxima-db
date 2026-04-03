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

/// Computes the Euclidean (L2) distance between two equal-length slices.
///
/// # Panics
///
/// Panics in debug builds if `a.len() != b.len()`.
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "l2_distance: slice length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
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
    debug_assert_eq!(a.len(), b.len(), "cosine_distance: slice length mismatch");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    let similarity = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
    1.0 - similarity
}
