/// Unique identifier for a vector within the database.
pub type VectorId = u64;

/// Unique identifier for a shard within the cluster.
pub type ShardId = u32;

/// A vector stored in the database, identified by a unique [`VectorId`].
#[derive(Debug, Clone)]
pub struct Vector {
    /// The unique identifier for this vector.
    pub id: VectorId,
    /// The embedding data as a slice of 32-bit floats.
    pub data: Vec<f32>,
}

/// A cluster centroid used for proximity-aware shard routing.
///
/// Each centroid defines the representative point of one Voronoi partition.
/// Vectors are assigned to the shard whose centroid is nearest in embedding space.
#[derive(Debug, Clone)]
pub struct Centroid {
    /// The unique identifier for this centroid.
    pub id: u32,
    /// The centroid coordinates in embedding space.
    pub data: Vec<f32>,
    /// The shard that owns this Voronoi partition.
    pub shard_id: ShardId,
}

/// The result of a nearest-neighbour search query.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The identifier of the matching vector.
    pub id: VectorId,
    /// The distance between the query vector and this result.
    pub distance: f32,
}