pub type VectorId = u64;
pub type ShardId = u32;

#[derive(Debug, Clone)]
pub struct Vector {
    pub id: VectorId,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Centroid {
    pub id: u32,
    pub data: Vec<f32>,
    pub shard_id: ShardId,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: VectorId,
    pub distance: f32,
}