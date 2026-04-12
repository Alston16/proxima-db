use std::{env, path::PathBuf, sync::Arc};

use shard::distance;
use shard::state::ShardState;
use tonic::{transport::Server, Request, Response, Status};

// Pull in the generated types for the `cluster` proto package.
pub mod cluster {
    tonic::include_proto!("cluster");
}

use cluster::{
    cluster_service_server::{ClusterService, ClusterServiceServer},
    PingRequest, PingResponse,
};

#[derive(Debug)]
pub struct ShardNode {
    shard_id: String,
    _state: Arc<ShardState>,
}

#[tonic::async_trait]
impl ClusterService for ShardNode {
    async fn ping(
        &self,
        request: Request<PingRequest>,
    ) -> Result<Response<PingResponse>, Status> {
        println!(
            "[shard {}] received ping from: {}",
            self.shard_id,
            request.into_inner().sender
        );

        Ok(Response::new(PingResponse {
            shard_id: self.shard_id.clone(),
            ok: true,
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shard_id = env::var("SHARD_ID").unwrap_or_else(|_| "shard-0".to_string());
    let addr = env::var("SHARD_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:50051".to_string())
        .parse()?;
    let store_path = env::var("SHARD_STORE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(format!("{shard_id}.vectors.bin")));
    let dimension = env::var("SHARD_DIMENSION")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(128);
    let initial_capacity = env::var("SHARD_INITIAL_CAPACITY")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1_024);
    let auto_simd_min_dim = env::var("SHARD_AUTO_SIMD_MIN_DIM")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(distance::DEFAULT_AUTO_SIMD_MIN_DIM);

    distance::set_auto_simd_min_dim(auto_simd_min_dim);

    let state = Arc::new(ShardState::open_or_create(
        &store_path,
        dimension,
        initial_capacity,
    )?);

    println!(
        "[{}] listening on {} with store {} (dim={}, capacity={}, auto_simd_min_dim={})",
        shard_id,
        addr,
        store_path.display(),
        dimension,
        initial_capacity,
        distance::auto_simd_min_dim(),
    );

    Server::builder()
        .add_service(ClusterServiceServer::new(ShardNode {
            shard_id,
            _state: state,
        }))
        .serve(addr)
        .await?;

    Ok(())
}