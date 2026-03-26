// shard/src/main.rs
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
    // In Stage 5, shard_id and port will come from CLI args or config.
    let shard_id = "shard-0".to_string();
    let addr = "127.0.0.1:50051".parse()?;

    println!("[{}] listening on {}", shard_id, addr);

    Server::builder()
        .add_service(ClusterServiceServer::new(ShardNode { shard_id }))
        .serve(addr)
        .await?;

    Ok(())
}