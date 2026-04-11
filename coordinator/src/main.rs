
pub mod cluster {
    tonic::include_proto!("cluster");
}

use cluster::{cluster_service_client::ClusterServiceClient, PingRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // In Stage 5 this list comes from a config file / service discovery.
    let shard_address = vec!["http://127.0.0.1:50051"];

    for addr in shard_address {
        let mut client = ClusterServiceClient::connect(addr).await?;

        let response = client
            .ping(PingRequest {
                sender: "coordinator".to_string(),
            })
            .await?;

        let pong = response.into_inner();
        println!(
            "[coordinator] pong from shard_id={} ok={}",
            pong.shard_id, pong.ok
        );
    }

    Ok(())
}