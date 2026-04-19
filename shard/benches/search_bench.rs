/// Benchmark: single-node brute-force k-NN search baseline.
///
/// Measures QPS and latency (sampled by Criterion) for `search_topk` over
/// varying dataset sizes and embedding dimensions. Run with:
///
///   cargo bench -p shard --bench search_bench
///
/// HTML reports land in `target/criterion/`.
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use common::distance::DistanceMetric;
use shard::storage::FlatVectorStore;
use common::Vector;

fn build_store(dim: usize, n: usize, rng: &mut StdRng) -> (FlatVectorStore, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.bin");
    let mut store = FlatVectorStore::open_or_create(&path, dim, n).unwrap();
    for id in 0..n as u64 {
        let data: Vec<f32> = (0..dim).map(|_| rng.gen_range(0.0_f32..1.0)).collect();
        store.insert(&Vector { id, data }).unwrap();
    }
    (store, dir)
}

fn bench_search(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let k = 10;

    let mut group = c.benchmark_group("search_topk_l2");
    for &(label, dim, n) in &[
        ("1k_d128", 128_usize, 1_000_usize),
        ("10k_d128", 128, 10_000),
        ("100k_d128", 128, 100_000),
        ("10k_d512", 512, 10_000),
    ] {
        let (store, _dir) = build_store(dim, n, &mut rng);
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(0.0_f32..1.0)).collect();

        group.bench_with_input(BenchmarkId::new(label, n), &n, |b, _| {
            b.iter(|| {
                store
                    .search_topk(&query, k, DistanceMetric::L2)
                    .expect("search failed")
            })
        });
    }
    group.finish();

    let mut group = c.benchmark_group("search_topk_cosine");
    for &(label, dim, n) in &[("1k_d128", 128_usize, 1_000_usize), ("10k_d128", 128, 10_000)] {
        let (store, _dir) = build_store(dim, n, &mut rng);
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(0.0_f32..1.0)).collect();

        group.bench_with_input(BenchmarkId::new(label, n), &n, |b, _| {
            b.iter(|| {
                store
                    .search_topk(&query, k, DistanceMetric::Cosine)
                    .expect("search failed")
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
