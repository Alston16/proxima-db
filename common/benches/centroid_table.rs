use common::CentroidTable;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn make_table(k: u32) -> CentroidTable {
    CentroidTable::new((0..k).map(|i| (i, format!("127.0.0.1:{}", 7000 + i))))
}

/// SC-003: lookup time must not grow linearly with K.
fn bench_lookup_constant_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup_constant_time");
    for k in [64_u32, 128, 256, 512] {
        let table = make_table(k);
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| table.get(black_box(k / 2)))
        });
    }
    group.finish();
}

/// SC-001: JSON round-trip at K=512 must complete in ≤ 50 ms.
fn bench_json_round_trip(c: &mut Criterion) {
    let table = make_table(512);
    c.bench_function("json_round_trip_k512", |b| {
        b.iter(|| {
            let json = table.to_json().unwrap();
            let _restored = CentroidTable::from_json(black_box(&json)).unwrap();
        })
    });
}

/// SC-002: bincode round-trip at K=512 must complete in ≤ 5 ms.
fn bench_bincode_round_trip(c: &mut Criterion) {
    let table = make_table(512);
    c.bench_function("bincode_round_trip_k512", |b| {
        b.iter(|| {
            let bytes = table.to_bincode().unwrap();
            let _restored = CentroidTable::from_bincode(black_box(&bytes)).unwrap();
        })
    });
}

criterion_group!(
    benches,
    bench_lookup_constant_time,
    bench_json_round_trip,
    bench_bincode_round_trip
);
criterion_main!(benches);
