use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use corrosive_tensor::prelude::*;

fn bench_matmul_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_square");

    for size in [16, 32, 64, 128, 256, 512].iter() {
        let a = Tensor::<f32>::ones(vec![*size, *size], Device::CPU).unwrap();
        let b = Tensor::<f32>::ones(vec![*size, *size], Device::CPU).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bencher, _| {
            bencher.iter(|| a.matmul(&b).unwrap());
        });
    }

    group.finish();
}

fn bench_matmul_rectangular(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_rectangular");

    let configs = vec![
        ("64x128x64", 64, 128, 64),
        ("128x64x128", 128, 64, 128),
        ("256x512x256", 256, 512, 256),
    ];

    for (name, m, k, n) in configs {
        let a = Tensor::<f32>::ones(vec![m, k], Device::CPU).unwrap();
        let b = Tensor::<f32>::ones(vec![k, n], Device::CPU).unwrap();

        group.bench_with_input(BenchmarkId::new("size", name), &(m, k, n), |bencher, _| {
            bencher.iter(|| a.matmul(&b).unwrap());
        });
    }

    group.finish();
}

criterion_group!(benches, bench_matmul_square, bench_matmul_rectangular);
criterion_main!(benches);