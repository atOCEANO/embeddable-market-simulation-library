//! Benchmarks for the compiled strategy paths: a single in-core backtest driven by
//! the engine through the `run` loop, and the parallel parameter sweep
//! (`run_sweep`), which forks one full backtest per parameter row across the Rayon
//! pool. The sweep is the coarse-grained parallel path, one long task per core
//! rather than one dispatch per bar, so it is where parameter-search throughput
//! actually comes from. Run with `cargo bench -p bar-engine`.

use std::hint::black_box;

use bar_engine::{builtin_strategy, run, run_sweep, Candles, Engine, EngineConfig, SmaCross};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use emsl_core::{Candle, Market};

const BARS: usize = 1_000;

/// An oscillating close so a moving-average cross actually trades, with real volume.
fn synthetic_series(bars: usize) -> Candles {
    let candles: Vec<Candle> = (0..bars)
        .map(|i| {
            let base = 100.0 + 20.0 * ((i as f64) / 25.0).sin();
            Candle {
                open: base,
                high: base + 2.0,
                low: base - 2.0,
                close: base,
                volume: 100_000.0,
            }
        })
        .collect();
    Candles::new(candles)
}

fn spot_config() -> EngineConfig {
    EngineConfig {
        market: Market::Spot,
        quote: 100_000.0,
        fee_taker: 0.0006,
        fee_maker: 0.0002,
        slippage_bps: 2.0,
        max_fill_fraction: 1.0,
        max_open_orders: 8,
        report: false,
        max_leverage: 0.0,
        impact: 0.0,
        funding_rate: 0.0,
        funding_interval: 0,
    }
}

/// A grid of `[fast, slow]` moving-average rows.
fn sma_params(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| vec![(5 + i % 20) as f64, (30 + i % 60) as f64])
        .collect()
}

fn bench_single_backtest(c: &mut Criterion) {
    let series = synthetic_series(BARS);
    let mut group = c.benchmark_group("compiled_backtest");
    group.throughput(Throughput::Elements(BARS as u64));
    group.bench_function("sma_cross", |b| {
        let cfg = spot_config();
        b.iter(|| {
            let mut engine = Engine::new(series.clone(), cfg);
            let mut strategy = SmaCross::from_params(&[10.0, 30.0]);
            black_box(run(&mut engine, &mut strategy));
        });
    });
    group.finish();
}

fn bench_sweep(c: &mut Criterion) {
    let series = synthetic_series(BARS);
    let factory = builtin_strategy("sma_cross").expect("sma_cross is built in");
    let mut group = c.benchmark_group("sweep");
    group.sample_size(10);
    for &n in &[256usize, 1_024, 4_096] {
        let params = sma_params(n);
        group.throughput(Throughput::Elements((n * BARS) as u64));
        group.bench_with_input(BenchmarkId::new("sma_cross", n), &n, |b, _| {
            b.iter(|| {
                black_box(run_sweep(
                    series.clone(),
                    spot_config(),
                    factory,
                    &params,
                    365.0,
                    0.0,
                ));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_single_backtest, bench_sweep);
criterion_main!(benches);
