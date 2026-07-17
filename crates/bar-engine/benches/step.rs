//! Throughput benchmarks for the bar engine. Two paths are measured over a
//! synthetic series: a single `Engine::step` loop, the pure-Rust hot path with no
//! Python boundary, and `EnvBatch::step_all` across many envs, the batched path
//! the RL throughput rests on. Run with `cargo bench -p bar-engine`. The figures
//! it prints are the only evidence any speed claim in the docs may cite.

use std::hint::black_box;

use bar_engine::{Candles, Engine, EngineConfig, EnvBatch};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use emsl_core::{Candle, Market};

const BARS: usize = 1_000;

/// A bounded oscillating OHLCV series with real volume, enough to drive the fill
/// model without the account running away over a long pass.
fn synthetic_series(bars: usize) -> Candles {
    let candles: Vec<Candle> = (0..bars)
        .map(|i| {
            let base = 100.0 + ((i % 40) as f64);
            Candle {
                open: base,
                high: base + 4.0,
                low: base - 4.0,
                close: base + 1.0,
                volume: 100_000.0,
            }
        })
        .collect();
    Candles::new(candles)
}

/// A representative perp config with honest costs on.
fn perp_config() -> EngineConfig {
    EngineConfig {
        market: Market::Perp,
        quote: 1_000_000.0,
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

fn bench_single_env(c: &mut Criterion) {
    let series = synthetic_series(BARS);
    let mut group = c.benchmark_group("single_env");
    group.throughput(Throughput::Elements(BARS as u64));

    group.bench_function("step_flat", |b| {
        let mut engine = Engine::new(series.clone(), perp_config());
        b.iter(|| {
            engine.reset();
            while !engine.done() {
                black_box(engine.step());
            }
        });
    });

    group.bench_function("step_trading", |b| {
        let mut engine = Engine::new(series.clone(), perp_config());
        b.iter(|| {
            engine.reset();
            while !engine.done() {
                engine.market_buy(black_box(0.001));
                black_box(engine.step());
            }
        });
    });

    group.finish();
}

fn bench_batch(c: &mut Criterion) {
    let series = synthetic_series(BARS);
    let mut group = c.benchmark_group("batch");
    group.sample_size(10);

    for &n in &[256usize, 1_024, 4_096] {
        group.throughput(Throughput::Elements((n * BARS) as u64));
        group.bench_with_input(BenchmarkId::new("step_all", n), &n, |b, &n| {
            let mut batch = EnvBatch::new(n, series.clone(), perp_config());
            b.iter(|| {
                batch.reset_all();
                while !batch.done() {
                    for engine in batch.engines_mut() {
                        engine.market_buy(black_box(0.001));
                    }
                    black_box(batch.step_all());
                }
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_single_env, bench_batch);
criterion_main!(benches);
