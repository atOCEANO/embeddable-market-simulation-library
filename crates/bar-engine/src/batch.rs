//! The batched runner: many independent engines stepped in parallel. They share
//! one `Arc`-backed candle series (no duplication) but each carries its own
//! account, book, and cursor, so the step is lock-free and, with no cross-env
//! reduction, batched stepping is bit-identical to stepping each engine in a
//! loop. That equality is the guarantee the fast path never diverges from the
//! correct sequential one.

use rayon::prelude::*;

use emsl_core::State;

use crate::candles::Candles;
use crate::engine::{Engine, EngineConfig};

/// A batch of independent engines over one shared candle series.
pub struct EnvBatch {
    engines: Vec<Engine>,
}

impl EnvBatch {
    /// Build `num_envs` engines over the same candles (shared by `Arc`) and
    /// config.
    pub fn new(num_envs: usize, candles: Candles, config: EngineConfig) -> EnvBatch {
        let engines = (0..num_envs)
            .map(|_| Engine::new(candles.clone(), config))
            .collect();
        EnvBatch { engines }
    }

    /// Build one engine per entry in `configs`, all over the one shared candle
    /// series. Lets each env carry its own config (per-env cost randomization,
    /// ADR 0014) while the candles stay shared with no duplication.
    pub fn with_configs(candles: Candles, configs: &[EngineConfig]) -> EnvBatch {
        let engines = configs
            .iter()
            .map(|config| Engine::new(candles.clone(), *config))
            .collect();
        EnvBatch { engines }
    }

    /// Number of envs.
    pub fn len(&self) -> usize {
        self.engines.len()
    }

    /// True when there are no envs.
    pub fn is_empty(&self) -> bool {
        self.engines.is_empty()
    }

    /// Number of bars in the shared series, zero when there are no envs.
    pub fn num_bars(&self) -> usize {
        self.engines.first().map(Engine::num_bars).unwrap_or(0)
    }

    /// True when no env has a next bar to step into. In the lockstep `Batch` use
    /// every env shares the cursor, so this is uniform; after per-env random-start
    /// resets it is true only once every env has reached the end. An empty batch is
    /// done.
    pub fn done(&self) -> bool {
        self.engines.iter().all(|engine| engine.done())
    }

    /// The engines, for placing per-env orders before a step.
    pub fn engines_mut(&mut self) -> &mut [Engine] {
        &mut self.engines
    }

    /// Reset every env, returning the initial state of each.
    pub fn reset_all(&mut self) -> Vec<State> {
        self.engines.par_iter_mut().map(Engine::reset).collect()
    }

    /// Step every env one bar in parallel, returning each new state. Because the
    /// envs are independent, this is bit-identical to stepping each in a loop.
    pub fn step_all(&mut self) -> Vec<State> {
        self.engines.par_iter_mut().map(Engine::step).collect()
    }

    /// Reset every env to its own start offset (`offsets[i]` for env `i`),
    /// returning each initial state. Gives each env a different slice of history.
    pub fn reset_at_each(&mut self, offsets: &[usize]) -> Vec<State> {
        self.engines
            .par_iter_mut()
            .zip(offsets.par_iter())
            .map(|(engine, &offset)| engine.reset_at(offset))
            .collect()
    }

    /// Reset only the envs where `mask[i]` is true, each to `offsets[i]`; the rest
    /// keep running. Returns every env's state afterward. This is the autoreset
    /// primitive: the finished envs restart at a fresh offset.
    pub fn reset_masked(&mut self, mask: &[bool], offsets: &[usize]) -> Vec<State> {
        self.engines
            .par_iter_mut()
            .enumerate()
            .map(|(i, engine)| {
                if mask[i] {
                    engine.reset_at(offsets[i])
                } else {
                    engine.current_state()
                }
            })
            .collect()
    }

    /// Batched observation: each env's `window` most recent candles as
    /// `num_envs * window * 5` row-major floats laid out (env, bar, ohlcv),
    /// front-padded with zeros when an env has fewer than `window` bars yet.
    pub fn observe_all(&self, window: usize) -> Vec<f64> {
        let cols = 5;
        let mut out = vec![0.0; self.engines.len() * window * cols];
        out.par_chunks_mut(window * cols)
            .zip(self.engines.par_iter())
            .for_each(|(chunk, engine)| {
                let bars = engine.candle_window(window);
                let pad = window - bars.len(); // fewer than `window` bars -> pad the front
                for (row, bar) in bars.iter().enumerate() {
                    let base = (pad + row) * cols;
                    chunk[base] = bar.open;
                    chunk[base + 1] = bar.high;
                    chunk[base + 2] = bar.low;
                    chunk[base + 3] = bar.close;
                    chunk[base + 4] = bar.volume;
                }
            });
        out
    }

    /// Per-env truncation flag: true where the env has reached the last bar.
    pub fn dones(&self) -> Vec<bool> {
        self.engines.iter().map(Engine::done).collect()
    }

    /// Per-env termination flag: true where the env's account died, by a perp
    /// liquidation or by equity reaching zero on either market (ADR 0019).
    pub fn busts(&self) -> Vec<bool> {
        self.engines.iter().map(Engine::is_bust).collect()
    }

    /// Per-env equity, marked at each env's current bar. The RL reward is built
    /// from this, so it uses the lean accessor rather than materializing a full
    /// state (with its open-orders vector) per env.
    pub fn equities(&self) -> Vec<f64> {
        self.engines.iter().map(Engine::equity).collect()
    }

    /// Batched feature observation: each env's `window` most recent rows of the
    /// shared `(T, n_features)` matrix passed as the row-major flat `features`, as
    /// `num_envs * window * n_features` floats laid out (env, row, feature),
    /// front-padded with zeros when an env has fewer than `window` rows yet.
    pub fn observe_features_all(
        &self,
        window: usize,
        features: &[f64],
        n_features: usize,
    ) -> Vec<f64> {
        let f = n_features;
        let mut out = vec![0.0; self.engines.len() * window * f];
        out.par_chunks_mut(window * f)
            .zip(self.engines.par_iter())
            .for_each(|(chunk, engine)| {
                let end = engine.tick() + 1; // inclusive of the current bar
                let start = end.saturating_sub(window);
                let pad = window - (end - start);
                for row in 0..(end - start) {
                    let src = (start + row) * f;
                    let dst = (pad + row) * f;
                    chunk[dst..dst + f].copy_from_slice(&features[src..src + f]);
                }
            });
        out
    }

    /// A batched snapshot of the per-env account fields a reward reads.
    pub fn snapshot(&self) -> BatchSnapshot {
        let states: Vec<State> = self.engines.iter().map(Engine::current_state).collect();
        BatchSnapshot {
            equity: states.iter().map(|s| s.equity).collect(),
            position: states.iter().map(|s| s.position).collect(),
            unrealized_pnl: states.iter().map(|s| s.unrealized_pnl).collect(),
            realized_pnl: states.iter().map(|s| s.realized_pnl).collect(),
            mark_price: states.iter().map(|s| s.mark_price).collect(),
        }
    }
}

/// Batched per-env account fields, one `Vec` per field, for building an RL reward.
pub struct BatchSnapshot {
    pub equity: Vec<f64>,
    pub position: Vec<f64>,
    pub unrealized_pnl: Vec<f64>,
    pub realized_pnl: Vec<f64>,
    pub mark_price: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::EnvBatch;
    use crate::candles::Candles;
    use crate::engine::{Engine, EngineConfig};
    use emsl_core::{Candle, Market, State};
    use proptest::prelude::*;

    fn series() -> Candles {
        Candles::new(vec![
            Candle {
                open: 100.0,
                high: 160.0,
                low: 90.0,
                close: 150.0,
                volume: 1000.0,
            },
            Candle {
                open: 200.0,
                high: 260.0,
                low: 190.0,
                close: 250.0,
                volume: 1000.0,
            },
            Candle {
                open: 300.0,
                high: 360.0,
                low: 290.0,
                close: 350.0,
                volume: 1000.0,
            },
        ])
    }

    fn cfg() -> EngineConfig {
        EngineConfig {
            market: Market::Spot,
            quote: 10_000.0,
            fee_taker: 0.0,
            fee_maker: 0.0,
            slippage_bps: 0.0,
            max_fill_fraction: 1.0,
            max_open_orders: 8,
            report: false,
            max_leverage: 0.0,
            impact: 0.0,
            funding_rate: 0.0,
            funding_interval: 0,
        }
    }

    #[test]
    fn len_and_reset() {
        let mut b = EnvBatch::new(4, series(), cfg());
        assert_eq!(b.len(), 4);
        assert!(!b.is_empty());
        let states = b.reset_all();
        assert_eq!(states.len(), 4);
        assert!(states
            .iter()
            .all(|s| s.tick_index == 0 && s.position == 0.0));
    }

    #[test]
    fn done_tracks_the_shared_cursor() {
        let mut b = EnvBatch::new(3, series(), cfg()); // 3 bars
        b.reset_all();
        assert!(!b.done());
        b.step_all();
        assert!(!b.done());
        b.step_all();
        assert!(b.done());
    }

    #[test]
    fn empty_batch_is_done() {
        let b = EnvBatch::new(0, series(), cfg());
        assert!(b.is_empty());
        assert!(b.done());
    }

    #[test]
    fn num_bars_reports_the_series_length() {
        assert_eq!(EnvBatch::new(2, series(), cfg()).num_bars(), 3); // series() has 3 bars
        assert_eq!(EnvBatch::new(0, series(), cfg()).num_bars(), 0);
    }

    #[test]
    fn reset_at_each_starts_each_env_at_its_offset() {
        let mut b = EnvBatch::new(3, series(), cfg());
        let states = b.reset_at_each(&[0, 1, 2]);
        assert_eq!(states[0].tick_index, 0);
        assert_eq!(states[1].tick_index, 1);
        assert_eq!(states[2].tick_index, 2);
    }

    #[test]
    fn observe_all_gathers_per_env_windows() {
        let mut b = EnvBatch::new(2, series(), cfg());
        b.reset_at_each(&[1, 2]); // env0 at bar 1, env1 at bar 2
        let obs = b.observe_all(2); // 2 envs * 2 bars * 5 cols
        assert_eq!(obs.len(), 2 * 2 * 5);
        assert_eq!(obs[0], 100.0); // env0 row0 open (bar 0)
        assert_eq!(obs[5], 200.0); // env0 row1 open (bar 1)
        assert_eq!(obs[10], 200.0); // env1 row0 open (bar 1)
        assert_eq!(obs[15], 300.0); // env1 row1 open (bar 2)
    }

    #[test]
    fn observe_all_front_pads_short_windows() {
        let mut b = EnvBatch::new(1, series(), cfg());
        b.reset_at_each(&[0]); // only one bar of history
        let obs = b.observe_all(3); // window 3 -> two padded rows
        assert_eq!(obs.len(), 15);
        assert_eq!(&obs[0..10], &[0.0; 10]); // first two rows padded
        assert_eq!(obs[10], 100.0); // last row = bar 0 open
    }

    #[test]
    fn reset_masked_resets_only_flagged_envs() {
        let mut b = EnvBatch::new(2, series(), cfg());
        b.reset_at_each(&[0, 0]);
        b.step_all(); // both at tick 1
        let states = b.reset_masked(&[true, false], &[0, 0]);
        assert_eq!(states[0].tick_index, 0); // reset
        assert_eq!(states[1].tick_index, 1); // kept running
    }

    #[test]
    fn dones_and_busts_report_per_env() {
        let mut b = EnvBatch::new(2, series(), cfg());
        b.reset_at_each(&[0, 2]); // env1 starts at the last bar
        assert_eq!(b.dones(), vec![false, true]);
        assert_eq!(b.busts(), vec![false, false]);
    }

    #[test]
    fn equities_start_at_the_configured_quote() {
        let mut b = EnvBatch::new(3, series(), cfg());
        b.reset_at_each(&[0, 1, 2]);
        assert_eq!(b.equities(), vec![10_000.0, 10_000.0, 10_000.0]);
    }

    #[test]
    fn observe_features_all_gathers_per_env_windows() {
        let mut b = EnvBatch::new(2, series(), cfg());
        b.reset_at_each(&[1, 2]); // env0 at row 1, env1 at row 2
        let features = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 rows x 2 features
        let obs = b.observe_features_all(2, &features, 2);
        // env0 rows 0,1 -> [1,2,3,4]; env1 rows 1,2 -> [3,4,5,6]
        assert_eq!(obs, vec![1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn observe_features_all_front_pads_short_windows() {
        let mut b = EnvBatch::new(1, series(), cfg());
        b.reset_at_each(&[0]); // one row of history
        let features = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let obs = b.observe_features_all(3, &features, 2); // window 3 -> two padded rows
        assert_eq!(obs, vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn snapshot_reports_account_fields() {
        let mut b = EnvBatch::new(2, series(), cfg());
        b.reset_at_each(&[0, 0]);
        let snap = b.snapshot();
        assert_eq!(snap.equity, vec![10_000.0, 10_000.0]);
        assert_eq!(snap.position, vec![0.0, 0.0]);
        assert_eq!(snap.mark_price, vec![150.0, 150.0]); // bar 0 close
    }

    #[test]
    fn batched_equals_looped_single() {
        let candles = series();
        let n = 16;

        // Parallel batch: each env buys a different size.
        let mut batch = EnvBatch::new(n, candles.clone(), cfg());
        batch.reset_all();
        for (i, e) in batch.engines_mut().iter_mut().enumerate() {
            e.market_buy(0.1 * (i as f64 + 1.0));
        }
        let parallel = batch.step_all();

        // Sequential reference doing the identical thing, one engine at a time.
        let mut refs: Vec<Engine> = (0..n)
            .map(|_| Engine::new(candles.clone(), cfg()))
            .collect();
        for e in refs.iter_mut() {
            e.reset();
        }
        for (i, e) in refs.iter_mut().enumerate() {
            e.market_buy(0.1 * (i as f64 + 1.0));
        }
        let sequential: Vec<State> = refs.iter_mut().map(Engine::step).collect();

        // Bit-for-bit: parallelism introduces no numerical drift.
        assert_eq!(parallel, sequential);
    }

    #[test]
    fn with_configs_gives_each_env_its_own_costs() {
        let mut cheap = cfg();
        cheap.fee_taker = 0.0;
        let mut dear = cfg();
        dear.fee_taker = 0.01;

        let mut batch = EnvBatch::with_configs(series(), &[cheap, dear]);
        assert_eq!(batch.len(), 2);
        batch.reset_all();
        for e in batch.engines_mut() {
            e.market_buy(1.0);
        }
        let states = batch.step_all();

        // both buy 1 at bar 1 open 200; env0 pays no taker fee, env1 pays 1%
        assert_eq!(states[0].quote, 9_800.0); // 10000 - 200
        assert_eq!(states[1].quote, 9_798.0); // 10000 - 200 - 0.01 * 200
    }

    fn long_series() -> Candles {
        let bars: Vec<Candle> = (0..8)
            .map(|i| {
                let base = 100.0 + 10.0 * ((i as f64) * 0.7).sin();
                Candle {
                    open: base,
                    high: base + 3.0,
                    low: base - 3.0,
                    close: base + 0.5,
                    volume: 5000.0,
                }
            })
            .collect();
        Candles::new(bars)
    }

    // A rich perp config so the invariant covers fees, slippage, impact, the
    // leverage cap, funding, and liquidation, not just a bare fill.
    fn prop_cfg() -> EngineConfig {
        EngineConfig {
            market: Market::Perp,
            quote: 10_000.0,
            fee_taker: 0.0006,
            fee_maker: 0.0002,
            slippage_bps: 2.0,
            max_fill_fraction: 1.0,
            max_open_orders: 8,
            report: false,
            max_leverage: 10.0,
            impact: 0.1,
            funding_rate: 0.0001,
            funding_interval: 2,
        }
    }

    proptest! {
        #[test]
        fn batched_equals_looped_multistep(
            (n, steps, actions) in (1usize..8, 1usize..5).prop_flat_map(|(n, steps)| {
                (Just(n), Just(steps), prop::collection::vec(-2.0f64..2.0, n * steps))
            }),
        ) {
            // the parallel batch must match a sequential loop of independent engines
            // bit-for-bit across many steps and mixed buy/sell actions
            let candles = long_series();
            let cfg = prop_cfg();

            let mut batch = EnvBatch::new(n, candles.clone(), cfg);
            batch.reset_all();

            let mut refs: Vec<Engine> =
                (0..n).map(|_| Engine::new(candles.clone(), cfg)).collect();
            for e in refs.iter_mut() {
                e.reset();
            }

            for step in 0..steps {
                for (env, e) in batch.engines_mut().iter_mut().enumerate() {
                    let a = actions[step * n + env];
                    if a > 0.0 {
                        e.market_buy(a);
                    } else if a < 0.0 {
                        e.market_sell(-a);
                    }
                }
                let parallel = batch.step_all();

                let sequential: Vec<State> = refs
                    .iter_mut()
                    .enumerate()
                    .map(|(env, e)| {
                        let a = actions[step * n + env];
                        if a > 0.0 {
                            e.market_buy(a);
                        } else if a < 0.0 {
                            e.market_sell(-a);
                        }
                        e.step()
                    })
                    .collect();

                prop_assert_eq!(parallel, sequential);
            }
        }
    }
}
