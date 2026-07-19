//! The compiled parameter sweep: run a built-in strategy over a grid of parameter
//! rows in parallel, returning the final stats of each. Because the strategies are
//! compiled Rust, the whole grid runs in the engine with no Python in the loop,
//! which is what makes brute-force search fast.
//!
//! This is deliberately not exposed to Python: it can only run the strategies
//! compiled in here, so as a public surface it would be one nobody could extend
//! without writing Rust. Parameter search from Python goes through `emsl.tune`;
//! this stays as the in-core path and the benchmark of the GIL-free sweep. See
//! ADR 0011.

use rayon::prelude::*;

use emsl_core::State;

use crate::candles::Candles;
use crate::engine::{Engine, EngineConfig};
use crate::stats::Stats;
use crate::strategy::{run, Strategy};

/// A moving-average crossover: go long when the fast simple moving average of the
/// close rises above the slow one, flatten when it falls back below. Parameter row
/// is `[fast, slow]`, both bar counts.
pub struct SmaCross {
    pub fast: usize,
    pub slow: usize,
    pub size: f64,
}

impl SmaCross {
    /// Build from a parameter row `[fast, slow]`. Each is clamped to at least one
    /// bar; the traded size is fixed at one unit.
    pub fn from_params(row: &[f64]) -> SmaCross {
        let fast = (row.first().copied().unwrap_or(1.0).max(1.0)) as usize;
        let slow = (row.get(1).copied().unwrap_or(1.0).max(1.0)) as usize;
        SmaCross {
            fast,
            slow,
            size: 1.0,
        }
    }
}

impl Strategy for SmaCross {
    fn next(&mut self, state: &State, engine: &mut Engine) {
        let (fast, slow) = {
            let bars = engine.candle_window(self.slow);
            if bars.len() < self.slow {
                return; // warmup: not enough bars for the slow average yet
            }
            let sma = |n: usize| {
                let n = n.min(bars.len());
                bars[bars.len() - n..].iter().map(|c| c.close).sum::<f64>() / n as f64
            };
            (sma(self.fast), sma(self.slow))
        };

        if state.position == 0.0 && fast > slow {
            engine.market_buy(self.size);
        } else if state.position != 0.0 && fast < slow {
            engine.close();
        }
    }
}

/// A factory that builds a fresh strategy from a parameter row.
pub type StrategyFactory = fn(&[f64]) -> Box<dyn Strategy>;

/// The factory for a built-in strategy by name, or `None` if unknown.
pub fn builtin_strategy(name: &str) -> Option<StrategyFactory> {
    match name {
        "sma_cross" => Some(|row| Box::new(SmaCross::from_params(row))),
        _ => None,
    }
}

/// Run `make(row)` over each parameter row to the end of the data, in parallel,
/// returning each run's stats. Reporting is forced on so stats are available; the
/// per-row engines are independent, so the sweep parallelizes cleanly.
pub fn run_sweep<F>(
    candles: Candles,
    config: EngineConfig,
    make: F,
    params: &[Vec<f64>],
    periods_per_year: f64,
    risk_free: f64,
) -> Vec<Stats>
where
    F: Fn(&[f64]) -> Box<dyn Strategy> + Sync,
{
    let mut config = config;
    config.report = true;
    params
        .par_iter()
        .map(|row| {
            let mut engine = Engine::new(candles.clone(), config);
            let mut strategy = make(row);
            run(&mut engine, strategy.as_mut());
            engine
                .stats(periods_per_year, risk_free)
                .expect("reporting is on, so stats are present")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{builtin_strategy, run_sweep, SmaCross};
    use crate::candles::Candles;
    use crate::engine::{Engine, EngineConfig};
    use crate::strategy::run;
    use emsl_core::{Candle, Market};

    fn rising(n: usize) -> Candles {
        let bars: Vec<Candle> = (0..n)
            .map(|i| {
                let c = 100.0 + i as f64;
                Candle {
                    open: c,
                    high: c + 0.5,
                    low: c - 0.5,
                    close: c,
                    volume: 1000.0,
                }
            })
            .collect();
        Candles::new(bars)
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
            report: true,
            max_leverage: 0.0,
            impact: 0.0,
            funding_rate: 0.0,
            funding_interval: 0,
        }
    }

    #[test]
    fn sma_cross_goes_long_on_a_rising_series() {
        let mut e = Engine::new(rising(10), cfg());
        let mut s = SmaCross::from_params(&[2.0, 3.0]);
        let final_state = run(&mut e, &mut s);
        // fast average stays above slow on a monotone rise, so it buys and holds
        assert!(final_state.position > 0.0);
    }

    #[test]
    fn builtin_strategy_resolves_known_names_only() {
        assert!(builtin_strategy("sma_cross").is_some());
        assert!(builtin_strategy("unknown").is_none());
    }

    #[test]
    fn run_sweep_returns_one_stats_per_param_row() {
        let params = vec![vec![2.0, 3.0], vec![3.0, 5.0], vec![5.0, 8.0]];
        let factory = builtin_strategy("sma_cross").unwrap();
        let stats = run_sweep(rising(20), cfg(), factory, &params, 365.0, 0.0);
        assert_eq!(stats.len(), 3);
        assert!(stats.iter().all(|s| s.total_return_pct.is_finite()));
    }
}
