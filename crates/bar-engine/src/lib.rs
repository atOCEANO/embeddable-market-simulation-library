//! bar-engine: the bar-level realization of the market simulator. It owns the
//! shared candle series, the next-bar fill model, the single step state machine,
//! and the batched runner. Pure Rust, no Python; it builds on emsl-core's
//! fidelity-neutral primitives.

pub mod batch;
pub mod candles;
pub mod engine;
pub mod execution;
pub mod reporter;
pub mod stats;
pub mod strategy;
pub mod sweep;

pub use batch::{BatchSnapshot, EnvBatch};
pub use candles::Candles;
pub use engine::{Engine, EngineConfig};
pub use execution::FillModel;
pub use reporter::{Reporter, Trade};
pub use stats::Stats;
pub use strategy::{run, Strategy};
pub use sweep::{builtin_strategy, run_sweep, SmaCross, StrategyFactory};
