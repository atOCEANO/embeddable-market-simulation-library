//! emsl-core: fidelity-neutral primitives for the market simulation engine.
//!
//! Pure Rust: no candle series, no Python, no parallelism. These primitives are
//! the reuse seam shared across fidelity tiers (the bar engine here, a future
//! tick and L2 engine later).

pub mod account;
pub mod book;
pub mod candle;
pub mod cost;
pub mod enums;
pub mod fill;
pub mod order;
pub mod position;
pub mod state;
pub mod units;

pub use account::Account;
pub use book::RestingOrderBook;
pub use candle::Candle;
pub use cost::{CostModel, FlatCostModel};
pub use enums::{Market, OrderType, Side, TimeInForce};
pub use fill::Fill;
pub use order::{Order, OrderId};
pub use position::Position;
pub use state::State;
pub use units::{Bps, Notional, Price, Qty};
