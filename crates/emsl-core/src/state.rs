//! The per-step snapshot the caller reads. Plain `f64` fields so it maps cleanly
//! onto the Python surface; the account computes it and the bar engine fills in
//! the bar fields.

use crate::order::Order;

/// Everything the caller reads after a step. Resting orders are listed; reward,
/// observation, and chart are deliberately not here.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct State {
    pub tick_index: usize,
    pub base: f64,
    pub quote: f64,
    /// Signed position in base units, negative is short.
    pub position: f64,
    /// Volume-weighted entry price, zero when flat.
    pub avg_entry: f64,
    /// Account value, marked in the quote asset.
    pub equity: f64,
    /// Close on spot, mark price on a perp.
    pub mark_price: f64,
    pub bar_open: f64,
    pub bar_high: f64,
    pub bar_low: f64,
    pub bar_close: f64,
    pub bar_volume: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub open_orders: Vec<Order>,
}

#[cfg(test)]
mod tests {
    use super::State;

    #[test]
    fn default_is_flat_and_empty() {
        let s = State::default();
        assert_eq!(s.position, 0.0);
        assert_eq!(s.equity, 0.0);
        assert_eq!(s.tick_index, 0);
        assert!(s.open_orders.is_empty());
    }
}
