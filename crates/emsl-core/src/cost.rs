//! The trading-fee schedule. A `CostModel` charges a fee on a fill; slippage is
//! applied separately at the fill stage (a worse fill price), not here.

use crate::fill::Fill;

/// Computes the fee on a fill, in the quote asset. `Send + Sync` so one model can
/// be shared read-only across the batched envs.
pub trait CostModel: Send + Sync {
    /// The fee on `fill`, in quote. It follows the sign of the rate: a positive
    /// rate is a cost, and a negative maker rate is a rebate (a credit to quote).
    fn fee(&self, fill: &Fill) -> f64;
}

/// A flat maker/taker schedule: a fixed fraction of notional per side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlatCostModel {
    /// Taker fee as a fraction of notional (0.0006 is 6 bps).
    pub fee_taker: f64,
    /// Maker fee as a fraction of notional (0.0002 is 2 bps); a negative rate is a
    /// maker rebate.
    pub fee_maker: f64,
}

impl CostModel for FlatCostModel {
    #[inline]
    fn fee(&self, fill: &Fill) -> f64 {
        let notional = fill.notional().get().abs();
        let rate = if fill.is_taker {
            self.fee_taker
        } else {
            self.fee_maker
        };
        notional * rate
    }
}

#[cfg(test)]
mod tests {
    use super::{CostModel, FlatCostModel};
    use crate::enums::Side;
    use crate::fill::Fill;
    use crate::units::{Price, Qty};

    fn fill(is_taker: bool) -> Fill {
        // notional = 100 * 2 = 200
        Fill {
            side: Side::Buy,
            size: Qty(2.0),
            price: Price(100.0),
            is_taker,
        }
    }

    #[test]
    fn taker_pays_the_taker_fee() {
        let m = FlatCostModel {
            fee_taker: 0.0006,
            fee_maker: 0.0002,
        };
        assert!((m.fee(&fill(true)) - 0.12).abs() < 1e-12);
    }

    #[test]
    fn maker_pays_the_maker_fee() {
        let m = FlatCostModel {
            fee_taker: 0.0006,
            fee_maker: 0.0002,
        };
        assert!((m.fee(&fill(false)) - 0.04).abs() < 1e-12);
    }

    #[test]
    fn negative_maker_rate_is_a_rebate() {
        let m = FlatCostModel {
            fee_taker: 0.0006,
            fee_maker: -0.0001,
        };
        assert!(m.fee(&fill(false)) < 0.0); // a maker rebate credits the account
    }
}
