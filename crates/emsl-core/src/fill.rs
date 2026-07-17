//! A fill record: what actually traded. Both the cost model and the account
//! read it.

use crate::enums::Side;
use crate::units::{Notional, Price, Qty};

/// A single fill: a quantity of base traded at a price on one side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fill {
    pub side: Side,
    /// Filled quantity, a positive magnitude.
    pub size: Qty,
    pub price: Price,
    /// Crossed the spread as a market taker, rather than resting as a maker.
    pub is_taker: bool,
}

impl Fill {
    /// The traded notional in the quote asset (price times size).
    #[inline]
    pub fn notional(&self) -> Notional {
        self.price.notional(self.size)
    }
}

#[cfg(test)]
mod tests {
    use super::Fill;
    use crate::enums::Side;
    use crate::units::{Notional, Price, Qty};

    #[test]
    fn notional_is_price_times_size() {
        let f = Fill {
            side: Side::Buy,
            size: Qty(2.0),
            price: Price(150.0),
            is_taker: true,
        };
        assert_eq!(f.notional(), Notional(300.0));
    }
}
