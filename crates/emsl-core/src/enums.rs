//! Small plain enums shared across the engine: order side, type, time in force,
//! lifecycle status, and market kind.

/// The direction of an order or a position change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    /// `+1.0` for a buy, `-1.0` for a sell: the sign a fill applies to a signed
    /// position.
    #[inline]
    pub const fn sign(self) -> f64 {
        match self {
            Side::Buy => 1.0,
            Side::Sell => -1.0,
        }
    }

    /// The opposite side.
    #[inline]
    pub const fn opposite(self) -> Side {
        match self {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }
}

/// How an order is matched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderType {
    /// Crosses the spread, fills at the next bar open, pays the taker fee.
    Market,
    /// Rests at a price, fills only when a bar reaches it, pays the maker fee.
    Limit,
    /// Becomes a market order once the trigger price is crossed.
    Stop,
}

/// How long an order stays live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeInForce {
    /// Good til canceled: rests until filled or canceled.
    Gtc,
    /// Immediate or cancel: fills what it can now, cancels the rest.
    Ioc,
    /// Fill or kill: fills completely now, or cancels entirely.
    Fok,
}

/// The lifecycle state of an order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderStatus {
    /// Live in the book, possibly partially filled.
    Resting,
    /// Fully filled.
    Filled,
    /// Removed without fully filling.
    Canceled,
}

/// The market a position trades in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Market {
    /// Spot: hold the base asset; equity is quote plus base times price.
    Spot,
    /// Perpetual future: a signed position with margin, no expiry.
    Perp,
}

#[cfg(test)]
mod tests {
    use super::Side;

    #[test]
    fn side_sign() {
        assert_eq!(Side::Buy.sign(), 1.0);
        assert_eq!(Side::Sell.sign(), -1.0);
    }

    #[test]
    fn side_opposite() {
        assert_eq!(Side::Buy.opposite(), Side::Sell);
        assert_eq!(Side::Sell.opposite(), Side::Buy);
        assert_eq!(Side::Buy.opposite().opposite(), Side::Buy);
    }
}
