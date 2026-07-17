//! The account: a cash balance plus a netted position, with spot and perp
//! equity. Cash moves differently per market. On spot the full notional moves on
//! every fill (a buy spends quote, a sell receives it), so realized PnL is
//! implicit in the cash. On a perp only realized PnL moves quote on a close, and
//! the open position is marked separately.

use crate::enums::{Market, Side};
use crate::fill::Fill;
use crate::position::Position;
use crate::units::{Price, Qty};

/// A cash balance and the netted position it backs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Account {
    pub market: Market,
    /// Cash balance in the quote asset.
    pub quote: f64,
    pub position: Position,
}

impl Account {
    /// A fresh account with `quote` cash and no position.
    pub fn new(market: Market, quote: f64) -> Account {
        Account {
            market,
            quote,
            position: Position::flat(),
        }
    }

    /// Apply a fill and its fee, moving cash by the market's convention.
    pub fn apply_fill(&mut self, fill: &Fill, fee: f64) {
        let realized = self.position.apply(fill);
        self.quote -= fee;
        match self.market {
            // Spot: the full notional moves; a buy spends, a sell receives.
            Market::Spot => {
                self.quote -= fill.side.sign() * fill.size.get() * fill.price.get();
            }
            // Perp: only realized PnL booked on closes and flips moves quote.
            Market::Perp => {
                self.quote += realized;
            }
        }
    }

    /// Account value marked in the quote asset: `quote + base * price` on spot,
    /// `quote + unrealized` on a perp.
    pub fn equity(&self, mark: Price) -> f64 {
        match self.market {
            Market::Spot => self.quote + self.position.qty.get() * mark.get(),
            Market::Perp => self.quote + self.position.unrealized(mark),
        }
    }

    /// Unrealized PnL of the open position at `mark`, in quote.
    #[inline]
    pub fn unrealized(&self, mark: Price) -> f64 {
        self.position.unrealized(mark)
    }

    /// Cumulative realized PnL, in quote.
    #[inline]
    pub fn realized(&self) -> f64 {
        self.position.realized
    }

    /// Base size worth `fraction` of current equity at `price`. `fraction` can
    /// exceed 1 for leverage on a perp; the margin cap is enforced elsewhere.
    pub fn qty_from_weight(&self, fraction: f64, price: Price) -> Qty {
        Qty(fraction * self.equity(price) / price.get())
    }

    /// Base size worth `cash` quote at `price`.
    pub fn qty_from_quote(&self, cash: f64, price: Price) -> Qty {
        Qty(cash / price.get())
    }

    /// Apply one funding charge at `rate` on the position marked at `mark`. A
    /// long pays when the rate is positive and a short receives; spot has no
    /// funding. Returns the amount paid (positive) or received (negative), in
    /// quote.
    pub fn apply_funding(&mut self, rate: f64, mark: Price) -> f64 {
        if self.market != Market::Perp || !rate.is_finite() || !mark.get().is_finite() {
            return 0.0;
        }
        let payment = self.position.qty.get() * mark.get() * rate;
        self.quote -= payment;
        payment
    }

    /// Check for liquidation at `mark`. On a perp, if the position's equity has
    /// fallen to zero or below, force-close it at the mark, book the loss, and
    /// return true. Spot and a flat position are never liquidated here.
    pub fn liquidate_if_bust(&mut self, mark: Price) -> bool {
        if self.market != Market::Perp || self.position.is_flat() || !mark.get().is_finite() {
            return false;
        }
        if self.equity(mark) > 0.0 {
            return false;
        }
        let side = if self.position.qty.get() > 0.0 {
            Side::Sell
        } else {
            Side::Buy
        };
        let closing = Fill {
            side,
            size: Qty(self.position.qty.get().abs()),
            price: mark,
            is_taker: true,
        };
        self.apply_fill(&closing, 0.0);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::Account;
    use crate::enums::{Market, Side};
    use crate::fill::Fill;
    use crate::units::{Price, Qty};
    use proptest::prelude::*;

    fn fill(side: Side, size: f64, price: f64) -> Fill {
        Fill {
            side,
            size: Qty(size),
            price: Price(price),
            is_taker: true,
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn spot_buy_spends_quote_and_holds_base() {
        let mut a = Account::new(Market::Spot, 1000.0);
        a.apply_fill(&fill(Side::Buy, 2.0, 100.0), 0.0);
        assert!(close(a.quote, 800.0));
        assert!(close(a.equity(Price(100.0)), 1000.0)); // no PnL at entry
        assert!(close(a.equity(Price(110.0)), 1020.0)); // +20 unrealized
    }

    #[test]
    fn spot_round_trip_realizes_in_cash() {
        let mut a = Account::new(Market::Spot, 1000.0);
        a.apply_fill(&fill(Side::Buy, 2.0, 100.0), 0.0);
        a.apply_fill(&fill(Side::Sell, 2.0, 110.0), 0.0);
        assert!(close(a.quote, 1020.0));
        assert!(a.position.is_flat());
        // Flat: equity is just the cash, independent of mark, and not double-counted.
        assert!(close(a.equity(Price(999.0)), 1020.0));
    }

    #[test]
    fn perp_holds_margin_and_marks_unrealized() {
        let mut a = Account::new(Market::Perp, 1000.0);
        a.apply_fill(&fill(Side::Buy, 2.0, 100.0), 0.0);
        assert!(close(a.quote, 1000.0)); // opening a perp moves no cash
        assert!(close(a.equity(Price(110.0)), 1020.0));
    }

    #[test]
    fn perp_close_books_pnl_to_quote() {
        let mut a = Account::new(Market::Perp, 1000.0);
        a.apply_fill(&fill(Side::Buy, 2.0, 100.0), 0.0);
        a.apply_fill(&fill(Side::Sell, 2.0, 110.0), 0.0);
        assert!(close(a.quote, 1020.0));
        assert!(a.position.is_flat());
    }

    #[test]
    fn fee_reduces_quote() {
        let mut a = Account::new(Market::Perp, 1000.0);
        a.apply_fill(&fill(Side::Buy, 2.0, 100.0), 0.12);
        assert!(close(a.quote, 999.88));
    }

    #[test]
    fn qty_from_weight_is_fraction_of_equity() {
        let a = Account::new(Market::Perp, 1000.0);
        // flat: equity is 1000; weight 0.5 at price 100 is 500 notional, 5 base
        assert!(close(a.qty_from_weight(0.5, Price(100.0)).get(), 5.0));
    }

    #[test]
    fn qty_from_quote_is_cash_over_price() {
        let a = Account::new(Market::Perp, 1000.0);
        assert!(close(a.qty_from_quote(250.0, Price(100.0)).get(), 2.5));
    }

    #[test]
    fn perp_long_pays_positive_funding() {
        let mut a = Account::new(Market::Perp, 1000.0);
        a.apply_fill(&fill(Side::Buy, 2.0, 100.0), 0.0);
        let paid = a.apply_funding(0.0001, Price(100.0));
        assert!(close(paid, 0.02)); // 2 * 100 * 0.0001
        assert!(close(a.quote, 999.98));
    }

    #[test]
    fn perp_short_receives_positive_funding() {
        let mut a = Account::new(Market::Perp, 1000.0);
        a.apply_fill(&fill(Side::Sell, 2.0, 100.0), 0.0);
        let paid = a.apply_funding(0.0001, Price(100.0));
        assert!(close(paid, -0.02)); // received
        assert!(close(a.quote, 1000.02));
    }

    #[test]
    fn spot_has_no_funding() {
        let mut a = Account::new(Market::Spot, 1000.0);
        a.apply_fill(&fill(Side::Buy, 2.0, 100.0), 0.0); // quote 800
        assert_eq!(a.apply_funding(0.01, Price(100.0)), 0.0);
        assert!(close(a.quote, 800.0));
    }

    #[test]
    fn flat_position_pays_no_funding() {
        let mut a = Account::new(Market::Perp, 1000.0);
        assert_eq!(a.apply_funding(0.01, Price(100.0)), 0.0);
        assert!(close(a.quote, 1000.0));
    }

    #[test]
    fn perp_long_liquidates_when_equity_hits_zero() {
        let mut a = Account::new(Market::Perp, 100.0);
        a.apply_fill(&fill(Side::Buy, 10.0, 100.0), 0.0); // 10x notional on 100 margin
                                                          // equity at 90 = 100 + 10*(90-100) = 0
        assert!(a.liquidate_if_bust(Price(90.0)));
        assert!(a.position.is_flat());
        assert!(close(a.quote, 0.0));
    }

    #[test]
    fn perp_short_liquidates_when_mark_rises() {
        let mut a = Account::new(Market::Perp, 100.0);
        a.apply_fill(&fill(Side::Sell, 10.0, 100.0), 0.0);
        // equity at 110 = 100 + (-10)*(110-100) = 0
        assert!(a.liquidate_if_bust(Price(110.0)));
        assert!(a.position.is_flat());
        assert!(close(a.quote, 0.0));
    }

    #[test]
    fn liquidation_can_leave_bad_debt_on_a_gap() {
        // a gap past the liquidation mark loses more than the margin: a long on 100
        // margin force-closed at 80 books -200, so quote runs negative (ADR 0003)
        let mut a = Account::new(Market::Perp, 100.0);
        a.apply_fill(&fill(Side::Buy, 10.0, 100.0), 0.0);
        assert!(a.liquidate_if_bust(Price(80.0)));
        assert!(a.position.is_flat());
        assert!(close(a.quote, -100.0));
    }

    #[test]
    fn solvent_position_is_not_liquidated() {
        let mut a = Account::new(Market::Perp, 1000.0);
        a.apply_fill(&fill(Side::Buy, 1.0, 100.0), 0.0);
        assert!(!a.liquidate_if_bust(Price(90.0))); // equity 990, fine
        assert_eq!(a.position.qty, Qty(1.0));
    }

    #[test]
    fn spot_and_flat_are_never_liquidated() {
        let mut spot = Account::new(Market::Spot, 1000.0);
        spot.apply_fill(&fill(Side::Buy, 2.0, 100.0), 0.0);
        assert!(!spot.liquidate_if_bust(Price(1.0)));

        let mut flat = Account::new(Market::Perp, 1000.0);
        assert!(!flat.liquidate_if_bust(Price(50.0)));
    }

    fn fill_seq(len: std::ops::Range<usize>) -> impl Strategy<Value = Vec<Fill>> {
        prop::collection::vec((any::<bool>(), 0.1f64..10.0, 1.0f64..500.0), len).prop_map(|rows| {
            rows.into_iter()
                .map(|(buy, size, price)| Fill {
                    side: if buy { Side::Buy } else { Side::Sell },
                    size: Qty(size),
                    price: Price(price),
                    is_taker: true,
                })
                .collect()
        })
    }

    proptest! {
        #[test]
        fn perp_quote_moves_only_by_realized(seq in fill_seq(1..30)) {
            // on a perp with no fees, cash changes only by booked PnL: opening and
            // marking never move quote, so quote stays at start plus realized
            let mut a = Account::new(Market::Perp, 10_000.0);
            for f in &seq {
                a.apply_fill(f, 0.0);
            }
            prop_assert!(a.quote.is_finite());
            prop_assert!((a.quote - (10_000.0 + a.position.realized)).abs() < 1e-6);
        }

        #[test]
        fn spot_round_trip_returns_quote_plus_pnl(
            size in 0.1f64..10.0,
            p in 1.0f64..500.0,
            q in 1.0f64..500.0,
        ) {
            // buy then sell the same size on spot: the account ends flat and quote
            // moves by exactly the price difference times size
            let mut a = Account::new(Market::Spot, 1_000_000.0);
            let buy = Fill { side: Side::Buy, size: Qty(size), price: Price(p), is_taker: true };
            let sell = Fill { side: Side::Sell, size: Qty(size), price: Price(q), is_taker: true };
            a.apply_fill(&buy, 0.0);
            a.apply_fill(&sell, 0.0);
            prop_assert!(a.position.is_flat());
            prop_assert!((a.quote - (1_000_000.0 + size * (q - p))).abs() < 1e-6);
        }
    }
}
