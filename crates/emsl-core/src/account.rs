//! The account: a cash balance plus a netted position, with spot and perp
//! equity. Cash moves differently per market. On spot the full notional moves on
//! every fill (a buy spends quote, a sell receives it), so realized PnL is
//! implicit in the cash. On a perp only realized PnL moves quote on a close, and
//! the open position is marked separately.

use crate::enums::Market;
use crate::fill::Fill;
use crate::position::{moves_position, Position, POS_EPS};
use crate::units::{Price, Qty};

/// True when `price` can divide a notional: finite and strictly positive. A bar
/// whose close is zero is accepted by the input contract (only non-finite values
/// are rejected), so the sizing helpers check it themselves.
#[inline]
fn tradeable(price: Price) -> bool {
    price.get().is_finite() && price.get() > 0.0
}

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

    /// Apply a fill and its fee, moving cash by the market's convention. A fill
    /// the position refuses (non-finite, or sized at or below the dust epsilon)
    /// moves no cash and pays no fee, so quote can never change without the
    /// position changing with it (ADR 0023).
    pub fn apply_fill(&mut self, fill: &Fill, fee: f64) {
        if !moves_position(fill) {
            return;
        }
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
    /// exceed 1 for leverage on a perp; the margin cap is enforced elsewhere. A
    /// price that is not finite and positive has no size to give, so it returns
    /// zero rather than an infinity the caller would have to notice (ADR 0023).
    pub fn qty_from_weight(&self, fraction: f64, price: Price) -> Qty {
        if !tradeable(price) {
            return Qty(0.0);
        }
        Qty(fraction * self.equity(price) / price.get())
    }

    /// Base size worth `cash` quote at `price`. Zero on a non-positive or
    /// non-finite price, for the same reason as `qty_from_weight`.
    pub fn qty_from_quote(&self, cash: f64, price: Price) -> Qty {
        if !tradeable(price) {
            return Qty(0.0);
        }
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

    /// True when the position is dead at `mark`: a perp whose equity has fallen to
    /// zero or below. Spot and a flat position are never liquidated (ADR 0003).
    ///
    /// The test is the trigger only. The price the position is then closed at is
    /// `bankruptcy_price`, which is not this mark: `mark` is the worst the bar
    /// printed and closing there is what used to leave the account owing money.
    pub fn is_bust_at(&self, mark: Price) -> bool {
        self.market == Market::Perp
            && !self.position.is_flat()
            && mark.get().is_finite()
            && self.equity(mark) <= 0.0
    }

    /// The price at which closing the whole position, and paying `fee_rate` on the
    /// closing notional, leaves the account with exactly nothing.
    ///
    /// Solving `quote + qty * (price - entry) - |qty| * price * fee_rate == 0` for
    /// price. This is what makes a liquidation unable to leave a debt: the loss is
    /// bounded by the margin because the exit is priced at the point the margin
    /// runs out, rather than at whatever the bar happened to print afterwards. It
    /// also answers where a fee comes from when nothing is left to pay it with:
    /// the close lands far enough before ruin that the fee fits inside what
    /// remains (ADR 0052). `None` when the position is dust or the arithmetic is
    /// degenerate.
    pub fn bankruptcy_price(&self, fee_rate: f64) -> Option<Price> {
        let qty = self.position.qty.get();
        let size = qty.abs();
        if size <= POS_EPS || !fee_rate.is_finite() {
            return None;
        }
        let denominator = qty - size * fee_rate;
        if denominator.abs() <= f64::EPSILON {
            return None;
        }
        let price = (qty * self.position.avg_entry.get() - self.quote) / denominator;
        if price.is_finite() && price > 0.0 {
            Some(Price(price))
        } else {
            None
        }
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
    fn perp_long_is_bust_where_its_margin_runs_out() {
        let mut a = Account::new(Market::Perp, 100.0);
        a.apply_fill(&fill(Side::Buy, 10.0, 100.0), 0.0); // 10x notional on 100 margin
                                                          // equity at 90 = 100 + 10*(90-100) = 0
        assert!(a.is_bust_at(Price(90.0)));
        assert!(!a.is_bust_at(Price(90.01)));
        assert!(close(a.bankruptcy_price(0.0).unwrap().get(), 90.0));
    }

    #[test]
    fn perp_short_is_bust_when_the_mark_rises_far_enough() {
        let mut a = Account::new(Market::Perp, 100.0);
        a.apply_fill(&fill(Side::Sell, 10.0, 100.0), 0.0);
        // equity at 110 = 100 + (-10)*(110-100) = 0
        assert!(a.is_bust_at(Price(110.0)));
        assert!(!a.is_bust_at(Price(109.99)));
        assert!(close(a.bankruptcy_price(0.0).unwrap().get(), 110.0));
    }

    #[test]
    fn closing_at_the_bankruptcy_price_leaves_exactly_nothing() {
        // whatever the bar printed, the position cannot lose more than the margin
        // behind it, so the exit is priced where that margin runs out (ADR 0052)
        for side in [Side::Buy, Side::Sell] {
            let mut a = Account::new(Market::Perp, 100.0);
            a.apply_fill(&fill(side, 10.0, 100.0), 0.0);
            let price = a.bankruptcy_price(0.0).unwrap();
            let closing = if side == Side::Buy {
                Side::Sell
            } else {
                Side::Buy
            };
            a.apply_fill(&fill(closing, 10.0, price.get()), 0.0);
            assert!(a.position.is_flat());
            assert!(close(a.quote, 0.0), "{side:?} left {}", a.quote);
        }
    }

    #[test]
    fn the_liquidation_fee_fits_inside_what_is_left() {
        // there is nothing to pay a fee from at the moment of ruin, so the exit is
        // priced far enough before it that the fee still lands the account at zero
        let mut a = Account::new(Market::Perp, 100.0);
        a.apply_fill(&fill(Side::Buy, 10.0, 100.0), 0.0);
        let price = a.bankruptcy_price(0.0006).unwrap();
        assert!(price.get() > 90.0, "priced at {}", price.get()); // before ruin, not after
        let fee = 10.0 * price.get() * 0.0006;
        a.apply_fill(&fill(Side::Sell, 10.0, price.get()), fee);
        assert!(close(a.quote, 0.0), "left {}", a.quote);
    }

    #[test]
    fn solvent_position_is_not_bust() {
        let mut a = Account::new(Market::Perp, 1000.0);
        a.apply_fill(&fill(Side::Buy, 1.0, 100.0), 0.0);
        assert!(!a.is_bust_at(Price(90.0))); // equity 990, fine
        assert_eq!(a.position.qty, Qty(1.0));
    }

    #[test]
    fn spot_and_flat_are_never_bust() {
        let mut spot = Account::new(Market::Spot, 1000.0);
        spot.apply_fill(&fill(Side::Buy, 2.0, 100.0), 0.0);
        assert!(!spot.is_bust_at(Price(1.0)));

        let flat = Account::new(Market::Perp, 1000.0);
        assert!(!flat.is_bust_at(Price(50.0)));
        assert!(flat.bankruptcy_price(0.0).is_none());
    }

    #[test]
    fn a_sub_dust_fill_moves_neither_position_nor_cash() {
        // the position ignores anything at or below POS_EPS, so the cash move and
        // the fee must be gated on the same test or quote drifts on its own (ADR 0023)
        let mut a = Account::new(Market::Spot, 1000.0);
        a.apply_fill(&fill(Side::Buy, 2.0, 100.0), 0.0); // quote 800, long 2
        let before = a;
        a.apply_fill(&fill(Side::Sell, 1e-10, 300.0), 0.5);
        assert_eq!(a, before);
    }

    #[test]
    fn a_non_finite_or_negative_fill_moves_no_cash() {
        let mut a = Account::new(Market::Spot, 1000.0);
        let before = a;
        a.apply_fill(&fill(Side::Buy, f64::NAN, 100.0), 1.0);
        a.apply_fill(&fill(Side::Buy, 1.0, f64::NAN), 1.0);
        a.apply_fill(&fill(Side::Buy, -100.0, 100.0), -20_000.0);
        assert_eq!(a, before);
        assert!(a.quote.is_finite());
    }

    #[test]
    fn sizing_helpers_return_zero_on_an_untradeable_price() {
        // a zero close passes the candle contract (only non-finite is rejected), and
        // dividing by it would hand the caller an infinity instead of a size
        let a = Account::new(Market::Spot, 1000.0);
        assert_eq!(a.qty_from_weight(0.5, Price(0.0)), Qty(0.0));
        assert_eq!(a.qty_from_quote(250.0, Price(0.0)), Qty(0.0));
        assert_eq!(a.qty_from_weight(0.5, Price(-1.0)), Qty(0.0));
        assert_eq!(a.qty_from_quote(250.0, Price(f64::NAN)), Qty(0.0));
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
