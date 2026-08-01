//! The netted, one-way position: signed size, a volume-weighted average entry,
//! and cumulative realized PnL. A single fill can add to, reduce, close, or flip
//! the position. The booking rule is ADR 0001 in `.Documentation/Decisions.md`.

use crate::fill::Fill;
use crate::units::{Price, Qty};

/// Positions with magnitude at or below this (in base units) are treated as
/// flat, so a close with non-representable fractional sizes (the classic
/// `0.1 + 0.2 != 0.3`) cannot strand a dust residual with a stale entry. Far
/// below any real exchange lot size.
pub const POS_EPS: f64 = 1e-9;

/// True when `fill` moves the position at all. A non-finite value or a size at
/// or below the dust epsilon changes nothing, so the account must not move cash
/// for it either (ADR 0023).
#[inline]
pub fn moves_position(fill: &Fill) -> bool {
    fill.size.get().is_finite() && fill.size.get() > POS_EPS && fill.price.get().is_finite()
}

/// A net long, net short, or flat position. There is never a simultaneous long
/// and short (no hedge mode).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Position {
    /// Signed size in base units: positive long, negative short, zero flat.
    pub qty: Qty,
    /// Volume-weighted entry price of the open position; zero when flat.
    pub avg_entry: Price,
    /// Cumulative realized PnL booked on closes and flips, in quote.
    pub realized: f64,
}

impl Position {
    /// A flat position.
    pub fn flat() -> Position {
        Position::default()
    }

    /// True when there is no open position (within the dust epsilon).
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.qty.get().abs() <= POS_EPS
    }

    /// Unrealized PnL of the open position marked at `mark`, in quote. Equal to
    /// `position * (mark - avg_entry)`, so it is correct for both sides.
    #[inline]
    pub fn unrealized(&self, mark: Price) -> f64 {
        self.qty.get() * (mark.get() - self.avg_entry.get())
    }

    /// Apply `fill` to the position. Books realized PnL on any quantity the fill
    /// closes, at the old average entry, and opens any remainder on the new side
    /// at the fill price. Returns the realized PnL booked by this fill, in quote
    /// (zero when only opening or adding).
    ///
    /// Non-finite fills and zero-size fills are ignored, so a bad value cannot
    /// poison the account; the engine validates orders before they reach here.
    /// `Account::apply_fill` gates its cash move on the same `moves_position`
    /// predicate, so a fill refused here never moves quote either (ADR 0023).
    pub fn apply(&mut self, fill: &Fill) -> f64 {
        // size is a positive magnitude; a non-finite, zero, or negative size is
        // ignored so a bad value cannot open the wrong side or poison the account
        if !moves_position(fill) {
            return 0.0;
        }
        let signed = fill.side.sign() * fill.size.get();
        let px = fill.price.get();

        let pos = self.qty.get();
        let entry = self.avg_entry.get();

        // Flat (or dust): open a fresh position at the fill price.
        if pos.abs() <= POS_EPS {
            self.qty = Qty(signed);
            self.avg_entry = fill.price;
            return 0.0;
        }

        // Same side: add and volume-weight the entry, no realized PnL.
        if (pos > 0.0) == (signed > 0.0) {
            let added = fill.size.get();
            let new_abs = pos.abs() + added;
            self.avg_entry = Price((pos.abs() * entry + added * px) / new_abs);
            self.qty = Qty(pos + signed);
            return 0.0;
        }

        // Opposite side: reduce, close, or flip. Book PnL on the closed part.
        let closed = fill.size.get().min(pos.abs());
        let realized = closed * (px - entry) * pos.signum();
        self.realized += realized;

        let new_pos = pos + signed;
        if new_pos.abs() <= POS_EPS {
            // Closed to flat, including a fractional close that left only dust.
            self.qty = Qty(0.0);
            self.avg_entry = Price(0.0);
        } else if (new_pos > 0.0) == (pos > 0.0) {
            // Reduced, still the same side: keep the entry on the remaining size.
            self.qty = Qty(new_pos);
        } else {
            // Flipped through zero: the remainder opens fresh at the fill price.
            self.qty = Qty(new_pos);
            self.avg_entry = fill.price;
        }

        realized
    }
}

#[cfg(test)]
mod tests {
    use super::Position;
    use crate::enums::Side;
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
    fn opens_long_from_flat() {
        let mut p = Position::flat();
        let r = p.apply(&fill(Side::Buy, 2.0, 100.0));
        assert_eq!(r, 0.0);
        assert_eq!(p.qty, Qty(2.0));
        assert_eq!(p.avg_entry, Price(100.0));
        assert_eq!(p.realized, 0.0);
    }

    #[test]
    fn opens_short_from_flat() {
        let mut p = Position::flat();
        p.apply(&fill(Side::Sell, 2.0, 100.0));
        assert_eq!(p.qty, Qty(-2.0));
        assert_eq!(p.avg_entry, Price(100.0));
    }

    #[test]
    fn adds_and_volume_weights_entry() {
        let mut p = Position::flat();
        p.apply(&fill(Side::Buy, 2.0, 100.0));
        let r = p.apply(&fill(Side::Buy, 3.0, 110.0));
        assert_eq!(r, 0.0);
        assert_eq!(p.qty, Qty(5.0));
        assert!(close(p.avg_entry.get(), 106.0)); // (2*100 + 3*110) / 5
    }

    #[test]
    fn reduce_books_realized_and_keeps_entry() {
        let mut p = Position::flat();
        p.apply(&fill(Side::Buy, 5.0, 106.0));
        let r = p.apply(&fill(Side::Sell, 2.0, 120.0));
        assert!(close(r, 28.0)); // 2 * (120 - 106)
        assert_eq!(p.qty, Qty(3.0));
        assert_eq!(p.avg_entry, Price(106.0));
        assert!(close(p.realized, 28.0));
    }

    #[test]
    fn exact_close_goes_flat() {
        let mut p = Position::flat();
        p.apply(&fill(Side::Buy, 3.0, 106.0));
        let r = p.apply(&fill(Side::Sell, 3.0, 90.0));
        assert!(close(r, -48.0)); // 3 * (90 - 106)
        assert!(p.is_flat());
        assert_eq!(p.avg_entry, Price(0.0));
    }

    #[test]
    fn flips_long_to_short() {
        let mut p = Position::flat();
        p.apply(&fill(Side::Buy, 2.0, 100.0));
        let r = p.apply(&fill(Side::Sell, 5.0, 110.0));
        assert!(close(r, 20.0)); // close 2 at (110 - 100)
        assert_eq!(p.qty, Qty(-3.0)); // remainder short 3
        assert_eq!(p.avg_entry, Price(110.0));
        assert!(close(p.realized, 20.0));
    }

    #[test]
    fn reduce_short_books_realized() {
        let mut p = Position::flat();
        p.apply(&fill(Side::Sell, 2.0, 100.0));
        let r = p.apply(&fill(Side::Buy, 1.0, 90.0));
        assert!(close(r, 10.0)); // short: 1 * (90 - 100) * -1
        assert_eq!(p.qty, Qty(-1.0));
        assert_eq!(p.avg_entry, Price(100.0));
    }

    #[test]
    fn flips_short_to_long() {
        let mut p = Position::flat();
        p.apply(&fill(Side::Sell, 2.0, 100.0));
        let r = p.apply(&fill(Side::Buy, 5.0, 90.0));
        assert!(close(r, 20.0)); // close 2 short at (90 - 100) * -1
        assert_eq!(p.qty, Qty(3.0)); // remainder long 3
        assert_eq!(p.avg_entry, Price(90.0));
    }

    #[test]
    fn unrealized_is_correct_both_sides() {
        let mut long = Position::flat();
        long.apply(&fill(Side::Buy, 2.0, 100.0));
        assert!(close(long.unrealized(Price(110.0)), 20.0));

        let mut short = Position::flat();
        short.apply(&fill(Side::Sell, 2.0, 100.0));
        assert!(close(short.unrealized(Price(90.0)), 20.0));
    }

    #[test]
    fn cumulative_realized_across_a_sequence() {
        let mut p = Position::flat();
        p.apply(&fill(Side::Buy, 5.0, 106.0)); // open long 5 @ 106
        p.apply(&fill(Side::Sell, 2.0, 120.0)); // reduce, +28
        p.apply(&fill(Side::Sell, 3.0, 90.0)); // close 3 @ (90-106) = -48
        assert!(close(p.realized, 28.0 - 48.0)); // -20
        assert!(p.is_flat());
    }

    #[test]
    fn fractional_close_goes_flat_not_dust() {
        // 0.1 + 0.2 != 0.3 in f64, so an exact == 0.0 check would strand dust.
        let mut p = Position::flat();
        p.apply(&fill(Side::Buy, 0.1, 100.0));
        p.apply(&fill(Side::Buy, 0.2, 100.0));
        let r = p.apply(&fill(Side::Sell, 0.3, 110.0));
        assert!(p.is_flat());
        assert_eq!(p.qty, Qty(0.0));
        assert_eq!(p.avg_entry, Price(0.0));
        assert!(close(r, 3.0)); // 0.3 * (110 - 100)
    }

    #[test]
    fn zero_size_fill_is_a_noop() {
        let mut p = Position::flat();
        p.apply(&fill(Side::Buy, 2.0, 100.0));
        let r = p.apply(&fill(Side::Buy, 0.0, 999.0));
        assert_eq!(r, 0.0);
        assert_eq!(p.qty, Qty(2.0));
        assert_eq!(p.avg_entry, Price(100.0));
    }

    #[test]
    fn same_price_close_books_zero() {
        let mut p = Position::flat();
        p.apply(&fill(Side::Sell, 3.0, 100.0));
        let r = p.apply(&fill(Side::Buy, 3.0, 100.0));
        assert_eq!(r, 0.0);
        assert!(p.is_flat());
    }

    #[test]
    fn flip_leaves_fractional_remainder_at_fill_price() {
        let mut p = Position::flat();
        p.apply(&fill(Side::Buy, 1.5, 100.0));
        let r = p.apply(&fill(Side::Sell, 4.0, 110.0));
        assert!(close(r, 15.0)); // close 1.5 @ (110 - 100)
        assert_eq!(p.qty, Qty(-2.5));
        assert_eq!(p.avg_entry, Price(110.0));
    }

    #[test]
    fn non_finite_fill_is_ignored() {
        let mut p = Position::flat();
        p.apply(&fill(Side::Buy, 2.0, 100.0));
        let before = p;
        assert_eq!(p.apply(&fill(Side::Buy, 1.0, f64::NAN)), 0.0);
        assert_eq!(p.apply(&fill(Side::Buy, f64::NAN, 100.0)), 0.0);
        assert_eq!(p, before);
    }

    #[test]
    fn negative_size_fill_is_ignored() {
        // a Buy of -2 is not a short: size is a positive magnitude, so a negative
        // one is a no-op rather than opening the opposite side
        let mut p = Position::flat();
        let before = p;
        assert_eq!(p.apply(&fill(Side::Buy, -2.0, 100.0)), 0.0);
        assert_eq!(p, before);
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
        fn realized_return_is_cumulative(seq in fill_seq(1..40)) {
            // the sum of per-fill booked PnL equals the running total, so the return
            // value and the accumulator never drift apart
            let mut p = Position::flat();
            let mut sum = 0.0;
            for f in &seq {
                sum += p.apply(f);
            }
            prop_assert!((p.realized - sum).abs() < 1e-6);
        }

        #[test]
        fn flat_has_zero_entry_else_stays_above_dust(seq in fill_seq(1..40)) {
            // after any sequence the position is either cleanly flat with a zeroed
            // entry, or a real position whose magnitude clears the dust epsilon
            let mut p = Position::flat();
            for f in &seq {
                p.apply(f);
            }
            if p.is_flat() {
                prop_assert_eq!(p.avg_entry.get(), 0.0);
            } else {
                prop_assert!(p.qty.get().abs() > 1e-9);
            }
        }
    }
}
