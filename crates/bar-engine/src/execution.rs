//! The next-bar fill model. An order decided at the close of one bar fills on the
//! NEXT bar, so a fill is never priced off information the decision could not have
//! seen. Market fills are takers, limit fills are makers, and a stop triggers and
//! then fills as a taker. See ADR 0004 (slippage) and ADR 0005 (volume cap).

use emsl_core::{Bps, Candle, Fill, Order, Price, Qty, Side};

/// The most a taker fill may be moved against itself, as a fraction of the
/// reference price. A slip of 1.0 would price a sell at zero and anything beyond
/// it below zero, which is not a trade, so the total is held just under 1 (ADR 0024).
const MAX_SLIP: f64 = 0.99;

/// The cost side of execution: how far a market order slips and how much of a bar
/// one order may consume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FillModel {
    /// Adverse slippage on a taker fill, in basis points off the reference price.
    pub slippage_bps: Bps,
    /// The most of a bar's volume ONE order may take, as a fraction (1.0 is the
    /// whole bar), so no single fill exceeds that slice of what the bar traded.
    /// Volume is in base units, as an order's size is (ADR 0005). The cap is per
    /// order, not a shared per-bar budget: several orders resolving against the
    /// same bar each get it, which is why the queue of pending market orders is
    /// bounded rather than unlimited (ADR 0047).
    pub max_fill_fraction: f64,
    /// Market-impact coefficient: extra adverse slippage equal to `impact` times
    /// the fill's fraction of the bar volume, so a larger order pays more. Zero
    /// disables it. See ADR 0013.
    pub impact: f64,
}

impl FillModel {
    /// The fillable size: the order's remaining, capped at the volume fraction.
    /// `f64::min` keeps the other operand when one side is NaN, so BOTH sides
    /// carry the guard: a non-finite remaining would launder a NaN size into a
    /// full-cap fill (ADR 0001), and a non-finite fraction would delete the cap
    /// altogether and let one order take more than the bar traded (ADR 0024).
    /// The available side must be finite as well as positive: an infinite fraction
    /// leaves `min` with a real operand and would hand back the whole remaining.
    fn cap(&self, order: &Order, bar: &Candle) -> f64 {
        let remaining = order.remaining().get();
        let available = self.max_fill_fraction * bar.volume;
        if !remaining.is_finite() || !available.is_finite() || available <= 0.0 {
            return 0.0;
        }
        remaining.min(available)
    }

    /// The adverse slip on a taker fill: the flat rate plus the impact term. Both
    /// are floored at zero, since slippage is adverse by definition (ADR 0004) and
    /// a negative rate would fill better than the market, and the total is capped
    /// below 1 so the reference price can never be dropped to zero or through it
    /// (ADR 0024).
    fn taker_slip(&self, size: f64, volume: f64) -> f64 {
        let flat = self.slippage_bps.as_fraction().max(0.0);
        let impact = if volume > 0.0 && volume.is_finite() {
            self.impact.max(0.0) * (size / volume)
        } else {
            0.0
        };
        let slip = flat + impact;
        if !slip.is_finite() {
            return MAX_SLIP;
        }
        slip.min(MAX_SLIP)
    }

    /// Fill a market order against `bar` (the bar after the decision). Priced off
    /// the bar open, lifted on a buy and dropped on a sell by the slippage, sized
    /// at the remaining capped to the volume fraction. `None` if the bar gives no
    /// volume.
    pub fn fill_market(&self, order: &Order, bar: &Candle) -> Option<Fill> {
        let size = self.cap(order, bar);
        if size <= 0.0 {
            return None;
        }
        let slip = self.taker_slip(size, bar.volume);
        let price = match order.side {
            Side::Buy => bar.open * (1.0 + slip),
            Side::Sell => bar.open * (1.0 - slip),
        };
        Some(taker(order.side, size, price))
    }

    /// Fill a resting limit against `bar`. A buy fills if the bar trades at or
    /// below its price, a sell if at or above. It fills AT the limit price, never
    /// better (a bar that gaps through still fills at the limit), pays no
    /// slippage, and is capped at the volume fraction. `None` if untouched or the
    /// bar gives no volume.
    ///
    /// Whether it pays the maker or the taker rate is decided against the bar's
    /// OPEN, not against the order type. A limit already through the market when
    /// the bar opens is marketable: it is lifted on arrival and takes liquidity,
    /// so it pays the taker rate. Booking every limit as a maker meant a buy and a
    /// sell resting at the same price both filled on one wide bar as makers, and
    /// since a maker rate is legally a rebate, that straddle ended flat and richer
    /// every bar, without bound (ADR 0045).
    pub fn fill_limit(&self, order: &Order, bar: &Candle) -> Option<Fill> {
        let limit = order.price?.get();
        let touched = match order.side {
            Side::Buy => bar.low <= limit,
            Side::Sell => bar.high >= limit,
        };
        if !touched {
            return None;
        }
        let size = self.cap(order, bar);
        if size <= 0.0 {
            return None;
        }
        Some(Fill {
            side: order.side,
            size: Qty(size),
            price: Price(limit),
            is_taker: self.limit_crosses(order.side, Price(limit), Price(bar.open)),
        })
    }

    /// Trigger a stop against `bar`. A buy stop triggers when the bar reaches its
    /// trigger from below, a sell stop from above. Once triggered it fills as a
    /// taker at the worse of the trigger and the bar open (so a gap through the
    /// stop fills at the gap), then takes slippage. `None` if not triggered or the
    /// bar gives no volume.
    pub fn fill_stop(&self, order: &Order, bar: &Candle) -> Option<Fill> {
        let trigger = order.trigger?.get();
        let triggered = match order.side {
            Side::Buy => bar.high >= trigger,
            Side::Sell => bar.low <= trigger,
        };
        if !triggered {
            return None;
        }
        let size = self.cap(order, bar);
        if size <= 0.0 {
            return None;
        }
        let slip = self.taker_slip(size, bar.volume);
        let price = match order.side {
            Side::Buy => bar.open.max(trigger) * (1.0 + slip),
            Side::Sell => bar.open.min(trigger) * (1.0 - slip),
        };
        Some(taker(order.side, size, price))
    }

    /// Would a limit at `price` on `side` cross immediately against `reference`,
    /// making it a taker? A post_only order that would cross is rejected at
    /// placement rather than turned taker.
    pub fn limit_crosses(&self, side: Side, price: Price, reference: Price) -> bool {
        match side {
            Side::Buy => price.get() >= reference.get(),
            Side::Sell => price.get() <= reference.get(),
        }
    }
}

/// Build a taker fill on `side` of `size` at `price`.
fn taker(side: Side, size: f64, price: f64) -> Fill {
    Fill {
        side,
        size: Qty(size),
        price: Price(price),
        is_taker: true,
    }
}

#[cfg(test)]
mod tests {
    use super::FillModel;
    use emsl_core::{Bps, Candle, Order, OrderId, Price, Qty, Side, TimeInForce};

    fn ohlc(open: f64, high: f64, low: f64, close: f64, volume: f64) -> Candle {
        Candle {
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// A bar whose extremes sit far from the open, for the market no-lookahead test.
    fn bar(open: f64, volume: f64) -> Candle {
        ohlc(open, open + 50.0, open - 50.0, open + 30.0, volume)
    }

    fn model(slip_bps: f64, cap: f64) -> FillModel {
        FillModel {
            slippage_bps: Bps(slip_bps),
            max_fill_fraction: cap,
            impact: 0.0,
        }
    }

    fn market(side: Side, size: f64) -> Order {
        Order::market(OrderId(0), side, Qty(size), false)
    }

    fn limit(side: Side, size: f64, price: f64) -> Order {
        Order::limit(
            OrderId(0),
            side,
            Qty(size),
            Price(price),
            TimeInForce::Gtc,
            false,
            false,
        )
    }

    fn stop(side: Side, size: f64, trigger: f64) -> Order {
        Order::stop(OrderId(0), side, Qty(size), Price(trigger), false)
    }

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    // market

    #[test]
    fn buy_market_fills_at_open_plus_slippage() {
        let f = model(10.0, 1.0)
            .fill_market(&market(Side::Buy, 1.0), &bar(100.0, 1000.0))
            .unwrap();
        assert!(approx(f.price.get(), 100.1));
        assert!(f.is_taker);
    }

    #[test]
    fn sell_market_fills_at_open_minus_slippage() {
        let f = model(10.0, 1.0)
            .fill_market(&market(Side::Sell, 1.0), &bar(100.0, 1000.0))
            .unwrap();
        assert!(approx(f.price.get(), 99.9));
    }

    #[test]
    fn market_uses_open_not_intrabar_extremes() {
        let b = bar(100.0, 1000.0); // high 150, low 50, close 130
        let buy = model(0.0, 1.0)
            .fill_market(&market(Side::Buy, 1.0), &b)
            .unwrap();
        let sell = model(0.0, 1.0)
            .fill_market(&market(Side::Sell, 1.0), &b)
            .unwrap();
        assert_eq!(buy.price.get(), 100.0);
        assert_eq!(sell.price.get(), 100.0);
    }

    #[test]
    fn volume_cap_limits_market_size() {
        let f = model(0.0, 0.1)
            .fill_market(&market(Side::Buy, 250.0), &bar(100.0, 1000.0))
            .unwrap();
        assert_eq!(f.size, Qty(100.0));
    }

    #[test]
    fn market_impact_scales_slippage_with_the_volume_fraction() {
        // impact 0.5, no flat slippage: a buy of 100 into a 1000-volume bar takes
        // 10% of it, so slip = 0.5 * 0.1 = 0.05; price = 100 * 1.05 = 105.
        let f = FillModel {
            slippage_bps: Bps(0.0),
            max_fill_fraction: 1.0,
            impact: 0.5,
        };
        let fill = f
            .fill_market(&market(Side::Buy, 100.0), &bar(100.0, 1000.0))
            .unwrap();
        assert!(approx(fill.price.get(), 105.0));
    }

    #[test]
    fn zero_volume_bar_fills_nothing() {
        assert!(model(0.0, 1.0)
            .fill_market(&market(Side::Buy, 1.0), &bar(100.0, 0.0))
            .is_none());
    }

    #[test]
    fn a_non_finite_fill_fraction_does_not_remove_the_cap() {
        // f64::min keeps the other operand when one side is NaN, so an unguarded
        // NaN or inf fraction would let one order take the whole order size
        // regardless of what the bar traded (ADR 0024)
        for fraction in [f64::NAN, f64::INFINITY] {
            let f =
                model(0.0, fraction).fill_market(&market(Side::Buy, 10_000.0), &bar(100.0, 5.0));
            assert!(
                f.is_none(),
                "fraction {fraction} filled against a 5-volume bar"
            );
        }
    }

    #[test]
    fn a_negative_volume_bar_fills_nothing() {
        assert!(model(0.0, 1.0)
            .fill_market(&market(Side::Buy, 1.0), &bar(100.0, -5.0))
            .is_none());
    }

    #[test]
    fn a_negative_slippage_cannot_improve_a_fill() {
        // slippage is adverse by definition (ADR 0004); a negative rate would fill
        // a buy below the open and mint equity out of the cost model
        let f = model(-20_000.0, 1.0)
            .fill_market(&market(Side::Buy, 1.0), &bar(100.0, 1000.0))
            .unwrap();
        assert_eq!(f.price.get(), 100.0);
        let s = model(-20_000.0, 1.0)
            .fill_market(&market(Side::Sell, 1.0), &bar(100.0, 1000.0))
            .unwrap();
        assert_eq!(s.price.get(), 100.0);
    }

    #[test]
    fn a_sell_is_never_priced_at_or_below_zero() {
        // a huge flat slip or a huge impact would otherwise reach 1.0 and price the
        // fill at zero, or beyond it and price it negative (ADR 0024)
        let flat = model(50_000.0, 1.0)
            .fill_market(&market(Side::Sell, 1.0), &bar(100.0, 1000.0))
            .unwrap();
        assert!(flat.price.get() > 0.0);

        let heavy = FillModel {
            slippage_bps: Bps(0.0),
            max_fill_fraction: 1.0,
            impact: 5.0,
        };
        let f = heavy
            .fill_market(&market(Side::Sell, 1000.0), &bar(100.0, 1000.0))
            .unwrap();
        assert!(f.price.get() > 0.0, "priced at {}", f.price.get());
    }

    // limit

    #[test]
    fn buy_limit_fills_when_bar_dips_to_it() {
        let f = model(50.0, 1.0)
            .fill_limit(
                &limit(Side::Buy, 1.0, 100.0),
                &ohlc(101.0, 102.0, 98.0, 101.0, 1000.0),
            )
            .unwrap();
        assert_eq!(f.price.get(), 100.0); // at the limit
        assert!(!f.is_taker); // maker, no slippage despite slippage_bps
    }

    #[test]
    fn buy_limit_untouched_does_not_fill() {
        assert!(model(0.0, 1.0)
            .fill_limit(
                &limit(Side::Buy, 1.0, 100.0),
                &ohlc(102.0, 103.0, 101.0, 102.0, 1000.0)
            )
            .is_none());
    }

    #[test]
    fn sell_limit_fills_when_bar_rises_to_it() {
        let f = model(0.0, 1.0)
            .fill_limit(
                &limit(Side::Sell, 1.0, 100.0),
                &ohlc(99.0, 101.0, 98.0, 100.0, 1000.0),
            )
            .unwrap();
        assert_eq!(f.price.get(), 100.0);
    }

    #[test]
    fn buy_limit_gap_through_still_fills_at_limit_never_better() {
        // limit 100; bar gaps down, opens 95 (low 90). Fills at 100, not 95.
        let f = model(0.0, 1.0)
            .fill_limit(
                &limit(Side::Buy, 1.0, 100.0),
                &ohlc(95.0, 96.0, 90.0, 95.0, 1000.0),
            )
            .unwrap();
        assert_eq!(f.price.get(), 100.0);
        // and it is marketable at that open, so it takes rather than makes
        assert!(f.is_taker);
    }

    #[test]
    fn a_limit_pays_maker_or_taker_by_where_the_bar_opened() {
        let m = model(0.0, 1.0);
        // the bar opens away from the limit and has to come to it: a maker
        let made = m
            .fill_limit(
                &limit(Side::Buy, 1.0, 100.0),
                &ohlc(101.0, 102.0, 98.0, 101.0, 1000.0),
            )
            .unwrap();
        assert!(!made.is_taker);
        // the bar opens at the limit, so it is lifted on arrival: a taker. This is
        // the straddle case, where a buy and a sell at one price both fill on one
        // bar and used to collect two maker rebates for a flat position (ADR 0045)
        let bar = ohlc(100.0, 101.0, 99.0, 100.0, 1000.0);
        let bought = m.fill_limit(&limit(Side::Buy, 1.0, 100.0), &bar).unwrap();
        let sold = m.fill_limit(&limit(Side::Sell, 1.0, 100.0), &bar).unwrap();
        assert!(bought.is_taker);
        assert!(sold.is_taker);
    }

    #[test]
    fn volume_cap_limits_limit_size() {
        let f = model(0.0, 0.1)
            .fill_limit(
                &limit(Side::Buy, 250.0, 100.0),
                &ohlc(101.0, 102.0, 98.0, 101.0, 1000.0),
            )
            .unwrap();
        assert_eq!(f.size, Qty(100.0));
    }

    // stop

    #[test]
    fn sell_stop_triggers_and_fills_at_trigger_with_slippage() {
        // sell stop 90; bar low 88 triggers, no gap (open 95). Fill at 90 - slip.
        let f = model(10.0, 1.0)
            .fill_stop(
                &stop(Side::Sell, 1.0, 90.0),
                &ohlc(95.0, 96.0, 88.0, 92.0, 1000.0),
            )
            .unwrap();
        assert!(approx(f.price.get(), 90.0 * 0.999));
        assert!(f.is_taker);
    }

    #[test]
    fn sell_stop_gap_fills_at_the_worse_open() {
        // sell stop 90; bar gaps down, opens 85 (worse). Fill at 85 - slip.
        let f = model(10.0, 1.0)
            .fill_stop(
                &stop(Side::Sell, 1.0, 90.0),
                &ohlc(85.0, 86.0, 84.0, 85.0, 1000.0),
            )
            .unwrap();
        assert!(approx(f.price.get(), 85.0 * 0.999));
    }

    #[test]
    fn buy_stop_triggers_at_trigger_with_slippage() {
        // buy stop 110; bar high 112 triggers, no gap (open 105). Fill at 110 + slip.
        let f = model(10.0, 1.0)
            .fill_stop(
                &stop(Side::Buy, 1.0, 110.0),
                &ohlc(105.0, 112.0, 104.0, 111.0, 1000.0),
            )
            .unwrap();
        assert!(approx(f.price.get(), 110.0 * 1.001));
    }

    #[test]
    fn buy_stop_gap_fills_at_the_worse_open() {
        // buy stop 110; bar gaps up, opens 115. Fill at 115 + slip.
        let f = model(10.0, 1.0)
            .fill_stop(
                &stop(Side::Buy, 1.0, 110.0),
                &ohlc(115.0, 116.0, 114.0, 115.0, 1000.0),
            )
            .unwrap();
        assert!(approx(f.price.get(), 115.0 * 1.001));
    }

    #[test]
    fn untriggered_stop_does_not_fill() {
        assert!(model(0.0, 1.0)
            .fill_stop(
                &stop(Side::Sell, 1.0, 90.0),
                &ohlc(95.0, 96.0, 92.0, 94.0, 1000.0)
            )
            .is_none());
    }

    // post_only cross check

    #[test]
    fn limit_crosses_detects_marketable_orders() {
        let m = model(0.0, 1.0);
        assert!(m.limit_crosses(Side::Buy, Price(105.0), Price(100.0))); // buy above market
        assert!(!m.limit_crosses(Side::Buy, Price(95.0), Price(100.0))); // buy below: maker
        assert!(m.limit_crosses(Side::Sell, Price(95.0), Price(100.0))); // sell below market
        assert!(!m.limit_crosses(Side::Sell, Price(105.0), Price(100.0))); // sell above: maker
    }
}
