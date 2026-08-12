//! The single-env step state machine. `reset()` positions at the first bar;
//! `step()` advances exactly one bar, resolves the orders placed since the last
//! step against the NEW bar, marks the account, and returns the new `State`. An
//! order decided on one bar fills on the next, never the bar the decision saw.
//!
//! Within a step the order of events follows the bar: pending market orders fill
//! at the open, resting limit and stop orders fill against the range, funding is
//! charged at each interval boundary, then liquidation is checked at the bar's
//! adverse extreme, then equity is marked at the close.

use emsl_core::{
    Account, Bps, Candle, CostModel, Fill, FlatCostModel, Market, Order, OrderId, OrderType, Price,
    Qty, RestingOrderBook, Side, State, TimeInForce,
};

use crate::candles::Candles;
use crate::execution::FillModel;
use crate::reporter::{Reporter, Trade};
use crate::stats::Stats;

/// A position moved toward zero by less than this counts as no close, mirroring
/// the position's own dust epsilon so a non-representable fractional close does
/// not log a phantom trade.
const CLOSE_EPS: f64 = 1e-9;

/// Construction parameters for an engine.
#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    pub market: Market,
    pub quote: f64,
    pub fee_taker: f64,
    pub fee_maker: f64,
    pub slippage_bps: f64,
    pub max_fill_fraction: f64,
    pub max_open_orders: usize,
    pub report: bool,
    /// Perp margin cap: a fill may not grow the position past `max_leverage`
    /// times equity in notional. Zero means no cap; the shipped default is a
    /// finite 10x (ADR 0012). Ignored on spot.
    pub max_leverage: f64,
    /// Market-impact coefficient for taker fills (ADR 0013). Zero disables it.
    pub impact: f64,
    /// Perp funding rate charged per event on the position notional; a long pays a
    /// positive rate, a short receives (ADR 0017). Ignored on spot.
    pub funding_rate: f64,
    /// Bars between funding events. Zero disables funding (ADR 0017).
    pub funding_interval: usize,
}

impl Default for EngineConfig {
    fn default() -> EngineConfig {
        EngineConfig {
            market: Market::Spot,
            quote: 10_000.0,
            fee_taker: 0.0006,
            fee_maker: 0.0002,
            slippage_bps: 0.0,
            max_fill_fraction: 1.0,
            max_open_orders: 8,
            report: false,
            max_leverage: 10.0,
            impact: 0.0,
            funding_rate: 0.0,
            funding_interval: 0,
        }
    }
}

/// The single-env bar engine: candles, an account, a resting book, the fill and
/// cost models, and an optional reporter.
pub struct Engine {
    // Set once in `new` and never reassigned or mutated. emsl-py hands out
    // zero-copy, read-only numpy views into this buffer (ADR 0008), so its
    // address and contents must stay stable for the engine's whole life; do not
    // add a method that reallocates it or vends a mutable candle view.
    candles: Candles,
    config: EngineConfig,
    fill_model: FillModel,
    cost: FlatCostModel,
    account: Account,
    book: RestingOrderBook,
    reporter: Option<Reporter>,
    pending: Vec<Order>,
    tick: usize,
    id_counter: u64,
    /// The tick at which the current position first became non-flat, used to
    /// stamp a closed trade's holding time. Meaningful only while a position is open.
    position_entry_tick: usize,
    /// The fee already paid to open the position now standing, in quote. A close
    /// consumes the share belonging to the size it closes, so a completed trade can
    /// carry its whole round-trip cost rather than the exit side alone (ADR 0030).
    /// Zero whenever the position is flat.
    open_fee: f64,
    /// Fills applied since the last reset. A run whose orders never filled, on a
    /// series with no volume say, is otherwise indistinguishable from one that never
    /// placed an order (ADR 0031).
    fills: usize,
    /// Funding paid since the last reset, in quote, positive when paid away. The
    /// account returned each payment and this is what keeps it (ADR 0017).
    funding_paid: f64,
    /// Set when the account is dead: force-closed by a perp liquidation, or marked
    /// at a non-positive equity on either market, which is terminal too (ADR 0019).
    /// A terminal event for the RL env. Cleared on reset.
    bust: bool,
}

impl Engine {
    /// Build an engine over `candles`. Panics if `candles` is empty, since a
    /// zero-bar series has no bar to reset or step to. Call `reset()` to get the
    /// first state.
    pub fn new(candles: Candles, config: EngineConfig) -> Engine {
        assert!(
            !candles.is_empty(),
            "engine requires a non-empty candle series"
        );
        Engine {
            fill_model: FillModel {
                slippage_bps: Bps(config.slippage_bps),
                max_fill_fraction: config.max_fill_fraction,
                impact: config.impact,
            },
            cost: FlatCostModel {
                fee_taker: config.fee_taker,
                fee_maker: config.fee_maker,
            },
            account: Account::new(config.market, config.quote),
            book: RestingOrderBook::new(config.max_open_orders),
            reporter: if config.report {
                Some(Reporter::new())
            } else {
                None
            },
            pending: Vec::new(),
            tick: 0,
            id_counter: 1,
            position_entry_tick: 0,
            open_fee: 0.0,
            fills: 0,
            funding_paid: 0.0,
            bust: false,
            candles,
            config,
        }
    }

    /// Reset to the first bar and a fresh account, returning the initial state.
    pub fn reset(&mut self) -> State {
        self.reset_at(0)
    }

    /// Reset the account and cursor, starting at `start_tick` instead of the first
    /// bar. The vectorized env uses this to give each env its own random start
    /// offset. `start_tick` is clamped to the last bar.
    pub fn reset_at(&mut self, start_tick: usize) -> State {
        self.account = Account::new(self.config.market, self.config.quote);
        self.book.cancel_all();
        self.pending.clear();
        self.tick = start_tick.min(self.candles.len().saturating_sub(1));
        // The id counter deliberately survives a reset. Restarting it made an id
        // handed out in one episode name a live order in the next, so a handle
        // carried across a reset cancelled somebody else's order (ADR 0028).
        self.position_entry_tick = self.tick;
        self.open_fee = 0.0;
        self.fills = 0;
        self.funding_paid = 0.0;
        self.bust = false;
        if self.reporter.is_some() {
            self.reporter = Some(Reporter::new());
        }
        self.state()
    }

    /// True when the account died since the last reset, by a perp liquidation or by
    /// equity reaching zero on either market. The RL env's terminal signal (ADR 0019).
    /// The engine does not stop on it; a driver that wants to halt checks it.
    pub fn is_bust(&self) -> bool {
        self.bust
    }

    /// The current state without advancing, for reading between steps.
    pub fn current_state(&self) -> State {
        self.state()
    }

    /// The current bar index (cursor position).
    pub fn tick(&self) -> usize {
        self.tick
    }

    /// True when there is no next bar to step into.
    pub fn done(&self) -> bool {
        self.tick + 1 >= self.candles.len()
    }

    /// The number of candles the engine holds.
    pub fn num_bars(&self) -> usize {
        self.candles.len()
    }

    /// The whole candle series as a slice, for a zero-copy full-series view. Like
    /// `candle_window`, it borrows the immutable `Arc` buffer with no copy.
    pub fn candles_all(&self) -> &[Candle] {
        self.candles.as_slice()
    }

    /// The `lookback` bars ending at (and including) the current tick, as a
    /// zero-copy borrow into the shared series. Clamped to what exists, so early
    /// in the series it returns the available bars. The Python layer turns this
    /// into a read-only numpy view (ADR 0008).
    pub fn candle_window(&self, lookback: usize) -> &[Candle] {
        self.candles.window(self.tick + 1, lookback)
    }

    fn next_id(&mut self) -> OrderId {
        let id = OrderId(self.id_counter);
        self.id_counter += 1;
        id
    }

    fn current_close(&self) -> f64 {
        self.candles.get(self.tick).map_or(0.0, |c| c.close)
    }

    fn place_market(
        &mut self,
        side: Side,
        size: f64,
        reduce_only: bool,
        tif: TimeInForce,
    ) -> Option<OrderId> {
        // The queue holds the same number of orders the resting book does. It was
        // unbounded, and since the volume cap is per order (ADR 0005), one order
        // split into a hundred took a hundred times the cap against a single bar,
        // every slice priced as though it were the only participant. Bounding it
        // does not make the cap honest, it makes the evasion finite; a liquidity
        // budget shared across a bar is the real answer (ADR 0047)
        if self.pending.len() >= self.config.max_open_orders {
            return None;
        }
        let id = self.next_id();
        let mut order = Order::market(id, side, Qty(size), reduce_only);
        order.tif = tif;
        self.pending.push(order);
        Some(id)
    }

    fn place_limit(
        &mut self,
        side: Side,
        size: f64,
        price: f64,
        reduce_only: bool,
        post_only: bool,
        tif: TimeInForce,
    ) -> Option<OrderId> {
        // A size that is not a positive finite number can never fill (ADR 0001), so
        // it is refused a slot rather than resting inert forever. A non-finite price
        // is refused for the same reason: every touch test against it compares with
        // NaN and is false, so the order would camp on a slot until the book filled
        // up and later placements were silently rejected (ADR 0027).
        if !size.is_finite() || size <= 0.0 || !price.is_finite() {
            return None;
        }
        // A post_only limit that would cross the prevailing price is rejected.
        if post_only
            && self
                .fill_model
                .limit_crosses(side, Price(price), Price(self.current_close()))
        {
            return None;
        }
        let id = self.next_id();
        let order = Order::limit(
            id,
            side,
            Qty(size),
            Price(price),
            tif,
            post_only,
            reduce_only,
        );
        self.book.place(order).ok()
    }

    fn place_stop(
        &mut self,
        side: Side,
        size: f64,
        trigger: f64,
        reduce_only: bool,
    ) -> Option<OrderId> {
        // The same guard as a limit: a non-positive or non-finite size never fills,
        // and a non-finite trigger never triggers (ADRs 0001, 0027).
        if !size.is_finite() || size <= 0.0 || !trigger.is_finite() {
            return None;
        }
        let id = self.next_id();
        let order = Order::stop(id, side, Qty(size), Price(trigger), reduce_only);
        self.book.place(order).ok()
    }

    /// The one order primitive the typed shortcuts wrap. Dispatches on `kind`: a
    /// market ignores `price` and `trigger`, a limit needs a `price`, a stop needs
    /// a `trigger`. `tif` applies to market and limit orders; a stop rests until it
    /// triggers (ADR 0016). `None` if the book is full, a post_only limit would
    /// cross, or a limit has no price or a stop no trigger.
    #[allow(clippy::too_many_arguments)]
    pub fn order(
        &mut self,
        side: Side,
        size: f64,
        kind: OrderType,
        price: Option<f64>,
        trigger: Option<f64>,
        reduce_only: bool,
        post_only: bool,
        tif: TimeInForce,
    ) -> Option<OrderId> {
        match kind {
            OrderType::Market => self.place_market(side, size, reduce_only, tif),
            OrderType::Limit => self.place_limit(side, size, price?, reduce_only, post_only, tif),
            OrderType::Stop => self.place_stop(side, size, trigger?, reduce_only),
        }
    }

    /// Queue a market buy; it fills on the next bar's open. `None` if the queue is
    /// already holding `max_open_orders` for this bar.
    pub fn market_buy(&mut self, size: f64) -> Option<OrderId> {
        self.place_market(Side::Buy, size, false, TimeInForce::Ioc)
    }

    /// Queue a market sell; it fills on the next bar's open. `None` if the queue is
    /// already holding `max_open_orders` for this bar.
    pub fn market_sell(&mut self, size: f64) -> Option<OrderId> {
        self.place_market(Side::Sell, size, false, TimeInForce::Ioc)
    }

    /// Rest a buy limit; it fills when a later bar trades down to `price`. `None`
    /// if the book is full.
    pub fn limit_buy(&mut self, size: f64, price: f64) -> Option<OrderId> {
        self.place_limit(Side::Buy, size, price, false, false, TimeInForce::Gtc)
    }

    /// Rest a sell limit; it fills when a later bar trades up to `price`.
    pub fn limit_sell(&mut self, size: f64, price: f64) -> Option<OrderId> {
        self.place_limit(Side::Sell, size, price, false, false, TimeInForce::Gtc)
    }

    /// Rest a stop that becomes a market order once `trigger` is crossed. `None`
    /// if the book is full or the size or trigger is not finite. `reduce_only`
    /// makes it a protective stop that can only shrink the position, never open
    /// one on the other side; a stop-loss wants it true (ADR 0028).
    pub fn stop(
        &mut self,
        side: Side,
        size: f64,
        trigger: f64,
        reduce_only: bool,
    ) -> Option<OrderId> {
        self.place_stop(side, size, trigger, reduce_only)
    }

    /// Base size for a fraction of current equity, marked at the current close.
    pub fn qty_from_weight(&self, fraction: f64) -> f64 {
        self.account
            .qty_from_weight(fraction, Price(self.current_close()))
            .get()
    }

    /// Base size for a cash amount in quote, at the current close.
    pub fn qty_from_quote(&self, cash: f64) -> f64 {
        self.account
            .qty_from_quote(cash, Price(self.current_close()))
            .get()
    }

    /// Cancel a resting order by id. True if it was found and removed.
    pub fn cancel(&mut self, id: OrderId) -> bool {
        self.book.cancel(id).is_some()
    }

    /// Move a resting order: cancel `id` and rest a replacement carrying the same
    /// side, kind and flags, with whichever of `size`, `price` and `trigger` are
    /// given. Returns the new id.
    ///
    /// `None` when `id` is not resting, and in that case NOTHING is placed. That is
    /// the point of it. Re-placing a trailing stop with `stop()` each bar rests a new
    /// order every time, so the one that fills leaves its siblings live, and on a
    /// perp those open a position on the other side; a trail written with `replace`
    /// cannot leave two orders alive, and once the stop has filled the replacement
    /// silently does nothing instead of arming a fresh one (ADR 0032).
    pub fn replace(
        &mut self,
        id: OrderId,
        size: Option<f64>,
        price: Option<f64>,
        trigger: Option<f64>,
    ) -> Option<OrderId> {
        let old = self.book.cancel(id)?;
        let size = size.unwrap_or(old.remaining().get());
        match old.kind {
            OrderType::Limit => self.place_limit(
                old.side,
                size,
                price.or(old.price.map(|p| p.get()))?,
                old.reduce_only,
                old.post_only,
                old.tif,
            ),
            OrderType::Stop => self.place_stop(
                old.side,
                size,
                trigger.or(old.trigger.map(|t| t.get()))?,
                old.reduce_only,
            ),
            // a market order never rests, so it can never be found to replace
            OrderType::Market => None,
        }
    }

    /// Cancel every resting order, returning how many were dropped. Pending market
    /// orders are not resting and fill on the next step regardless.
    pub fn cancel_all(&mut self) -> usize {
        let n = self.book.iter().count();
        self.book.cancel_all();
        n
    }

    /// Flatten the position with a reduce-only market order. `None` when flat.
    pub fn close(&mut self) -> Option<OrderId> {
        let pos = self.account.position.qty.get();
        if pos == 0.0 {
            return None;
        }
        let (side, size) = if pos > 0.0 {
            (Side::Sell, pos)
        } else {
            (Side::Buy, -pos)
        };
        self.place_market(side, size, true, TimeInForce::Ioc)
    }

    /// Advance one bar and resolve the bar, then mark and return the state. At the
    /// end of the data the tick does not advance and the state is returned as is.
    pub fn step(&mut self) -> State {
        let advanced = self.tick + 1 < self.candles.len();
        if advanced {
            self.tick += 1;
            let bar = self.candles.get(self.tick).expect("tick in range");

            // 1. pending market orders fill at the open
            let pending = std::mem::take(&mut self.pending);
            for order in pending {
                if let Some(fill) = self.fill_model.fill_market(&order, &bar) {
                    // FOK is all or nothing against everything that can shrink the
                    // fill, not just the bar's liquidity: the margin cap and the
                    // spot clamps are applied to a copy first, so an order that
                    // could only fill in part books nothing at all (ADR 0025).
                    if order.tif == TimeInForce::Fok
                        && self.clamp_fill(order.reduce_only, &fill).size.get() + CLOSE_EPS
                            < order.size.get()
                    {
                        continue;
                    }
                    self.apply_fill_clamped(order.reduce_only, fill);
                }
            }

            // 2. resting limit and stop orders fill against the bar range
            self.resolve_resting(&bar);

            // 3. funding at each interval boundary, on the position held at the
            //    funding event (so after this bar's fills, not the position carried
            //    into the bar), marked at the close, and before liquidation so a
            //    funding debit can bust the account this bar (ADR 0017)
            if self.config.funding_interval > 0 && self.tick % self.config.funding_interval == 0 {
                self.funding_paid += self
                    .account
                    .apply_funding(self.config.funding_rate, Price(bar.close));
            }

            // 4. liquidation at the bar's adverse extreme (long at the low, short
            //    at the high)
            let adverse = if self.account.position.qty.get() >= 0.0 {
                bar.low
            } else {
                bar.high
            };
            if self.account.is_bust_at(Price(adverse)) {
                self.liquidate();
                self.bust = true;
            }
        }

        let state = self.state();
        // A dead account is terminal, marked at the close, for both markets: perp
        // liquidation force-closes above, but a spot account (or a perp drained by
        // fees or funding) can still reach zero equity without a forced close, and
        // that is a true terminal, not a truncation (ADR 0019).
        if state.equity <= 0.0 {
            self.bust = true;
        }
        // Record only on a real advance, so a driver that keeps calling step past
        // the last bar does not re-append the same equity point and distort stats.
        if advanced {
            if let Some(reporter) = &mut self.reporter {
                reporter.record_equity(state.equity, state.position != 0.0);
            }
        }
        state
    }

    /// Clamp a fill so it does not grow the position past the perp margin cap
    /// (`max_leverage` times equity in notional). Reductions are never capped, and
    /// a position already over the cap (from an equity drop) is not force-reduced,
    /// only blocked from growing. No-op on spot or with no cap set.
    fn cap_leverage(&self, fill: &mut Fill) {
        if self.config.market != Market::Perp || self.config.max_leverage <= 0.0 {
            return;
        }
        let mark = fill.price.get();
        if !mark.is_finite() || mark <= 0.0 {
            return;
        }
        let equity = self.account.equity(Price(mark));
        // No equity backs new notional, so an insolvent account may only shrink. It
        // used to be waved through here on the grounds that liquidation handles it,
        // but liquidation cannot force-close an account that is flat, so a position
        // opened from negative equity was closed at the bar's adverse extreme and
        // deepened the bad debt instead of being refused (ADR 0026).
        let max_size = if equity > 0.0 {
            self.config.max_leverage * equity / mark
        } else {
            0.0
        };
        let pos = self.account.position.qty.get();
        let new_pos = pos + fill.side.sign() * fill.size.get();
        // A same-side grow may keep an existing over-cap position (from an equity
        // drop) but not extend it, so the allowance is the larger of the cap and
        // the current size. A flip closes the old side and opens a fresh one, which
        // must fit the cap outright, so it gets no such allowance (ADR 0012).
        let flipping = pos != 0.0 && new_pos != 0.0 && (new_pos > 0.0) != (pos > 0.0);
        let max_abs = if flipping {
            max_size
        } else {
            max_size.max(pos.abs())
        };
        if new_pos.abs() > max_abs {
            let capped_new = max_abs * new_pos.signum();
            fill.size = Qty((capped_new - pos).abs());
        }
    }

    /// On spot, keep the position from going below zero: a sell may not exceed the
    /// current long, so it can flatten but never open or extend a short (ADR 0015).
    /// A sell from flat is clamped to nothing. No-op on perp, where a short is a
    /// normal signed position, and on a buy.
    fn cap_spot_short(&self, fill: &mut Fill) {
        if self.config.market != Market::Spot || fill.side != Side::Sell {
            return;
        }
        let pos = self.account.position.qty.get();
        let sellable = pos.max(0.0);
        if fill.size.get() > sellable {
            fill.size = Qty(sellable);
        }
    }

    /// On spot, keep a buy within the cash on hand: its cost, notional plus fee,
    /// may not exceed the quote balance, so the balance never goes negative. This
    /// is the buy-side mirror of `cap_spot_short`; spot has no borrow and no margin,
    /// so credit is refused rather than pretended (ADR 0018). No-op on perp, where
    /// the margin cap governs, and on a sell.
    fn cap_spot_buy(&self, fill: &mut Fill) {
        if self.config.market != Market::Spot || fill.side != Side::Buy {
            return;
        }
        let price = fill.price.get();
        if !price.is_finite() || price <= 0.0 {
            return;
        }
        if self.account.quote <= 0.0 {
            fill.size = Qty(0.0);
            return;
        }
        let fee_rate = if fill.is_taker {
            self.cost.fee_taker
        } else {
            self.cost.fee_maker
        };
        let affordable = self.account.quote / (price * (1.0 + fee_rate));
        if fill.size.get() > affordable {
            fill.size = Qty(affordable);
        }
    }

    /// Every risk clamp applied to a copy of `fill` without booking anything: the
    /// perp margin cap, the spot short and cash clamps, then reduce_only. Returns
    /// the fill that would actually be booked, so an all-or-nothing order can be
    /// judged against the real fillable size rather than the raw liquidity the
    /// fill model offered (ADR 0025). A refused fill comes back sized zero.
    fn clamp_fill(&self, reduce_only: bool, fill: &Fill) -> Fill {
        let mut fill = *fill;
        // size is a positive magnitude; a non-positive or non-finite fill is not a
        // real trade, so it moves nothing and pays no fee
        let size = fill.size.get();
        if !size.is_finite() || size <= 0.0 || !fill.price.get().is_finite() {
            fill.size = Qty(0.0);
            return fill;
        }
        self.cap_leverage(&mut fill);
        self.cap_spot_short(&mut fill);
        self.cap_spot_buy(&mut fill);
        if reduce_only {
            let pos = self.account.position.qty.get();
            let reduces =
                (pos > 0.0 && fill.side == Side::Sell) || (pos < 0.0 && fill.side == Side::Buy);
            if !reduces {
                fill.size = Qty(0.0);
                return fill;
            }
            fill.size = Qty(fill.size.get().min(pos.abs()));
        }
        // a clamp can leave NaN or a negative remainder; normalize so the caller only
        // has to test for a positive size. The gate is the dust epsilon, not zero,
        // because the position refuses anything at or below it: a size in between
        // moved no cash and no position, but still counted a fill and banked a fee,
        // which shows up as a phantom fill in the dead-feed canary and as drift in
        // the round-trip fee identity (ADRs 0030, 0031)
        let clamped = fill.size.get();
        if clamped.is_nan() || clamped <= CLOSE_EPS {
            fill.size = Qty(0.0);
        }
        fill
    }

    /// Force-close the whole position at the price that leaves nothing.
    ///
    /// Not at the bar's adverse extreme, which is what the trigger was tested
    /// against. Closing there charged the account for however far the bar happened
    /// to run past the point its margin was gone, so a long on 100 of margin
    /// against a bar that wicked to 80 booked a loss of 200 and left the account
    /// owing 100. A position cannot lose more than the margin behind it, so the
    /// exit is priced where the margin runs out, and the liquidation fee is inside
    /// that price rather than charged past it (ADR 0052).
    fn liquidate(&mut self) {
        let price = match self.account.bankruptcy_price(self.cost.fee_taker) {
            Some(price) => price,
            None => return,
        };
        let size = self.account.position.qty.get().abs();
        let fill = Fill {
            side: if self.account.position.qty.get() > 0.0 {
                Side::Sell
            } else {
                Side::Buy
            },
            size: Qty(size),
            price,
            is_taker: true,
        };
        let fee = self.cost.fee(&fill);
        let before = self.before_fill();
        self.account.apply_fill(&fill, fee);
        self.book_fill(size, price.get(), fee, before, true);
    }

    /// The position fields a fill has to be compared against once it has landed.
    fn before_fill(&self) -> (f64, f64, f64) {
        (
            self.account.position.qty.get(),
            self.account.position.avg_entry.get(),
            self.account.realized(),
        )
    }

    fn apply_fill_clamped(&mut self, reduce_only: bool, fill: Fill) -> f64 {
        let fill = self.clamp_fill(reduce_only, &fill);
        if fill.size.get() <= 0.0 {
            return 0.0;
        }
        let applied = fill.size.get();
        let fee = self.cost.fee(&fill);
        let before = self.before_fill();
        self.account.apply_fill(&fill, fee);
        self.book_fill(applied, fill.price.get(), fee, before, false);
        applied
    }

    /// Book what a fill just did to the account: count it, log whatever it closed
    /// as a trade, carry the entry-side fee of whatever is left, and stamp a new
    /// position's open tick.
    ///
    /// The single place this happens, which is the point. A liquidation used to
    /// close the position on the account directly and reach none of it, so the
    /// forced close appeared in no trade row, counted toward no fill, and left the
    /// dead position's entry fee standing to be charged against the NEXT trade
    /// (ADRs 0030, 0031).
    fn book_fill(
        &mut self,
        applied: f64,
        exit_price: f64,
        fee: f64,
        before: (f64, f64, f64),
        liquidated: bool,
    ) {
        let (pos_before, entry_before, realized_before) = before;
        self.fills += 1;
        let pos_after = self.account.position.qty.get();

        // The base amount the fill moved the position toward zero: a reduce keeps
        // the same-side remainder, a close or flip keeps none of the old side.
        let kept = if (pos_before > 0.0) == (pos_after > 0.0) {
            pos_after.abs()
        } else {
            0.0
        };
        // Clamped at zero: a same-side add keeps MORE than it started with, so the
        // raw difference goes negative and `opened` would then exceed the fill,
        // charging the entry fee twice over. Nothing is closed by an add.
        let closed = (pos_before.abs() - kept).max(0.0);
        let opened = applied - closed;

        // Split this fill's fee between the part that closed (the exit side of a
        // completed trade) and the part that opened (the entry side of the position
        // now standing, to be charged when it closes later).
        let exit_fee = if applied > 0.0 {
            fee * closed / applied
        } else {
            0.0
        };
        // The share of the position's own entry fee that this close consumes.
        let entry_fee = if closed > CLOSE_EPS && pos_before.abs() > CLOSE_EPS {
            self.open_fee * closed / pos_before.abs()
        } else {
            0.0
        };

        if closed > CLOSE_EPS && self.reporter.is_some() {
            let pnl = self.account.realized() - realized_before;
            // The whole round trip: the fee paid to open this size and the fee paid
            // to close it. Charging only the exit side made a strategy whose edge is
            // smaller than its costs look profitable per trade (ADR 0030).
            let fees = entry_fee + exit_fee;
            let trade = Trade {
                entry_tick: self.position_entry_tick,
                exit_tick: self.tick,
                side: if pos_before > 0.0 {
                    Side::Buy
                } else {
                    Side::Sell
                },
                size: closed,
                entry_price: entry_before,
                exit_price,
                fees,
                pnl,
                net_pnl: pnl - fees,
                bars_held: self.tick.saturating_sub(self.position_entry_tick),
                liquidated,
            };
            if let Some(reporter) = self.reporter.as_mut() {
                reporter.record_trade(trade);
            }
        }

        // Carry the entry-side fee of whatever position is left: what this close
        // consumed is gone, and the part of this fill that opened size pays in.
        self.open_fee = if pos_after.abs() <= CLOSE_EPS {
            0.0
        } else if applied > 0.0 {
            (self.open_fee - entry_fee) + fee * opened / applied
        } else {
            self.open_fee - entry_fee
        };

        // Stamp the open tick whenever a new side is established, from flat or
        // through a flip, so the next close measures its holding time from here.
        let opened_new_side =
            pos_after != 0.0 && (pos_before == 0.0 || (pos_before > 0.0) != (pos_after > 0.0));
        if opened_new_side {
            self.position_entry_tick = self.tick;
        }
    }

    /// Resolve the resting book against `bar`, in slot order.
    ///
    /// A stop is consumed only when it produces a fill. On a bar that trades no
    /// volume the trigger may be crossed and nothing can fill, and treating that as
    /// a trigger would delete a stop-loss precisely on the illiquid data where it
    /// matters; it stays armed for the next bar that can trade instead (ADR 0035).
    fn resolve_resting(&mut self, bar: &Candle) {
        let mut fills: Vec<(OrderId, Fill, OrderType, bool, TimeInForce, f64)> = Vec::new();
        for order in self.book.iter() {
            let fill = match order.kind {
                OrderType::Limit => self.fill_model.fill_limit(order, bar),
                OrderType::Stop => self.fill_model.fill_stop(order, bar),
                OrderType::Market => None,
            };
            if let Some(fill) = fill {
                fills.push((
                    order.id,
                    fill,
                    order.kind,
                    order.reduce_only,
                    order.tif,
                    order.remaining().get(),
                ));
            }
        }

        for (id, fill, kind, reduce_only, tif, requested) in fills {
            // FOK limit: all or nothing against every clamp, not just the bar's
            // liquidity, and judged against the account as it stands when this
            // order is reached rather than the snapshot the loop above saw. The
            // sweep below then cancels it unfilled (ADRs 0016, 0025).
            if kind == OrderType::Limit
                && tif == TimeInForce::Fok
                && self.clamp_fill(reduce_only, &fill).size.get() + CLOSE_EPS < requested
            {
                continue;
            }
            let applied = self.apply_fill_clamped(reduce_only, fill);
            match kind {
                // A triggered stop is a market order; it does not rest afterward.
                OrderType::Stop => {
                    self.book.cancel(id);
                }
                // A limit accumulates fills and rests until fully filled.
                OrderType::Limit => {
                    if applied > 0.0 {
                        if let Some(order) = self.book.get_mut(id) {
                            order.filled = Qty(order.filled.get() + applied);
                        }
                    }
                    let done = self
                        .book
                        .get(id)
                        .is_some_and(|o| o.remaining().get() <= 0.0);
                    if done {
                        self.book.cancel(id);
                    }
                }
                OrderType::Market => {}
            }
        }

        // IOC and FOK limits get one bar of exposure, then expire whether or not
        // they filled: they never rest for a later bar (ADR 0016).
        let expiring: Vec<OrderId> = self
            .book
            .iter()
            .filter(|o| {
                o.kind == OrderType::Limit && matches!(o.tif, TimeInForce::Ioc | TimeInForce::Fok)
            })
            .map(|o| o.id)
            .collect();
        for id in expiring {
            self.book.cancel(id);
        }
    }

    /// The reporter, when reporting is on.
    pub fn reporter(&self) -> Option<&Reporter> {
        self.reporter.as_ref()
    }

    /// Performance statistics for the run, or `None` when reporting is off. The
    /// starting equity is the configured quote.
    pub fn stats(&self, periods_per_year: f64, risk_free: f64) -> Option<Stats> {
        self.reporter.as_ref().map(|reporter| {
            reporter.stats(
                self.config.quote,
                periods_per_year,
                risk_free,
                self.fills,
                self.funding_paid,
            )
        })
    }

    /// Fills applied since the last reset, whatever the reporter is set to. Zero
    /// beside orders you placed means none of them ever filled (ADR 0031).
    pub fn num_fills(&self) -> usize {
        self.fills
    }

    /// The account equity marked at the current bar's close, without building a
    /// full `State`. The batched RL reward reads this per env, so it skips the
    /// open-orders vector a snapshot would allocate. Matches `state().equity`.
    pub fn equity(&self) -> f64 {
        let mark = self.candles.get(self.tick).expect("tick in range").close;
        self.account.equity(Price(mark))
    }

    fn state(&self) -> State {
        let bar = self.candles.get(self.tick).expect("tick in range");
        let mark = Price(bar.close);
        let position = self.account.position.qty.get();
        State {
            tick_index: self.tick,
            base: if self.config.market == Market::Spot {
                position
            } else {
                0.0
            },
            quote: self.account.quote,
            position,
            avg_entry: self.account.position.avg_entry.get(),
            equity: self.account.equity(mark),
            mark_price: bar.close,
            bar_open: bar.open,
            bar_high: bar.high,
            bar_low: bar.low,
            bar_close: bar.close,
            bar_volume: bar.volume,
            realized_pnl: self.account.realized(),
            unrealized_pnl: self.account.unrealized(mark),
            funding_paid: self.funding_paid,
            open_orders: self.book.iter().copied().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Engine, EngineConfig, CLOSE_EPS};
    use crate::candles::Candles;
    use emsl_core::{Candle, Fill, Market, OrderType, Price, Qty, Side, TimeInForce};

    fn ohlc(open: f64, high: f64, low: f64, close: f64, volume: f64) -> Candle {
        Candle {
            open,
            high,
            low,
            close,
            volume,
        }
    }

    fn series() -> Candles {
        Candles::new(vec![
            ohlc(100.0, 160.0, 90.0, 150.0, 1000.0),
            ohlc(200.0, 260.0, 190.0, 250.0, 1000.0),
            ohlc(300.0, 360.0, 290.0, 350.0, 1000.0),
        ])
    }

    // A first bar at 100, then a bar that dips to a low of 94, then a calm bar.
    fn dip_series() -> Candles {
        Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
            ohlc(100.0, 101.0, 94.0, 100.0, 1000.0),
            ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
        ])
    }

    // A first bar at 100, then a crash to a low of 80.
    fn crash_series() -> Candles {
        Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
            ohlc(100.0, 101.0, 80.0, 85.0, 1000.0),
            ohlc(85.0, 86.0, 84.0, 85.0, 1000.0),
        ])
    }

    /// Spot, no fees, no slippage, so fills land on round numbers.
    fn cfg() -> EngineConfig {
        EngineConfig {
            market: Market::Spot,
            quote: 10_000.0,
            fee_taker: 0.0,
            fee_maker: 0.0,
            slippage_bps: 0.0,
            max_fill_fraction: 1.0,
            max_open_orders: 8,
            report: false,
            max_leverage: 0.0,
            impact: 0.0,
            funding_rate: 0.0,
            funding_interval: 0,
        }
    }

    fn perp_cfg() -> EngineConfig {
        EngineConfig {
            market: Market::Perp,
            quote: 100.0,
            ..cfg()
        }
    }

    fn report_cfg() -> EngineConfig {
        EngineConfig {
            report: true,
            ..cfg()
        }
    }

    #[test]
    fn default_config_caps_perp_leverage_at_ten() {
        // the shipped default is a finite 10x cap, not uncapped notional (ADR 0012)
        assert_eq!(EngineConfig::default().max_leverage, 10.0);
    }

    #[test]
    #[should_panic]
    fn new_rejects_an_empty_series() {
        // a zero-bar series has no bar to reset or step to
        Engine::new(Candles::new(Vec::<Candle>::new()), cfg());
    }

    #[test]
    fn a_negative_market_size_is_a_no_op() {
        // size is a positive magnitude; a negative buy must not open a short, move
        // the position, or pay a fee
        let cfg = EngineConfig {
            fee_taker: 0.001,
            ..cfg()
        };
        let mut e = Engine::new(series(), cfg);
        e.reset();
        e.market_buy(-2.0);
        let s = e.step();
        assert_eq!(s.position, 0.0);
        assert_eq!(s.quote, 10_000.0); // no fill, so no fee charged
    }

    #[test]
    fn a_nan_market_size_is_a_no_op() {
        // f64::min keeps the other operand when one side is NaN, so without the cap
        // guard a NaN size would fill the whole volume cap; it must fill nothing and
        // pay nothing (ADR 0001)
        let cfg = EngineConfig {
            fee_taker: 0.001,
            ..cfg()
        };
        let mut e = Engine::new(series(), cfg);
        e.reset();
        e.market_buy(f64::NAN);
        let s = e.step();
        assert_eq!(s.position, 0.0);
        assert_eq!(s.quote, 10_000.0); // no fill, so no fee charged
    }

    #[test]
    fn a_non_finite_size_limit_or_stop_is_refused_a_slot() {
        // a NaN-size resting order could never complete (NaN minus any fill stays
        // NaN), so placement refuses it instead of letting it camp on a book slot
        // and refill the cap on every touching bar (ADR 0001)
        let mut e = Engine::new(series(), cfg());
        e.reset();
        assert_eq!(e.limit_buy(f64::NAN, 50.0), None);
        assert_eq!(e.stop(Side::Sell, f64::INFINITY, 90.0, false), None);
        let s = e.step();
        assert!(s.open_orders.is_empty());
        assert_eq!(s.position, 0.0);
    }

    #[test]
    fn a_gtc_limit_fills_across_bars_under_the_volume_cap() {
        // ADR 0005: a resting limit fills up to the volume cap each bar and keeps its
        // remainder resting for the next, so a large limit completes over several bars
        let candles = Candles::new(vec![
            ohlc(100.0, 100.0, 50.0, 100.0, 10.0), // decision bar
            ohlc(60.0, 60.0, 40.0, 60.0, 10.0),    // touches the 60 limit, low volume
            ohlc(60.0, 60.0, 40.0, 60.0, 10.0),    // touches again
        ]);
        let cfg = EngineConfig {
            max_fill_fraction: 0.5, // cap each fill at half the bar volume (5 units)
            ..cfg()
        };
        let mut e = Engine::new(candles, cfg);
        e.reset();
        e.limit_buy(9.0, 60.0); // wants 9 at 60; the cap allows 5 per bar
        let s1 = e.step(); // bar 1: fills 5, 4 remain resting
        assert_eq!(s1.position, 5.0);
        assert_eq!(s1.open_orders.len(), 1);
        let s2 = e.step(); // bar 2: fills the remaining 4 and completes
        assert_eq!(s2.position, 9.0);
        assert!(s2.open_orders.is_empty());
    }

    #[test]
    fn reset_starts_flat_at_the_first_bar() {
        let mut e = Engine::new(series(), cfg());
        let s = e.reset();
        assert_eq!(s.tick_index, 0);
        assert_eq!(s.position, 0.0);
        assert_eq!(s.quote, 10_000.0);
        assert_eq!(s.bar_open, 100.0);
    }

    #[test]
    fn reset_at_starts_the_cursor_at_the_offset() {
        let mut e = Engine::new(series(), cfg());
        let s = e.reset_at(1);
        assert_eq!(s.tick_index, 1);
        assert_eq!(s.bar_open, 200.0); // bar 1
        assert_eq!(s.position, 0.0);
        assert!(!e.done());
    }

    #[test]
    fn reset_at_clamps_past_the_end() {
        let mut e = Engine::new(series(), cfg());
        let s = e.reset_at(99);
        assert_eq!(s.tick_index, 2); // clamped to the last bar
        assert!(e.done());
    }

    #[test]
    fn market_order_fills_on_the_next_bar_open_not_the_decision_bar() {
        let mut e = Engine::new(series(), cfg());
        let s0 = e.reset();
        assert_eq!(s0.tick_index, 0); // looking at bar 0
        e.market_buy(1.0); // decided here
        let s1 = e.step(); // advance to bar 1
        assert_eq!(s1.tick_index, 1);
        // Filled at bar 1 OPEN (200), not bar 0 close (150) or bar 1 close (250).
        assert_eq!(s1.avg_entry, 200.0);
        assert_eq!(s1.position, 1.0);
    }

    #[test]
    fn done_only_at_the_last_bar() {
        let mut e = Engine::new(series(), cfg()); // 3 bars
        e.reset();
        assert!(!e.done());
        e.step();
        assert!(!e.done());
        e.step();
        assert!(e.done());
    }

    #[test]
    fn candle_window_ends_at_the_current_tick_and_clamps_early() {
        let mut e = Engine::new(series(), cfg());
        e.reset(); // tick 0
        let w = e.candle_window(2);
        assert_eq!(w.len(), 1); // only bar 0 exists so far
        assert_eq!(w[0].open, 100.0);
        e.step(); // tick 1
        let w = e.candle_window(2);
        assert_eq!(w.len(), 2); // bars 0 and 1
        assert_eq!(w[0].open, 100.0);
        assert_eq!(w[1].open, 200.0);
    }

    #[test]
    fn step_without_orders_just_advances_and_marks() {
        let mut e = Engine::new(series(), cfg());
        e.reset();
        let s = e.step();
        assert_eq!(s.tick_index, 1);
        assert_eq!(s.position, 0.0);
        assert_eq!(s.equity, 10_000.0);
    }

    #[test]
    fn close_flattens_the_position() {
        let mut e = Engine::new(series(), cfg());
        e.reset();
        e.market_buy(2.0);
        let s1 = e.step(); // long 2 @ 200
        assert_eq!(s1.position, 2.0);
        e.close();
        let s2 = e.step(); // close fills at bar 2 open (300)
        assert_eq!(s2.position, 0.0);
        // spot: bought 2 @ 200 (spent 400), sold 2 @ 300 (got 600), +200
        assert_eq!(s2.equity, 10_200.0);
    }

    #[test]
    fn spot_sell_cannot_open_a_short() {
        let mut e = Engine::new(series(), cfg());
        e.reset();
        e.market_sell(1.0); // sell from flat on spot
        let s = e.step();
        assert_eq!(s.position, 0.0); // clamped to flat, no short opened
        assert_eq!(s.quote, 10_000.0); // nothing sold, quote untouched
    }

    #[test]
    fn spot_sell_clamps_to_the_current_long() {
        let mut e = Engine::new(series(), cfg());
        e.reset();
        e.market_buy(1.0);
        e.step(); // long 1 @ 200
        e.market_sell(3.0); // wants 3, holds only 1
        let s = e.step(); // fills at bar 2 open 300
        assert_eq!(s.position, 0.0); // sold exactly the 1 held, not -2
        assert_eq!(s.quote, 10_100.0); // 10000 - 200 (buy) + 300 (sell 1)
    }

    #[test]
    fn spot_buy_clamps_to_available_cash() {
        let mut e = Engine::new(series(), cfg()); // spot, 10k, no fee
        e.reset();
        e.market_buy(1000.0); // wants 1000 base at bar 1 open 200, can afford 50
        let s = e.step();
        assert_eq!(s.position, 50.0); // 10000 / 200
        assert_eq!(s.quote, 0.0); // spent exactly the cash, never negative
    }

    #[test]
    fn spot_buy_clamp_accounts_for_the_fee() {
        let mut c = cfg();
        c.fee_taker = 0.01; // 1% taker
        let mut e = Engine::new(series(), c);
        e.reset();
        e.market_buy(1000.0); // affordable = 10000 / (200 * 1.01)
        let s = e.step();
        assert!((s.position - 10_000.0 / 202.0).abs() < 1e-9);
        assert!(s.quote >= -1e-9 && s.quote < 1e-6); // lands at ~0, never negative
    }

    #[test]
    fn perp_sell_still_opens_a_short() {
        let mut e = Engine::new(series(), perp_cfg());
        e.reset();
        e.market_sell(0.1); // perp: a short is a normal signed position
        let s = e.step();
        assert_eq!(s.position, -0.1);
    }

    #[test]
    fn perp_funding_charges_a_held_long_each_interval() {
        // funding every bar at 0.001 of notional; a long pays, a short receives
        let mut config = perp_cfg();
        config.quote = 10_000.0;
        config.funding_rate = 0.001;
        config.funding_interval = 1;
        let mut e = Engine::new(series(), config);
        e.reset();
        e.market_buy(1.0);
        let s1 = e.step(); // long 1 @ 200; funding at bar 1 close 250: pay 0.25
        let q1 = s1.quote;
        assert!((10_000.0 - q1 - 0.25).abs() < 1e-9); // 1 * 250 * 0.001
        let s2 = e.step(); // funding at bar 2 close 350: pay another 0.35
        assert!((q1 - s2.quote - 0.35).abs() < 1e-9);
    }

    #[test]
    fn funding_is_off_by_default_and_on_spot() {
        // default config funds nothing (interval 0); spot never funds either
        let mut e = Engine::new(series(), cfg()); // spot, no funding
        e.reset();
        e.market_buy(1.0);
        let s = e.step();
        assert_eq!(s.quote, 10_000.0 - 200.0); // only the buy, no funding debit
    }

    #[test]
    fn a_reduce_only_order_only_shrinks_the_position() {
        let mut e = Engine::new(series(), cfg());
        e.reset();
        e.market_buy(1.0);
        e.step(); // long 1 @ 200
                  // a reduce_only buy cannot grow the long, so it applies nothing
        e.order(
            Side::Buy,
            1.0,
            OrderType::Market,
            None,
            None,
            true,
            false,
            TimeInForce::Ioc,
        );
        let s = e.step();
        assert_eq!(s.position, 1.0);
    }

    #[test]
    fn an_ioc_limit_expires_after_its_one_bar() {
        let mut e = Engine::new(dip_series(), cfg());
        e.reset();
        e.order(
            Side::Buy,
            1.0,
            OrderType::Limit,
            Some(50.0), // never reached, so it does not fill
            None,
            false,
            false,
            TimeInForce::Ioc,
        );
        let s = e.step(); // its one bar of exposure passes
        assert_eq!(s.position, 0.0);
        assert!(s.open_orders.is_empty()); // IOC did not rest, unlike a GTC limit
    }

    #[test]
    fn a_fok_limit_fills_fully_or_not_at_all() {
        // bar 1 dips to 94 and trades 1000; a FOK for more than the volume cap
        // cannot fill in full, so it fills nothing and does not rest
        let mut e = Engine::new(dip_series(), cfg());
        e.reset();
        e.order(
            Side::Buy,
            2000.0,
            OrderType::Limit,
            Some(95.0),
            None,
            false,
            false,
            TimeInForce::Fok,
        );
        let s = e.step();
        assert_eq!(s.position, 0.0);
        assert!(s.open_orders.is_empty());
    }

    #[test]
    fn a_fok_limit_fills_when_the_whole_size_is_available() {
        let mut e = Engine::new(dip_series(), cfg());
        e.reset();
        e.order(
            Side::Buy,
            10.0,
            OrderType::Limit,
            Some(95.0),
            None,
            false,
            false,
            TimeInForce::Fok,
        );
        let s = e.step();
        assert_eq!(s.position, 10.0); // 10 within the cap, filled whole at 95
    }

    #[test]
    fn a_fok_market_fills_fully_or_nothing() {
        let mut config = cfg();
        config.max_fill_fraction = 0.1; // cap at 100 of a 1000-volume bar
        let mut e = Engine::new(series(), config);
        e.reset();
        e.order(
            Side::Buy,
            250.0,
            OrderType::Market,
            None,
            None,
            false,
            false,
            TimeInForce::Fok,
        );
        let s = e.step();
        assert_eq!(s.position, 0.0); // 250 over the cap of 100, so FOK fills nothing
    }

    #[test]
    fn cancel_all_drops_every_resting_order() {
        let mut e = Engine::new(dip_series(), cfg());
        e.reset();
        e.limit_buy(1.0, 50.0);
        e.limit_buy(1.0, 40.0);
        assert_eq!(e.cancel_all(), 2);
        let s = e.step();
        assert!(s.open_orders.is_empty());
    }

    #[test]
    fn qty_helpers_size_from_weight_and_quote() {
        let mut e = Engine::new(series(), cfg());
        e.reset(); // bar 0 close 150, equity 10000, flat
        assert!((e.qty_from_weight(0.5) - (5_000.0 / 150.0)).abs() < 1e-9);
        assert!((e.qty_from_quote(300.0) - 2.0).abs() < 1e-9); // 300 / 150
    }

    #[test]
    fn the_order_primitive_dispatches_by_kind() {
        let mut e = Engine::new(dip_series(), cfg());
        e.reset();
        // a limit with no price is rejected
        assert!(e
            .order(
                Side::Buy,
                1.0,
                OrderType::Limit,
                None,
                None,
                false,
                false,
                TimeInForce::Gtc
            )
            .is_none());
        let lim = e.order(
            Side::Buy,
            1.0,
            OrderType::Limit,
            Some(50.0),
            None,
            false,
            false,
            TimeInForce::Gtc,
        );
        let stp = e.order(
            Side::Sell,
            1.0,
            OrderType::Stop,
            None,
            Some(40.0),
            false,
            false,
            TimeInForce::Gtc,
        );
        assert!(lim.is_some() && stp.is_some());
        let s = e.step();
        assert_eq!(s.open_orders.len(), 2); // the GTC limit and the resting stop
    }

    #[test]
    fn reporter_records_equity_each_step_when_on() {
        let mut config = cfg();
        config.report = true;
        let mut e = Engine::new(series(), config);
        e.reset();
        e.step();
        e.step();
        assert_eq!(e.reporter().unwrap().equity_curve().len(), 2);
    }

    #[test]
    fn a_close_records_a_trade_with_realized_pnl() {
        let mut e = Engine::new(series(), report_cfg());
        e.reset();
        e.market_buy(2.0);
        e.step(); // long 2 @ 200 (bar 1 open)
        e.close();
        e.step(); // close 2 @ 300 (bar 2 open)
        let trades = e.reporter().unwrap().trades();
        assert_eq!(trades.len(), 1);
        let t = trades[0];
        assert_eq!(t.side, Side::Buy);
        assert_eq!(t.size, 2.0);
        assert_eq!(t.entry_price, 200.0);
        assert_eq!(t.exit_price, 300.0);
        assert_eq!(t.pnl, 200.0); // 2 * (300 - 200)
        assert_eq!(t.entry_tick, 1);
        assert_eq!(t.exit_tick, 2);
        assert_eq!(t.bars_held, 1);
    }

    #[test]
    fn a_partial_close_records_only_the_reduced_size() {
        let mut e = Engine::new(series(), report_cfg());
        e.reset();
        e.market_buy(5.0);
        e.step(); // long 5 @ 200
        e.market_sell(2.0);
        e.step(); // sell 2 @ 300, still long 3
        let trades = e.reporter().unwrap().trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].size, 2.0);
        assert_eq!(trades[0].pnl, 200.0); // 2 * (300 - 200)
    }

    #[test]
    fn a_flip_records_only_the_closed_portion() {
        // a flip through zero reopens on the opposite side; spot cannot go short
        // (ADR 0015), so the flip is exercised on a perp
        let mut e = Engine::new(
            series(),
            EngineConfig {
                report: true,
                ..perp_cfg()
            },
        );
        e.reset();
        e.market_buy(2.0);
        e.step(); // long 2 @ 200
        e.market_sell(5.0);
        let s = e.step(); // sell 5 @ 300: close 2, open short 3
        let trades = e.reporter().unwrap().trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].size, 2.0); // only the closed 2, not the whole 5
        assert_eq!(trades[0].pnl, 200.0);
        assert_eq!(s.position, -3.0); // flipped to short 3
    }

    #[test]
    fn adding_to_a_position_records_no_trade() {
        let mut e = Engine::new(series(), report_cfg());
        e.reset();
        e.market_buy(1.0);
        e.step(); // long 1 @ 200
        e.market_buy(1.0);
        e.step(); // add 1 @ 300, no close
        assert!(e.reporter().unwrap().trades().is_empty());
    }

    #[test]
    fn stats_count_the_closed_trades() {
        let mut e = Engine::new(series(), report_cfg());
        e.reset();
        e.market_buy(1.0);
        e.step(); // long 1 @ 200
        e.close();
        e.step(); // close @ 300, +100: a winning trade
        let stats = e.stats(365.0, 0.0).unwrap();
        assert_eq!(stats.num_trades, 1);
        assert_eq!(stats.win_rate, 1.0);
    }

    #[test]
    fn limit_rests_then_fills_when_the_bar_reaches_it() {
        let mut e = Engine::new(dip_series(), cfg());
        e.reset();
        e.limit_buy(1.0, 95.0); // placed at bar 0
        let s1 = e.step(); // bar 1 dips to 94, reaching 95
        assert_eq!(s1.position, 1.0);
        assert_eq!(s1.avg_entry, 95.0); // at the limit, maker, no slippage
        assert!(s1.open_orders.is_empty()); // fully filled, removed
    }

    #[test]
    fn untouched_limit_stays_resting_and_can_be_canceled() {
        let mut e = Engine::new(dip_series(), cfg());
        e.reset();
        let id = e.limit_buy(1.0, 50.0).unwrap(); // never reached
        let s1 = e.step();
        assert_eq!(s1.position, 0.0);
        assert_eq!(s1.open_orders.len(), 1); // still resting
        assert_eq!(s1.open_orders[0].id, id);
        assert!(e.cancel(id));
        let s2 = e.step();
        assert!(s2.open_orders.is_empty());
    }

    #[test]
    fn sell_stop_triggers_when_a_bar_crosses_it() {
        // a sell stop from flat opens a short, a perp behavior (ADR 0015)
        let mut e = Engine::new(crash_series(), perp_cfg());
        e.reset();
        e.stop(Side::Sell, 1.0, 95.0, false); // placed at bar 0
        let s1 = e.step(); // bar 1 low 80 crosses 95; open 100 so fills at trigger 95
        assert_eq!(s1.position, -1.0);
        assert_eq!(s1.avg_entry, 95.0);
        assert!(s1.open_orders.is_empty());
    }

    #[test]
    fn post_only_limit_that_would_cross_is_rejected() {
        let mut e = Engine::new(dip_series(), cfg());
        e.reset(); // bar 0 close 100
        assert!(e
            .place_limit(Side::Buy, 1.0, 105.0, false, true, TimeInForce::Gtc)
            .is_none()); // post_only that would cross
        assert!(e
            .place_limit(Side::Buy, 1.0, 95.0, false, true, TimeInForce::Gtc)
            .is_some()); // maker, accepted
    }

    #[test]
    fn leveraged_long_is_liquidated_on_a_crashing_bar() {
        let mut e = Engine::new(crash_series(), perp_cfg());
        e.reset();
        e.market_buy(10.0); // 10x notional on 100 margin, fills at bar 1 open 100
        let s1 = e.step(); // bar 1 low 80: equity 100 + 10*(80-100) = -100 -> liquidated
        assert_eq!(s1.position, 0.0);
    }

    #[test]
    fn is_bust_is_set_by_liquidation_and_cleared_by_reset() {
        let mut e = Engine::new(crash_series(), perp_cfg());
        e.reset();
        assert!(!e.is_bust());
        e.market_buy(10.0);
        e.step(); // liquidated on the crash bar
        assert!(e.is_bust());
        e.reset();
        assert!(!e.is_bust());
    }

    #[test]
    fn leverage_caps_the_perp_position() {
        let mut config = perp_cfg(); // perp, quote 100
        config.max_leverage = 2.0;
        let mut e = Engine::new(series(), config);
        e.reset();
        // fill at bar 1 open 200; equity 100, so max size = 2 * 100 / 200 = 1.0
        e.market_buy(10.0);
        let s = e.step();
        assert_eq!(s.position, 1.0); // capped at the margin limit
    }

    #[test]
    fn leverage_is_ignored_on_spot() {
        let mut config = cfg(); // spot
        config.max_leverage = 1.0;
        let mut e = Engine::new(series(), config);
        e.reset();
        e.market_buy(10.0);
        let s = e.step();
        assert_eq!(s.position, 10.0); // spot is unleveraged; no cap
    }

    #[test]
    fn leverage_caps_the_opened_side_of_a_flip() {
        let mut config = perp_cfg(); // quote 100
        config.max_leverage = 3.0;
        let mut e = Engine::new(series(), config);
        e.reset();
        e.market_buy(1.0);
        e.step(); // long 1 @ 200
                  // at bar 2 (fill 300) equity is 100 + 1*(300-200) = 200, so max size = 3*200/300 = 2
        e.market_sell(10.0);
        let s = e.step(); // flips: closes 1, opens short capped at 2
        assert_eq!(s.position, -2.0);
    }

    #[test]
    fn leverage_caps_the_new_side_of_a_flip_from_over_cap() {
        // bar 1 fills the open long at 100; by bar 2 the price has fallen to 80, so
        // equity drops and the 2-unit long is over the 2x cap. A big sell flips: the
        // fresh short must fit the cap (1.5), not inherit the old side's size (2).
        let candles = Candles::new(vec![
            Candle {
                open: 100.0,
                high: 100.0,
                low: 100.0,
                close: 100.0,
                volume: 1_000.0,
            },
            Candle {
                open: 100.0,
                high: 100.0,
                low: 100.0,
                close: 100.0,
                volume: 1_000.0,
            },
            Candle {
                open: 80.0,
                high: 80.0,
                low: 80.0,
                close: 80.0,
                volume: 1_000.0,
            },
        ]);
        let mut config = perp_cfg(); // quote 100
        config.max_leverage = 2.0;
        let mut e = Engine::new(candles, config);
        e.reset();
        e.market_buy(2.0);
        let s1 = e.step(); // long 2 @ 100 (2x on 100 equity)
        assert_eq!(s1.position, 2.0);
        // bar 2 fill 80: equity = 100 + 2*(80-100) = 60; max size = 2*60/80 = 1.5
        e.market_sell(20.0);
        let s2 = e.step(); // flips: closes 2, opens short capped at 1.5, not 2
        assert_eq!(s2.position, -1.5);
    }

    #[test]
    fn reduce_only_market_cannot_flip_the_position() {
        let mut e = Engine::new(series(), cfg());
        e.reset();
        e.market_buy(2.0);
        e.step(); // long 2 @ 200
        e.place_market(Side::Sell, 5.0, true, TimeInForce::Ioc); // reduce_only sell of 5
        let s = e.step(); // clamped to 2, so it closes rather than flipping to short 3
        assert_eq!(s.position, 0.0);
    }

    #[test]
    fn reduce_only_same_side_is_a_noop() {
        let mut e = Engine::new(series(), cfg());
        e.reset();
        e.market_buy(2.0);
        e.step(); // long 2
        e.place_market(Side::Buy, 1.0, true, TimeInForce::Ioc); // reduce_only buy cannot increase
        let s = e.step();
        assert_eq!(s.position, 2.0);
    }

    #[test]
    fn logged_trade_pnl_reconciles_with_realized() {
        // the Validation Guide's reconciliation invariant: with no liquidation, the
        // sum of logged trade PnL equals the account's cumulative realized PnL
        let mut config = perp_cfg(); // perp, quote 100, no cap
        config.report = true;
        let candles = Candles::new(vec![
            ohlc(100.0, 110.0, 90.0, 100.0, 1_000.0),
            ohlc(100.0, 110.0, 90.0, 105.0, 1_000.0),
            ohlc(105.0, 115.0, 95.0, 110.0, 1_000.0),
            ohlc(110.0, 120.0, 100.0, 108.0, 1_000.0),
        ]);
        let mut e = Engine::new(candles, config);
        e.reset();
        e.market_buy(1.0);
        e.step(); // long 1
        e.market_sell(2.0);
        e.step(); // flip to short 1, books the long's PnL
        e.market_buy(1.0);
        let last = e.step(); // close the short, books its PnL

        let trade_sum: f64 = e.reporter().unwrap().trades().iter().map(|t| t.pnl).sum();
        assert!(!e.reporter().unwrap().trades().is_empty()); // trades were logged
        assert!((trade_sum - last.realized_pnl).abs() < 1e-9);
    }

    #[test]
    fn net_trade_pnl_accounts_for_the_whole_run_when_it_ends_flat() {
        // the invariant the entry-fee attribution buys: with the position closed and
        // no liquidation, the net PnL of the logged trades IS the change in equity.
        // Charging only the exit fee left about half the cost attributed to nothing
        // (ADR 0030).
        let mut config = cfg(); // spot
        config.report = true;
        config.fee_taker = 0.001;
        config.fee_maker = 0.001;
        let bars: Vec<Candle> = (0..30)
            .map(|i| {
                let c = 100.0 + (i as f64 * 0.7).sin() * 5.0;
                ohlc(c, c + 1.0, c - 1.0, c, 10_000.0)
            })
            .collect();
        let mut e = Engine::new(Candles::new(bars), config);
        let mut state = e.reset();
        let mut hold = 0;
        while !e.done() {
            // an order decided on the last bar has no bar to fill against, so the
            // flattening close has to be decided on the one before it
            if e.tick() + 2 >= e.num_bars() {
                e.close();
            } else if state.position == 0.0 {
                e.market_buy(2.0);
                hold = 0;
            } else {
                hold += 1;
                if hold >= 2 {
                    e.close();
                }
            }
            state = e.step();
        }
        assert_eq!(
            state.position, 0.0,
            "the run must end flat for the identity"
        );

        let trades = e.reporter().unwrap().trades();
        assert!(trades.len() > 3, "only {} trades", trades.len());
        let net: f64 = trades.iter().map(|t| t.net_pnl).sum();
        let equity_change = state.equity - 10_000.0;
        assert!(
            (net - equity_change).abs() < 1e-9,
            "sum(net_pnl) {net} != equity change {equity_change}"
        );
        // and the gross figure alone does not reconcile, which is the whole point
        let gross: f64 = trades.iter().map(|t| t.pnl).sum();
        assert!((gross - equity_change).abs() > 1e-6);
        for t in trades {
            assert!((t.net_pnl - (t.pnl - t.fees)).abs() < 1e-12);
        }
    }

    #[test]
    fn a_trade_carries_both_sides_of_its_fee() {
        let mut config = cfg(); // spot, bar 1 open 200, bar 2 open 300
        config.report = true;
        config.fee_taker = 0.01;
        let mut e = Engine::new(series(), config);
        e.reset();
        e.market_buy(1.0);
        e.step(); // buy 1 at 200, fee 2.0
        e.close();
        e.step(); // sell 1 at 300, fee 3.0
        let t = e.reporter().unwrap().trades()[0];
        assert!((t.pnl - 100.0).abs() < 1e-9); // gross price PnL
        assert!((t.fees - 5.0).abs() < 1e-9); // 2.0 entry plus 3.0 exit
        assert!((t.net_pnl - 95.0).abs() < 1e-9);
    }

    #[test]
    fn adding_to_a_position_does_not_double_charge_its_entry_fee() {
        // buy 1 at 200 (fee 2), buy 1 at 400 (fee 4), then close both at 300 (fee 6).
        // The whole position closes, so the row must carry every fee paid: 12, not 16.
        // A same-side add keeps more than it started with, so the closed amount goes
        // negative unless it is clamped, and the opening share of the fee is counted
        // twice.
        let candles = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1_000_000.0),
            ohlc(200.0, 200.0, 200.0, 200.0, 1_000_000.0),
            ohlc(400.0, 400.0, 400.0, 400.0, 1_000_000.0),
            ohlc(300.0, 300.0, 300.0, 300.0, 1_000_000.0),
            ohlc(300.0, 300.0, 300.0, 300.0, 1_000_000.0),
        ]);
        let mut config = perp_cfg();
        config.quote = 1_000_000.0;
        config.report = true;
        config.fee_taker = 0.01;
        let mut e = Engine::new(candles, config);
        e.reset();
        e.market_buy(1.0);
        e.step(); // 1 @ 200, fee 2
        e.market_buy(1.0);
        e.step(); // 1 @ 400, fee 4; position 2, average entry 300
        e.market_sell(2.0);
        let last = e.step(); // 2 @ 300, fee 6

        let trades = e.reporter().unwrap().trades();
        assert_eq!(trades.len(), 1);
        assert!(
            (trades[0].fees - 12.0).abs() < 1e-9,
            "fees {} should be 2 + 4 + 6",
            trades[0].fees
        );
        assert!((trades[0].pnl - 0.0).abs() < 1e-9);
        assert!((trades[0].net_pnl + 12.0).abs() < 1e-9);
        // the identity must survive a scale-in as well as a single entry
        assert!((trades[0].net_pnl - (last.equity - 1_000_000.0)).abs() < 1e-9);
    }

    #[test]
    fn net_pnl_reconciles_through_scale_ins_and_partial_exits() {
        let mut config = cfg(); // spot
        config.report = true;
        config.fee_taker = 0.002;
        config.fee_maker = 0.002;
        let bars: Vec<Candle> = (0..40)
            .map(|i| {
                let c = 100.0 + (i as f64 * 0.9).sin() * 8.0;
                ohlc(c, c + 1.0, c - 1.0, c, 1_000_000.0)
            })
            .collect();
        let mut e = Engine::new(Candles::new(bars), config);
        let mut state = e.reset();
        while !e.done() {
            let tick = e.tick();
            if tick + 2 >= e.num_bars() {
                e.close();
            } else if state.position < 3.0 {
                e.market_buy(1.0); // scale in, one unit at a time
            } else if tick % 5 == 0 {
                e.market_sell(1.0); // and scale out again
            }
            state = e.step();
        }
        assert_eq!(state.position, 0.0);
        let trades = e.reporter().unwrap().trades();
        assert!(trades.len() > 3, "only {} trades", trades.len());
        let net: f64 = trades.iter().map(|t| t.net_pnl).sum();
        assert!(
            (net - (state.equity - 10_000.0)).abs() < 1e-9,
            "sum(net_pnl) {net} != equity change {}",
            state.equity - 10_000.0
        );
    }

    #[test]
    fn a_partial_close_takes_only_its_share_of_the_entry_fee() {
        let mut config = cfg();
        config.report = true;
        config.fee_taker = 0.01;
        let mut e = Engine::new(series(), config);
        e.reset();
        e.market_buy(4.0);
        e.step(); // buy 4 at 200, entry fee 8.0 for the whole position
        e.market_sell(1.0);
        e.step(); // sell 1 at 300, exit fee 3.0; entry share is 8.0 * 1/4 = 2.0
        let t = e.reporter().unwrap().trades()[0];
        assert!((t.size - 1.0).abs() < 1e-9);
        assert!((t.fees - 5.0).abs() < 1e-9);
        assert!((t.net_pnl - 95.0).abs() < 1e-9);
    }

    #[test]
    fn num_fills_separates_a_dead_feed_from_a_quiet_strategy() {
        // a zero-volume series fills nothing, and without a counter the result is
        // identical to a strategy that never placed an order (ADR 0031)
        let dead = Candles::new(vec![ohlc(100.0, 100.0, 100.0, 100.0, 0.0); 6]);
        let mut config = cfg();
        config.report = true;
        let mut e = Engine::new(dead, config);
        e.reset();
        while !e.done() {
            e.market_buy(1.0);
            e.step();
        }
        assert_eq!(e.num_fills(), 0);
        assert_eq!(e.stats(365.0, 0.0).unwrap().num_fills, 0);

        let mut live = Engine::new(series(), config);
        live.reset();
        live.market_buy(1.0);
        live.step();
        assert_eq!(live.num_fills(), 1);
    }

    #[test]
    fn replace_moves_one_order_and_refuses_to_arm_a_second() {
        // the trailing-stop trap: re-placing with stop() rests a new order every
        // bar, and the one that fills leaves its siblings live (ADR 0032)
        let mut e = Engine::new(dip_series(), perp_cfg());
        e.reset();
        let first = e.stop(Side::Sell, 1.0, 90.0, true).unwrap();
        let second = e.replace(first, None, None, Some(92.0)).unwrap();
        let s = e.step();
        assert_eq!(s.open_orders.len(), 1, "replace must not leave two live");
        assert_eq!(s.open_orders[0].id, second);
        assert_eq!(s.open_orders[0].trigger.unwrap().get(), 92.0);
        assert!(s.open_orders[0].reduce_only, "flags carry over");

        // once the order is gone, replacing places nothing at all
        assert!(e.cancel(second));
        assert_eq!(e.replace(second, None, None, Some(95.0)), None);
        assert!(e.step().open_orders.is_empty());
    }

    #[test]
    fn funding_can_bust_the_account_on_the_same_bar() {
        // a punitive funding rate on a leveraged long drains equity below zero within
        // the step, before the liquidation check, so the account busts (ADR 0017)
        let mut config = perp_cfg(); // quote 100
        config.funding_rate = 1.0; // 100% of notional per event, for the test
        config.funding_interval = 1;
        let candles = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1_000.0),
            ohlc(100.0, 100.0, 100.0, 100.0, 1_000.0),
        ]);
        let mut e = Engine::new(candles, config);
        e.reset();
        e.market_buy(2.0); // long 2 @ 100, notional 200 on 100 equity
        let s = e.step(); // funding 2*100*1.0 = 200 debited -> equity -100
        assert!(e.is_bust());
        assert!(s.equity <= 0.0);
    }

    #[test]
    fn a_fok_market_blocked_by_the_cash_clamp_fills_nothing() {
        // the bar has the liquidity, but spot cash affords only 50 of the 1000, and
        // FOK is all or nothing against every clamp, not just volume (ADR 0025)
        let mut e = Engine::new(series(), cfg()); // spot, quote 10_000, bar 1 open 200
        e.reset();
        e.order(
            Side::Buy,
            1000.0,
            OrderType::Market,
            None,
            None,
            false,
            false,
            TimeInForce::Fok,
        );
        let s = e.step();
        assert_eq!(s.position, 0.0);
        assert_eq!(s.quote, 10_000.0);
    }

    #[test]
    fn a_fok_market_blocked_by_the_leverage_cap_fills_nothing() {
        let mut config = perp_cfg(); // quote 100
        config.max_leverage = 2.0;
        let mut e = Engine::new(series(), config);
        e.reset();
        // 2x on 100 equity at the bar 1 open of 200 allows 1.0, so a FOK for 10 dies
        e.order(
            Side::Buy,
            10.0,
            OrderType::Market,
            None,
            None,
            false,
            false,
            TimeInForce::Fok,
        );
        assert_eq!(e.step().position, 0.0);
    }

    #[test]
    fn an_insolvent_perp_cannot_open_a_position() {
        // after a gap liquidation the account is flat with negative equity. Nothing
        // backs new notional, and liquidation cannot force-close a flat account, so
        // the fill is refused rather than opened and closed at the low (ADR 0026)
        let candles = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1_000_000.0),
            ohlc(100.0, 100.0, 100.0, 100.0, 1_000_000.0),
            ohlc(40.0, 40.0, 40.0, 40.0, 1_000_000.0),
            ohlc(40.0, 41.0, 25.0, 30.0, 1_000_000.0),
        ]);
        let mut config = perp_cfg(); // quote 100
        config.max_leverage = 10.0;
        let mut e = Engine::new(candles, config);
        e.reset();
        e.market_buy(10.0);
        e.step();
        let busted = e.step();
        // dead, and holding exactly nothing rather than owing anything: the bar
        // crashed to 40 but the position was closed where its margin ran out at 90
        assert!(busted.equity.abs() < CLOSE_EPS, "left {}", busted.equity);
        assert_eq!(busted.position, 0.0);
        e.market_buy(5_000.0);
        let after = e.step();
        assert_eq!(after.position, 0.0);
        // the refused fill costs nothing; equity only marks the same dead account
        assert!(
            (after.equity - busted.equity).abs() < 1e-9,
            "equity moved {} -> {}",
            busted.equity,
            after.equity
        );
    }

    #[test]
    fn two_resting_orders_filling_on_one_bar_are_clamped_in_sequence() {
        // the fills are collected against a pre-fill snapshot and applied one at a
        // time, so the second must be clamped against the account the first left
        let candles = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1_000.0),
            ohlc(100.0, 100.0, 80.0, 100.0, 1_000.0),
        ]);
        let mut config = perp_cfg(); // quote 100
        config.max_leverage = 2.0;
        let mut e = Engine::new(candles, config);
        e.reset();
        e.limit_buy(2.0, 90.0).unwrap();
        e.limit_buy(2.0, 90.0).unwrap();
        let s = e.step();
        // 2x on 100 equity at a fill price of 90 allows 2.222 units in total, so the
        // first limit fills whole and the second is clamped to the remainder
        assert!(
            (s.position - 2.0 * 100.0 / 90.0).abs() < 1e-9,
            "position {}",
            s.position
        );
    }

    #[test]
    fn two_pending_market_orders_are_clamped_in_sequence() {
        let mut e = Engine::new(series(), cfg()); // spot, quote 10_000, bar 1 open 200
        e.reset();
        e.market_buy(40.0);
        e.market_buy(40.0);
        let s = e.step();
        // 10_000 of cash at 200 buys 50 in total, not 80
        assert_eq!(s.position, 50.0);
        assert_eq!(s.quote, 0.0);
    }

    #[test]
    fn close_flattens_a_short_and_records_the_trade() {
        // every other close test runs on a long, leaving the short arm of close()
        // and the pos < 0 half of the reduce_only predicate unexecuted
        let mut config = perp_cfg(); // quote 100, no cap
        config.report = true;
        let mut e = Engine::new(series(), config);
        e.reset();
        e.market_sell(1.0);
        let s1 = e.step(); // short 1 @ bar 1 open 200
        assert_eq!(s1.position, -1.0);
        e.close();
        let s2 = e.step(); // buys back at bar 2 open 300
        assert_eq!(s2.position, 0.0);
        let trades = e.reporter().unwrap().trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].side, Side::Sell);
        assert_eq!(trades[0].pnl, -100.0); // short 1 from 200 bought back at 300
        let stats = e.stats(365.0, 0.0).unwrap();
        assert_eq!(stats.win_rate, 0.0);
        assert_eq!(stats.profit_factor, 0.0);
    }

    #[test]
    fn a_reduce_only_buy_closes_a_short() {
        let mut e = Engine::new(series(), perp_cfg());
        e.reset();
        e.market_sell(2.0);
        e.step(); // short 2
        e.place_market(Side::Buy, 5.0, true, TimeInForce::Ioc);
        let s = e.step(); // clamped to 2, so it closes rather than flipping long
        assert_eq!(s.position, 0.0);
    }

    #[test]
    fn funding_is_charged_only_on_interval_bars() {
        // every other funding test uses interval 1, where the modulo is trivially
        // true on every bar, so the schedule itself is never exercised (ADR 0017)
        let mut config = perp_cfg();
        config.quote = 10_000.0;
        config.funding_rate = 0.001;
        config.funding_interval = 3;
        let flat = ohlc(100.0, 100.0, 100.0, 100.0, 10_000.0);
        let mut e = Engine::new(Candles::new(vec![flat; 8]), config);
        e.reset();
        e.market_buy(1.0);
        let mut charged = Vec::new();
        let mut prev = 10_000.0;
        for _ in 0..7 {
            let s = e.step();
            charged.push((prev - s.quote).abs() > 1e-12);
            prev = s.quote;
        }
        // ticks 1..7; funding fires where tick % 3 == 0, so on ticks 3 and 6 only
        assert_eq!(
            charged,
            vec![false, false, true, false, false, true, false],
            "charged on {charged:?}"
        );
    }

    #[test]
    fn a_perp_marks_equity_at_the_close_on_an_open_position() {
        // the only numeric equity assertions were on spot, so a perp marked with the
        // spot formula would have gone unnoticed
        let mut e = Engine::new(series(), perp_cfg()); // quote 100
        e.reset();
        e.market_buy(1.0);
        let s1 = e.step(); // long 1 @ bar 1 open 200, bar 1 close 250
        assert_eq!(s1.equity, 100.0 + 1.0 * (250.0 - 200.0));
        let s2 = e.step(); // bar 2 close 350
        assert_eq!(s2.equity, 100.0 + 1.0 * (350.0 - 200.0));
    }

    #[test]
    fn order_ids_stay_unique_across_a_reset() {
        // an id handed out in one episode must not name a live order in the next,
        // or a handle carried across a reset cancels somebody else's order (ADR 0028)
        let mut e = Engine::new(dip_series(), cfg());
        e.reset();
        let first = e.limit_buy(1.0, 50.0).unwrap();
        e.reset();
        let second = e.limit_buy(1.0, 50.0).unwrap();
        assert_ne!(first, second);
        assert!(!e.cancel(first)); // the stale handle matches nothing
        assert!(e.cancel(second));
    }

    #[test]
    fn a_stop_crossed_on_a_dead_bar_stays_armed() {
        // bar 1 crosses the trigger but trades nothing, so the stop cannot fill. It
        // must survive: consuming it here would delete the protection on exactly the
        // illiquid data where a stop matters, and it fills on bar 2 instead (ADR 0035)
        let candles = Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1_000.0),
            ohlc(100.0, 101.0, 80.0, 85.0, 0.0), // crosses 95, no volume
            ohlc(85.0, 86.0, 84.0, 85.0, 1_000.0),
        ]);
        let mut e = Engine::new(candles, perp_cfg());
        e.reset();
        e.stop(Side::Sell, 1.0, 95.0, false).unwrap();
        let s1 = e.step();
        assert_eq!(s1.position, 0.0, "nothing can fill on a bar with no volume");
        assert_eq!(s1.open_orders.len(), 1, "the stop must still be armed");
        let s2 = e.step(); // bar 2 trades, and the trigger is still crossed
        assert_eq!(s2.position, -1.0);
        assert!(s2.open_orders.is_empty(), "and is consumed once it fills");
    }

    #[test]
    fn a_reduce_only_stop_cannot_open_the_other_side() {
        // the shortcut carries the flag now, so a protective stop that outlives the
        // one that closed the position fills nothing instead of opening a short
        let mut e = Engine::new(crash_series(), perp_cfg());
        e.reset();
        assert!(e.stop(Side::Sell, 1.0, 95.0, true).is_some());
        let s = e.step(); // bar 1 low 80 crosses 95, but there is nothing to reduce
        assert_eq!(s.position, 0.0);
    }

    #[test]
    fn a_non_finite_limit_price_or_stop_trigger_is_refused_a_slot() {
        // every touch test against NaN is false, so such an order never fills and
        // never expires; it would hold a slot until the book was full (ADR 0027)
        let mut e = Engine::new(series(), cfg());
        e.reset();
        assert_eq!(e.limit_buy(1.0, f64::NAN), None);
        assert_eq!(e.limit_sell(1.0, f64::INFINITY), None);
        assert_eq!(e.stop(Side::Sell, 1.0, f64::NAN, false), None);
        let s = e.step();
        assert!(s.open_orders.is_empty());
    }

    #[test]
    fn ioc_limit_partial_fill_applies_then_cancels_the_rest() {
        // an IOC limit larger than the bar's fillable volume takes what it can this
        // bar and cancels the remainder rather than resting
        let mut config = cfg(); // spot
        config.max_fill_fraction = 0.1; // at most 10% of a bar's volume per order
        let candles = Candles::new(vec![
            ohlc(100.0, 160.0, 90.0, 150.0, 1_000.0),
            ohlc(100.0, 160.0, 40.0, 150.0, 1_000.0), // low 40 reaches the limit at 50
            ohlc(100.0, 160.0, 90.0, 150.0, 1_000.0),
        ]);
        let mut e = Engine::new(candles, config);
        e.reset();
        e.place_limit(Side::Buy, 500.0, 50.0, false, false, TimeInForce::Ioc);
        let s = e.step(); // reaches 50, fills 100 (10% of 1000), cancels the other 400
        assert_eq!(s.position, 100.0);
        assert!(s.open_orders.is_empty()); // IOC: nothing rests
    }

    #[test]
    fn funding_is_kept_where_the_result_can_read_it() {
        // the account returned every payment and nothing kept it, so the one cost
        // unique to a perp was unmeasurable from a finished run, and a carry could
        // not be told apart from a direction (ADR 0017)
        let config = EngineConfig {
            market: Market::Perp,
            funding_rate: 0.001,
            funding_interval: 1,
            report: true,
            ..cfg()
        };
        let mut long = Engine::new(series(), config);
        long.reset();
        long.market_buy(1.0);
        let s1 = long.step(); // fills at the open of 200, funded at the close of 250
        assert!((s1.funding_paid - 0.25).abs() < CLOSE_EPS);
        let s2 = long.step(); // funded again at the close of 350
        assert!((s2.funding_paid - 0.60).abs() < CLOSE_EPS);
        let stats = long.stats(365.0, 0.0).expect("reporting is on");
        assert!((stats.funding_paid - 0.60).abs() < CLOSE_EPS);

        // a positive rate is paid BY the long and TO the short, so the sign flips
        let mut short = Engine::new(series(), config);
        short.reset();
        short.market_sell(1.0);
        let s = short.step();
        assert!((s.funding_paid + 0.25).abs() < CLOSE_EPS);
    }

    #[test]
    fn a_liquidation_is_booked_like_any_other_close() {
        // the forced close ran on the account directly and reached none of the
        // engine's bookkeeping: it appeared in no trade row, counted toward no
        // fill, and left the dead position's entry fee standing to be charged
        // against the NEXT position (ADRs 0030, 0031)
        let config = EngineConfig {
            market: Market::Perp,
            quote: 100.0,
            fee_taker: 0.001,
            max_leverage: 10.0,
            report: true,
            ..cfg()
        };
        let mut e = Engine::new(crash_series(), config);
        e.reset();
        e.market_buy(10.0); // 1000 of notional on 100 of quote, exactly at the cap
        let s = e.step(); // fills at the open of 100, then the low of 80 busts it
        assert!(e.is_bust());
        assert_eq!(s.position, 0.0);
        assert_eq!(e.num_fills(), 2); // the buy, and the forced close
        let trades = e.reporter().expect("reporting is on").trades();
        assert_eq!(trades.len(), 1);
        assert!(trades[0].liquidated);
        // closed where the margin ran out, not at the 80 the bar printed. The entry
        // fee left 99 of margin, so (10*100 - 99) / (10 - 10*0.001) = 90.1902
        assert!(
            (trades[0].exit_price - 901.0 / 9.99).abs() < 1e-9,
            "closed at {}",
            trades[0].exit_price
        );
        // and the account is left with exactly nothing, never owing (ADR 0052)
        assert!(s.equity.abs() < CLOSE_EPS, "left {}", s.equity);
        assert!(trades[0].fees > 1.0); // the entry fee, plus a real fee on the exit
        assert_eq!(e.open_fee, 0.0); // nothing left over to charge the next position
    }

    #[test]
    fn a_liquidation_can_never_leave_the_account_owing() {
        // whatever the bar prints past the point the margin is gone, the loss is
        // bounded by the margin: closing at the extreme booked 200 on a 100 account
        for low in [90.0, 80.0, 40.0, 1.0] {
            let config = EngineConfig {
                market: Market::Perp,
                quote: 100.0,
                fee_taker: 0.0006,
                max_leverage: 10.0,
                ..cfg()
            };
            let candles = Candles::new(vec![
                ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
                ohlc(100.0, 101.0, low, low, 1000.0),
                ohlc(low, low + 1.0, low - 1.0, low, 1000.0),
            ]);
            let mut e = Engine::new(candles, config);
            e.reset();
            e.market_buy(10.0);
            let s = e.step();
            assert!(s.equity >= -CLOSE_EPS, "a low of {low} left {}", s.equity);
            assert!(
                s.quote >= -CLOSE_EPS,
                "a low of {low} left quote {}",
                s.quote
            );
        }
    }

    #[test]
    fn splitting_an_order_cannot_take_more_than_its_slots_of_a_bar() {
        // the volume cap is per order (ADR 0005) and the pending queue was
        // unbounded, so a hundred small market buys took the whole bar while every
        // slice was priced as though it were the only participant (ADR 0047)
        let config = EngineConfig {
            max_open_orders: 3,
            max_fill_fraction: 0.01, // 10 base units per order, against volume 1000
            quote: 1_000_000.0,
            ..cfg()
        };
        let mut e = Engine::new(series(), config);
        e.reset();
        let placed = (0..100).filter(|_| e.market_buy(50.0).is_some()).count();
        assert_eq!(placed, 3);
        let s = e.step();
        assert_eq!(s.position, 30.0); // three slots at the 10-unit cap, not a hundred
                                      // the queue drains with the bar, so the next bar gets its slots back
        assert!(e.market_buy(50.0).is_some());
    }

    #[test]
    fn a_straddle_of_limits_at_one_price_never_ends_richer() {
        // ADR 0006 accepts that a buy and a sell resting at the same price can both
        // fill against one wide bar, which is a bar-fidelity artifact. Booking both
        // as makers turned that artifact into a money printer, because a maker rate
        // is legally a rebate: the account came back flat and richer every bar,
        // without bound, and a search over anything placing limits walks into it
        for maker in [-0.0002, 0.0, 0.0002] {
            for market in [Market::Spot, Market::Perp] {
                let config = EngineConfig {
                    market,
                    quote: 10_000.0,
                    fee_maker: maker,
                    fee_taker: 0.0006,
                    max_leverage: 10.0,
                    ..cfg()
                };
                let candles = Candles::new(vec![
                    ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
                    ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
                    ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
                ]);
                let mut e = Engine::new(candles, config);
                e.reset();
                e.place_limit(Side::Buy, 1.0, 100.0, false, false, TimeInForce::Gtc);
                e.place_limit(Side::Sell, 1.0, 100.0, false, false, TimeInForce::Gtc);
                let s = e.step();
                assert!(s.position.abs() <= CLOSE_EPS, "{market:?} left a position");
                assert!(
                    s.equity <= 10_000.0 + CLOSE_EPS,
                    "{market:?} at maker {maker} ended with {}",
                    s.equity
                );
            }
        }
    }

    #[test]
    fn a_sub_dust_fill_moves_nothing_and_counts_nothing() {
        // the position refuses a size at or below its dust epsilon, so one in
        // (0, 1e-9] used to pass the engine's own gate, move no cash and no
        // position, and still count a fill and bank its fee. The leverage cap
        // produces exactly such a residual when a position sits a sub-ulp under it
        let mut e = Engine::new(series(), cfg());
        e.reset();
        let dust = Fill {
            side: Side::Buy,
            size: Qty(5e-13),
            price: Price(100.0),
            is_taker: true,
        };
        assert_eq!(e.clamp_fill(false, &dust).size.get(), 0.0);
        assert_eq!(e.apply_fill_clamped(false, dust), 0.0);
        assert_eq!(e.num_fills(), 0);
        assert_eq!(e.state().position, 0.0);
    }
}
