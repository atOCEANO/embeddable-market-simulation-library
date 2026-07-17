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
    /// Set when the account was force-closed by liquidation, a terminal event for
    /// the RL env. Cleared on reset.
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
        self.id_counter = 1;
        self.position_entry_tick = self.tick;
        self.bust = false;
        if self.reporter.is_some() {
            self.reporter = Some(Reporter::new());
        }
        self.state()
    }

    /// True when the account was force-closed by liquidation since the last reset,
    /// the RL env's terminal signal.
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
    ) -> OrderId {
        let id = self.next_id();
        let mut order = Order::market(id, side, Qty(size), reduce_only);
        order.tif = tif;
        self.pending.push(order);
        id
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
        // it is refused a slot rather than resting inert forever.
        if !(size.is_finite() && size > 0.0) {
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
        // The same guard as a limit: a non-positive or non-finite size never fills.
        if !(size.is_finite() && size > 0.0) {
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
            OrderType::Market => Some(self.place_market(side, size, reduce_only, tif)),
            OrderType::Limit => self.place_limit(side, size, price?, reduce_only, post_only, tif),
            OrderType::Stop => self.place_stop(side, size, trigger?, reduce_only),
        }
    }

    /// Queue a market buy; it fills on the next bar's open.
    pub fn market_buy(&mut self, size: f64) -> OrderId {
        self.place_market(Side::Buy, size, false, TimeInForce::Ioc)
    }

    /// Queue a market sell; it fills on the next bar's open.
    pub fn market_sell(&mut self, size: f64) -> OrderId {
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
    /// if the book is full.
    pub fn stop(&mut self, side: Side, size: f64, trigger: f64) -> Option<OrderId> {
        self.place_stop(side, size, trigger, false)
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
        Some(self.place_market(side, size, true, TimeInForce::Ioc))
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
                    // FOK: the whole size fills against this bar's liquidity or none
                    if order.tif == TimeInForce::Fok
                        && fill.size.get() + CLOSE_EPS < order.size.get()
                    {
                        continue;
                    }
                    self.apply_fill_clamped(order.reduce_only, fill);
                }
            }

            // 2. resting limit and stop orders fill against the bar range
            self.resolve_resting(&bar);

            // 3. funding at each interval boundary, on the position held into the
            //    bar, marked at the close, before liquidation so a funding debit can
            //    bust the account this bar (ADR 0017)
            if self.config.funding_interval > 0 && self.tick % self.config.funding_interval == 0 {
                self.account
                    .apply_funding(self.config.funding_rate, Price(bar.close));
            }

            // 4. liquidation at the bar's adverse extreme (long at the low, short
            //    at the high)
            let adverse = if self.account.position.qty.get() >= 0.0 {
                bar.low
            } else {
                bar.high
            };
            if self.account.liquidate_if_bust(Price(adverse)) {
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
        if equity <= 0.0 {
            return; // insolvent; liquidation handles it
        }
        let max_size = self.config.max_leverage * equity / mark;
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

    fn apply_fill_clamped(&mut self, reduce_only: bool, fill: Fill) -> f64 {
        let mut fill = fill;
        // size is a positive magnitude; a non-positive or non-finite fill is not a
        // real trade, so it moves nothing and pays no fee
        let size = fill.size.get();
        if !size.is_finite() || size <= 0.0 || !fill.price.get().is_finite() {
            return 0.0;
        }
        self.cap_leverage(&mut fill);
        self.cap_spot_short(&mut fill);
        self.cap_spot_buy(&mut fill);
        if reduce_only {
            let pos = self.account.position.qty.get();
            let reduces =
                (pos > 0.0 && fill.side == Side::Sell) || (pos < 0.0 && fill.side == Side::Buy);
            if !reduces {
                return 0.0;
            }
            let size = fill.size.get().min(pos.abs());
            if size <= 0.0 {
                return 0.0;
            }
            fill.size = Qty(size);
        }
        let applied = fill.size.get();
        let fee = self.cost.fee(&fill);

        // Snapshot the position before the fill so any portion it closes can be
        // booked as a trade.
        let pos_before = self.account.position.qty.get();
        let entry_before = self.account.position.avg_entry.get();
        let realized_before = self.account.realized();

        self.account.apply_fill(&fill, fee);

        let pos_after = self.account.position.qty.get();

        // The base amount the fill moved the position toward zero: a reduce keeps
        // the same-side remainder, a close or flip keeps none of the old side.
        let kept = if (pos_before > 0.0) == (pos_after > 0.0) {
            pos_after.abs()
        } else {
            0.0
        };
        let closed = pos_before.abs() - kept;

        if closed > CLOSE_EPS && self.reporter.is_some() {
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
                exit_price: fill.price.get(),
                // Prorate the fill's fee onto the portion that closed.
                fees: fee * closed / applied,
                pnl: self.account.realized() - realized_before,
                bars_held: self.tick.saturating_sub(self.position_entry_tick),
            };
            if let Some(reporter) = self.reporter.as_mut() {
                reporter.record_trade(trade);
            }
        }

        // Stamp the open tick whenever a new side is established, from flat or
        // through a flip, so the next close measures its holding time from here.
        let opened_new_side =
            pos_after != 0.0 && (pos_before == 0.0 || (pos_before > 0.0) != (pos_after > 0.0));
        if opened_new_side {
            self.position_entry_tick = self.tick;
        }

        applied
    }

    fn resolve_resting(&mut self, bar: &Candle) {
        let mut fills: Vec<(OrderId, Fill, OrderType, bool)> = Vec::new();
        for order in self.book.iter() {
            let fill = match order.kind {
                OrderType::Limit => self.fill_model.fill_limit(order, bar),
                OrderType::Stop => self.fill_model.fill_stop(order, bar),
                OrderType::Market => None,
            };
            if let Some(fill) = fill {
                // FOK limit: skip a fill that cannot cover the whole remaining; the
                // sweep below then cancels it unfilled.
                if order.kind == OrderType::Limit
                    && order.tif == TimeInForce::Fok
                    && fill.size.get() + CLOSE_EPS < order.remaining().get()
                {
                    continue;
                }
                fills.push((order.id, fill, order.kind, order.reduce_only));
            }
        }

        for (id, fill, kind, reduce_only) in fills {
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
        self.reporter
            .as_ref()
            .map(|reporter| reporter.stats(self.config.quote, periods_per_year, risk_free))
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
            open_orders: self.book.iter().copied().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Engine, EngineConfig};
    use crate::candles::Candles;
    use emsl_core::{Candle, Market, OrderType, Side, TimeInForce};

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
        assert_eq!(e.stop(Side::Sell, f64::INFINITY, 90.0), None);
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
        e.stop(Side::Sell, 1.0, 95.0); // placed at bar 0
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
}
