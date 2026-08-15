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

/// How badly a fill ends for the position currently held: the price PnL per unit
/// it would realize, so the most adverse sorts first. A fill that does not reduce
/// the position is not part of the question a wide bar asks, and sorts last.
fn adversity(position: f64, entry: f64, fill: &Fill) -> f64 {
    let reduces =
        (position > 0.0 && fill.side == Side::Sell) || (position < 0.0 && fill.side == Side::Buy);
    if !reduces {
        return f64::INFINITY;
    }
    (fill.price.get() - entry) * position.signum()
}

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
        // The same guard the resting book has carried since ADR 0027, extended to
        // the queue on the day ADR 0047 made the queue a bounded resource too. A
        // size that can never fill used to take a slot anyway, so two of them ahead
        // of a real order refused it: `qty_from_weight` returns exactly zero on a
        // flat mark, which is how a strategy places one without meaning to.
        if !size.is_finite() || size <= 0.0 {
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
        let (slot, old) = self.book.cancel_at(id)?;
        let size = size.unwrap_or(old.remaining().get());
        let placed = match old.kind {
            OrderType::Limit => price.or(old.price.map(|p| p.get())).and_then(|price| {
                self.place_limit(
                    old.side,
                    size,
                    price,
                    old.reduce_only,
                    old.post_only,
                    old.tif,
                )
            }),
            OrderType::Stop => trigger
                .or(old.trigger.map(|t| t.get()))
                .and_then(|trigger| self.place_stop(old.side, size, trigger, old.reduce_only)),
            // a market order never rests, so it can never be found to replace
            OrderType::Market => None,
        };
        if placed.is_none() {
            // The cancel came first, so a replacement the book refuses would
            // otherwise leave nothing behind and still answer `None`, which this
            // method defines as "nothing happened". A caller trailing a stop reads
            // that as "it has already filled" and stops arming one, so a refused
            // size or a non-finite trigger silently disarmed a live stop-loss. The
            // slot it came from is still free, since nothing was placed on this
            // path, so putting the old order back cannot fail and its id stays
            // valid (ADR 0032).
            //
            // Back in that same slot, not wherever `place` would put it. `place`
            // takes the FIRST free slot (ADR 0006), so an order whose neighbour had
            // been cancelled was silently promoted up the queue by a replacement
            // that was REFUSED, and resolution follows slot order: on a spot book
            // with the cash clamp binding, the promoted order took the cash the
            // other one would have had. "Nothing happened" has to include the
            // queue (ADR 0083).
            self.book.restore(slot, old);
        }
        placed
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

            // 0. the price a position's margin runs out at is a point on this bar's
            //    own path, and everything past it belongs to a market the account
            //    had already been closed out of. So the fence is worked out before
            //    any order is resolved. A bar that OPENED past it killed the account
            //    before it opened, so it is closed first and nothing on the bar
            //    fills; on any other bar the orders are resolved against a candle
            //    clipped at the fence, so an exit that would have filled beyond it
            //    simply never triggers. Checking only afterwards, as this used to,
            //    left ADR 0052's guarantee resting on the position still being open
            //    at the end of the bar: a `close()` on a bar that gapped past the
            //    liquidation booked the whole gap and left the account owing money,
            //    and a partial close left a "liquidation" priced outside the bar
            //    that booked a profit (ADR 0067).
            let long = self.account.position.qty.get() >= 0.0;
            let fence = self.liquidation_fence(&bar);
            let gapped = fence.is_some_and(|f| if long { bar.open <= f } else { bar.open >= f });
            if gapped {
                self.liquidate();
                self.bust = true;
            }
            let resolving = match fence {
                Some(f) if !gapped => Engine::fenced(&bar, f, long),
                _ => bar,
            };

            // 1. pending market orders fill at the open
            let pending = std::mem::take(&mut self.pending);
            for order in pending {
                if let Some(fill) = self.fill_model.fill_market(&order, &resolving) {
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
            self.resolve_resting(&resolving);

            // 3. funding at each interval boundary, on the position held at the
            //    funding event (so after this bar's fills, not the position carried
            //    into the bar), marked at the close, and before liquidation so a
            //    funding debit can bust the account this bar (ADR 0017).
            //
            //    At the close of the RESOLVING bar, which is the fenced one: this
            //    read `bar.close` and so marked a liquidated account at a price on
            //    the far side of the point its margin ran out. `fenced` calls
            //    itself the honest picture of a market the account was not in, and
            //    every order on the bar is resolved against it; funding was the one
            //    event that reached past it. A short liquidated on a bar that kept
            //    rising was CREDITED for the whole rise, which overstates
            //    funding_paid on any run with a liquidation in it and, at the
            //    margin, hands the account enough to clear the bust check it had
            //    already failed (ADR 0082)
            if self.config.funding_interval > 0 && self.tick % self.config.funding_interval == 0 {
                self.funding_paid += self
                    .account
                    .apply_funding(self.config.funding_rate, Price(resolving.close));
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
        // real trade, so it moves nothing and pays no fee. Neither is a price at or
        // below zero, and that one is the input both risk clamps have to give up
        // on: each divides by the mark, so each returns without clamping anything,
        // and the fill that follows buys an unbounded position for nothing or for a
        // credit. MAX_SLIP exists so a slipped fill cannot reach zero (ADR 0024);
        // this is the same rule applied to the bar's own price, which the candle
        // validator lets through because it checks only for finiteness (ADR 0027)
        let size = fill.size.get();
        let price = fill.price.get();
        if !size.is_finite() || size <= 0.0 || !price.is_finite() || price <= 0.0 {
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

    /// The price this bar would take the position's margin to zero at, or `None`
    /// when nothing on the bar can kill the account.
    ///
    /// The trigger is the bar's adverse extreme and the answer is the bankruptcy
    /// price, which are two different numbers on purpose (ADR 0052). Read before
    /// the bar's orders are resolved, because it is the point past which this
    /// account was no longer in the market.
    fn liquidation_fence(&self, bar: &Candle) -> Option<f64> {
        let adverse = if self.account.position.qty.get() >= 0.0 {
            bar.low
        } else {
            bar.high
        };
        if !self.account.is_bust_at(Price(adverse)) {
            return None;
        }
        self.account
            .bankruptcy_price(self.cost.fee_taker)
            .map(|price| price.get())
    }

    /// The bar as far as this account got: a long sees nothing below `fence` and a
    /// short nothing above it. Clamping is monotone, so the OHLC ordering survives
    /// it, and a bar lying entirely past the fence collapses to the fence itself,
    /// which is the honest picture of a market the account was not in.
    fn fenced(bar: &Candle, fence: f64, long: bool) -> Candle {
        let clip = |price: f64| {
            if long {
                price.max(fence)
            } else {
                price.min(fence)
            }
        };
        Candle {
            open: clip(bar.open),
            high: clip(bar.high),
            low: clip(bar.low),
            close: clip(bar.close),
            volume: bar.volume,
        }
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

    /// No fill closes a perp past the point the margin behind it ran out.
    ///
    /// `fenced` clips the whole candle, and it is read once from the position
    /// carried INTO the bar, which is the approximation ADR 0067 accepted on the
    /// grounds that its error runs pessimistic. It does not, in two shapes. A
    /// position GROWN during the bar and exited on the same bar is bounded by a
    /// fence computed for a smaller position, or by no fence at all when the
    /// account entered the bar flat, so a market entry and a protective stop
    /// armed in one decision booked straight through the margin and left the
    /// account owing money. And a resting limit prices at its own limit, which
    /// never passes through the taker clamp, so it walks out of the fenced candle
    /// from underneath. Bounding the fill instead of the bar covers both, because
    /// the bound is read at the moment the fill lands rather than a step earlier
    /// (ADR 0094).
    ///
    /// Closing at the bankruptcy price leaves equity exactly zero AT that price,
    /// so a partial close is bounded by the same number as a full one and no
    /// ordering of fills within a bar can book past it.
    fn bound_by_margin(&self, fill: &mut Fill) {
        if self.config.market != Market::Perp {
            return;
        }
        let position = self.account.position.qty.get();
        let long = position > 0.0;
        let reduces =
            (long && fill.side == Side::Sell) || (position < 0.0 && fill.side == Side::Buy);
        if !reduces {
            return;
        }
        if let Some(bound) = self.account.bankruptcy_price(self.cost.fee_taker) {
            let price = fill.price.get();
            fill.price = Price(if long {
                price.max(bound.get())
            } else {
                price.min(bound.get())
            });
        }
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
        let mut fill = self.clamp_fill(reduce_only, &fill);
        if fill.size.get() <= 0.0 {
            return 0.0;
        }
        self.bound_by_margin(&mut fill);
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
    ///
    /// Thirteen boundary mutants survive here and ALL THIRTEEN are equivalent or
    /// sit on an unreachable branch. Measured, not argued: probes asserting the
    /// three invariants below ran against 179 Rust tests, 547 Python tests and
    /// 3,600 randomised differential cases without firing once.
    ///
    ///   `applied` is never zero or less. `apply_fill_clamped` returns before
    ///   calling this on a non-positive size and `liquidate` never books a flat
    ///   position, so the two `applied > 0.0` guards and the `else` that carries
    ///   `open_fee` without a fee share are all dead. Four mutants.
    ///
    ///   `closed` is never between zero and CLOSE_EPS. It comes from the
    ///   position's own dust rule (ADR 0023), so it is either exactly zero or
    ///   comfortably above the epsilon, and `>` cannot be told from `>=`. That
    ///   also makes the `&&` in the entry-fee guard equivalent to `||`: at
    ///   `closed == 0` the share is `open_fee * 0 / pos` either way. Four mutants.
    ///
    ///   `kept` is only ever read through `(pos_before.abs() - kept).max(0.0)`,
    ///   and every sign comparison in it collapses at exactly zero: opening from
    ///   flat closes nothing under both readings, and a full close closes the
    ///   whole old side under both. Two mutants, and two more on the new-side
    ///   stamp, which is guarded by `pos_after != 0.0` before either sign is read.
    ///
    /// So the count here overstates the gap. What this function actually does is
    /// pinned by value: `a_trade_carries_both_sides_of_its_fee`,
    /// `adding_to_a_position_does_not_double_charge_its_entry_fee` and the two
    /// partial-close tests.
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

        // A bar that reaches both a stop and a target says nothing about which it
        // reached first, and the answer used to be slot order, which is the order
        // those two lines happen to appear in the strategy: place the stop first
        // and every ambiguous bar books a loss, swap the two lines and every one
        // books a win. They are ordered by how badly they end for the position
        // instead, worst first, which does not depend on how the strategy was
        // typed and is the same pessimism the rest of the engine already applies
        // (ADR 0056). The sort is stable, so anything that is not an exit keeps
        // its slot order.
        let position = self.account.position.qty.get();
        if fills.len() > 1 && position != 0.0 {
            let entry = self.account.position.avg_entry.get();
            // The exits are reordered among the slots the exits already hold, and
            // everything else stays exactly where it was. Sorting the whole list
            // instead moved every non-exit behind every exit, which is a different
            // rule than the one written down and not a harmless one: it let an
            // entry be funded by the proceeds of a sale that may just as well have
            // come after it, so the account ended a wide bar richer than slot order
            // (ADR 0006) would have left it, which is the optimism ADR 0056 exists
            // to remove rather than relocate.
            let slots: Vec<usize> = (0..fills.len())
                .filter(|&i| adversity(position, entry, &fills[i].1).is_finite())
                .collect();
            let mut ordered: Vec<usize> = slots.clone();
            ordered.sort_by(|&a, &b| {
                adversity(position, entry, &fills[a].1)
                    .partial_cmp(&adversity(position, entry, &fills[b].1))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let exits: Vec<_> = ordered.iter().map(|&i| fills[i]).collect();
            for (slot, exit) in slots.iter().zip(exits) {
                fills[*slot] = exit;
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
                // A stop is consumed by filling and by nothing else, which is what
                // ADR 0035 says and what it used to say only for a bar with no
                // volume at all. Triggering and being cancelled regardless meant a
                // bar carrying a thousandth of a unit filled a token size and threw
                // the rest of the protection away, while a bar carrying exactly
                // nothing kept all of it: on one crash that discontinuity was the
                // difference between exiting a 10-lot and riding 9.999 of it down.
                // It also deleted a breakout stop outright whenever a cash or margin
                // clamp took the fill to zero. So it accumulates and rests like a
                // limit, and the remainder stays armed for the next bar that can
                // actually trade it (ADR 0068).
                OrderType::Stop | OrderType::Limit => {
                    if applied > 0.0 {
                        if let Some(order) = self.book.get_mut(id) {
                            order.filled = Qty(order.filled.get() + applied);
                        }
                    }
                    // retired at the dust epsilon, not at zero, because that is
                    // where `clamp_fill` stops booking: an order whose remaining
                    // lands in between can never fill again and never expires, so
                    // it holds a book slot for the rest of the run. Ten fills of
                    // 0.1 against a size of 1.0 leave 1.1e-16, which is all it
                    // takes. Enough of those and the book is full of orders that
                    // will never trade, and every later placement is refused in
                    // silence, which is the harm ADR 0027 names (ADR 0079)
                    let done = self
                        .book
                        .get(id)
                        .is_some_and(|o| o.remaining().get() <= CLOSE_EPS);
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
    use super::{adversity, Engine, EngineConfig, CLOSE_EPS};
    use crate::candles::Candles;
    use emsl_core::{Candle, Fill, Market, OrderType, Price, Qty, Side, State, TimeInForce};
    use proptest::prelude::*;

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
    fn impact_scales_with_the_fraction_actually_filled() {
        // both existing fixtures take exactly a tenth of their bar, so they are
        // degenerate in the one dimension the claim is about and two wrong
        // implementations reproduce them exactly: a hard-coded tenth, and the
        // REQUESTED size over the volume rather than the filled one, which are the
        // same number whenever max_fill_fraction does not bind. Here it binds, so
        // requested, capped and a tenth are three different fractions (ADR 0108)
        let wide = Candles::new(vec![
            ohlc(100.0, 200.0, 90.0, 100.0, 1000.0),
            ohlc(100.0, 200.0, 90.0, 100.0, 1000.0),
            ohlc(100.0, 200.0, 90.0, 100.0, 1000.0),
        ]);
        // both caps have to BIND on a request of 400, or the fraction is the
        // request's own and the second mutant is indistinguishable again
        for (cap, filled, price) in [(0.25, 250.0, 125.0), (0.2, 200.0, 120.0)] {
            let mut config = cfg();
            config.quote = 1_000_000.0;
            config.impact = 1.0;
            config.max_fill_fraction = cap;
            let mut e = Engine::new(wide.clone(), config);
            e.reset();
            e.market_buy(400.0); // more than either cap allows
            let s = e.step();
            assert_eq!(s.position, filled, "cap {cap} filled the wrong size");
            // impact 1.0 times the FILLED fraction, off an open of 100
            assert_eq!(
                s.quote,
                1_000_000.0 - filled * price,
                "cap {cap} priced the fill at something other than {price}"
            );
        }
    }

    #[test]
    fn a_fok_limit_the_cash_cannot_afford_fills_nothing() {
        // ADR 0025 says all or nothing against EVERY clamp, not just the bar's
        // liquidity, and the resting path judges that with a second copy of the
        // rule the pending path already has. Only the pending one was pinned:
        // a_fok_limit_fills_fully_or_not_at_all is refused by the volume cap
        // before any clamp is consulted, so the case the decision actually names,
        // where the bar HAS the liquidity and a clamp shrinks the fill, was never
        // reached on a limit. Comparing the pre-clamp size here books the five
        // units the quote affords out of the twenty asked (ADR 0108)
        let mut config = cfg();
        config.quote = 1_000.0;
        config.report = true;
        let mut e = Engine::new(dip_series(), config);
        e.reset();
        e.order(
            Side::Buy,
            20.0,
            OrderType::Limit,
            Some(200.0),
            None,
            false,
            false,
            TimeInForce::Fok,
        );
        let s = e.step();
        assert_eq!(
            s.position, 0.0,
            "a FOK the cash cannot cover filled in part"
        );
        assert_eq!(s.quote, 1_000.0);
        assert_eq!(e.num_fills(), 0);
        assert!(s.open_orders.is_empty());
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

    /// Perp at 10x with a fee, reporting on, over `crash_series`.
    fn perp_10x() -> EngineConfig {
        EngineConfig {
            market: Market::Perp,
            quote: 100.0,
            fee_taker: 0.001,
            max_leverage: 10.0,
            report: true,
            ..cfg()
        }
    }

    #[test]
    fn a_stop_armed_with_the_entry_cannot_book_past_the_margin() {
        // the fence is read from the position carried INTO the bar, and an account
        // that entered flat carries none, so the stop reduced a position the
        // margin no longer stood behind and booked the whole distance. Placing a
        // stop was four times worse than placing nothing (ADR 0094)
        let mut e = Engine::new(crash_series(), perp_10x());
        e.reset();
        e.market_buy(10.0);
        e.stop(Side::Sell, 10.0, 90.0, true);
        let s = e.step();
        assert!(s.equity >= -CLOSE_EPS, "account owes {}", s.equity);

        // and it is no worse than the same bar with nothing resting on it
        let mut bare = Engine::new(crash_series(), perp_10x());
        bare.reset();
        bare.market_buy(10.0);
        let alone = bare.step();
        assert!((s.equity - alone.equity).abs() < CLOSE_EPS);
    }

    #[test]
    fn a_partial_exit_that_empties_the_margin_books_no_winning_liquidation() {
        // the residual was liquidated off a NEGATIVE quote, which solves to a price
        // on the far side of the entry: a profit booked on a bar that never traded
        // there, lifting win_rate and profit_factor while the return read -100
        let mut e = Engine::new(crash_series(), perp_10x());
        e.reset();
        // the trigger is BELOW the bankruptcy price of 901/9.99, about 90.19, so
        // the partial exit alone loses more than the whole margin: six units at
        // 82 is 108 against 99 of it, which leaves the quote negative and four
        // units still open for the forced close to price off
        e.market_buy(10.0);
        e.stop(Side::Sell, 6.0, 82.0, true);
        let s = e.step();
        assert!(s.equity >= -CLOSE_EPS, "account owes {}", s.equity);
        for trade in e.reporter().expect("reporting is on").trades() {
            assert!(
                !(trade.liquidated && trade.net_pnl > 0.0),
                "liquidated at {} for a profit of {}",
                trade.exit_price,
                trade.net_pnl
            );
        }
    }

    #[test]
    fn a_resting_limit_cannot_price_an_exit_past_the_margin() {
        // fill_limit returns the limit itself and never passes through the taker
        // clamp, so it got out from under the fenced candle the trigger respected
        let mut e = Engine::new(crash_series(), perp_10x());
        e.reset();
        e.market_buy(10.0);
        e.limit_sell(10.0, 20.0);
        let s = e.step();
        assert!(s.equity >= -CLOSE_EPS, "account owes {}", s.equity);
    }

    #[test]
    fn a_healthy_account_prices_its_exit_where_it_asked() {
        // the bound must bite only where the margin has actually run out, or it
        // would quietly improve every ordinary exit
        let mut e = Engine::new(crash_series(), perp_10x());
        e.reset();
        e.market_buy(1.0); // a tenth of the cap, nowhere near the fence
        e.limit_sell(1.0, 99.0);
        e.step();
        let trades = e.reporter().expect("reporting is on").trades();
        assert_eq!(trades.len(), 1);
        assert!(
            (trades[0].exit_price - 99.0).abs() < 1e-9,
            "exited at {}",
            trades[0].exit_price
        );
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
    fn a_bar_reaching_both_a_stop_and_a_target_books_the_stop_either_way() {
        // the engine cannot know which the bar touched first. It used to answer
        // with slot order, so placing the stop before the target booked a loss on
        // every ambiguous bar and swapping the two lines booked a win: the sign of
        // a bracket strategy's result was decided by the order it was typed in
        for stop_first in [true, false] {
            let candles = Candles::new(vec![
                ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
                ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
                // one wide bar that reaches the stop at 90 AND the target at 110
                ohlc(100.0, 120.0, 80.0, 100.0, 1000.0),
                ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
            ]);
            let mut e = Engine::new(candles, report_cfg()); // spot, so a sell clamps
            e.reset();
            e.market_buy(1.0);
            e.step(); // long 1 at 100
            if stop_first {
                e.stop(Side::Sell, 1.0, 90.0, true);
                e.limit_sell(1.0, 110.0);
            } else {
                e.limit_sell(1.0, 110.0);
                e.stop(Side::Sell, 1.0, 90.0, true);
            }
            let s = e.step();
            assert_eq!(s.position, 0.0, "stop_first={stop_first}");
            let trades = e.reporter().expect("reporting is on").trades();
            assert_eq!(trades.len(), 1, "stop_first={stop_first}");
            assert_eq!(trades[0].exit_price, 90.0, "stop_first={stop_first}");
            assert!(trades[0].pnl < 0.0, "stop_first={stop_first}");
        }
    }

    #[test]
    fn ordering_the_exits_leaves_an_unambiguous_bar_alone() {
        // only a bar reaching BOTH legs is ambiguous; one that reaches the target
        // and never the stop still books the target
        let candles = Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
            ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
            ohlc(100.0, 120.0, 99.0, 115.0, 1000.0), // reaches 110, never 90
            ohlc(100.0, 101.0, 99.0, 100.0, 1000.0),
        ]);
        let mut e = Engine::new(candles, report_cfg());
        e.reset();
        e.market_buy(1.0);
        e.step();
        e.stop(Side::Sell, 1.0, 90.0, true);
        e.limit_sell(1.0, 110.0);
        e.step();
        let trades = e.reporter().expect("reporting is on").trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].exit_price, 110.0);
        assert!(trades[0].pnl > 0.0);
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

    // A first bar at 100, then a bar that opens at 50, far past where a 10x long
    // opened at 100 runs out of margin.
    fn gap_series() -> Candles {
        Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1_000.0),
            ohlc(100.0, 101.0, 99.0, 100.0, 1_000.0),
            ohlc(50.0, 50.0, 20.0, 30.0, 1_000.0),
            ohlc(30.0, 31.0, 29.0, 30.0, 1_000.0),
        ])
    }

    fn levered() -> EngineConfig {
        EngineConfig {
            max_leverage: 10.0,
            report: true,
            ..perp_cfg()
        }
    }

    #[test]
    fn closing_on_a_bar_that_gapped_past_the_liquidation_leaves_nothing_owed() {
        // ADR 0052 says bad debt is structurally unreachable rather than clamped,
        // and that rested on the position still being open when the check ran at
        // the end of the bar. An exit order flattened it first, so nothing was left
        // to liquidate and the account booked the whole gap: a 10x long on 100 of
        // margin ended at -400. The run with the exit must land where the run
        // without one does, because the account died before either could matter
        for exit in [false, true] {
            let mut e = Engine::new(gap_series(), levered());
            e.reset();
            e.market_buy(10.0);
            e.step();
            if exit {
                e.close();
            }
            let state = e.step();
            assert_eq!(state.equity, 0.0, "exit={exit} left {}", state.equity);
            let trades = e.reporter().expect("reporting").trades();
            assert_eq!(trades.len(), 1);
            assert_eq!(trades[0].exit_price, 90.0);
            assert!(trades[0].liquidated, "exit={exit} booked no liquidation");
        }
    }

    #[test]
    fn a_stop_below_the_bankruptcy_price_never_fills_and_one_above_it_does() {
        // the fence is a price, not a moment: coming down from the open the market
        // reaches 95 before 90, so a stop at 95 saves the account and one at 60 is
        // beyond a point it had already been closed at
        let bars = Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1_000.0),
            ohlc(100.0, 101.0, 99.0, 100.0, 1_000.0),
            ohlc(100.0, 101.0, 50.0, 60.0, 1_000.0),
            ohlc(60.0, 61.0, 59.0, 60.0, 1_000.0),
        ]);
        for (trigger, want_equity, want_exit) in [(95.0, 50.0, 95.0), (60.0, 0.0, 90.0)] {
            let mut e = Engine::new(bars.clone(), levered());
            e.reset();
            e.market_buy(10.0);
            e.step();
            e.stop(Side::Sell, 10.0, trigger, true);
            let state = e.step();
            assert_eq!(state.equity, want_equity, "stop at {trigger}");
            let trades = e.reporter().expect("reporting").trades();
            assert_eq!(trades[0].exit_price, want_exit, "stop at {trigger}");
        }
    }

    #[test]
    fn a_liquidation_never_books_a_price_outside_the_bar_that_triggered_it() {
        // a partial close on the gap bar drove quote negative, and the bankruptcy
        // solve is derived assuming quote is positive, so it landed on the far side
        // of the entry: the forced close booked a PROFIT at 120 on a bar whose high
        // was 72, and the log showed a liquidated winner
        let bars = Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
            ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
            ohlc(70.0, 72.0, 60.0, 65.0, 1e6),
            ohlc(65.0, 66.0, 64.0, 65.0, 1e6),
        ]);
        let mut e = Engine::new(bars, levered());
        e.reset();
        e.market_buy(10.0);
        e.step();
        e.market_sell(6.0);
        let state = e.step();
        assert_eq!(state.equity, 0.0);
        for trade in e.reporter().expect("reporting").trades() {
            assert!(trade.pnl <= 0.0, "a liquidation booked {}", trade.pnl);
            assert!(trade.exit_price <= 101.0, "exited at {}", trade.exit_price);
        }
    }

    #[test]
    fn a_fill_priced_at_or_below_zero_is_refused_rather_than_booked() {
        // both risk clamps divide by the mark, so each returns without clamping when
        // it is zero or negative, and the fill behind them bought an unbounded
        // position for nothing. The candle validator lets those prices through
        // because it checks only finiteness (ADRs 0024, 0027)
        for price in [0.0, -100.0] {
            let bars = Candles::new(vec![
                ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
                ohlc(price, price, price, price, 1e6),
                ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ]);
            let mut e = Engine::new(bars, cfg());
            e.reset();
            e.market_buy(1_000.0);
            e.step();
            let state = e.step();
            assert_eq!(state.position, 0.0, "price {price} opened a position");
            assert_eq!(state.equity, 10_000.0, "price {price} minted equity");
        }
    }

    #[test]
    fn a_refused_replacement_leaves_the_original_order_resting() {
        // replace cancels before it places, so a replacement the book refuses left
        // nothing behind and still answered None, which this method defines as
        // "nothing happened". A trail reads that as "it already filled" and stops
        // arming one, so the protective stop was silently gone (ADR 0032)
        let mut e = Engine::new(series(), perp_cfg());
        e.reset();
        e.market_buy(1.0);
        e.step();
        let id = e.stop(Side::Sell, 1.0, 90.0, true).expect("stop rests");
        for refused in [
            e.replace(id, None, None, Some(f64::NAN)),
            e.replace(id, Some(0.0), None, None),
        ] {
            assert!(refused.is_none());
        }
        let resting = e.current_state().open_orders;
        assert_eq!(resting.len(), 1);
        assert_eq!(resting[0].id, id);
        assert_eq!(
            resting[0].trigger.expect("a stop keeps its trigger").get(),
            90.0
        );
    }

    #[test]
    fn a_refused_replacement_leaves_the_original_where_it_was_in_the_queue() {
        // ADR 0032 defines a None from `replace` as "nothing happened", and ADR 0069
        // restores the original so that is true of the book too. It was not true of
        // the order's PRIORITY. The restore goes through `place`, which takes the
        // FIRST free slot (ADR 0006), so an order whose neighbour had been cancelled
        // was silently promoted up the queue by a replacement that was refused.
        //
        // Resolution follows slot order, and slot order decides who gets the cash
        // when the spot clamp binds, so the promotion moves a number. Sixty of cash
        // against a 40 and a 50: the one resolved first fills whole and the other
        // takes what is left.
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 35.0, 100.0, 1e6),
        ]);
        let config = EngineConfig {
            quote: 60.0,
            ..cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        let first = e.limit_buy(1.0, 10.0).expect("rests in the first slot");
        let kept = e.limit_buy(1.0, 50.0).expect("rests in the second");
        e.step();

        // the first slot is free now, and a refused replacement must not take it
        assert!(e.cancel(first));
        assert_eq!(e.replace(kept, Some(-1.0), None, None), None);
        e.limit_buy(1.0, 40.0)
            .expect("takes the slot the cancel freed");
        e.step();

        // the 40 resolves first and fills whole, leaving 20 of the 60, so the 50
        // fills 0.4 of its 1.0. Promote the 50 and it fills first for the whole 50,
        // leaving 10 against the 40, which is 1.25
        let state = e.current_state();
        assert!(
            (state.position - 1.4).abs() < 1e-9,
            "position {} says the restored order jumped the queue",
            state.position
        );
    }

    #[test]
    fn a_stop_is_consumed_by_filling_and_by_nothing_else() {
        // ADR 0035 binds triggering to filling, and that held only for a bar with no
        // volume at all. A cash clamp that took the fill to zero deleted the stop
        // just as thoroughly, and it is the same protection either way
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 150.0, 100.0, 150.0, 1e6),
            ohlc(150.0, 150.0, 150.0, 150.0, 1e6),
        ]);
        let mut e = Engine::new(
            bars,
            EngineConfig {
                quote: 100.0,
                ..cfg()
            },
        );
        e.reset();
        e.market_buy(1.0);
        e.stop(Side::Buy, 1.0, 110.0, false);
        let state = e.step();
        assert_eq!(e.num_fills(), 1, "the market order took all the cash");
        assert_eq!(state.open_orders.len(), 1, "the breakout stop was deleted");
    }

    #[test]
    fn a_partly_filled_stop_keeps_the_rest_of_its_protection() {
        // consuming a stop on any fill at all made a bar carrying a thousandth of a
        // unit far more dangerous than a bar carrying none: the token fill threw the
        // remaining protection away, while zero volume kept all of it
        for volume in [0.0, 0.001, 1.0] {
            let bars = Candles::new(vec![
                ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
                ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
                ohlc(100.0, 101.0, 80.0, 85.0, volume),
                ohlc(85.0, 86.0, 60.0, 60.0, 1e6),
                ohlc(60.0, 60.0, 60.0, 60.0, 1e6),
            ]);
            let mut e = Engine::new(
                bars,
                EngineConfig {
                    quote: 100_000.0,
                    ..perp_cfg()
                },
            );
            e.reset();
            e.market_buy(10.0);
            e.step();
            e.stop(Side::Sell, 10.0, 95.0, true);
            e.step();
            e.step();
            let state = e.step();
            assert!(
                state.position.abs() <= CLOSE_EPS,
                "volume {volume} left {} riding the crash",
                state.position
            );
        }
    }

    #[test]
    fn a_market_size_that_cannot_fill_is_refused_a_queue_slot() {
        // the queue became a bounded resource with ADR 0047 and did not inherit
        // ADR 0027's guard, so two sizes that can never fill refused a real order.
        // qty_from_weight returns exactly zero on a flat mark, which is how a
        // strategy places one without meaning to
        let mut e = Engine::new(
            series(),
            EngineConfig {
                max_open_orders: 2,
                ..cfg()
            },
        );
        e.reset();
        assert!(e.market_buy(f64::NAN).is_none());
        assert!(e.market_buy(-1.0).is_none());
        assert!(e.market_buy(0.0).is_none());
        assert!(e.market_buy(1.0).is_some());
        assert_eq!(e.step().position, 1.0);
    }

    #[test]
    fn the_worst_first_sort_leaves_everything_that_is_not_an_exit_where_it_was() {
        // ADR 0056 says the sort applies only to fills that reduce the position, and
        // ranking a non-exit at infinity moved every one of them behind every exit.
        // That let an entry be funded by the proceeds of a sale that may well have
        // come after it, so the account ended the bar richer than slot order leaves it
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 120.0, 80.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
        ]);
        let mut e = Engine::new(
            bars,
            EngineConfig {
                quote: 150.0,
                ..cfg()
            },
        );
        e.reset();
        e.market_buy(1.0);
        e.step();
        e.limit_buy(1.0, 90.0); // slot 0, an increase
        e.limit_sell(1.0, 110.0); // slot 1, an exit
        let state = e.step();
        // slot order: the buy is clamped to the 50 of cash on hand, then the sell
        assert!(
            (state.position - 50.0 / 90.0).abs() < 1e-9,
            "{}",
            state.position
        );
        assert!(
            (state.equity - 165.5555555555).abs() < 1e-6,
            "{}",
            state.equity
        );
    }

    // eight flat bars, so nothing moves but the position
    fn flat_series() -> Candles {
        Candles::new(
            (0..8)
                .map(|_| ohlc(100.0, 100.0, 100.0, 100.0, 1e6))
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn an_order_filled_down_to_float_dust_gives_its_slot_back() {
        // the fill gate refuses a size at or below the dust epsilon and the retire
        // gate asked for zero, so an order left holding the difference could never
        // trade again and never expired. Ten fills of 0.1 against a size of 1.0
        // leave 1.1e-16 behind, which is enough to hold a slot for the whole run;
        // a strategy resting orders on a long series eventually finds every
        // placement silently refused (ADR 0079)
        let bars: Vec<Candle> = (0..14)
            .map(|_| ohlc(100.0, 100.0, 100.0, 100.0, 0.1))
            .collect();
        let mut e = Engine::new(Candles::new(bars), cfg());
        e.reset();
        e.limit_buy(1.0, 100.0);
        for _ in 0..12 {
            e.step();
        }
        let state = e.current_state();
        assert!(
            (state.position - 1.0).abs() < 1e-9,
            "the order should have filled out: {}",
            state.position
        );
        assert!(
            state.open_orders.is_empty(),
            "a filled order kept its slot with {:?} left",
            state.open_orders.first().map(|o| o.remaining().get())
        );
    }

    #[test]
    fn a_holding_time_is_measured_from_when_the_side_was_opened() {
        // bars_held rests on one predicate deciding whether a fill opened a new
        // side, and every branch of it was reachable only through a whole run, so
        // six single-line mutations of it survived. Adding to a position must not
        // restart the clock, a partial close must not either, and a flip must
        let mut e = Engine::new(
            flat_series(),
            EngineConfig {
                report: true,
                ..perp_cfg()
            },
        );
        e.reset();
        e.market_buy(1.0);
        e.step(); // tick 1: long 1 opens here
        e.market_buy(1.0);
        e.step(); // tick 2: long 2, still opened at 1
        e.market_sell(1.0);
        e.step(); // tick 3: closes 1 of it, held 3 - 1
        e.market_sell(3.0);
        e.step(); // tick 4: closes the last 1, held 4 - 1, and opens short 2 here
        e.market_buy(2.0);
        e.step(); // tick 5: closes the short, held 5 - 4

        let held: Vec<usize> = e
            .reporter()
            .expect("reporting")
            .trades()
            .iter()
            .map(|t| t.bars_held)
            .collect();
        assert_eq!(held, vec![2, 3, 1]);
    }

    #[test]
    fn a_partial_close_takes_its_share_of_the_entry_fee_and_no_more() {
        // ADR 0030's whole point: a trade carries its round trip, so the entry fee
        // is split across the closes by size and consumed exactly once. The
        // arithmetic that does it had no test of its own
        let config = EngineConfig {
            fee_taker: 0.01,
            report: true,
            ..perp_cfg()
        };
        let mut e = Engine::new(flat_series(), config);
        e.reset();
        e.market_buy(2.0);
        e.step(); // entry fee is 2 * 100 * 1% = 2.0
        e.market_sell(1.0);
        e.step(); // exit fee 1.0, plus half the entry fee
        e.market_sell(1.0);
        e.step(); // exit fee 1.0, plus the other half

        let fees: Vec<f64> = e
            .reporter()
            .expect("reporting")
            .trades()
            .iter()
            .map(|t| t.fees)
            .collect();
        assert_eq!(fees, vec![2.0, 2.0]);
        // and the whole round trip is exactly what the account paid
        let paid: f64 = fees.iter().sum();
        assert!((paid - 4.0).abs() < 1e-9, "{paid}");
    }

    #[test]
    fn a_flip_must_fit_the_cap_outright_where_a_grow_may_keep_what_it_has() {
        // ADR 0012's allowance: a position already over the cap because equity
        // fell is not force-reduced, so a same-side add may keep it, but a flip
        // closes that side and opens a fresh one which gets no such credit. The
        // two branches only differ when the position IS over the cap, and nothing
        // reached that state, so the predicate deciding it was unpinned
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(90.0, 90.0, 90.0, 90.0, 1e6),
            ohlc(90.0, 90.0, 90.0, 90.0, 1e6),
        ]);
        let config = EngineConfig {
            quote: 1_000.0,
            max_leverage: 3.0,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(30.0);
        let state = e.step(); // long 30 at 100, exactly the cap
        assert_eq!(state.position, 30.0);

        // the mark falls to 90, so equity is 700 and the cap is now 23.33 while
        // the position is still 30: over it, and a flip gets no allowance for that
        e.market_sell(60.0);
        let state = e.step();
        assert!(
            (state.position + 3.0 * 700.0 / 90.0).abs() < 1e-9,
            "flipped to {} rather than the cap",
            state.position
        );
    }

    #[test]
    fn a_reduce_only_order_on_the_wrong_side_fills_nothing_either_way() {
        // the predicate has a branch per side and only one of them was ever taken
        for (opening, wrong) in [(Side::Buy, Side::Buy), (Side::Sell, Side::Sell)] {
            let mut e = Engine::new(flat_series(), perp_cfg());
            e.reset();
            e.order(
                opening,
                1.0,
                OrderType::Market,
                None,
                None,
                false,
                false,
                TimeInForce::Ioc,
            );
            let state = e.step();
            let held = state.position;
            assert!(held.abs() > 0.0);

            e.order(
                wrong,
                5.0,
                OrderType::Market,
                None,
                None,
                true,
                false,
                TimeInForce::Ioc,
            );
            let state = e.step();
            assert_eq!(
                state.position, held,
                "a reduce-only {wrong:?} grew the position"
            );

            // and the right side reduces, clamped to what is actually held
            let closing = if held > 0.0 { Side::Sell } else { Side::Buy };
            e.order(
                closing,
                5.0,
                OrderType::Market,
                None,
                None,
                true,
                false,
                TimeInForce::Ioc,
            );
            let state = e.step();
            assert_eq!(state.position, 0.0);
        }
    }

    fn exit_at(side: Side, price: f64) -> Fill {
        Fill {
            side,
            size: Qty(1.0),
            price: Price(price),
            is_taker: true,
        }
    }

    #[test]
    fn adversity_ranks_an_exit_by_what_it_realizes_and_sorts_everything_else_last() {
        // the whole of ADR 0056 rests on this eight-line function, and the suite
        // reached it only through the engine, where one wide bar exercises one
        // branch. Every comparison and every term is pinned here instead
        let long_loss = adversity(2.0, 100.0, &exit_at(Side::Sell, 90.0));
        let long_win = adversity(2.0, 100.0, &exit_at(Side::Sell, 110.0));
        assert_eq!(long_loss, -10.0);
        assert_eq!(long_win, 10.0);
        assert!(long_loss < long_win, "a long's worse exit must sort first");

        // a short is the mirror: the HIGHER price is the worse exit
        let short_loss = adversity(-2.0, 100.0, &exit_at(Side::Buy, 110.0));
        let short_win = adversity(-2.0, 100.0, &exit_at(Side::Buy, 90.0));
        assert_eq!(short_loss, -10.0);
        assert_eq!(short_win, 10.0);
        assert!(
            short_loss < short_win,
            "a short's worse exit must sort first"
        );

        // anything that is not an exit sorts last and never reorders (ADR 0071)
        for (position, side) in [
            (2.0, Side::Buy),   // adding to a long
            (-2.0, Side::Sell), // adding to a short
            (0.0, Side::Buy),   // opening from flat
            (0.0, Side::Sell),
        ] {
            assert_eq!(
                adversity(position, 100.0, &exit_at(side, 90.0)),
                f64::INFINITY,
                "position {position} with a {side:?} is not an exit"
            );
        }
        assert!(long_loss < f64::INFINITY);
    }

    #[test]
    fn a_wick_that_busts_a_long_liquidates_even_when_the_bar_closes_back_above_it() {
        // ADR 0003 puts the trigger at the bar's ADVERSE EXTREME, the low for a
        // long, because a bar that traded there is a bar the account was closed out
        // on whatever it printed afterwards. Every other liquidation fixture ends
        // at or below its own bust level, so marking the check at bar.close passes
        // all of them; this bar wicks to 85 and closes back at 100, where a
        // close-marked check reads an equity of 100 and lets a dead account trade on.
        //
        // The favourable wick reaches 130, which is TWICE as far from the entry as
        // the adverse one: a check reading whichever extreme is furthest from the
        // entry, rather than the one that is adverse to this side, marks the long
        // at 130 and finds it healthy. On a bar wicking to 85 and 101 the two
        // readings agree, which is why the high is 130 and not 101
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 130.0, 85.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
        ]);
        let mut e = Engine::new(bars, levered());
        e.reset();
        e.market_buy(10.0); // 1000 of notional on 100 of margin, filled at the open of 100
        let state = e.step();
        // margin at the low is 100 + 10 * (85 - 100) = -50, so the account died on
        // the way down and is closed where the margin ran out, at
        // (10 * 100 - 100) / 10 = 90, leaving exactly nothing (ADR 0052)
        assert!(e.is_bust());
        assert_eq!(state.position, 0.0);
        assert_eq!(state.equity, 0.0);
        assert_eq!(state.bar_close, 100.0); // the bar recovered, the account did not
        let trades = e.reporter().expect("reporting").trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].exit_price, 90.0);
        assert!(trades[0].liquidated);
    }

    #[test]
    fn a_wick_that_busts_a_short_liquidates_even_when_the_bar_closes_back_below_it() {
        // the mirror of the rule above, and no engine test has ever liquidated a
        // short at all: a short's adverse extreme is the HIGH, so a wick up that
        // takes the margin kills it however calmly the bar ends (ADR 0003). A check
        // reading bar.close, or one that kept the long's low for both sides, reads
        // 100 here and carries a dead short into the next bar. The favourable wick
        // reaches 70, twice as far from the entry as the adverse one, so a check
        // reading the furthest extreme rather than the adverse one marks the short
        // at 70 and finds it richer than ever
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 115.0, 70.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
        ]);
        let mut e = Engine::new(bars, levered());
        e.reset();
        e.market_sell(10.0); // short 10 at the open of 100, on 100 of margin
        let state = e.step();
        // margin at the high is 100 - 10 * (115 - 100) = -50, and the margin runs
        // out at (-10 * 100 - 100) / -10 = 110, which is where it is closed
        assert!(e.is_bust());
        assert_eq!(state.position, 0.0);
        assert_eq!(state.equity, 0.0);
        assert_eq!(state.bar_close, 100.0); // the bar came back, the account did not
        let trades = e.reporter().expect("reporting").trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].exit_price, 110.0);
        assert!(trades[0].liquidated);
    }

    #[test]
    fn funding_on_a_liquidating_bar_marks_at_the_fence_and_not_at_the_raw_close() {
        // ADR 0082. Every order on a fenced bar resolves against the clipped
        // candle, which `fenced` calls the honest picture of a market the account
        // was not in, and funding reached past it to the raw close. This short
        // dies at 110 on a bar that closes at 128, so the two readings are 11.0
        // and 12.8: the second credits the account for 18 points of a rise it had
        // already been closed out of, and funding_paid is reported
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 130.0, 100.0, 128.0, 1e6),
        ]);
        let config = EngineConfig {
            funding_rate: 0.01,
            funding_interval: 2, // tick 1 is not a boundary, tick 2 is
            ..levered()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_sell(10.0); // short 10 at the open of 100, on 100 of margin
        let state = e.step();
        assert_eq!(state.position, -10.0);
        assert_eq!(state.funding_paid, 0.0, "tick 1 is not a funding boundary");

        // the margin runs out at 110, so the bar the account was in closes there
        let state = e.step();
        assert!(
            (state.funding_paid + 11.0).abs() < 1e-9,
            "funding was marked at {}, not at the fence",
            state.funding_paid / -0.1
        );
        assert!(
            (state.funding_paid + 12.8).abs() > 1e-9,
            "funding was marked at the raw close of a bar the account never saw"
        );
        // and the credit lands before the liquidation, so the exit moves with it:
        // 100 of margin plus 11 of funding runs out at 111.1 rather than 110
        assert!(e.is_bust());
        assert_eq!(state.position, 0.0);
        // ADR 0052's nothing left, to the last bit the funding arithmetic allows:
        // 100 of margin, 11 of credit and an exit at 111.1 land 5.7e-14 off zero
        assert!(state.equity.abs() < 1e-9);
        let trades = e.reporter().expect("reporting").trades();
        assert_eq!(trades.len(), 1);
        assert!((trades[0].exit_price - 111.1).abs() < 1e-9);
        assert!(trades[0].liquidated);
    }

    #[test]
    fn an_over_cap_position_is_kept_whole_refuses_an_add_and_reduces_only_what_was_asked() {
        // ADR 0012's allowance runs one way only: a position pushed over its cap by
        // a fall in equity is not force-reduced, and it may not be extended either.
        // Dropping the max(pos.abs()) from the cap turns this refusal into a fill,
        // and the fill is on the SAME side, so the order that was meant to be
        // blocked is the one that grows the long from 30 to 36.67.
        //
        // The refusal alone does not pin "kept whole", because an add and a cap
        // that clamps to max_size both answer with a zero fill here. The reduce at
        // the end is what separates them: it asks for 5 of the 30 and must get
        // exactly 5, where a cap applied to every fill sells 6.67 to land the
        // position on the cap itself, which is a liquidation the account did not owe
        let bars = Candles::new(vec![
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(100.0, 100.0, 100.0, 100.0, 1e6),
            ohlc(90.0, 90.0, 90.0, 90.0, 1e6),
            ohlc(90.0, 90.0, 90.0, 90.0, 1e6),
        ]);
        let config = EngineConfig {
            quote: 1_000.0,
            max_leverage: 3.0,
            ..perp_cfg()
        };
        let mut e = Engine::new(bars, config);
        e.reset();
        e.market_buy(30.0);
        let state = e.step(); // long 30 at 100: 3000 of notional on 1000, exactly the cap
        assert_eq!(state.position, 30.0);
        assert_eq!(e.num_fills(), 1);

        // the mark falls to 90, so equity is 1000 + 30 * (90 - 100) = 700 and the
        // cap is 3 * 700 / 90 = 23.33 against a position of 30: over it, and kept
        e.market_buy(5.0);
        let state = e.step();
        assert_eq!(
            state.position, 30.0,
            "the add moved a position it may not move"
        );
        assert_eq!(state.equity, 700.0);
        assert_eq!(state.quote, 1_000.0);
        assert_eq!(e.num_fills(), 1, "the refused add booked a fill");

        // and a reduce is the size that was asked for, not the size that would put
        // the position back inside the cap: 30 less 5 is 25, still over the 23.33
        // the cap allows, and it stays there
        e.market_sell(5.0);
        let state = e.step();
        assert_eq!(
            state.position, 25.0,
            "the reduce was widened to land the position on the cap"
        );
        assert_eq!(state.equity, 700.0);
        assert_eq!(e.num_fills(), 2);
        assert!(!e.is_bust());
    }

    #[test]
    fn funding_fires_on_the_absolute_bar_index_not_on_bars_since_the_reset() {
        // ADRs 0002 and 0017: the funding schedule belongs to the series, so every
        // env in a vectorized run charges the same bars however its episode was
        // offset. Only an offset start separates the two counters, and nothing did
        // that: starting at bar 2 with an interval of 3 funds ticks 3 and 6, while
        // a per-episode counter funds the third bar of the episode, tick 5, and
        // nothing else in this window
        let config = EngineConfig {
            quote: 10_000.0,
            funding_rate: 0.0625, // one sixteenth, so each charge is exactly 6.25
            funding_interval: 3,
            ..perp_cfg()
        };
        let mut e = Engine::new(flat_series(), config); // eight flat bars at 100
        e.reset_at(2);
        e.market_buy(1.0); // fills at the open of bar 3, ahead of that bar's funding
        let mut charged = Vec::new();
        for _ in 0..5 {
            let state = e.step();
            charged.push((state.tick_index, state.funding_paid));
        }
        // 1 unit at 100 pays 6.25 a time, on tick 3 and on tick 6 and on no other
        assert_eq!(
            charged,
            vec![(3, 6.25), (4, 6.25), (5, 6.25), (6, 12.5), (7, 12.5)]
        );
        assert_eq!(e.current_state().quote, 9_987.5);
    }

    #[test]
    fn a_bar_that_gaps_up_past_a_shorts_bankruptcy_price_closes_it_at_the_fence() {
        // ADR 0067 works the fence out before the bar's orders are resolved, and it
        // faces the position's OWN direction: a short dies going up, so a bar that
        // OPENS above its bankruptcy price is a bar the account was already closed
        // out on. Every fence fixture is a 10x long, so the short side of that
        // comparison was never taken; with the long's test a close() here books the
        // raw open of 150 and leaves the account owing 400 (ADR 0052)
        let bars = Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
            ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
            ohlc(150.0, 180.0, 140.0, 160.0, 1e6),
            ohlc(160.0, 161.0, 159.0, 160.0, 1e6),
        ]);
        for exit in [false, true] {
            let mut e = Engine::new(bars.clone(), levered());
            e.reset();
            e.market_sell(10.0); // short 10 at 100, on 100 of margin
            e.step();
            if exit {
                e.close();
            }
            let state = e.step();
            // the margin runs out at (-10 * 100 - 100) / -10 = 110 and the bar
            // opened past it, so the exit is priced there and leaves nothing
            assert_eq!(state.position, 0.0, "exit={exit}");
            assert_eq!(state.equity, 0.0, "exit={exit} left {}", state.equity);
            let trades = e.reporter().expect("reporting").trades();
            assert_eq!(trades.len(), 1, "exit={exit}");
            assert_eq!(trades[0].exit_price, 110.0, "exit={exit}");
            assert!(trades[0].liquidated, "exit={exit} booked no liquidation");
        }
    }

    #[test]
    fn a_shorts_stop_above_the_bankruptcy_price_never_fills_and_one_below_it_does() {
        // the mirror of the long fence: a short's bar is clipped at the fence from
        // ABOVE, so coming up from the open the market reaches 105 before 110 and a
        // buy stop there saves the account, while one at 140 sits beyond a price
        // the account had already been closed at. Clipping a short's bar with the
        // long's max() leaves that 140 reachable and books an exit 30 past the
        // point the margin was gone (ADR 0067)
        let bars = Candles::new(vec![
            ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
            ohlc(100.0, 101.0, 99.0, 100.0, 1e6),
            ohlc(100.0, 150.0, 99.0, 140.0, 1e6),
            ohlc(140.0, 141.0, 139.0, 140.0, 1e6),
        ]);
        for (trigger, want_equity, want_exit) in [(105.0, 50.0, 105.0), (140.0, 0.0, 110.0)] {
            let mut e = Engine::new(bars.clone(), levered());
            e.reset();
            e.market_sell(10.0);
            e.step(); // short 10 at 100
            e.stop(Side::Buy, 10.0, trigger, true);
            let state = e.step();
            assert_eq!(state.equity, want_equity, "stop at {trigger}");
            let trades = e.reporter().expect("reporting").trades();
            assert_eq!(trades[0].exit_price, want_exit, "stop at {trigger}");
        }
    }

    /// A random walk built into consistent OHLC bars, violent enough to liquidate a
    /// 10x perp in a single step, because that is the path the guarantees have to
    /// survive rather than the one they are comfortable on.
    fn walk(len: usize) -> impl Strategy<Value = Vec<Candle>> {
        prop::collection::vec(
            (-0.3f64..0.3, 0.0f64..0.25, 0.0f64..0.25, 0.0f64..2_000.0),
            len,
        )
        .prop_map(|steps| {
            let mut price = 100.0;
            steps
                .into_iter()
                .map(|(ret, up, down, volume)| {
                    let open = price;
                    let close = (open * (1.0 + ret)).max(0.01);
                    let bar = ohlc(
                        open,
                        open.max(close) * (1.0 + up),
                        (open.min(close) * (1.0 - down)).max(0.001),
                        close,
                        volume,
                    );
                    price = close;
                    bar
                })
                .collect()
        })
    }

    fn walk_and_actions(
        bars: usize,
        acts: usize,
    ) -> impl Strategy<Value = (Vec<Candle>, Vec<f64>)> {
        (walk(bars), prop::collection::vec(-3.0f64..3.0, acts))
    }

    /// A perp with every cost switched on and reporting, so nothing is left out of
    /// the paths these properties have to hold across.
    ///
    /// `impact` is deliberately extreme. At 0.5 against a bar the order takes all
    /// of, the raw slip is 50%, which is what found ADR 0074: the price left the
    /// bar entirely and the loss it booked was bounded by nothing, so the bad-debt
    /// guarantee fell over from outside the liquidation. Holding a taker price
    /// inside its bar is what makes the property below hold at this coefficient.
    fn every_cost() -> EngineConfig {
        EngineConfig {
            market: Market::Perp,
            quote: 10_000.0,
            fee_taker: 0.0006,
            fee_maker: 0.0002,
            slippage_bps: 5.0,
            max_fill_fraction: 1.0,
            max_open_orders: 8,
            report: true,
            max_leverage: 10.0,
            impact: 0.5,
            funding_rate: 0.0001,
            funding_interval: 3,
        }
    }

    // one signed action per bar: positive buys, negative sells, and the magnitude is
    // the fraction of equity to put on
    /// One action per bar, and the magnitude picks the ORDER KIND as well as the
    /// size, so the same generator reaches a resting book.
    ///
    /// This drove market orders alone for as long as it existed, which is what
    /// let three separate bad-debt paths live under a property test written to
    /// say there was no fourth one: every one of them needed an order still
    /// resting when the bar resolved, and no run this ever produced had one.
    /// A stop armed away from the close and a limit armed through it are the two
    /// shapes that reach the book, and the protective pair is what a reader of
    /// the guide actually writes (ADR 0094).
    fn drive(engine: &mut Engine, actions: &[f64]) -> Vec<State> {
        let mut seen = Vec::with_capacity(actions.len());
        for action in actions {
            let weight = action.abs();
            let size = engine.qty_from_weight(weight);
            let close = engine.current_close();
            let buying = *action > 0.0;
            if buying {
                engine.market_buy(size);
            } else if *action < 0.0 {
                engine.market_sell(size);
            }
            // a protective stop armed in the SAME decision as the entry, which is
            // what the guide teaches and what nothing here used to place. The
            // trigger is further out than the liquidation distance at this
            // leverage on purpose: that is the case where the account dies before
            // the stop does, so the stop is the fill that reduces a position the
            // margin no longer stands behind (ADR 0094)
            if weight > 1.0 {
                let (side, trigger) = if buying {
                    (Side::Sell, close * 0.90)
                } else {
                    (Side::Buy, close * 1.10)
                };
                engine.stop(side, size, trigger, true);
            }
            // and a marketable resting exit, which prices at its own limit rather
            // than at anything the bar clamped
            if weight > 2.0 {
                if buying {
                    engine.limit_sell(size, close * 0.90);
                } else {
                    engine.limit_buy(size, close * 1.10);
                }
            }
            seen.push(engine.step());
            if engine.done() {
                break;
            }
        }
        seen
    }

    proptest! {
        #[test]
        fn a_perp_never_ends_a_bar_owing_money((bars, actions) in walk_and_actions(50, 40)) {
            // ADRs 0052 and 0067: bad debt is unreachable rather than caught, so no
            // path through a violent series at full leverage may take equity below
            // zero on any bar. The point tests pin the three shapes that broke it;
            // this is the one that says there is no fourth shape
            let mut e = Engine::new(Candles::new(bars), every_cost());
            e.reset();
            for state in drive(&mut e, &actions) {
                prop_assert!(
                    state.equity >= -1e-6,
                    "equity {} at tick {}",
                    state.equity,
                    state.tick_index
                );
            }
        }

        #[test]
        fn a_state_reconciles_its_own_cash_and_position(
            (bars, actions) in walk_and_actions(40, 30),
        ) {
            // the fields the caller reads have to agree with each other: on a perp
            // equity is cash plus the open position's unrealized PnL, and on spot it
            // is cash plus the base holding marked at the close
            for market in [Market::Spot, Market::Perp] {
                let config = EngineConfig { market, ..every_cost() };
                let mut e = Engine::new(Candles::new(bars.clone()), config);
                e.reset();
                for state in drive(&mut e, &actions) {
                    let expected = if market == Market::Spot {
                        state.quote + state.position * state.mark_price
                    } else {
                        state.quote + state.unrealized_pnl
                    };
                    prop_assert!(
                        (state.equity - expected).abs() <= 1e-6 * state.equity.abs().max(1.0),
                        "{:?} equity {} against {}",
                        market,
                        state.equity,
                        expected
                    );
                }
            }
        }

        #[test]
        fn no_state_depends_on_a_bar_after_it(
            (bars, actions) in walk_and_actions(50, 40),
            cut in 5usize..35,
        ) {
            // ADR 0065 promises an order decided on bar i fills against bar i + 1 and
            // never a later one, so rewriting every bar past the cut has to leave
            // every state up to it identical. Lookahead has exactly this shape
            let mut altered = bars.clone();
            for bar in altered.iter_mut().skip(cut + 1) {
                *bar = ohlc(7.0, 900.0, 0.5, 3.0, 5.0);
            }
            let mut first = Engine::new(Candles::new(bars), every_cost());
            first.reset();
            let mut second = Engine::new(Candles::new(altered), every_cost());
            second.reset();
            let left = drive(&mut first, &actions);
            let right = drive(&mut second, &actions);
            for i in 0..cut.min(left.len()).min(right.len()) {
                prop_assert_eq!(&left[i], &right[i], "diverged at index {}", i);
            }
        }

        #[test]
        fn every_statistic_stays_finite_or_deliberately_infinite(
            (bars, actions) in walk_and_actions(50, 40),
            ppy in prop_oneof![
                Just(0.0), Just(-1.0), Just(f64::NAN), Just(365.0), Just(525_600.0)
            ],
        ) {
            // ADRs 0007, 0029, 0046 and 0072: nothing is ever NaN, and the only
            // infinity is the deliberate one a reward earned against no measured risk
            // goes to. A high annualization is ordinary input, since ADR 0048 reads
            // it off the candles and minute bars are a normal feed
            let mut e = Engine::new(Candles::new(bars), every_cost());
            e.reset();
            drive(&mut e, &actions);
            let s = e.stats(ppy, 0.02).expect("reporting is on");
            for (name, value) in [
                ("total_return_pct", s.total_return_pct),
                ("cagr_pct", s.cagr_pct),
                ("max_drawdown_pct", s.max_drawdown_pct),
                ("volatility_pct", s.volatility_pct),
                ("exposure_pct", s.exposure_pct),
                ("win_rate", s.win_rate),
                ("avg_trade_pct", s.avg_trade_pct),
                ("funding_paid", s.funding_paid),
            ] {
                prop_assert!(value.is_finite(), "{} was {}", name, value);
            }
            for (name, value) in [
                ("sharpe", s.sharpe),
                ("sortino", s.sortino),
                ("calmar", s.calmar),
                ("profit_factor", s.profit_factor),
            ] {
                prop_assert!(!value.is_nan(), "{} was NaN", name);
                prop_assert!(
                    value.is_finite() || value == f64::INFINITY,
                    "{} was {}",
                    name,
                    value
                );
            }
            prop_assert!(s.max_drawdown_pct <= 100.0 + 1e-9);
        }
    }
}
