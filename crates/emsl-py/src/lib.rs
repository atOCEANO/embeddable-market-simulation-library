//! emsl-py: the PyO3 binding layer, compiled to the `emsl._emsl` extension. This
//! is the only Python-linked crate, and it is a thin shell over `bar-engine`: it
//! parses arguments, copies the candles once from a numpy array into the
//! Rust-owned series, and hands `State` back as a plain dict. No simulation logic
//! lives here.
//!
//! The crate-level allow silences a clippy false positive: pyo3's `#[pymethods]`
//! macro emits an identity `PyErr` conversion in the trampolines it generates for
//! `PyResult`-returning methods, which clippy reports against our return types.
//! The conversion is the macro's, not ours.
#![allow(clippy::useless_conversion)]

use numpy::ndarray::{Array3, ArrayView2};
use numpy::npyffi::flags::NPY_ARRAY_WRITEABLE;
use numpy::{
    PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use bar_engine::{
    builtin_strategy, run_sweep, Candles, Engine as BarEngine, EngineConfig, EnvBatch, Stats, Trade,
};
use emsl_core::{Candle, Market, Order, OrderId, OrderStatus, OrderType, Side, State, TimeInForce};

/// Parse a market string into the core enum.
fn parse_market(market: &str) -> PyResult<Market> {
    match market.to_ascii_lowercase().as_str() {
        "spot" => Ok(Market::Spot),
        "perp" => Ok(Market::Perp),
        other => Err(PyValueError::new_err(format!(
            "market must be 'spot' or 'perp', got '{other}'"
        ))),
    }
}

/// Parse a side string into the core enum.
fn parse_side(side: &str) -> PyResult<Side> {
    match side.to_ascii_lowercase().as_str() {
        "buy" => Ok(Side::Buy),
        "sell" => Ok(Side::Sell),
        other => Err(PyValueError::new_err(format!(
            "side must be 'buy' or 'sell', got '{other}'"
        ))),
    }
}

/// Parse an order-type string into the core enum.
fn parse_order_type(kind: &str) -> PyResult<OrderType> {
    match kind.to_ascii_lowercase().as_str() {
        "market" => Ok(OrderType::Market),
        "limit" => Ok(OrderType::Limit),
        "stop" => Ok(OrderType::Stop),
        other => Err(PyValueError::new_err(format!(
            "type must be 'market', 'limit', or 'stop', got '{other}'"
        ))),
    }
}

/// Parse a time-in-force string into the core enum.
fn parse_tif(tif: &str) -> PyResult<TimeInForce> {
    match tif.to_ascii_uppercase().as_str() {
        "GTC" => Ok(TimeInForce::Gtc),
        "IOC" => Ok(TimeInForce::Ioc),
        "FOK" => Ok(TimeInForce::Fok),
        other => Err(PyValueError::new_err(format!(
            "tif must be 'GTC', 'IOC', or 'FOK', got '{other}'"
        ))),
    }
}

fn side_str(side: Side) -> &'static str {
    match side {
        Side::Buy => "buy",
        Side::Sell => "sell",
    }
}

fn kind_str(kind: OrderType) -> &'static str {
    match kind {
        OrderType::Market => "market",
        OrderType::Limit => "limit",
        OrderType::Stop => "stop",
    }
}

fn status_str(status: OrderStatus) -> &'static str {
    match status {
        OrderStatus::Resting => "resting",
        OrderStatus::Filled => "filled",
        OrderStatus::Canceled => "canceled",
    }
}

/// Copy a `(T, 5)` OHLCV array into an owned candle series. Columns are open,
/// high, low, close, volume. This is the one input copy; observation windows
/// handed back later are the zero-copy path.
fn candles_from_array(array: PyReadonlyArray2<'_, f64>) -> PyResult<Candles> {
    let view = array.as_array();
    let cols = view.ncols();
    if cols != 5 {
        return Err(PyValueError::new_err(format!(
            "candles must have 5 columns (open, high, low, close, volume), got {cols}"
        )));
    }
    let bars: Vec<Candle> = view
        .outer_iter()
        .map(|row| Candle {
            open: row[0],
            high: row[1],
            low: row[2],
            close: row[3],
            volume: row[4],
        })
        .collect();
    if bars.is_empty() {
        return Err(PyValueError::new_err("candles must have at least one row"));
    }
    // Reject NaN and infinity: the spec's input contract says NaN rows are rejected,
    // and a non-finite candle would silently poison every mark and equity downstream.
    for (i, bar) in bars.iter().enumerate() {
        if !(bar.open.is_finite()
            && bar.high.is_finite()
            && bar.low.is_finite()
            && bar.close.is_finite()
            && bar.volume.is_finite())
        {
            return Err(PyValueError::new_err(format!(
                "candles contain a non-finite value at row {i}; clean NaN and inf rows first"
            )));
        }
    }
    Ok(Candles::new(bars))
}

/// Read an optional per-env cost-override array, checked to length `num_envs`.
/// `None` leaves the batch on its shared scalar for that field (ADR 0014).
fn per_env_vec(
    name: &str,
    num_envs: usize,
    array: &Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Option<Vec<f64>>> {
    match array {
        Some(a) => {
            let slice = a.as_slice().map_err(|_| {
                PyValueError::new_err(format!("{name} must be a contiguous 1-D float64 array"))
            })?;
            if slice.len() != num_envs {
                return Err(PyValueError::new_err(format!(
                    "{name} length {} must equal num_envs {num_envs}",
                    slice.len()
                )));
            }
            Ok(Some(slice.to_vec()))
        }
        None => Ok(None),
    }
}

fn order_to_dict<'py>(py: Python<'py>, order: &Order) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("id", order.id.0)?;
    d.set_item("side", side_str(order.side))?;
    d.set_item("kind", kind_str(order.kind))?;
    d.set_item("price", order.price.map(|p| p.get()))?;
    d.set_item("trigger", order.trigger.map(|t| t.get()))?;
    d.set_item("size", order.size.get())?;
    d.set_item("filled", order.filled.get())?;
    d.set_item("remaining", order.remaining().get())?;
    d.set_item("status", status_str(order.status))?;
    d.set_item("reduce_only", order.reduce_only)?;
    d.set_item("post_only", order.post_only)?;
    Ok(d)
}

fn state_to_dict<'py>(py: Python<'py>, state: &State) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("tick_index", state.tick_index)?;
    d.set_item("base", state.base)?;
    d.set_item("quote", state.quote)?;
    d.set_item("position", state.position)?;
    d.set_item("avg_entry", state.avg_entry)?;
    d.set_item("equity", state.equity)?;
    d.set_item("mark_price", state.mark_price)?;
    d.set_item("bar_open", state.bar_open)?;
    d.set_item("bar_high", state.bar_high)?;
    d.set_item("bar_low", state.bar_low)?;
    d.set_item("bar_close", state.bar_close)?;
    d.set_item("bar_volume", state.bar_volume)?;
    d.set_item("realized_pnl", state.realized_pnl)?;
    d.set_item("unrealized_pnl", state.unrealized_pnl)?;
    let orders = PyList::empty_bound(py);
    for order in &state.open_orders {
        orders.append(order_to_dict(py, order)?)?;
    }
    d.set_item("open_orders", orders)?;
    Ok(d)
}

fn states_to_list<'py>(py: Python<'py>, states: &[State]) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty_bound(py);
    for state in states {
        list.append(state_to_dict(py, state)?)?;
    }
    Ok(list)
}

fn stats_to_dict<'py>(py: Python<'py>, stats: &Stats) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("total_return_pct", stats.total_return_pct)?;
    d.set_item("net_profit_pct", stats.net_profit_pct)?;
    d.set_item("cagr_pct", stats.cagr_pct)?;
    d.set_item("sharpe", stats.sharpe)?;
    d.set_item("sortino", stats.sortino)?;
    d.set_item("calmar", stats.calmar)?;
    d.set_item("max_drawdown_pct", stats.max_drawdown_pct)?;
    d.set_item("volatility_pct", stats.volatility_pct)?;
    d.set_item("exposure_pct", stats.exposure_pct)?;
    d.set_item("win_rate", stats.win_rate)?;
    d.set_item("profit_factor", stats.profit_factor)?;
    d.set_item("num_trades", stats.num_trades)?;
    d.set_item("avg_trade_pct", stats.avg_trade_pct)?;
    Ok(d)
}

fn trade_to_dict<'py>(py: Python<'py>, trade: &Trade) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("entry_tick", trade.entry_tick)?;
    d.set_item("exit_tick", trade.exit_tick)?;
    d.set_item("side", side_str(trade.side))?;
    d.set_item("size", trade.size)?;
    d.set_item("entry_price", trade.entry_price)?;
    d.set_item("exit_price", trade.exit_price)?;
    d.set_item("fees", trade.fees)?;
    d.set_item("pnl", trade.pnl)?;
    d.set_item("bars_held", trade.bars_held)?;
    Ok(d)
}

/// Build a read-only, zero-copy `(rows, 5)` numpy view over `window`, a slice of an
/// engine's immutable candle buffer. `container` (the engine pyobject, which owns
/// the `Arc`) becomes the array's base so the data outlives the view, and the array
/// is made read-only because the buffer is shared across envs. This is the one place
/// the crate reinterprets candle memory as floats; both `observation` and `data` go
/// through it (ADR 0008).
fn candle_view<'py>(window: &[Candle], container: Bound<'py, PyAny>) -> Bound<'py, PyArray2<f64>> {
    let rows = window.len();

    // SAFETY: Candle is repr(C) with five f64 fields and no padding (guarded by
    // emsl-core's layout test), so a `&[Candle]` of length `rows` aliases a `&[f64]`
    // of length `rows * 5`, row-major in open/high/low/close/volume.
    let flat: &[f64] =
        unsafe { std::slice::from_raw_parts(window.as_ptr() as *const f64, rows * 5) };
    let view =
        ArrayView2::from_shape((rows, 5), flat).expect("candle slice length is exactly rows * 5");

    // SAFETY: `view` points into the immutable candle buffer, which outlives the
    // array because `container` (the engine, holding the Arc) is its base. The buffer
    // is never mutated and the array is made read-only just below, so this alias into
    // shared data can never be written through.
    let array = unsafe { PyArray2::borrow_from_array_bound(&view, container) };
    // SAFETY: the array was just created and is not yet shared with Python, so
    // clearing its WRITEABLE flag races with nothing; a read-only view is required
    // because the candle buffer is shared across every env in a batch.
    unsafe {
        (*array.as_array_ptr()).flags &= !(NPY_ARRAY_WRITEABLE as std::os::raw::c_int);
    }
    array
}

/// The market simulation engine, exposed to Python as `emsl.Engine`. It wraps one
/// single-env bar engine; orders decided on a bar fill on the next.
#[pyclass]
struct Engine {
    inner: BarEngine,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (
        candles,
        market = "spot",
        quote = 10_000.0,
        fee_taker = 0.0006,
        fee_maker = 0.0002,
        slippage_bps = 0.0,
        max_fill_fraction = 1.0,
        max_open_orders = 8,
        report = false,
        leverage = 10.0,
        impact = 0.0,
        funding_rate = 0.0,
        funding_interval = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        candles: PyReadonlyArray2<'_, f64>,
        market: &str,
        quote: f64,
        fee_taker: f64,
        fee_maker: f64,
        slippage_bps: f64,
        max_fill_fraction: f64,
        max_open_orders: usize,
        report: bool,
        leverage: f64,
        impact: f64,
        funding_rate: f64,
        funding_interval: usize,
    ) -> PyResult<Engine> {
        let config = EngineConfig {
            market: parse_market(market)?,
            quote,
            fee_taker,
            fee_maker,
            slippage_bps,
            max_fill_fraction,
            max_open_orders,
            report,
            max_leverage: leverage,
            impact,
            funding_rate,
            funding_interval,
        };
        let series = candles_from_array(candles)?;
        Ok(Engine {
            inner: BarEngine::new(series, config),
        })
    }

    /// Reset to the first bar and return the initial state.
    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = self.inner.reset();
        state_to_dict(py, &state)
    }

    /// Advance one bar, resolve the orders placed since the last step, mark the
    /// account, and return the new state.
    fn step<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = self.inner.step();
        state_to_dict(py, &state)
    }

    /// True when there is no next bar to step into.
    fn done(&self) -> bool {
        self.inner.done()
    }

    /// The shape of the loaded candles as `(T, 5)`, the honesty check on how many
    /// bars the engine actually holds.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.inner.num_bars(), 5)
    }

    /// A zero-copy, read-only numpy view of the `lookback` bars ending at the
    /// current tick: shape `(rows, 5)`, columns open, high, low, close, volume.
    /// `rows` is `lookback`, or fewer early in the series. The array borrows the
    /// engine's immutable candle buffer directly with no copy; it is read-only
    /// because that buffer is shared across envs, and it holds the engine alive
    /// for as long as the view exists (ADR 0008).
    fn observation<'py>(
        slf: Bound<'py, Self>,
        lookback: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // The engine owns the Arc-backed candle buffer, so keeping it as the
        // array's base keeps the data alive for the whole life of the view.
        let container = slf.clone().into_any();
        let engine = slf.borrow();
        Ok(candle_view(engine.inner.candle_window(lookback), container))
    }

    /// A zero-copy, read-only numpy view of the whole candle series: shape
    /// `(T, 5)`, columns open, high, low, close, volume by position (close is
    /// column 3). Like `observation` it borrows the engine's immutable buffer with
    /// no copy and holds the engine alive for the view's life; it is the
    /// full-series companion, so `eng.data[:, 3]` reads every close (ADR 0008).
    #[getter]
    fn data<'py>(slf: Bound<'py, Self>) -> Bound<'py, PyArray2<f64>> {
        let container = slf.clone().into_any();
        let engine = slf.borrow();
        candle_view(engine.inner.candles_all(), container)
    }

    /// Queue a market buy; it fills on the next bar's open. Returns the order id.
    fn market_buy(&mut self, size: f64) -> u64 {
        self.inner.market_buy(size).0
    }

    /// Queue a market sell; it fills on the next bar's open. Returns the order id.
    fn market_sell(&mut self, size: f64) -> u64 {
        self.inner.market_sell(size).0
    }

    /// Rest a buy limit. Returns the order id, or None if the book is full.
    fn limit_buy(&mut self, size: f64, price: f64) -> Option<u64> {
        self.inner.limit_buy(size, price).map(|id| id.0)
    }

    /// Rest a sell limit. Returns the order id, or None if the book is full.
    fn limit_sell(&mut self, size: f64, price: f64) -> Option<u64> {
        self.inner.limit_sell(size, price).map(|id| id.0)
    }

    /// Rest a stop on `side` that triggers at `trigger`. None if the book is full.
    fn stop(&mut self, side: &str, size: f64, trigger: f64) -> PyResult<Option<u64>> {
        let side = parse_side(side)?;
        Ok(self.inner.stop(side, size, trigger).map(|id| id.0))
    }

    /// Flatten the position with a reduce-only market order. None when flat.
    fn close(&mut self) -> Option<u64> {
        self.inner.close().map(|id| id.0)
    }

    /// Cancel a resting order by id. True if it was found and removed.
    fn cancel(&mut self, order_id: u64) -> bool {
        self.inner.cancel(OrderId(order_id))
    }

    /// Cancel every resting order, returning how many were dropped. Pending market
    /// orders are not resting and still fill on the next step.
    fn cancel_all(&mut self) -> usize {
        self.inner.cancel_all()
    }

    /// The one order primitive the shortcuts wrap. `side` is "buy" or "sell",
    /// `type` is "market", "limit", or "stop", `tif` is "GTC", "IOC", or "FOK". A
    /// limit needs `price`, a stop needs `trigger`. Returns the order id, or None if
    /// the book is full or a post_only limit would cross. `tif` applies to market
    /// and limit orders; a stop rests until it triggers (ADR 0016).
    #[pyo3(signature = (
        side,
        size,
        r#type = "market",
        price = None,
        trigger = None,
        reduce_only = false,
        post_only = false,
        tif = "GTC",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn order(
        &mut self,
        side: &str,
        size: f64,
        r#type: &str,
        price: Option<f64>,
        trigger: Option<f64>,
        reduce_only: bool,
        post_only: bool,
        tif: &str,
    ) -> PyResult<Option<u64>> {
        let side = parse_side(side)?;
        let kind = parse_order_type(r#type)?;
        let tif = parse_tif(tif)?;
        if kind == OrderType::Limit && price.is_none() {
            return Err(PyValueError::new_err("a limit order requires a price"));
        }
        if kind == OrderType::Stop && trigger.is_none() {
            return Err(PyValueError::new_err("a stop order requires a trigger"));
        }
        Ok(self
            .inner
            .order(
                side,
                size,
                kind,
                price,
                trigger,
                reduce_only,
                post_only,
                tif,
            )
            .map(|id| id.0))
    }

    /// Base size for `fraction` of current equity, marked at the current close.
    fn qty_from_weight(&self, fraction: f64) -> f64 {
        self.inner.qty_from_weight(fraction)
    }

    /// Base size for a `cash` amount in quote, at the current close.
    fn qty_from_quote(&self, cash: f64) -> f64 {
        self.inner.qty_from_quote(cash)
    }

    /// Performance stats when reporting is on, else None. `periods_per_year`
    /// annualizes to the bar interval; `risk_free` is an annual rate.
    #[pyo3(signature = (periods_per_year = 365.0, risk_free = 0.0))]
    fn stats<'py>(
        &self,
        py: Python<'py>,
        periods_per_year: f64,
        risk_free: f64,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.inner.stats(periods_per_year, risk_free) {
            Some(stats) => Ok(Some(stats_to_dict(py, &stats)?)),
            None => Ok(None),
        }
    }

    /// The recorded equity curve as a numpy array, one point per step, or None
    /// when reporting is off.
    fn equity_curve<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .reporter()
            .map(|reporter| PyArray1::from_slice_bound(py, reporter.equity_curve()))
    }

    /// The closed trades as a list of dicts, or None when reporting is off.
    fn trades<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyList>>> {
        match self.inner.reporter() {
            Some(reporter) => {
                let list = PyList::empty_bound(py);
                for trade in reporter.trades() {
                    list.append(trade_to_dict(py, trade)?)?;
                }
                Ok(Some(list))
            }
            None => Ok(None),
        }
    }

    /// Drive `strategy` over the whole series and return the final state. Resets,
    /// calls `strategy.init(engine)` if it exists, then for each bar calls
    /// `strategy.next(state, engine)` and steps. The strategy places its orders
    /// through the engine's own methods (market, limit, stop, close), so it can
    /// express any action; it should not call `step` itself, since the driver
    /// does. This is the single-env callback path: `next` runs under the GIL, so
    /// it is the slow path next to `Batch.step_all`.
    fn run<'py>(
        slf: Bound<'py, Self>,
        strategy: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();

        // Each engine borrow below is scoped and dropped before Python runs, so
        // the callback is free to re-enter the engine (place orders) without a
        // double-borrow panic.
        let mut state = {
            let mut engine = slf.borrow_mut();
            let initial = engine.inner.reset();
            state_to_dict(py, &initial)?
        };

        if strategy.hasattr("init")? {
            strategy.call_method1("init", (slf.clone(),))?;
        }

        while !slf.borrow().inner.done() {
            strategy.call_method1("next", (state.clone(), slf.clone()))?;
            state = {
                let mut engine = slf.borrow_mut();
                let next = engine.inner.step();
                state_to_dict(py, &next)?
            };
        }

        Ok(state)
    }
}

/// A batch of independent envs over one shared candle series, stepped in
/// parallel. Every env sees the same candles (shared by `Arc`, no duplication)
/// but carries its own account and orders, so the batched step is bit-identical
/// to stepping each env in a loop. `step_all` releases the GIL around the
/// parallel fills, so many envs advance while Python threads run.
#[pyclass]
struct Batch {
    inner: EnvBatch,
}

#[pymethods]
impl Batch {
    #[new]
    #[pyo3(signature = (
        candles,
        num_envs,
        market = "spot",
        quote = 10_000.0,
        fee_taker = 0.0006,
        fee_maker = 0.0002,
        slippage_bps = 0.0,
        max_fill_fraction = 1.0,
        max_open_orders = 8,
        report = false,
        leverage = 10.0,
        impact = 0.0,
        funding_rate = 0.0,
        funding_interval = 0,
        fee_taker_per_env = None,
        fee_maker_per_env = None,
        slippage_bps_per_env = None,
        impact_per_env = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        candles: PyReadonlyArray2<'_, f64>,
        num_envs: usize,
        market: &str,
        quote: f64,
        fee_taker: f64,
        fee_maker: f64,
        slippage_bps: f64,
        max_fill_fraction: f64,
        max_open_orders: usize,
        report: bool,
        leverage: f64,
        impact: f64,
        funding_rate: f64,
        funding_interval: usize,
        fee_taker_per_env: Option<PyReadonlyArray1<'_, f64>>,
        fee_maker_per_env: Option<PyReadonlyArray1<'_, f64>>,
        slippage_bps_per_env: Option<PyReadonlyArray1<'_, f64>>,
        impact_per_env: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Batch> {
        let config = EngineConfig {
            market: parse_market(market)?,
            quote,
            fee_taker,
            fee_maker,
            slippage_bps,
            max_fill_fraction,
            max_open_orders,
            report,
            max_leverage: leverage,
            impact,
            funding_rate,
            funding_interval,
        };
        let series = candles_from_array(candles)?;

        // Per-env cost randomization (ADR 0014): each override, if given, replaces
        // its scalar field for the matching env; absent overrides keep the scalar.
        let ft = per_env_vec("fee_taker_per_env", num_envs, &fee_taker_per_env)?;
        let fm = per_env_vec("fee_maker_per_env", num_envs, &fee_maker_per_env)?;
        let sb = per_env_vec("slippage_bps_per_env", num_envs, &slippage_bps_per_env)?;
        let im = per_env_vec("impact_per_env", num_envs, &impact_per_env)?;

        let inner = if ft.is_some() || fm.is_some() || sb.is_some() || im.is_some() {
            let configs: Vec<EngineConfig> = (0..num_envs)
                .map(|i| {
                    let mut c = config;
                    if let Some(v) = &ft {
                        c.fee_taker = v[i];
                    }
                    if let Some(v) = &fm {
                        c.fee_maker = v[i];
                    }
                    if let Some(v) = &sb {
                        c.slippage_bps = v[i];
                    }
                    if let Some(v) = &im {
                        c.impact = v[i];
                    }
                    c
                })
                .collect();
            EnvBatch::with_configs(series, &configs)
        } else {
            EnvBatch::new(num_envs, series, config)
        };

        Ok(Batch { inner })
    }

    /// Number of envs.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Number of envs.
    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.len()
    }

    /// True when the envs have no next bar to step into.
    fn done(&self) -> bool {
        self.inner.done()
    }

    /// Reset every env, returning the per-env initial state dicts.
    fn reset_all<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let states = py.allow_threads(|| self.inner.reset_all());
        states_to_list(py, &states)
    }

    /// Apply the per-env `actions` as market orders (positive buys, negative
    /// sells, zero holds), then step every env one bar in parallel with the GIL
    /// released, returning the per-env state dicts. `actions` is a 1-D float64
    /// array of length num_envs, or None to step placing no new orders.
    #[pyo3(signature = (actions = None))]
    fn step_all<'py>(
        &mut self,
        py: Python<'py>,
        actions: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Bound<'py, PyList>> {
        // Read the actions into an owned Vec under the GIL; the numpy array must
        // not be touched once the GIL is released.
        let action_vec = self.action_vec(&actions)?;

        // GIL released: apply the actions and run the parallel step. The closure
        // touches only Rust data, never a Python object.
        let states = py.allow_threads(|| {
            if let Some(sizes) = &action_vec {
                for (engine, &size) in self.inner.engines_mut().iter_mut().zip(sizes) {
                    if size > 0.0 {
                        engine.market_buy(size);
                    } else if size < 0.0 {
                        engine.market_sell(-size);
                    }
                }
            }
            self.inner.step_all()
        });

        states_to_list(py, &states)
    }

    // --- Batched RL surface (used by emsl.rl.VectorEnv) ---

    /// Reset every env to its own start offset in `offsets` (length num_envs).
    fn reset_at_each(
        &mut self,
        py: Python<'_>,
        offsets: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<()> {
        let offs = self.offsets_vec(offsets)?;
        py.allow_threads(|| {
            self.inner.reset_at_each(&offs);
        });
        Ok(())
    }

    /// Reset only the envs where `mask` is true, each to its entry in `offsets`;
    /// the rest keep running. The autoreset primitive.
    fn reset_masked(
        &mut self,
        py: Python<'_>,
        mask: PyReadonlyArray1<'_, bool>,
        offsets: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<()> {
        let mask_slice = mask
            .as_slice()
            .map_err(|_| PyValueError::new_err("mask must be a contiguous 1-D bool array"))?;
        if mask_slice.len() != self.inner.len() {
            return Err(PyValueError::new_err(
                "mask length must equal num_envs".to_string(),
            ));
        }
        let mask_vec = mask_slice.to_vec();
        let offs = self.offsets_vec(offsets)?;
        py.allow_threads(|| {
            self.inner.reset_masked(&mask_vec, &offs);
        });
        Ok(())
    }

    /// Apply the per-env `actions` (positive buys, negative sells, zero holds) and
    /// step every env in parallel with the GIL released, discarding the states.
    /// The lean step for the RL loop, which reads observation and equity after.
    #[pyo3(signature = (actions = None))]
    fn advance(
        &mut self,
        py: Python<'_>,
        actions: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<()> {
        let action_vec = self.action_vec(&actions)?;
        py.allow_threads(|| {
            if let Some(sizes) = &action_vec {
                for (engine, &size) in self.inner.engines_mut().iter_mut().zip(sizes) {
                    if size > 0.0 {
                        engine.market_buy(size);
                    } else if size < 0.0 {
                        engine.market_sell(-size);
                    }
                }
            }
            self.inner.step_all();
        });
        Ok(())
    }

    /// Batched observation: the `window` most recent candles per env as an
    /// `(num_envs, window, 5)` float64 array, front-padded with zeros at warmup.
    fn observe<'py>(&self, py: Python<'py>, window: usize) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let n = self.inner.len();
        let flat = py.allow_threads(|| self.inner.observe_all(window));
        let array = Array3::from_shape_vec((n, window, 5), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray3::from_owned_array_bound(py, array))
    }

    /// Per-env equity as a `(num_envs,)` array, for building the reward.
    fn equities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec_bound(py, self.inner.equities())
    }

    /// Per-env truncation flags (reached the last bar) as a `(num_envs,)` array.
    fn dones<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_vec_bound(py, self.inner.dones())
    }

    /// Per-env termination flags (liquidated) as a `(num_envs,)` array.
    fn busts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_vec_bound(py, self.inner.busts())
    }

    /// Batched feature observation: the `window` most recent rows of the shared
    /// `(T, F)` feature matrix per env, as `(num_envs, window, F)` float64. The
    /// gather runs on the Rayon pool; the feature buffer is read in place.
    fn observe_features<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'_, f64>,
        window: usize,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let n_features = features.as_array().ncols();
        let n_rows = features.as_array().nrows();
        let num_bars = self.inner.num_bars();
        if n_rows != num_bars {
            return Err(PyValueError::new_err(format!(
                "features must have one row per candle: got {n_rows} rows for {num_bars} bars"
            )));
        }
        // Copy the feature buffer to an owned Vec under the GIL so the parallel
        // gather can run with the GIL released, like observe; the numpy array must
        // not be touched once the GIL is dropped.
        let owned: Vec<f64> = features
            .as_slice()
            .map_err(|_| PyValueError::new_err("features must be a contiguous 2-D float64 array"))?
            .to_vec();
        let n = self.inner.len();
        let out = py.allow_threads(|| self.inner.observe_features_all(window, &owned, n_features));
        let array = Array3::from_shape_vec((n, window, n_features), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray3::from_owned_array_bound(py, array))
    }

    /// A batched per-env account snapshot as a dict of `(num_envs,)` arrays, for
    /// building a reward: equity, position, unrealized_pnl, realized_pnl,
    /// mark_price.
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snap = self.inner.snapshot();
        let d = PyDict::new_bound(py);
        d.set_item("equity", PyArray1::from_vec_bound(py, snap.equity))?;
        d.set_item("position", PyArray1::from_vec_bound(py, snap.position))?;
        d.set_item(
            "unrealized_pnl",
            PyArray1::from_vec_bound(py, snap.unrealized_pnl),
        )?;
        d.set_item(
            "realized_pnl",
            PyArray1::from_vec_bound(py, snap.realized_pnl),
        )?;
        d.set_item("mark_price", PyArray1::from_vec_bound(py, snap.mark_price))?;
        Ok(d)
    }
}

impl Batch {
    /// Validate and convert a per-env `i64` offset array into clamped `usize`.
    fn offsets_vec(&self, offsets: PyReadonlyArray1<'_, i64>) -> PyResult<Vec<usize>> {
        let slice = offsets
            .as_slice()
            .map_err(|_| PyValueError::new_err("offsets must be a contiguous 1-D int64 array"))?;
        if slice.len() != self.inner.len() {
            return Err(PyValueError::new_err(format!(
                "offsets length {} must equal num_envs {}",
                slice.len(),
                self.inner.len()
            )));
        }
        Ok(slice.iter().map(|&x| x.max(0) as usize).collect())
    }

    /// Validate and copy an optional per-env signed-size action array.
    fn action_vec(
        &self,
        actions: &Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Option<Vec<f64>>> {
        match actions {
            Some(array) => {
                let slice = array.as_slice().map_err(|_| {
                    PyValueError::new_err("actions must be a contiguous 1-D float64 array")
                })?;
                if slice.len() != self.inner.len() {
                    return Err(PyValueError::new_err(format!(
                        "actions length {} must equal num_envs {}",
                        slice.len(),
                        self.inner.len()
                    )));
                }
                Ok(Some(slice.to_vec()))
            }
            None => Ok(None),
        }
    }
}

/// Runs a built-in compiled strategy over a grid of parameter rows in parallel,
/// returning final stats per row. Exposed to Python as `emsl.batch.BatchRunner`.
#[pyclass]
struct BatchRunner {
    candles: Candles,
    config: EngineConfig,
}

#[pymethods]
impl BatchRunner {
    #[new]
    #[pyo3(signature = (
        candles,
        market = "spot",
        quote = 10_000.0,
        fee_taker = 0.0006,
        fee_maker = 0.0002,
        slippage_bps = 0.0,
        max_fill_fraction = 1.0,
        max_open_orders = 8,
        leverage = 10.0,
        impact = 0.0,
        funding_rate = 0.0,
        funding_interval = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        candles: PyReadonlyArray2<'_, f64>,
        market: &str,
        quote: f64,
        fee_taker: f64,
        fee_maker: f64,
        slippage_bps: f64,
        max_fill_fraction: f64,
        max_open_orders: usize,
        leverage: f64,
        impact: f64,
        funding_rate: f64,
        funding_interval: usize,
    ) -> PyResult<BatchRunner> {
        let config = EngineConfig {
            market: parse_market(market)?,
            quote,
            fee_taker,
            fee_maker,
            slippage_bps,
            max_fill_fraction,
            max_open_orders,
            report: false, // the sweep forces reporting on per run
            max_leverage: leverage,
            impact,
            funding_rate,
            funding_interval,
        };
        Ok(BatchRunner {
            candles: candles_from_array(candles)?,
            config,
        })
    }

    /// Run `strategy` over each row of `params` (shape `(G, P)`) to the end of the
    /// data, in parallel with the GIL released, returning a dict of `(G,)` stat
    /// arrays aligned to the rows.
    #[pyo3(signature = (strategy, params, periods_per_year = 365.0, risk_free = 0.0))]
    fn run_all<'py>(
        &self,
        py: Python<'py>,
        strategy: &str,
        params: PyReadonlyArray2<'_, f64>,
        periods_per_year: f64,
        risk_free: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let factory = builtin_strategy(strategy).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown strategy '{strategy}'; built-in strategies: sma_cross"
            ))
        })?;
        let rows: Vec<Vec<f64>> = params
            .as_array()
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();
        let candles = self.candles.clone();
        let config = self.config;
        let stats = py.allow_threads(move || {
            run_sweep(candles, config, factory, &rows, periods_per_year, risk_free)
        });

        let d = PyDict::new_bound(py);
        let column = |get: fn(&Stats) -> f64| stats.iter().map(get).collect::<Vec<f64>>();
        d.set_item(
            "total_return_pct",
            PyArray1::from_vec_bound(py, column(|s| s.total_return_pct)),
        )?;
        d.set_item(
            "net_profit_pct",
            PyArray1::from_vec_bound(py, column(|s| s.net_profit_pct)),
        )?;
        d.set_item(
            "cagr_pct",
            PyArray1::from_vec_bound(py, column(|s| s.cagr_pct)),
        )?;
        d.set_item("sharpe", PyArray1::from_vec_bound(py, column(|s| s.sharpe)))?;
        d.set_item(
            "sortino",
            PyArray1::from_vec_bound(py, column(|s| s.sortino)),
        )?;
        d.set_item("calmar", PyArray1::from_vec_bound(py, column(|s| s.calmar)))?;
        d.set_item(
            "max_drawdown_pct",
            PyArray1::from_vec_bound(py, column(|s| s.max_drawdown_pct)),
        )?;
        d.set_item(
            "volatility_pct",
            PyArray1::from_vec_bound(py, column(|s| s.volatility_pct)),
        )?;
        d.set_item(
            "exposure_pct",
            PyArray1::from_vec_bound(py, column(|s| s.exposure_pct)),
        )?;
        d.set_item(
            "win_rate",
            PyArray1::from_vec_bound(py, column(|s| s.win_rate)),
        )?;
        d.set_item(
            "profit_factor",
            PyArray1::from_vec_bound(py, column(|s| s.profit_factor)),
        )?;
        d.set_item(
            "num_trades",
            PyArray1::from_vec_bound(
                py,
                stats
                    .iter()
                    .map(|s| s.num_trades as i64)
                    .collect::<Vec<i64>>(),
            ),
        )?;
        d.set_item(
            "avg_trade_pct",
            PyArray1::from_vec_bound(py, column(|s| s.avg_trade_pct)),
        )?;
        Ok(d)
    }
}

/// The `_emsl` extension module, imported by Python as `emsl._emsl`.
#[pymodule]
fn _emsl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_class::<Batch>()?;
    m.add_class::<BatchRunner>()?;
    Ok(())
}
