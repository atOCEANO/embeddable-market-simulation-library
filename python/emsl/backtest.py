"""Single-env backtesting over the Rust engine.

A thin convenience layer: ``Backtester`` builds a reporting ``Engine``, drives a
``Strategy`` over the series with the engine's own loop, and returns a
``BacktestResult`` bundling the stats, equity curve, and trade log. No simulation
logic lives here; the engine does all of it.
"""

from __future__ import annotations

from ._data import index_of, to_ohlcv
from ._emsl import Engine


class Strategy:
    """Base class for a backtest strategy.

    Override ``next``: it is called once per bar with the current state dict and
    the engine, and places orders through the engine's methods (``market_buy``,
    ``limit_buy``, ``stop``, ``close``, ...). ``init`` is optional and runs once
    after reset, before the first bar.
    """

    def init(self, engine):
        pass

    def next(self, state, engine):
        raise NotImplementedError("Strategy.next must be overridden")


class BacktestResult:
    """The outcome of a run: performance stats, the equity curve, and trades."""

    def __init__(self, stats, equity_curve, trades):
        self.stats = stats
        self.equity_curve = equity_curve
        self.trades = trades

    def __repr__(self):
        stats = self.stats or {}
        ret = stats.get("total_return_pct", 0.0)
        n = stats.get("num_trades", 0)
        return f"BacktestResult(total_return_pct={ret:.2f}, num_trades={n})"


class Backtester:
    """Runs a strategy over one candle series and reports the result.

    ``candles`` is a numpy array, a pandas DataFrame, or a parquet path (see
    ``to_ohlcv``). The remaining arguments are the engine's configuration;
    reporting is always on so the result carries stats, the equity curve, and the
    trade log.
    """

    def __init__(
        self,
        candles,
        market="spot",
        quote=10_000.0,
        fee_taker=0.0006,
        fee_maker=0.0002,
        slippage_bps=0.0,
        max_fill_fraction=1.0,
        max_open_orders=8,
        leverage=10.0,
        impact=0.0,
        funding_rate=0.0,
        funding_interval=0,
        periods_per_year=365.0,
        risk_free=0.0,
    ):
        self._candles = to_ohlcv(candles)
        self._index = index_of(candles)
        self._config = dict(
            market=market,
            quote=quote,
            fee_taker=fee_taker,
            fee_maker=fee_maker,
            slippage_bps=slippage_bps,
            max_fill_fraction=max_fill_fraction,
            max_open_orders=max_open_orders,
            leverage=leverage,
            impact=impact,
            funding_rate=funding_rate,
            funding_interval=funding_interval,
        )
        self._periods_per_year = periods_per_year
        self._risk_free = risk_free

    def run(self, strategy):
        engine = Engine(self._candles, report=True, **self._config)
        engine.run(strategy)
        trades = engine.trades()
        self._stamp_times(trades)
        return BacktestResult(
            stats=engine.stats(self._periods_per_year, self._risk_free),
            equity_curve=engine.equity_curve(),
            trades=trades,
        )

    def _stamp_times(self, trades):
        # when the input carried a datetime index, map each trade's entry and exit
        # tick to a timestamp; a numpy input has no index, so trades stay tick-only
        if self._index is None:
            return
        n = len(self._index)
        for t in trades:
            entry, exit_ = t["entry_tick"], t["exit_tick"]
            if 0 <= entry < n:
                t["entry_time"] = self._index[entry]
            if 0 <= exit_ < n:
                t["exit_time"] = self._index[exit_]
