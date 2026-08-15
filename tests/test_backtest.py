"""Tests for the emsl.backtest wrapper: it should drive a Strategy through the
engine and bundle stats, the equity curve, and the trade log.
"""

import numpy as np
import pytest

from emsl.backtest import Backtester, BacktestResult, Strategy


def series():
    return np.array(
        [
            [100.0, 160.0, 90.0, 150.0, 1000.0],
            [200.0, 260.0, 190.0, 250.0, 1000.0],
            [300.0, 360.0, 290.0, 350.0, 1000.0],
        ],
        dtype=np.float64,
    )


class BuyThenClose(Strategy):
    def next(self, state, engine):
        if state["tick_index"] == 0:
            engine.market_buy(1.0)
        elif state["position"] != 0.0:
            engine.close()


def test_backtester_returns_stats_equity_and_trades():
    bt = Backtester(series(), market="spot", fee_taker=0.0, fee_maker=0.0)
    result = bt.run(BuyThenClose())
    assert isinstance(result, BacktestResult)

    # one closed trade: bought 1 @ 200, closed @ 300 -> +100
    assert result.stats["num_trades"] == 1
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade["side"] == "buy"
    assert trade["pnl"] == 100.0

    # equity curve has one point per step (2 steps for a 3-bar series)
    assert len(result.equity_curve) == 2


def test_base_strategy_next_must_be_overridden():
    bt = Backtester(series())
    with pytest.raises(NotImplementedError):
        bt.run(Strategy())


def test_trades_carry_times_when_a_datetime_index_is_given():
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2024-01-01", periods=3, freq="1D")
    df = pd.DataFrame(
        series(), columns=["open", "high", "low", "close", "volume"], index=idx
    )
    result = Backtester(df, market="spot", fee_taker=0.0, fee_maker=0.0).run(BuyThenClose())
    trade = result.trades[0]
    # entry filled on bar 1, closed on bar 2 (see the numpy test above)
    assert trade["entry_time"] == np.datetime64(idx[1])
    assert trade["exit_time"] == np.datetime64(idx[2])


def test_trades_stay_tick_only_for_numpy_input():
    result = Backtester(series(), market="spot", fee_taker=0.0, fee_maker=0.0).run(BuyThenClose())
    assert "entry_time" not in result.trades[0]  # no index, so no timestamps


def test_the_result_carries_the_annualization_its_stats_were_computed_at():
    # anything reading the curve a second time, emsl.metrics most of all, has to
    # reach the numbers this result already reports, and inferring it twice is how
    # two readings come to disagree (ADR 0048)
    with pytest.warns(UserWarning):
        result = Backtester(series()).run(BuyThenClose())
    assert result.periods_per_year == 365.0

    stated = Backtester(series(), periods_per_year=8760.0).run(BuyThenClose())
    assert stated.periods_per_year == 8760.0
    assert stated.stats["sharpe"] != result.stats["sharpe"]


def test_hourly_candles_are_annualized_hourly_without_being_told():
    pd = pytest.importorskip("pandas")
    raw = series()
    data = pd.DataFrame(raw, columns=["open", "high", "low", "close", "volume"])
    data.index = pd.date_range("2025-01-01", periods=len(raw), freq="1h", tz="UTC")
    result = Backtester(data).run(BuyThenClose())
    assert result.periods_per_year == 8760.0


def test_the_named_price_views_are_the_columns_of_data():
    from emsl import Engine

    eng = Engine(series())
    for name, column in (("opens", 0), ("highs", 1), ("lows", 2),
                         ("closes", 3), ("volumes", 4)):
        view = getattr(eng, name)
        assert view.shape == (len(series()),)
        assert np.array_equal(view, eng.data[:, column])


def test_a_price_view_is_read_only_and_copies_nothing():
    from emsl import Engine

    eng = Engine(series())
    assert not eng.closes.flags.writeable
    with pytest.raises(ValueError):
        eng.closes[0] = 1.0
    # a view, not a copy: it shares the engine's buffer rather than duplicating it
    assert eng.closes.base is not None


class Late(Strategy):
    """Reads five bars of history, and declares it rather than guarding by hand."""

    warmup = 5

    def init(self, engine):
        self.closes = engine.closes
        self.seen = []

    def next(self, state, engine):
        i = state["tick_index"]
        self.seen.append(i)
        # with the warm-up declared this can never be a negative index, which numpy
        # would resolve from the END of the array and read the future from
        assert self.closes[i - 5] == self.closes[max(i - 5, 0)]


def test_a_declared_warmup_skips_the_bars_that_could_index_backwards():
    long_series = np.repeat(series(), 4, axis=0)
    strategy = Late()
    Backtester(long_series, periods_per_year=365.0).run(strategy)
    assert strategy.seen[0] == 5
    assert strategy.seen == list(range(5, len(long_series) - 1))


def test_a_warmup_longer_than_the_series_simply_never_decides():
    strategy = Late()
    strategy.warmup = 10_000
    result = Backtester(series(), periods_per_year=365.0).run(strategy)
    assert strategy.seen == []
    assert result.stats["num_trades"] == 0


def test_a_warmup_that_is_not_a_count_of_bars_is_refused():
    strategy = Late()
    strategy.warmup = "soon"
    with pytest.raises(ValueError) as excinfo:
        Backtester(series(), periods_per_year=365.0).run(strategy)
    assert "non-negative whole number of bars" in str(excinfo.value)


def test_a_run_that_ended_at_zero_says_so_on_the_result():
    # a forced close books a row carrying liquidated, but an account drained by
    # fees or funding books nothing at all, so the trade log cannot always say
    # it happened and the run-level flag is the only place that can (ADR 0091)
    bars = np.array([
        [100.0, 100.0, 100.0, 100.0, 1_000.0],
        [100.0, 100.0, 100.0, 100.0, 1_100.0],
        [100.0, 100.0, 80.0, 85.0, 1_200.0],
        [85.0, 90.0, 85.0, 90.0, 1_300.0],
        [90.0, 95.0, 90.0, 95.0, 1_400.0],
    ], dtype=np.float64)

    class BuyAndDie(Strategy):
        def init(self, engine):
            self.placed = False

        def next(self, state, engine):
            if not self.placed:
                engine.market_buy(9.9)
                self.placed = True

    dead = Backtester(bars, market="perp", quote=100.0, fee_taker=0.001,
                      fee_maker=0.001, leverage=10.0).run(BuyAndDie())
    assert dead.bust is True
    assert dead.to_dict()["bust"] is True
    assert any(t["liquidated"] for t in dead.trades)

    alive = Backtester(series(), market="spot", fee_taker=0.0,
                       fee_maker=0.0).run(BuyThenClose())
    assert alive.bust is False
    assert alive.to_dict()["bust"] is False


def test_editing_the_array_after_handing_it_over_cannot_change_a_run():
    # ascontiguousarray hands back the caller's own object when it is already
    # contiguous float64, and run() builds a fresh Engine off it every call, so a
    # second run read bars the fingerprint no longer described (ADR 0104)
    data = series()
    bt = Backtester(data, market="spot", fee_taker=0.0, fee_maker=0.0)
    first = bt.run(BuyThenClose())
    data *= 2.0
    second = bt.run(BuyThenClose())
    assert first.data_hash == second.data_hash
    assert first.stats["total_return_pct"] == second.stats["total_return_pct"]
    assert np.array_equal(bt._candles, series())


def test_a_frame_was_never_aliased_and_still_is_not():
    # prepare already materialises a fresh array on the DataFrame path, so this is
    # the control that says the copy above changed nothing for it
    pd = pytest.importorskip("pandas")
    close = 100.0 + np.arange(12, dtype=np.float64)
    frame = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0,
                          "close": close, "volume": np.full(12, 1000.0)})
    bt = Backtester(frame, market="spot", fee_taker=0.0, fee_maker=0.0)
    before = bt.run(BuyThenClose()).stats["total_return_pct"]
    frame.loc[:, "close"] = frame["close"] * 2.0
    assert bt.run(BuyThenClose()).stats["total_return_pct"] == before
