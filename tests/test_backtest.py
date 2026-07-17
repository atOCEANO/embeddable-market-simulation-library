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
