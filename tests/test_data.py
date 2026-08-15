"""Tests for DataFrame and parquet input via emsl.to_ohlcv, and that the
high-level entry points accept a DataFrame.
"""

import numpy as np
import pandas as pd
import pytest

from emsl import to_ohlcv
from emsl._data import annualization, bars_per_year


def frame(n=10):
    close = 100.0 + np.arange(n, dtype=np.float64)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    )


def test_to_ohlcv_from_numpy_passes_through():
    out = to_ohlcv(np.zeros((5, 5), dtype=np.float64))
    assert out.shape == (5, 5)
    assert out.dtype == np.float64


def test_to_ohlcv_from_dataframe_selects_the_columns():
    out = to_ohlcv(frame(10))
    assert out.shape == (10, 5)
    assert out[0, 0] == 100.0  # open of row 0, columns ordered OHLCV


def test_to_ohlcv_missing_columns_raises():
    with pytest.raises(ValueError):
        to_ohlcv(pd.DataFrame({"open": [1.0], "close": [1.0]}))


def test_to_ohlcv_from_parquet(tmp_path):
    path = tmp_path / "data.parquet"
    frame(8).to_parquet(path)
    out = to_ohlcv(str(path))
    assert out.shape == (8, 5)


def test_backtester_accepts_a_dataframe():
    from emsl.backtest import Backtester, Strategy

    class Noop(Strategy):
        def next(self, state, engine):
            pass

    result = Backtester(frame(12)).run(Noop())
    assert "sharpe" in result.stats


def test_vectorenv_accepts_a_dataframe():
    from emsl.rl import VectorEnv

    env = VectorEnv(frame(40), num_envs=2, window=4, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2, 4, 5)


def test_unsorted_datetime_index_is_rejected():
    df = frame(5)
    df.index = pd.date_range("2024-01-01", periods=5, freq="1D")[::-1]  # descending
    with pytest.raises(ValueError):
        to_ohlcv(df)


def test_duplicate_datetime_index_is_rejected():
    df = frame(3)
    df.index = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"])
    with pytest.raises(ValueError):
        to_ohlcv(df)


def stamped(n, freq):
    data = frame(n)
    data.index = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    return data


def test_hourly_candles_annualize_at_eight_thousand_seven_hundred_and_sixty():
    assert bars_per_year(stamped(200, "1h").index.to_numpy()) == 8760.0


def test_a_handful_of_missing_bars_does_not_move_the_annualization():
    # the median gap decides it, so dropping bars leaves the interval intact
    data = stamped(200, "1h")
    data = data.drop(data.index[[10, 11, 12, 80, 150]])
    assert bars_per_year(data.index.to_numpy()) == 8760.0


def test_a_spacing_that_is_no_real_interval_is_reported_rather_than_rounded():
    # five hours is not a candle interval anyone ships, so this is a mixed or
    # decimated feed. Rounding it to the four-hour neighbour would state a bar
    # count nothing in the data supports, so it says what it saw instead
    data = stamped(40, "1h")
    with pytest.warns(UserWarning) as caught:
        inferred = bars_per_year(data.index[::5].to_numpy())
    assert "not within one percent" in str(caught[0].message)
    assert inferred == 365.0 * 24.0 * 60.0 * 60.0 / (5 * 3600.0)


def test_an_outage_leaves_the_interval_alone():
    # a whole day missing out of a year of hourly candles does not make them
    # something other than hourly, and the median is what keeps that true
    data = stamped(200, "1h")
    data = data.drop(data.index[40:64])
    assert bars_per_year(data.index.to_numpy()) == 8760.0


def test_five_minute_and_daily_candles_are_read_too():
    assert bars_per_year(stamped(50, "5min").index.to_numpy()) == 105_120.0
    assert bars_per_year(stamped(50, "1D").index.to_numpy()) == 365.0


def test_candles_with_no_datetime_index_have_no_annualization_to_read():
    assert bars_per_year(None) is None
    assert bars_per_year(np.arange(10)) is None
    assert bars_per_year(stamped(2, "1h").index.to_numpy()) is None  # too few gaps


def test_the_annualization_the_caller_states_is_used_unchanged():
    assert annualization(252.0, stamped(50, "1h").index.to_numpy()) == 252.0


def test_a_series_without_times_says_so_instead_of_assuming_daily():
    with pytest.warns(UserWarning) as caught:
        assert annualization(None, None) == 365.0
    assert "no datetime index" in str(caught[0].message)


def test_a_row_whose_prices_are_not_a_bar_is_refused():
    # the taker clamp bounds a fill by price.min(high) on a buy and price.max(low)
    # on a sell, so on an unordered row those move the fill toward the account
    # rather than away from it, silently and always favourably (ADR 0096)
    import emsl

    data = np.array([
        [100.0, 101.0, 99.0, 100.0, 1000.0],
        [100.0, 99.0, 101.0, 100.0, 1000.0],   # high and low transposed
    ])
    with pytest.raises(ValueError) as excinfo:
        emsl.Engine(data)
    said = str(excinfo.value)
    assert "row 1" in said
    assert "column order" in said


def test_a_column_order_that_keeps_the_high_above_the_low_is_refused_too():
    # the case that decides how much of the ordering has to be checked. Read a
    # low, high, open, close feed positionally and the two columns emsl calls
    # high and low are the real high and the real OPEN, so the high is at or
    # above the low on every row, every value is finite, and a high >= low check
    # passes it untouched while the fill model still prices off a bar that never
    # traded. Only the full ordering catches it (ADR 0096)
    import emsl

    real = [(100.0, 101.0, 99.0, 100.0), (100.0, 102.0, 98.0, 101.0)]
    shuffled = np.array([[low, high, opened, close, 1000.0]
                         for opened, high, low, close in real])
    assert (shuffled[:, 1] >= shuffled[:, 2]).all()      # the naive check passes
    with pytest.raises(ValueError) as excinfo:
        emsl.Engine(shuffled)
    assert "not a bar" in str(excinfo.value)


def test_a_bar_that_never_moved_is_still_a_bar():
    # every price equal is a real, if dull, candle and the rule is inclusive at
    # both ends, so a dead feed must not be refused as malformed
    import emsl

    flat = np.array([
        [100.0, 100.0, 100.0, 100.0, 0.0],
        [100.0, 100.0, 100.0, 100.0, 0.0],
    ])
    engine = emsl.Engine(flat)
    assert engine.reset()["equity"] > 0.0
