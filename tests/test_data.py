"""Tests for DataFrame and parquet input via emsl.to_ohlcv, and that the
high-level entry points accept a DataFrame.
"""

import numpy as np
import pandas as pd
import pytest

from emsl import to_ohlcv


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
