"""Tests for emsl.batch.BatchRunner: the compiled-strategy parameter sweep should
run a built-in strategy over a grid and return a stat array per row.
"""

import numpy as np
import pytest

from emsl.batch import BatchRunner


def rising(n=40):
    close = 100.0 + np.arange(n, dtype=np.float64)
    return np.stack(
        [close, close + 0.5, close - 0.5, close, np.full(n, 1000.0)], axis=1
    )


def test_run_all_returns_a_stat_array_per_param_row():
    runner = BatchRunner(rising(), market="spot", fee_taker=0.0, fee_maker=0.0)
    params = np.array([[2, 3], [3, 5], [5, 8]], dtype=np.float64)
    results = runner.run_all("sma_cross", params)

    assert {"sharpe", "total_return_pct", "num_trades", "max_drawdown_pct"} <= set(results)
    assert results["sharpe"].shape == (3,)
    assert results["num_trades"].shape == (3,)
    # every crossover buys and holds into a rising series, so all are profitable
    assert (results["total_return_pct"] > 0).all()


def test_unknown_strategy_raises():
    runner = BatchRunner(rising())
    with pytest.raises(ValueError):
        runner.run_all("does_not_exist", np.array([[2, 3]], dtype=np.float64))
