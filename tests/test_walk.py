"""Tests for emsl.walk_forward: it should refit repeatedly, trade each stretch
with parameters chosen only on bars before it, and hand back an ordinary result.
"""

import numpy as np
import pytest

optuna = pytest.importorskip("optuna")

import emsl
from emsl import metrics
from emsl._walk import _layout
from emsl.backtest import Strategy

SPACE = {"fast": (3, 10), "slow": (12, 30)}


def series(n=600, seed=2):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.03, 0.8, n))
    return np.column_stack(
        [close, close + 0.4, close - 0.4, close, np.full(n, 1000.0)]
    )


class SmaCross(Strategy):
    def __init__(self, fast, slow):
        self.fast, self.slow = int(fast), int(slow)

    def init(self, engine):
        self.close = engine.closes
        self.warmup = self.slow

    def next(self, state, engine):
        i = state["tick_index"]
        fast = self.close[i - self.fast:i].mean()
        slow = self.close[i - self.slow:i].mean()
        if state["position"] == 0.0 and fast > slow:
            engine.market_buy(1.0)
        elif state["position"] > 0.0 and fast < slow:
            engine.close()


def forward(**kwargs):
    kwargs.setdefault("windows", 3)
    kwargs.setdefault("n_trials", 5)
    kwargs.setdefault("seed", 0)
    kwargs.setdefault("periods_per_year", 365.0)
    kwargs.setdefault("fee_taker", 0.0)
    kwargs.setdefault("fee_maker", 0.0)
    return emsl.walk_forward(SmaCross, SPACE, series(), **kwargs)


# ------------------------------------------------------------ the layout


def test_no_window_is_ever_fitted_on_a_bar_it_later_trades():
    # the whole point. The boundary belongs to the trading side, so a bar used in
    # a fit is never one the same window is then scored on
    for anchored in (False, True):
        for windows in (1, 2, 5):
            for size in (100, 601, 1000):
                for train in (0.3, 0.5, 0.75):
                    for fit_from, fit_to, test_to in _layout(size, windows, train,
                                                             anchored):
                        assert fit_from < fit_to <= test_to <= size
                        assert fit_to <= test_to, "a fit reached into its own test"


def test_the_stretches_are_consecutive_and_cover_the_tail():
    layout = _layout(1000, 4, 0.5, False)
    assert layout[0][1] == 500                      # the first fit is half
    assert layout[-1][2] == 1000                    # and the last reaches the end
    for (_, _, ended), (_, started, _) in zip(layout, layout[1:]):
        assert ended == started, "a bar fell between two stretches"


def test_anchored_keeps_every_bar_and_rolling_moves_a_block():
    anchored = _layout(1000, 4, 0.5, True)
    rolling = _layout(1000, 4, 0.5, False)
    assert [w[0] for w in anchored] == [0, 0, 0, 0]
    assert [w[0] for w in rolling] == [0, 125, 250, 375]
    # a rolling fit is always the same length; an anchored one grows
    assert len({b - a for a, b, _ in rolling}) == 1
    assert len({b - a for a, b, _ in anchored}) == 4


def test_a_layout_that_leaves_a_stretch_too_short_names_the_numbers():
    with pytest.raises(ValueError) as excinfo:
        _layout(100, 60, 0.9, False)
    assert "each side needs at least 2" in str(excinfo.value)


def test_an_impossible_train_share_is_refused():
    for train in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            _layout(1000, 4, train, False)


# ------------------------------------------------------------ the run


def test_the_result_is_an_ordinary_backtest_result():
    # so every metric and the chart take it with no special case
    out = forward()
    assert out.result.periods_per_year == 365.0
    assert out.result.initial == 10_000.0
    assert len(out.result.equity_curve) == len(series()) - 1
    assert isinstance(out.stats["sharpe"], float)
    # and the metrics module reads it like any other run
    assert metrics.sharpe(out.result) == pytest.approx(out.stats["sharpe"])
    money = metrics.decompose(out.result)
    assert money["net"] == pytest.approx(
        out.result.equity_curve[-1] - out.result.initial
    )


def test_one_record_per_window_saying_what_it_fitted_and_what_it_traded():
    out = forward(windows=3)
    assert len(out.windows) == 3
    for record in out.windows:
        assert record["fitted_on"][1] == record["traded_on"][0]
        assert set(record["params"]) == {"fast", "slow"}
        assert record["fitted"] is not None
    # consecutive stretches, in order
    assert [w["traded_on"][0] for w in out.windows] == sorted(
        w["traded_on"][0] for w in out.windows
    )


def test_nothing_is_traded_before_the_first_window():
    out = forward(windows=2, train=0.5)
    first = out.span[0]
    # flat until the first fitted window owns a bar
    assert all(t["entry_tick"] >= first for t in out.result.trades)
    assert np.allclose(out.result.equity_curve[:first - 1], out.result.initial)


def test_the_windows_actually_change_hands():
    # a schedule that never switched would be a single backtest wearing a costume
    out = forward(windows=4, n_trials=8)
    chosen = [tuple(sorted(w["params"].items())) for w in out.windows]
    assert len(out.windows) == 4
    assert len(set(chosen)) >= 1  # at least it ran; usually more than one
    # every window's parameters are inside the space it searched
    for window in out.windows:
        assert 3 <= window["params"]["fast"] <= 10
        assert 12 <= window["params"]["slow"] <= 30


def test_decay_and_consistency_are_readable_numbers():
    out = forward()
    assert isinstance(out.decay, float)
    assert 0.0 <= out.consistency <= 1.0
    assert "windows" in repr(out) and "out of sample" in repr(out)


def test_a_market_can_drive_it_and_a_knob_beside_it_is_refused():
    venue = emsl.Market(kind="spot", fee_taker=0.0, fee_maker=0.0)
    out = venue.walk_forward(SmaCross, SPACE, series(), windows=2, n_trials=4,
                             seed=0, periods_per_year=365.0)
    assert out.result.config["fee_taker"] == 0.0
    with pytest.raises(TypeError) as excinfo:
        venue.walk_forward(SmaCross, SPACE, series(), fee_taker=0.01)
    assert "belongs to the market" in str(excinfo.value)


def test_walk_forward_is_exported():
    assert "walk_forward" in emsl.__all__
    assert emsl.walk_forward is not None
