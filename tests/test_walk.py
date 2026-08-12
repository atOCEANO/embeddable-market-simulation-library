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
                    layout = _layout(size, windows, train, anchored)
                    for fit_from, fit_to, test_to in layout:
                        assert fit_from < fit_to < test_to <= size
                        # the two ranges are disjoint, which is the claim. The
                        # second assertion here used to be `fit_to <= test_to`,
                        # implied by the line above it and therefore free
                        fitted = set(range(fit_from, fit_to))
                        traded = set(range(fit_to, test_to))
                        assert not fitted & traded


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
    # a schedule that never switched would be a single backtest wearing a costume.
    # `len(set(chosen)) >= 1` used to stand here, which is true of any non-empty
    # list and so tested nothing the name claims. The delegation is what matters,
    # so drive the composite directly and record which part answers each bar
    from emsl._walk import _Scheduled

    schedule = [(0, 10, {"fast": 3, "slow": 12}),
                (10, 20, {"fast": 9, "slow": 30})]
    seen = []

    class Spy(SmaCross):
        def init(self, engine):
            # the second window's part cannot decide until bar 15, which is what
            # keeps the composite honest about each part's own warm-up
            self.warmup = 0 if self.slow == 12 else 15

        def next(self, state, engine):
            seen.append((state["tick_index"], self.fast, self.slow))

    composite = _Scheduled(Spy, schedule)
    composite.init(emsl.Engine(series(n=30)))
    assert composite.warmup == 0
    for bar in range(25):
        composite.next({"tick_index": bar, "position": 0.0}, None)
    early = {(f, s) for b, f, s in seen if b < 10}
    late = {(f, s) for b, f, s in seen if 10 <= b < 20}
    after = [b for b, _f, _s in seen if b >= 20]
    assert early == {(3, 12)}, f"the first window answered with {early}"
    assert late == {(9, 30)}, f"the second window answered with {late}"
    assert not after, "a bar past the last window was still delegated"
    # and the second part is silent through its own warm-up, even though the bars
    # belong to its window and the composite is past its own warm-up
    warming = [b for b, _f, s in seen if s == 30 and b < 15]
    assert not warming, f"a part decided during its own warm-up at bars {warming}"
    assert min(b for b, _f, s in seen if s == 30) == 15

    out = forward(windows=4, n_trials=8)
    assert len(out.windows) == 4
    # every window's parameters are inside the space it searched
    for window in out.windows:
        assert 3 <= window["params"]["fast"] <= 10
        assert 12 <= window["params"]["slow"] <= 30


def test_a_window_is_scored_on_the_run_that_happened():
    # the score used to come from a fresh isolated Backtester over the window
    # alone, which opens its own account and restarts every warm-up inside the
    # window. It is read off the composite curve now, so it must agree with
    # metrics.segment over the same bars and with nothing else (ADR 0060)
    out = forward(windows=3)
    for record in out.windows:
        first, last = record["traded_on"]
        expected = metrics.segment(out.result, first, last)["sharpe"]
        assert record["traded"] == pytest.approx(expected)


def test_a_window_keeps_the_history_before_it_rather_than_restarting():
    # the failure this replaced: each window used to be scored by a fresh
    # Backtester over its own bars, which restarts the warm-up inside the window,
    # so a winner needing 150 bars of history never had `next` called on a
    # 60-bar stretch and the window reported a tidy 0.0 (ADR 0060)
    data = series(n=600)
    space = {"fast": (3, 10), "slow": (150, 170)}
    out = emsl.walk_forward(SmaCross, space, data, windows=4, train=0.6,
                            n_trials=4, seed=0, periods_per_year=365.0,
                            fee_taker=0.0)
    for record in out.windows:
        first, last = record["traded_on"]
        warmup = record["params"]["slow"]
        assert warmup > last - first, "this layout is meant to outrun its windows"
        # the composite warms up on the bars before the window, so every bar of
        # the stretch is decidable
        assert record["bars_traded"] == last - first
        # while the isolated run the score used to come from could not trade at all
        alone = emsl.backtest.Backtester(
            data[first:last], periods_per_year=365.0, fee_taker=0.0
        ).run(SmaCross(**record["params"]))
        assert alone.stats["num_trades"] == 0
        assert alone.stats["sharpe"] == 0.0
    # and the scores that came back are not that row of zeros
    assert any(w["traded"] not in (None, 0.0) for w in out.windows)


def test_the_windows_compound_to_the_run():
    # each window is seeded from the balance carried into it, so the segments
    # multiply back to the whole rather than each starting from a fresh 10,000
    out = forward(windows=3)
    whole = metrics.segment(out.result)["total_return_pct"]
    product = 1.0
    edges = [out.windows[0]["traded_on"][0]] + [w["traded_on"][1] for w in out.windows]
    for first, last in zip(edges, edges[1:]):
        product *= 1.0 + metrics.segment(out.result, first, last)["total_return_pct"] / 100.0
    # the stretch before the first window is flat, so the windows carry it all
    assert (product - 1.0) * 100.0 == pytest.approx(whole, abs=1e-9)


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
