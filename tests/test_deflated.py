"""Tests for metrics.deflated_sharpe: it should measure the winner against what
the best of that many random looks would have reached, and refuse every input
that would make the number mean something other than it says.
"""

import numpy as np
import pytest

optuna = pytest.importorskip("optuna")

import emsl
from emsl import metrics
from emsl.backtest import Strategy

SPACE = {"fast": (3, 12), "slow": (15, 40)}


def series(n=400, seed=5):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.04, 0.9, n))
    return np.column_stack(
        [close, close + 0.5, close - 0.5, close, np.full(n, 1000.0)]
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


def search(sampler="tpe", n_trials=12, data=None, objective="sharpe", seed=0):
    return emsl.tune(SmaCross, SPACE, series() if data is None else data,
                     objective=objective, n_trials=n_trials, seed=seed,
                     sampler=sampler, oos=0, periods_per_year=365.0,
                     fee_taker=0.0, fee_maker=0.0)


def test_a_search_now_carries_its_winner_and_what_it_searched_for():
    study = search()
    assert study.best_result is not None
    assert study.best_result.stats["sharpe"] == pytest.approx(study.best_stats["sharpe"])
    assert study.objective == "sharpe"
    assert study.sampler == "tpe"
    assert len(study.data_hash) == 8


def test_the_deflated_probability_is_a_probability():
    study, null = search(), search(sampler="random", seed=1)
    value = metrics.deflated_sharpe(study, null)
    assert 0.0 <= value <= 1.0


def test_the_bar_rises_with_the_spread_and_with_the_number_of_looks():
    # the whole intuition, asserted rather than gestured at: the more times you
    # looked and the more the scores scattered, the higher a winner has to be
    assert metrics.deflation_threshold(2.0, 20) > metrics.deflation_threshold(1.0, 20)
    assert metrics.deflation_threshold(1.0, 200) > metrics.deflation_threshold(1.0, 20)
    assert metrics.deflation_threshold(0.0, 200) == 0.0  # no scatter, nothing to beat
    for looks in (2, 5, 50, 5_000):
        assert metrics.deflation_threshold(1.0, looks) > 0.0
    with pytest.raises(ValueError):
        metrics.deflation_threshold(1.0, 1)


def test_deflating_harder_lowers_the_probability():
    # a bar twice as high cannot make the same winner look better
    study = search()
    null = search(sampler="random", seed=1, n_trials=20)
    _trials, looks, spread = metrics._null_shape(study, null)
    gentle = metrics.probabilistic_sharpe(
        study.best_result, benchmark=metrics.deflation_threshold(spread, looks)
    )
    harsh = metrics.probabilistic_sharpe(
        study.best_result, benchmark=metrics.deflation_threshold(spread * 3.0, looks)
    )
    assert harsh <= gentle
    assert metrics.deflated_sharpe(study, null) == pytest.approx(gentle)


def test_more_looks_in_the_null_means_more_to_beat():
    study = search()
    few = search(sampler="random", seed=3, n_trials=6)
    many = search(sampler="random", seed=3, n_trials=48)
    _t, looks_few, _s = metrics._null_shape(study, few)
    _t, looks_many, _s = metrics._null_shape(study, many)
    assert looks_many > looks_few
    # and every trial counts as a look, including one that scored nothing
    assert looks_many == len(many.trials)


def test_a_tpe_study_cannot_be_its_own_null():
    # the whole reason the null is required: a converging search understates its
    # own spread, so this number would rise as the overfitting got worse
    study = search()
    with pytest.raises(ValueError) as excinfo:
        metrics.deflated_sharpe(study, search(sampler="tpe", seed=1))
    assert "sampler='random'" in str(excinfo.value)


def test_a_null_over_different_bars_is_not_a_null():
    study = search()
    elsewhere = emsl.tune(SmaCross, SPACE, series(seed=99), n_trials=8, seed=1,
                          sampler="random", oos=0, periods_per_year=365.0)
    with pytest.raises(ValueError) as excinfo:
        metrics.deflated_sharpe(study, elsewhere)
    assert "different bars" in str(excinfo.value)


def test_deflating_a_search_that_optimised_something_else_is_refused():
    study = search(objective="calmar")
    null = search(sampler="random", seed=1)
    with pytest.raises(ValueError) as excinfo:
        metrics.deflated_sharpe(study, null)
    assert "selected on 'calmar'" in str(excinfo.value)


def test_a_null_with_nothing_in_it_sets_no_threshold():
    study = search()
    null = search(sampler="random", seed=1, n_trials=1)
    with pytest.raises(ValueError) as excinfo:
        metrics.deflated_sharpe(study, null)
    assert "at least 2" in str(excinfo.value)


def test_an_unknown_sampler_is_refused_before_the_search_runs():
    with pytest.raises(ValueError) as excinfo:
        emsl.tune(SmaCross, SPACE, series(), n_trials=2, sampler="grid", oos=0,
                  periods_per_year=365.0)
    assert "'tpe' or 'random'" in str(excinfo.value)


def test_a_random_search_is_reproducible_from_its_seed():
    one = search(sampler="random", seed=7)
    two = search(sampler="random", seed=7)
    assert one.best_params == two.best_params
    assert one.best_value == two.best_value


def test_a_market_can_drive_the_null_too():
    venue = emsl.Market(kind="spot", fee_taker=0.0)
    data = series()
    study = venue.tune(SmaCross, SPACE, data, n_trials=10, seed=0, oos=0,
                       periods_per_year=365.0)
    null = venue.tune(SmaCross, SPACE, data, n_trials=10, seed=1, oos=0,
                      sampler="random", periods_per_year=365.0)
    assert 0.0 <= metrics.deflated_sharpe(study, null) <= 1.0
