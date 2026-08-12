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


def search(sampler="tpe", n_trials=12, data=None, objective="sharpe", seed=0,
           min_trades=0, oos=0):
    return emsl.tune(SmaCross, SPACE, series() if data is None else data,
                     objective=objective, n_trials=n_trials, seed=seed,
                     sampler=sampler, oos=oos, periods_per_year=365.0,
                     min_trades=min_trades, fee_taker=0.0, fee_maker=0.0)


def test_a_search_now_carries_its_winner_and_what_it_searched_for():
    study = search()
    assert study.best_result is not None
    assert study.best_result.stats["sharpe"] == pytest.approx(study.best_stats["sharpe"])
    assert study.objective == "sharpe"
    assert study.sampler == "tpe"
    assert study.min_trades == 0
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


def test_the_looks_are_the_searchs_own_trials_and_not_the_nulls():
    # the null estimates how far a sharpe scatters over this space. It does not
    # decide how hard YOU searched, and reading the count off it meant a small
    # null flattered the winner: the same 12-trial study scored as though it had
    # looked 6 times against one null and 48 times against another (ADR 0058)
    study = search(n_trials=12)
    few = search(sampler="random", seed=3, n_trials=12)
    many = search(sampler="random", seed=3, n_trials=48)
    _t, looks_few, _s = metrics._null_shape(study, few)
    _t, looks_many, _s = metrics._null_shape(study, many)
    assert looks_few == looks_many == len(study.trials) == 12


def test_a_bigger_search_has_more_to_beat_than_a_smaller_one():
    # the count that moves the bar is the search's, so this is where the
    # intuition has to hold: look harder, clear a higher bar
    data = series()
    null = search(sampler="random", seed=3, n_trials=40, data=data)
    small = search(n_trials=4, data=data)
    large = search(n_trials=40, data=data)
    _t, few, spread_a = metrics._null_shape(small, null)
    _t, many, spread_b = metrics._null_shape(large, null)
    assert few == 4 and many == 40
    assert spread_a == spread_b            # one null, one spread
    assert metrics.deflation_threshold(spread_b, many) > metrics.deflation_threshold(
        spread_a, few
    )


def test_a_null_carrying_an_activity_floor_is_refused():
    # an activity floor fails the thin cells, and the thin cells hold the extreme
    # sharpes, so the survivors scatter less than the space does. Reading the
    # spread off them lowered the bar the stricter the floor, which is the
    # inversion ADR 0054 was written to prevent arriving through the other term
    study = search()
    floored = search(sampler="random", seed=1, n_trials=20, min_trades=3)
    assert floored.min_trades == 3
    with pytest.raises(ValueError) as excinfo:
        metrics.deflated_sharpe(study, floored)
    assert "min_trades=0" in str(excinfo.value)
    # and the search being deflated may still carry one
    strict = search(n_trials=12, min_trades=1)
    clean = search(sampler="random", seed=1, n_trials=20)
    assert 0.0 <= metrics.deflated_sharpe(strict, clean) <= 1.0


def test_a_floor_on_the_null_would_have_lowered_the_bar():
    # the measurement behind the refusal above, asserted rather than asserted
    # about: the survivors of a floor really do scatter less
    data = series()
    loose = search(sampler="random", seed=1, n_trials=30, data=data)
    rows = [t["stats"] for t in loose.trials if t["stats"]]
    # a floor set at the median activity, so it bites on about half of them
    floor = int(np.median([r["num_trades"] for r in rows])) + 1
    scored = [r["sharpe"] for r in rows]
    kept = [r["sharpe"] for r in rows if r["num_trades"] >= floor]
    assert 2 <= len(kept) < len(scored)
    assert np.std(kept, ddof=1) < np.std(scored, ddof=1)


def test_a_null_thinner_than_the_search_says_so():
    study = search(n_trials=20)
    thin = search(sampler="random", seed=3, n_trials=6)
    with pytest.warns(UserWarning, match="fewer draws"):
        metrics.deflated_sharpe(study, thin)


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


def test_the_documented_pairing_actually_runs():
    # the example shipped in the docstring and on the API page held out a tail in
    # the study and not in the null, so the two ran on different bars and the
    # function's own fingerprint guard rejected it. Nothing caught that, because
    # the docs test parses examples and does not run them
    data = series(n=600)
    study = emsl.tune(SmaCross, SPACE, data, n_trials=8, seed=0, oos=0.3,
                      periods_per_year=365.0)
    null = emsl.tune(SmaCross, SPACE, data, n_trials=8, seed=1, oos=0.3,
                     sampler="random", periods_per_year=365.0)
    assert 0.0 <= metrics.deflated_sharpe(study, null) <= 1.0


def test_holding_out_in_one_and_not_the_other_is_still_refused():
    data = series(n=600)
    study = emsl.tune(SmaCross, SPACE, data, n_trials=6, seed=0, oos=0.3,
                      periods_per_year=365.0)
    whole = emsl.tune(SmaCross, SPACE, data, n_trials=6, seed=1, oos=0,
                      sampler="random", periods_per_year=365.0)
    with pytest.raises(ValueError) as excinfo:
        metrics.deflated_sharpe(study, whole)
    assert "different bars" in str(excinfo.value)


def test_a_market_can_drive_the_null_too():
    venue = emsl.Market(kind="spot", fee_taker=0.0)
    data = series()
    study = venue.tune(SmaCross, SPACE, data, n_trials=10, seed=0, oos=0,
                       periods_per_year=365.0)
    null = venue.tune(SmaCross, SPACE, data, n_trials=10, seed=1, oos=0,
                      sampler="random", periods_per_year=365.0)
    assert 0.0 <= metrics.deflated_sharpe(study, null) <= 1.0
