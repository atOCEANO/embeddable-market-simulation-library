"""Tests for emsl.tune: it should search a Strategy's parameters by running many
backtests, honor the objective and direction, sample the search space from the
shorthand and explicit forms, and carry the strategy and objective into worker
processes. optuna and cloudpickle back it, so the whole module skips without them.
"""

import pickle

import numpy as np
import pytest

optuna = pytest.importorskip("optuna")
pytest.importorskip("cloudpickle")

import emsl
from emsl.backtest import Strategy
from emsl._tune import Categorical, Float, Int, _CloudPickled, _evaluate, _to_distribution


def series(n=300):
    # a rising, oscillating close so a moving-average crossover both trades and
    # varies with its fast/slow lengths; open equals close so a next-bar fill lands
    # on the same value, and the high/low bracket it
    t = np.arange(n, dtype=np.float64)
    close = 100.0 + 20.0 * np.sin(t / 15.0) + 0.05 * t
    high = close + 1.0
    low = close - 1.0
    volume = np.full(n, 1000.0)
    return np.column_stack([close, high, low, close, volume])


def tuned(*args, **kwargs):
    # the tests below are about the search itself, not about the annualization or
    # the holdout, and both of those warn on purpose when left unstated. Stating
    # them here keeps the suite clear of warnings it is not testing, so a new one
    # stands out instead of joining a crowd
    kwargs.setdefault("periods_per_year", 365.0)
    kwargs.setdefault("oos", 0)
    return emsl.tune(*args, **kwargs)


class SmaCross(Strategy):
    """Fast/slow moving-average crossover; the two lengths are the tunables, taken
    as constructor arguments and stored as fields.
    """

    def __init__(self, fast, slow):
        self.fast = int(fast)
        self.slow = int(slow)

    def init(self, engine):
        self.close = engine.data[:, 3]

    def next(self, state, engine):
        i = state["tick_index"]
        if i < self.slow:
            return
        fast = self.close[i - self.fast:i].mean()
        slow = self.close[i - self.slow:i].mean()
        if state["position"] == 0 and fast > slow:
            engine.market_buy(1.0)
        elif state["position"] > 0 and fast < slow:
            engine.close()


SPACE = {"fast": (3, 20), "slow": (21, 60)}


def test_tune_returns_the_best_trial_within_the_space():
    result = tuned(
        SmaCross, SPACE, series(), objective="sharpe", n_trials=15, seed=0,
        fee_taker=0.0, fee_maker=0.0,
    )
    assert isinstance(result, emsl.tune.TuneResult)
    assert set(result.best_params) == {"fast", "slow"}
    assert 3 <= result.best_params["fast"] <= 20
    assert 21 <= result.best_params["slow"] <= 60
    assert np.isfinite(result.best_value)
    # maximize: the reported best equals the largest completed trial value
    values = [t["value"] for t in result.trials if t["value"] is not None]
    assert result.best_value == max(values)
    assert len(result.trials) == 15
    # the winning run's full stats travel with the result, and the objective stat
    # equals the optimized value
    assert result.best_stats["sharpe"] == result.best_value
    assert "max_drawdown_pct" in result.best_stats


def test_tune_is_reproducible_with_a_seed_and_one_job():
    kwargs = dict(objective="sharpe", n_trials=12, n_jobs=1, seed=7, fee_taker=0.0, fee_maker=0.0)
    a = tuned(SmaCross, SPACE, series(), **kwargs)
    b = tuned(SmaCross, SPACE, series(), **kwargs)
    assert a.best_params == b.best_params
    assert a.best_value == b.best_value


def test_tune_runs_across_worker_processes():
    # exercises the process pool, the per-worker initializer, and the cloudpickle
    # carry of the strategy and objective; completion order is not deterministic, so
    # only the shape of the result is asserted
    result = tuned(
        SmaCross, SPACE, series(), objective="sharpe", n_trials=6, n_jobs=2, seed=1,
        fee_taker=0.0, fee_maker=0.0,
    )
    assert np.isfinite(result.best_value)
    assert set(result.best_params) == {"fast", "slow"}
    assert len(result.trials) == 6


def test_objective_can_be_a_callable_over_the_result():
    result = tuned(
        SmaCross, SPACE, series(), objective=lambda r: r.stats["total_return_pct"],
        n_trials=10, seed=0, fee_taker=0.0, fee_maker=0.0,
    )
    values = [t["value"] for t in result.trials if t["value"] is not None]
    assert result.best_value == max(values)


def test_direction_minimize_selects_the_lowest_value():
    result = tuned(
        SmaCross, SPACE, series(), objective="max_drawdown_pct", direction="minimize",
        n_trials=12, seed=0, fee_taker=0.0, fee_maker=0.0,
    )
    values = [t["value"] for t in result.trials if t["value"] is not None]
    assert result.best_value == min(values)


def test_best_strategy_builds_a_strategy_from_the_best_params():
    result = tuned(SmaCross, SPACE, series(), n_trials=8, seed=0, fee_taker=0.0, fee_maker=0.0)
    strat = result.best_strategy()
    assert isinstance(strat, SmaCross)
    assert strat.fast == result.best_params["fast"]
    assert strat.slow == result.best_params["slow"]


def test_unknown_objective_key_raises_before_the_search():
    # a bad objective key is checked against the known stats up front, not left to
    # surface as every trial silently failing
    with pytest.raises(KeyError):
        tuned(SmaCross, SPACE, series(), objective="not_a_stat", n_trials=5)


def test_misspelled_space_name_raises_before_the_search():
    # a name the strategy constructor does not accept is caught against its signature,
    # not by running any particular parameter combination
    with pytest.raises(TypeError):
        tuned(SmaCross, {"fst": (3, 20), "slow": (21, 60)}, series(), n_trials=5)


def test_bad_market_raises():
    with pytest.raises(ValueError):
        tuned(SmaCross, SPACE, series(), market="options", n_trials=5)


def test_evaluate_keeps_inf_but_fails_on_nan():
    # profit_factor with no losing trades is a legitimate +inf that must rank, not
    # fail; only a NaN objective is an undefined result the search skips
    class FakeResult:
        def __init__(self, stats):
            self.stats = stats

    class FakeBacktester:
        def __init__(self, stats):
            self._stats = stats

        def run(self, strategy):
            return FakeResult(self._stats)

    def ctor(**params):
        return object()

    value, stats = _evaluate(
        FakeBacktester({"pf": float("inf")}), ctor, lambda r: r.stats["pf"], {}
    )
    assert value == float("inf")
    assert stats == {"pf": float("inf")}

    with pytest.raises(ValueError):
        _evaluate(FakeBacktester({"pf": float("nan")}), ctor, lambda r: r.stats["pf"], {})


def test_bad_direction_raises():
    with pytest.raises(ValueError):
        tuned(SmaCross, SPACE, series(), direction="up", n_trials=5)


def test_empty_space_raises():
    with pytest.raises(ValueError):
        tuned(SmaCross, {}, series(), n_trials=5)


def test_strategy_must_be_callable():
    with pytest.raises(TypeError):
        tuned(object(), SPACE, series(), n_trials=5)


def test_space_shorthands_map_to_distributions():
    from optuna.distributions import (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )

    assert isinstance(_to_distribution("a", (5, 40)), IntDistribution)  # two ints
    assert isinstance(_to_distribution("b", (1.0, 2.0)), FloatDistribution)  # a float end
    log = _to_distribution("c", (0.001, 0.1, "log"))
    assert isinstance(log, FloatDistribution) and log.log
    assert isinstance(_to_distribution("d", [1, 2, 3]), CategoricalDistribution)


def test_explicit_specs_carry_step_and_log():
    from optuna.distributions import (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )

    assert isinstance(Int(5, 40, step=5)._distribution(), IntDistribution)
    f = Float(1e-4, 1e-1, log=True)._distribution()
    assert isinstance(f, FloatDistribution) and f.log
    assert isinstance(Categorical(["a", "b"])._distribution(), CategoricalDistribution)


def test_bad_space_entries_raise():
    with pytest.raises(ValueError):
        _to_distribution("x", (1, 2, "nope"))  # third element must be 'log'
    with pytest.raises(TypeError):
        _to_distribution("x", 5)  # not a range, list, or spec


def test_cloudpickled_wrapper_round_trips_a_closure_through_stdlib_pickle():
    # the mechanism that lets a strategy or objective defined in __main__ or a
    # notebook reach a spawn worker: a stdlib pickle of the wrapper routes through
    # cloudpickle, so a closure (which stdlib pickle cannot handle) survives
    factor = 3

    def scale(x):
        return x * factor

    restored = pickle.loads(pickle.dumps(_CloudPickled(scale)))
    assert restored.obj(10) == 30


def test_tune_is_exposed_at_the_top_level():
    assert callable(emsl.tune)
    assert emsl.Strategy is Strategy
    assert emsl.tune.Int is Int
    assert emsl.tune.Float is Float
    assert emsl.tune.Categorical is Categorical


def test_a_search_that_holds_nothing_back_says_the_score_is_in_sample():
    with pytest.warns(UserWarning) as caught:
        result = emsl.tune(SmaCross, SPACE, series(), n_trials=5, seed=0,
                           periods_per_year=365.0)
    assert any("in-sample" in str(w.message) for w in caught)
    assert result.oos_stats is None
    assert "in-sample" in repr(result)


def test_the_winner_is_scored_on_bars_no_trial_ever_saw():
    result = emsl.tune(SmaCross, SPACE, series(), n_trials=8, seed=0, oos=0.3,
                       periods_per_year=365.0, fee_taker=0.0, fee_maker=0.0)
    assert result.oos_stats is not None
    assert result.oos_result.periods_per_year == 365.0
    # the held-out run is a real backtest over the tail, so it covers those bars
    # and only those: 300 bars split at 210 leaves 90
    assert len(result.oos_result.equity_curve) == 89
    # and it is a different number from the in-sample one, which is the point
    assert result.oos_stats["total_return_pct"] != result.best_stats["total_return_pct"]


def test_stating_a_zero_holdout_is_not_the_same_as_not_saying():
    # oos=0 is a decision and passes quietly; leaving it out is not, and warns
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        result = emsl.tune(SmaCross, SPACE, series(), n_trials=5, seed=0, oos=0,
                           periods_per_year=365.0)
    assert result.oos_stats is None


def test_a_holdout_that_leaves_either_side_too_short_names_both_numbers():
    with pytest.raises(ValueError) as excinfo:
        emsl.tune(SmaCross, SPACE, series(6), n_trials=2, oos=0.9,
                  periods_per_year=365.0)
    assert "each side needs at least 2" in str(excinfo.value)


def test_an_out_of_range_holdout_is_refused():
    with pytest.raises(ValueError) as excinfo:
        emsl.tune(SmaCross, SPACE, series(), n_trials=2, oos=1.0,
                  periods_per_year=365.0)
    assert "at least 0 and below 1" in str(excinfo.value)
