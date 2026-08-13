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


def test_a_bad_market_is_refused_before_any_trial_runs():
    # the name used to be shared with the engine-level check in test_smoke, so a
    # failure report named a test that existed twice and said which of the two
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


def test_the_holdout_is_the_end_of_the_series_and_never_the_start():
    # the load-bearing claim of ADR 0049: a strategy is a claim about what comes
    # NEXT. Asserting the holdout's length alone is satisfied just as well by an
    # inverted split, so this pins which bars each half actually saw
    from emsl._tune import _split

    data = series()
    head, tail = _split(data, 0.3)
    assert len(head) == 210 and len(tail) == 90
    assert np.array_equal(head, data[:210])
    assert np.array_equal(tail, data[210:])
    # and the winner really is re-run on that tail rather than on the head
    result = emsl.tune(SmaCross, SPACE, data, n_trials=6, seed=0, oos=0.3,
                       periods_per_year=365.0, fee_taker=0.0, fee_maker=0.0)
    expected = emsl.backtest.Backtester(
        data[210:], periods_per_year=365.0, fee_taker=0.0, fee_maker=0.0
    ).run(result.best_strategy())
    assert result.oos_stats["total_return_pct"] == pytest.approx(
        expected.stats["total_return_pct"]
    )
    assert result.oos_result.data_hash == expected.data_hash


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


def test_every_trial_was_fitted_on_the_first_seventy_percent_and_saw_no_more():
    # the two holdout tests above read _split's own return and the winner's re-run,
    # and a tune that split correctly and then handed the WHOLE series to every
    # trial would satisfy both of them: the tail would still be 90 bars and would
    # still be scored on its own. The claim is about what the TRIALS were given, so
    # this strategy records the series each run was handed. A violation reads as a
    # trial that saw 300 bars, or one whose last close is bar 299 and not bar 209,
    # which is a fit that already knows how the test period ends (ADR 0049)
    data = series()
    seen = []

    class Recorder(Strategy):
        """Records the length and the two endpoints of the series it is run over,
        then buys once so the trial is an ordinary scored backtest.
        """

        def __init__(self, entry):
            self.entry = int(entry)

        def init(self, engine):
            closes = engine.closes
            seen.append((len(closes), float(closes[0]), float(closes[-1])))

        def next(self, state, engine):
            if state["tick_index"] == self.entry:
                engine.market_buy(1.0)

    result = emsl.tune(
        Recorder, {"entry": (1, 5)}, data, objective="total_return_pct",
        n_trials=5, seed=0, oos=0.3, periods_per_year=365.0,
        fee_taker=0.0, fee_maker=0.0,
    )
    # 300 bars at oos=0.3 fit on bars 0..209 and hold back bars 210..299
    head = (210, float(data[0, 3]), float(data[209, 3]))
    tail = (90, float(data[210, 3]), float(data[299, 3]))
    # the five trials, then the winner re-run on that same head, then the tail
    assert seen == [head] * 6 + [tail]
    assert result.oos_stats is not None


# a venue no default agrees with on any knob, so a knob dropped from the trial
# config shows up as a stat that does not match
VENUE = dict(
    market="perp",
    quote=25_000.0,
    fee_taker=0.001,
    fee_maker=0.0005,
    slippage_bps=250.0,
    max_fill_fraction=0.5,
    max_open_orders=3,
    leverage=2.0,
    impact=2.0,
    funding_rate=0.001,
    funding_interval=8,
)


def test_the_whole_venue_reaches_every_trial_and_not_only_the_fees():
    # a knob missing from the trial config scores every trial at a DEFAULT venue,
    # and the winner is then optimal for a market nobody trades: 250 bps of
    # slippage and a perp funding charge would be searched away as free. best_stats
    # is the TRIAL's own stats, not the winner's re-run, so matching it key by key
    # against a Backtester built here at the same venue is the reading that pins
    # the config onto the trials themselves. A violation reads as best_stats
    # agreeing with the default-venue run below instead (ADR 0021)
    data = series()
    result = tuned(SmaCross, SPACE, data, objective="sharpe", n_trials=10, seed=0,
                   **VENUE)
    direct = emsl.backtest.Backtester(
        data, periods_per_year=365.0, risk_free=0.0, **VENUE
    ).run(SmaCross(**result.best_params))
    assert direct.data_hash == result.best_result.data_hash  # the same bars
    assert set(result.best_stats) == set(direct.stats)
    for key in sorted(direct.stats):
        assert result.best_stats[key] == direct.stats[key], key
    # and the venue is distinctive, so the equality above is not two runs that
    # would agree at any settings at all: at the defaults the same parameters pay
    # no funding whatsoever and reach a different return
    default = emsl.backtest.Backtester(data, periods_per_year=365.0).run(
        SmaCross(**result.best_params)
    )
    assert default.stats["funding_paid"] == 0.0
    assert result.best_stats["funding_paid"] != 0.0
    assert result.best_stats["total_return_pct"] != default.stats["total_return_pct"]
    # every knob by value, including the two a market-order crossover cannot feel:
    # max_fill_fraction and max_open_orders move no number on this series, so
    # nothing but the config itself can say they were carried to the trials
    assert result.best_result.config == VENUE


def test_a_trade_floor_refuses_the_thin_winner_an_unfloored_search_takes():
    # min_trades fails a trial outright, and nothing else in the suite runs a
    # search with a floor: delete the raise in _evaluate and the suite stays green
    # while every floored search walks to its thinnest cell, where a handful of
    # samples gives the best-looking number. The objective here is minus the trade
    # count, so the thinnest configuration wins by construction and the floor is
    # the only thing that can stop it. A violation reads as the floored search
    # returning pulses=1 as well, or scoring those trials instead of failing them
    # (ADR 0034)
    data = series()

    class Pulses(Strategy):
        """Opens on bar 10k and closes on bar 10k + 5, so the number of trades a
        trial closes is exactly its parameter and nothing else.
        """

        def __init__(self, pulses):
            self.pulses = int(pulses)
            self.entries = {10 * k for k in range(self.pulses)}
            self.exits = {10 * k + 5 for k in range(self.pulses)}

        def next(self, state, engine):
            i = state["tick_index"]
            if i in self.entries:
                engine.market_buy(1.0)
            elif i in self.exits:
                engine.close()

    def fewest_trades(result):
        return -float(result.stats["num_trades"])

    kwargs = dict(objective=fewest_trades, n_trials=20, seed=3, sampler="random",
                  fee_taker=0.0, fee_maker=0.0)
    loose = tuned(Pulses, {"pulses": [1, 5]}, data, min_trades=0, **kwargs)
    floored = tuned(Pulses, {"pulses": [1, 5]}, data, min_trades=5, **kwargs)
    # both choices were drawn in both searches, so neither winner is an artifact
    # of a space one of them never explored
    assert {t["params"]["pulses"] for t in loose.trials} == {1, 5}
    assert {t["params"]["pulses"] for t in floored.trials} == {1, 5}
    # unfloored, the single-trade configuration is the maximum of -num_trades
    assert loose.best_params == {"pulses": 1}
    assert loose.best_value == -1.0
    assert loose.best_stats["num_trades"] == 1
    # with the floor, every trial that drew it is failed rather than scored, so
    # the winner is the only configuration that clears five closed trades
    assert floored.best_params == {"pulses": 5}
    assert floored.best_value == -5.0
    assert floored.best_stats["num_trades"] == 5
    assert floored.min_trades == 5
    thin = [t for t in floored.trials if t["params"]["pulses"] == 1]
    assert thin and all(t["state"] == "FAIL" for t in thin)
    assert all(t["value"] is None and t["stats"] is None for t in thin)
