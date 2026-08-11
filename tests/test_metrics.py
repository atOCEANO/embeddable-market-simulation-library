"""Tests for emsl.metrics: it should reproduce the numbers the engine already
reports, account for every unit of quote a run moved, and refuse to attach a
probability to a sample that cannot carry one.
"""

import math

import numpy as np
import pytest

import emsl
from emsl import metrics
from emsl.backtest import Backtester, BacktestResult, Strategy


def series(n=120, seed=3):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.05, 1.0, n))
    return np.column_stack(
        [close, close + 1.0, close - 1.0, close, np.full(n, 1000.0)]
    )


class Alternate(Strategy):
    """Long for five bars, flat for five, so the run ends flat and both a trade log
    and an equity curve exist."""

    def next(self, state, engine):
        i = state["tick_index"]
        if i % 10 == 0:
            engine.market_buy(1.0)
        elif i % 10 == 5:
            engine.close()


class Hold(Strategy):
    def next(self, state, engine):
        if state["tick_index"] == 0:
            engine.market_buy(1.0)


def run(strategy=None, **kwargs):
    kwargs.setdefault("periods_per_year", 365.0)
    return Backtester(series(), **kwargs).run(strategy or Alternate())


# ------------------------------------------------------------ the agreement


def test_the_sharpe_recomputed_here_is_the_sharpe_the_engine_reported():
    # two paths to one number. The drawdown panel once disagreed with the engine's
    # own maximum for exactly this reason, and this is the guard against the whole
    # class of it
    result = run()
    assert math.isclose(metrics.sharpe(result), result.stats["sharpe"], rel_tol=1e-9)


def test_the_agreement_holds_at_another_annualization_and_a_risk_free_rate():
    result = run(periods_per_year=8760.0, risk_free=0.02)
    assert math.isclose(metrics.sharpe(result), result.stats["sharpe"], rel_tol=1e-9)


def test_the_returns_are_seeded_from_the_opening_balance():
    # the curve records a point per advance, so it does not contain the balance the
    # run started at, and anything reading it alone misses the first bar's move
    result = run()
    assert len(metrics.returns(result)) == len(result.equity_curve)
    first = result.equity_curve[0] / result.initial - 1.0
    assert math.isclose(metrics.returns(result)[0], first, rel_tol=1e-12)


def test_a_result_built_by_hand_says_what_it_is_missing():
    bare = BacktestResult(stats={}, equity_curve=np.array([1.0, 2.0]), trades=[])
    with pytest.raises(ValueError) as excinfo:
        metrics.returns(bare)
    assert "no opening balance" in str(excinfo.value)


# ------------------------------------------------------------ the decomposition


def test_every_unit_of_quote_a_flat_run_moved_is_accounted_for():
    # gross price pnl, minus fees, minus funding, equals the change in equity. This
    # is the ADR 0030 identity extended past the trade log to the whole account
    result = run()
    money = metrics.decompose(result)
    assert math.isclose(
        money["gross_pnl"] - money["fees"] - money["funding"],
        money["net"],
        abs_tol=1e-9,
    )
    assert math.isclose(money["unrealized"], 0.0, abs_tol=1e-9)
    assert math.isclose(
        money["net"], result.equity_curve[-1] - result.initial, abs_tol=1e-9
    )


def test_a_run_still_holding_at_the_end_puts_the_remainder_in_unrealized():
    result = run(Hold())
    money = metrics.decompose(result)
    assert result.stats["num_trades"] == 0
    assert money["unrealized"] != 0.0
    assert math.isclose(
        money["gross_pnl"] - money["fees"] - money["funding"] + money["unrealized"],
        money["net"],
        abs_tol=1e-9,
    )


def test_funding_shows_up_in_the_decomposition_of_a_perp_run():
    result = run(market="perp", funding_rate=0.001, funding_interval=4)
    assert result.stats["funding_paid"] != 0.0
    money = metrics.decompose(result)
    assert money["funding"] == result.stats["funding_paid"]
    assert math.isclose(
        money["gross_pnl"] - money["fees"] - money["funding"] + money["unrealized"],
        money["net"],
        abs_tol=1e-9,
    )


# ------------------------------------------------------------ shape


def test_the_drawdown_series_matches_the_engines_own_maximum():
    result = run()
    assert math.isclose(
        -min(metrics.drawdown(result)), result.stats["max_drawdown_pct"], abs_tol=1e-9
    )


def test_the_drawdown_table_orders_the_worst_first_and_bounds_each_episode():
    table = metrics.drawdown_table(run(), top=3)
    assert len(table) <= 3
    assert table == sorted(table, key=lambda e: e["depth_pct"])
    for episode in table:
        assert episode["start_bar"] <= episode["trough_bar"]
        assert episode["bars_under"] >= episode["bars_to_trough"]


def test_time_under_water_agrees_with_the_drawdown_series():
    result = run()
    water = metrics.time_under_water(result)
    below = sum(1 for v in metrics.drawdown(result) if v < 0.0)
    assert math.isclose(
        water["share_pct"], below / len(result.equity_curve) * 100.0, abs_tol=1e-9
    )
    assert water["longest_bars"] <= below


def test_the_sides_split_covers_every_trade():
    result = run()
    sides = metrics.long_short_split(result)
    assert sides["long"]["trades"] + sides["short"]["trades"] == len(result.trades)
    assert math.isclose(
        sides["long"]["net_pnl"] + sides["short"]["net_pnl"],
        sum(t["net_pnl"] for t in result.trades),
        abs_tol=1e-9,
    )


def test_holding_is_compared_against_on_the_same_bars():
    data = series()
    result = Backtester(data, periods_per_year=365.0).run(Alternate())
    hold = metrics.buy_and_hold(result, data)
    expected = (data[-1, 3] / data[0, 3] - 1.0) * 100.0
    assert math.isclose(hold["hold_return_pct"], expected, rel_tol=1e-9)
    assert math.isclose(
        hold["excess_return_pct"],
        result.stats["total_return_pct"] - expected,
        abs_tol=1e-9,
    )


# ------------------------------------------------------------ the inference tier


def test_the_probability_is_computed_on_the_per_period_sharpe():
    # the annualized figure is what the result reports, and handing it to the
    # estimator unchanged is the standard way to get a confidently wrong answer.
    # Same run, two annualizations: the probability must not move
    daily = run(periods_per_year=365.0)
    hourly = run(periods_per_year=8760.0)
    assert daily.stats["sharpe"] != hourly.stats["sharpe"]
    assert math.isclose(
        metrics.probabilistic_sharpe(daily),
        metrics.probabilistic_sharpe(hourly),
        rel_tol=1e-9,
    )


def test_the_probability_is_between_zero_and_one_and_rises_with_the_sharpe():
    good = run()
    assert 0.0 <= metrics.probabilistic_sharpe(good) <= 1.0
    assert metrics.probabilistic_sharpe(good, benchmark=0.0) >= metrics.probabilistic_sharpe(
        good, benchmark=5.0
    )


def test_a_sample_too_short_to_carry_a_probability_says_so():
    short = BacktestResult(
        stats={}, equity_curve=np.array([101.0, 102.0]), trades=[],
        initial=100.0, periods_per_year=365.0,
    )
    with pytest.raises(ValueError) as excinfo:
        metrics.probabilistic_sharpe(short)
    assert "at least 4 returns" in str(excinfo.value)


def test_a_flat_curve_has_no_probability_rather_than_a_nan():
    flat = BacktestResult(
        stats={}, equity_curve=np.array([100.0] * 20), trades=[],
        initial=100.0, periods_per_year=365.0,
    )
    with pytest.raises(ValueError):
        metrics.probabilistic_sharpe(flat)


def profitable(n=200):
    # a curve built to have a real, positive sharpe, so the length test measures
    # the function rather than skipping whenever a seed's backtest happens to lose
    rng = np.random.default_rng(0)
    curve = 100.0 * np.cumprod(1.0 + rng.normal(0.01, 0.01, n))
    return BacktestResult(
        stats={"num_trades": 12}, equity_curve=curve, trades=[],
        initial=100.0, periods_per_year=365.0,
    )


def test_the_track_record_length_says_how_many_bars_would_be_enough():
    result = profitable()
    assert metrics.sharpe(result) > 0.0
    need = metrics.min_track_record_length(result)
    assert need["bars"] > 0.0
    assert need["have_bars"] == len(result.equity_curve)
    assert need["enough"] == (need["have_bars"] >= need["bars"])
    assert need["num_trades"] == 12
    # a stiffer benchmark always needs more bars to clear
    far = metrics.min_track_record_length(result, benchmark=metrics.sharpe(result) * 0.9)
    assert far["bars"] > need["bars"]


def test_a_sharpe_below_the_benchmark_cannot_be_made_significant_by_waiting():
    result = run()
    with pytest.raises(ValueError) as excinfo:
        metrics.min_track_record_length(result, benchmark=1e6)
    assert "not above the benchmark" in str(excinfo.value)


def test_the_inverse_normal_matches_the_values_everyone_knows():
    assert math.isclose(metrics._normal_ppf(0.95), 1.6448536, abs_tol=1e-6)
    assert math.isclose(metrics._normal_ppf(0.975), 1.9599640, abs_tol=1e-6)
    assert math.isclose(metrics._normal_ppf(0.5), 0.0, abs_tol=1e-9)
    assert math.isclose(metrics._normal_cdf(1.6448536), 0.95, abs_tol=1e-6)


def test_kurtosis_is_not_excess_unless_asked():
    result = run()
    assert math.isclose(
        metrics.kurtosis(result), metrics.kurtosis(result, excess=True) + 3.0,
        abs_tol=1e-9,
    )


# ------------------------------------------------------------ presentation


def test_the_report_carries_the_engine_stats_and_everything_derived():
    data = series()
    result = Backtester(data, periods_per_year=365.0).run(Alternate())
    out = metrics.report(result, data)
    for key in result.stats:
        assert out[key] == result.stats[key]
    for key in ("decompose_net", "under_water_longest_bars", "skew", "autocorr_1",
                "long_trades", "short_trades", "hold_beta"):
        assert key in out


def test_the_summary_prints_and_returns_the_same_numbers(capsys):
    result = run()
    out = metrics.summary(result)
    printed = capsys.readouterr().out
    assert "net" in printed and "sharpe" in printed
    assert out["decompose_net"] == metrics.decompose(result)["net"]


def test_metrics_is_exported_from_the_package():
    assert emsl.metrics is metrics
    assert "metrics" in emsl.__all__
