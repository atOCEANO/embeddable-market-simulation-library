"""Tests for emsl.metrics: it should reproduce the numbers the engine already
reports, account for every unit of quote a run moved, and refuse to attach a
probability to a sample that cannot carry one.
"""

import math

import numpy as np
import pytest

import emsl
from emsl import metrics
from emsl._walk import WalkForward
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


# ------------------------------------------------------------ costs and shape


def test_the_cost_curve_charges_more_as_the_friction_rises():
    curve = metrics.cost_curve(Alternate, series(), costs=(0.0, 10.0, 50.0),
                               periods_per_year=365.0)
    assert [row["round_trip_bps"] for row in curve] == [0.0, 10.0, 50.0]
    assert curve[0]["fees"] == 0.0
    assert curve[1]["fees"] < curve[2]["fees"]
    assert curve[0]["total_return_pct"] > curve[2]["total_return_pct"]


def test_a_strategy_class_is_rebuilt_for_each_run_and_an_instance_is_reused():
    # a class comes back fresh per run, so nothing carries over between them
    assert metrics._fresh(Alternate) is not metrics._fresh(Alternate)
    instance = Alternate()
    assert metrics._fresh(instance) is instance


def trending(n=120):
    # a straight line up, so the strategy is reliably profitable at zero cost and
    # the breakeven is a real number rather than a coin flip on the seed
    close = 100.0 + np.arange(n, dtype=np.float64) * 0.5
    return np.column_stack(
        [close, close + 1.0, close - 1.0, close, np.full(n, 1000.0)]
    )


def test_breakeven_finds_the_cost_that_kills_the_edge():
    data = trending()
    breakeven = metrics.breakeven_bps(Alternate, data, ceiling=500.0,
                                      periods_per_year=365.0)
    assert breakeven is not None and 0.0 < breakeven < 500.0
    # comfortably under it the run is up, comfortably over it the run is down
    below = metrics.cost_curve(Alternate, data, costs=(breakeven * 0.5,),
                               periods_per_year=365.0)[0]
    above = metrics.cost_curve(Alternate, data, costs=(breakeven * 2.0,),
                               periods_per_year=365.0)[0]
    assert below["total_return_pct"] > 0.0
    assert above["total_return_pct"] < 0.0


def test_a_strategy_that_survives_every_cost_reports_the_ceiling():
    # the ceiling is an answer, not a failure: it says the edge outlives the sweep
    assert metrics.breakeven_bps(Alternate, trending(), ceiling=1.0,
                                 periods_per_year=365.0) == 1.0


def test_a_strategy_that_loses_for_free_has_no_breakeven():
    # slippage is a cost the sweep does not set, so it goes through and makes this
    # run a loser even at a zero fee
    assert metrics.breakeven_bps(Alternate, series(), slippage_bps=200.0,
                                 periods_per_year=365.0) is None


def test_setting_a_fee_on_a_cost_sweep_is_a_contradiction_not_an_override():
    for call in (metrics.cost_curve, metrics.breakeven_bps):
        with pytest.raises(TypeError) as excinfo:
            call(Alternate, series(), fee_taker=0.001)
        assert "sweeping the cost is what it does" in str(excinfo.value)


def test_every_trade_lived_through_at_least_what_it_finished_with():
    data = series()
    result = Backtester(data, periods_per_year=365.0).run(Alternate())
    rows = metrics.excursions(result, data)
    assert len(rows) == len(result.trades)
    for row, trade in zip(rows, result.trades):
        # the best it ever showed is never below the worst it ever showed. The
        # second assertion here used to be `worst <= 0 or best >= 0`, which the
        # line above it already implies and which therefore tested nothing
        assert row["best_pct"] >= row["worst_pct"]
        # and the excursion brackets what the trade actually finished at, which
        # is the property that catches best and worst being swapped on one side
        finished = (trade["exit_price"] / trade["entry_price"] - 1.0) * 100.0
        if trade["side"] == "sell":
            finished = -finished
        assert row["worst_pct"] - 1e-9 <= finished <= row["best_pct"] + 1e-9


def test_a_shorts_excursions_are_not_a_longs_read_backwards():
    # a short is hurt by the high and helped by the low, and swapping the two
    # puts a stop on exactly the wrong side of the entry
    data = np.array([
        [100.0, 100.0, 100.0, 100.0, 1e6],
        [100.0, 100.0, 100.0, 100.0, 1e6],
        [100.0, 130.0, 70.0, 100.0, 1e6],
        [100.0, 100.0, 100.0, 100.0, 1e6],
        [100.0, 100.0, 100.0, 100.0, 1e6],
    ])

    class ShortOnce(Strategy):
        def next(self, state, engine):
            i = state["tick_index"]
            if i == 0:
                engine.market_sell(1.0)
            elif i == 3:
                engine.close()

    result = Backtester(data, market="perp", periods_per_year=365.0).run(ShortOnce())
    row = metrics.excursions(result, data)[0]
    assert row["side"] == "sell"
    # entered at 100, the bar ran to 130 (against) and 70 (for)
    assert row["worst_pct"] == pytest.approx(-30.0)
    assert row["best_pct"] == pytest.approx(30.0)


def test_excursions_needs_the_highs_and_lows():
    result = run()
    with pytest.raises(TypeError) as excinfo:
        metrics.excursions(result, np.zeros((10, 2)))
    assert "high and low" in str(excinfo.value)


def test_trades_bucket_by_the_hour_they_closed_on():
    pd = pytest.importorskip("pandas")
    raw = series()
    data = pd.DataFrame(raw, columns=["open", "high", "low", "close", "volume"])
    data.index = pd.date_range("2025-01-01", periods=len(raw), freq="1h", tz="UTC")
    result = Backtester(data).run(Alternate())
    buckets = metrics.session_buckets(result, data, by="hour")
    assert set(buckets) <= set(range(24))
    assert sum(b["trades"] for b in buckets.values()) == len(result.trades)
    assert math.isclose(
        sum(b["net_pnl"] for b in buckets.values()),
        sum(t["net_pnl"] for t in result.trades),
        abs_tol=1e-9,
    )
    # every bucket key is the hour pandas reads off the same stamp. `set(buckets)
    # <= set(range(24))` used to stand alone here, which is true of anything
    # modulo 24 and pinned no convention at all
    expected = {}
    for tick in (t["exit_tick"] for t in result.trades):
        hour = int(data.index[tick].hour)
        expected[hour] = expected.get(hour, 0) + 1
    assert {k: v["trades"] for k, v in buckets.items()} == expected


def test_weekdays_are_numbered_from_monday_like_every_other_clock():
    # epoch day zero is a Thursday, and the offset was one out, so this numbered
    # from Sunday while datetime.weekday and pandas.dayofweek number from Monday.
    # buckets[0] read as Monday and was Sunday, and nothing said so (ADR 0059)
    pd = pytest.importorskip("pandas")
    raw = series(n=400)
    data = pd.DataFrame(raw, columns=["open", "high", "low", "close", "volume"])
    data.index = pd.date_range("2025-01-01", periods=len(raw), freq="1h", tz="UTC")
    result = Backtester(data).run(Alternate())
    buckets = metrics.session_buckets(result, data, by="weekday")
    expected = {}
    for tick in (t["exit_tick"] for t in result.trades):
        day = int(data.index[tick].dayofweek)
        expected[day] = expected.get(day, 0) + 1
    assert {k: v["trades"] for k, v in buckets.items()} == expected
    # and 2025-01-06 is a Monday, so a trade closing that day lands in bucket 0
    assert int(pd.Timestamp("2025-01-06", tz="UTC").dayofweek) == 0


def test_bucketing_without_a_clock_says_so_rather_than_counting_bars():
    result = run()
    with pytest.raises(TypeError) as excinfo:
        metrics.session_buckets(result, series())
    assert "timestamps" in str(excinfo.value)


def test_an_unknown_bucket_is_refused():
    result = run()
    with pytest.raises(ValueError) as excinfo:
        metrics.session_buckets(result, series(), by="minute")
    assert "'hour' or 'weekday'" in str(excinfo.value)


# ------------------------------------------------------- the frame has to match


def test_a_metric_that_takes_a_frame_refuses_one_the_run_never_saw():
    # excursions silently returned 35 rows of 1,516 given a slice of its own
    # frame, and buy_and_hold trimmed the two series from opposite ends, comparing
    # the run's last returns against the frame's first bars (ADR 0059)
    data = series(n=200)
    result = Backtester(data, periods_per_year=365.0).run(Alternate())
    short = data[:80]
    for call in (metrics.excursions, metrics.buy_and_hold):
        with pytest.raises(ValueError) as excinfo:
            call(result, short)
        assert "200" in str(excinfo.value) and "80" in str(excinfo.value)


def test_a_different_asset_of_the_same_length_is_caught_where_it_matters():
    # a length check cannot see this one, and the trade ticks would be read
    # against another asset's prices
    data = series(n=200)
    result = Backtester(data, periods_per_year=365.0).run(Alternate())
    elsewhere = data.copy()
    elsewhere[:, :4] *= 3.0
    with pytest.raises(ValueError) as excinfo:
        metrics.excursions(result, elsewhere)
    assert "different series" in str(excinfo.value)


def test_but_a_benchmark_against_another_asset_is_the_whole_point():
    # beta and the information ratio are only interesting against something other
    # than what you traded, so buy_and_hold checks the length and not the identity
    data = series(n=200)
    result = Backtester(data, periods_per_year=365.0).run(Alternate())
    other = series(n=200, seed=99)
    hold = metrics.buy_and_hold(result, other)
    expected = (other[-1, 3] / other[0, 3] - 1.0) * 100.0
    assert hold["hold_return_pct"] == pytest.approx(expected)


def test_beta_is_measured_against_the_benchmark_and_not_against_itself():
    # computed against the run's own returns it is exactly 1.0 for every input,
    # which no test noticed
    data = series(n=200)
    result = Backtester(data, periods_per_year=365.0).run(Alternate())
    hold = metrics.buy_and_hold(result, data)
    assert hold["beta"] != pytest.approx(1.0)
    mine = metrics.returns(result)
    held = data[1:, 3] / data[:-1, 3] - 1.0
    expected = np.cov(mine, held, ddof=1)[0, 1] / np.var(held, ddof=1)
    assert hold["beta"] == pytest.approx(expected)


def test_the_benchmark_sharpe_is_net_of_the_same_rate_the_run_is():
    data = series(n=200)
    free = Backtester(data, periods_per_year=365.0).run(Alternate())
    charged = Backtester(data, periods_per_year=365.0, risk_free=0.5).run(Alternate())
    assert (metrics.buy_and_hold(charged, data)["hold_sharpe"]
            < metrics.buy_and_hold(free, data)["hold_sharpe"])


# ------------------------------------------------------------ over a stretch


def test_a_segment_of_the_whole_run_is_the_run():
    # segment is a second path to arithmetic the engine owns, which is the drift
    # this codebase keeps paying for, so it is pinned key by key (ADR 0060)
    result = run()
    whole = metrics.segment(result)
    for key, value in whole.items():
        assert value == pytest.approx(result.stats[key], rel=1e-9, abs=1e-9), key
    assert "exposure_pct" not in whole and "num_fills" not in whole


def test_consecutive_segments_compound_to_the_run():
    # each is seeded from the balance carried into it, so no return falls between
    # two of them and none is counted twice
    result = run()
    bars = len(result.equity_curve) + 1
    edges = [0, 40, 90, bars]
    product = 1.0
    for first, last in zip(edges, edges[1:]):
        product *= 1.0 + metrics.segment(result, first, last)["total_return_pct"] / 100.0
    assert (product - 1.0) * 100.0 == pytest.approx(
        result.stats["total_return_pct"], abs=1e-9
    )


def test_every_trade_belongs_to_exactly_one_segment():
    result = run()
    bars = len(result.equity_curve) + 1
    edges = [0, 40, 90, bars]
    counted = sum(metrics.segment(result, a, b)["num_trades"]
                  for a, b in zip(edges, edges[1:]))
    assert counted == result.stats["num_trades"]


def test_a_segment_outside_the_run_is_refused():
    result = run()
    bars = len(result.equity_curve) + 1
    for start, stop in ((0, bars + 1), (-1, 10), (50, 50), (60, 20)):
        with pytest.raises(ValueError):
            metrics.segment(result, start, stop)


def test_period_returns_split_the_run_by_the_calendar():
    pd = pytest.importorskip("pandas")
    raw = series(n=2_000)
    data = pd.DataFrame(raw, columns=["open", "high", "low", "close", "volume"])
    data.index = pd.date_range("2025-01-01", periods=len(raw), freq="1h", tz="UTC")
    result = Backtester(data).run(Alternate())
    months = metrics.period_returns(result, data, by="month")
    assert len(months) >= 2
    assert [m["period"] for m in months] == sorted(m["period"] for m in months)
    assert sum(m["bars"] for m in months) == len(raw)
    assert sum(m["num_trades"] for m in months) == result.stats["num_trades"]
    quarters = metrics.period_returns(result, data, by="quarter")
    assert len(quarters) < len(months)


def test_a_run_ending_a_bar_into_a_month_still_accounts_for_that_bar():
    # periods under two bars were skipped, which read as tidying away a degenerate
    # statistic and was a hole in the one thing the function promises. A year of
    # hourly candles ending just into a month is the ordinary shape of real data,
    # and those bars vanished: the table stopped adding up to the headline above
    # it, quietly (ADRs 0062, 0081)
    pd = pytest.importorskip("pandas")
    hours = 24 * 59 + 1        # two whole months, then a single bar of the third
    raw = series(n=hours)
    data = pd.DataFrame(raw, columns=["open", "high", "low", "close", "volume"])
    data.index = pd.date_range("2025-01-01", periods=hours, freq="1h", tz="UTC")
    result = Backtester(data).run(Alternate())

    months = metrics.period_returns(result, data, by="month")
    assert [m["period"] for m in months] == ["2025-01", "2025-02", "2025-03"]
    assert sum(m["bars"] for m in months) == hours
    compounded = 1.0
    for month in months:
        compounded *= 1.0 + month["total_return_pct"] / 100.0
    assert compounded == pytest.approx(
        1.0 + result.stats["total_return_pct"] / 100.0, rel=1e-9
    )


def test_the_calendar_periods_compound_back_to_the_whole_run():
    # each period is computed on the equity the account actually carried into it,
    # so the periods multiply back to the run. A period seeded from a fresh
    # balance passes every other check here: the labels are the same, the bars
    # still add up, and only the compounding identity says that a bad January
    # shrank the size February had to trade with (ADR 0062).
    #
    # All three groupings are read, over daily bars running into a third calendar
    # year so that each one has several rows. On the hourly run this used to use,
    # the quarterly table was a single row and the yearly table was never asked
    # for at all: one row compounds back to the run whatever seeded it, so the
    # quarter and the year were carried on the month's evidence
    pd = pytest.importorskip("pandas")
    raw = series(n=900)
    data = pd.DataFrame(raw, columns=["open", "high", "low", "close", "volume"])
    data.index = pd.date_range("2025-01-01", periods=len(raw), freq="1D", tz="UTC")
    result = Backtester(data).run(Alternate())
    path = np.concatenate(([result.initial], result.equity_curve))

    counted = {}
    for by in ("month", "quarter", "year"):
        periods = metrics.period_returns(result, data, by=by)
        counted[by] = len(periods)
        assert sum(p["bars"] for p in periods) == len(raw), by
        assert [p["period"] for p in periods] == sorted(p["period"] for p in periods)
        product = 1.0
        for period in periods:
            first, last = period["start_bar"], period["start_bar"] + period["bars"]
            # the balance carried in, the opening balance only for the first
            opening = path[first - 1] if first else path[0]
            assert period["total_return_pct"] == pytest.approx(
                (path[last - 1] / opening - 1.0) * 100.0
            ), (by, period["period"])
            product *= 1.0 + period["total_return_pct"] / 100.0
        assert (product - 1.0) * 100.0 == pytest.approx(
            result.stats["total_return_pct"], abs=1e-9
        ), by
    # 900 daily bars from new year reach 19 June 2027, so no grouping here is the
    # single row that cannot fail
    assert counted == {"month": 30, "quarter": 10, "year": 3}


def test_a_rolling_sharpe_is_aligned_like_an_indicator():
    result = run()
    bars = len(result.equity_curve) + 1
    line = metrics.rolling_sharpe(result, window=30)
    assert line.shape == (bars,)
    assert not np.isfinite(line[:30]).any()
    assert np.isfinite(line[30])
    # the last window is the sharpe of a run over exactly those bars
    tail = metrics.segment(result, bars - 30, bars)["sharpe"]
    assert line[-1] == pytest.approx(tail, rel=1e-6)


# ------------------------------------------------------------ identity


def test_a_result_knows_what_data_and_what_costs_produced_it():
    data = series()
    result = Backtester(data, market="perp", fee_taker=0.001,
                        periods_per_year=365.0).run(Alternate())
    assert result.config["market"] == "perp"
    assert result.config["fee_taker"] == 0.001
    assert result.strategy.startswith("Alternate")
    assert len(result.data_hash) == 8
    assert result.version == emsl.__version__


def test_the_same_bars_fingerprint_the_same_and_different_bars_do_not():
    same = Backtester(series(), periods_per_year=365.0).run(Alternate())
    again = Backtester(series(), periods_per_year=365.0).run(Alternate())
    other = Backtester(series(seed=9), periods_per_year=365.0).run(Alternate())
    assert same.data_hash == again.data_hash
    assert same.data_hash != other.data_hash


def test_a_single_changed_bar_changes_the_fingerprint():
    data = series()
    nudged = data.copy()
    nudged[50, 3] += 1e-9
    assert (Backtester(data, periods_per_year=365.0).run(Alternate()).data_hash
            != Backtester(nudged, periods_per_year=365.0).run(Alternate()).data_hash)


def test_to_dict_carries_the_identity_and_the_headline_but_not_the_curve():
    out = run().to_dict()
    for key in ("strategy", "data_hash", "version", "initial", "bars",
                "config_market", "config_fee_taker", "sharpe", "total_return_pct"):
        assert key in out
    assert "equity_curve" not in out
    assert "trades" not in out


def test_compare_lines_runs_up_and_says_which_saw_the_same_bars(capsys):
    data = series()
    cheap = Backtester(data, fee_taker=0.0, periods_per_year=365.0).run(Alternate())
    dear = Backtester(data, fee_taker=0.01, periods_per_year=365.0).run(Alternate())
    rows = metrics.compare({"cheap": cheap, "dear": dear})
    printed = capsys.readouterr().out
    assert "cheap" in printed and "dear" in printed
    assert printed.count(data_hash := cheap.data_hash) == 2  # same bars, both rows
    assert dear.data_hash == data_hash
    assert [r["name"] for r in rows] == ["cheap", "dear"]
    assert rows[0]["config_fee_taker"] == 0.0


def test_compare_takes_a_plain_list_too():
    rows = metrics.compare([run(), run()])
    assert len(rows) == 2
    assert all(r["name"].startswith("Alternate") for r in rows)


def test_comparing_nothing_is_not_an_error():
    assert metrics.compare([]) == []


def test_comparing_on_a_key_nothing_reports_is_refused():
    # it printed the word None in every row, which reads as a run that scored
    # nothing rather than as a column that does not exist
    with pytest.raises(KeyError) as excinfo:
        metrics.compare([run()], keys=["sharpe", "alpha"])
    assert "alpha" in str(excinfo.value)


# ------------------------------------------------- the estimators, checked hard


def test_the_probability_uses_plain_kurtosis_and_not_the_excess_kind():
    # the trap `kurtosis`'s own docstring exists to prevent, asserted against a
    # hand-computed value rather than against the implementation
    result = run()
    values = metrics.returns(result)
    ppy = result.periods_per_year
    observed = metrics.sharpe(result) / math.sqrt(ppy)
    plain = metrics.kurtosis(result)
    assert plain == pytest.approx(metrics.kurtosis(result, excess=True) + 3.0)
    variance = 1.0 - metrics.skew(result) * observed + (plain - 1.0) / 4.0 * observed ** 2
    size = metrics.effective_sample(result)
    z = observed * math.sqrt(size - 1) / math.sqrt(variance)
    expected = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    assert metrics.probabilistic_sharpe(result) == pytest.approx(expected, rel=1e-12)
    assert values.size >= 4


def test_confidence_is_counted_in_bets_rather_than_bars():
    # the module reported autocorrelation, said in prose that every confidence
    # figure computed from the bar count was overstated, and then computed them
    # from the bar count (ADR 0061)
    result = run()
    assert metrics.effective_sample(result) <= len(metrics.returns(result))
    adjusted = metrics.probabilistic_sharpe(result)
    raw = metrics.probabilistic_sharpe(result, independent=True)
    if metrics.autocorrelation(result) > 0.0:
        assert metrics.effective_sample(result) < len(metrics.returns(result))
        # fewer independent observations can only make a positive sharpe less
        # certain, never more
        assert (adjusted < raw) == (metrics.sharpe(result) > 0.0)
    else:
        assert adjusted == pytest.approx(raw)


def persistent():
    # a run whose returns cluster, which is what holding a position across bars
    # produces and what makes a bar count an overstatement of the bets taken
    rng = np.random.default_rng(17)
    steps = rng.normal(0.0004, 0.004, 400)
    smoothed = np.convolve(steps, np.full(20, 1.0 / 20.0), mode="same")
    return BacktestResult(stats={}, equity_curve=100.0 * np.cumprod(1.0 + smoothed),
                          trades=[], initial=100.0, periods_per_year=8760.0)


def test_autocorrelation_is_the_ordinary_estimator():
    # pinned against the arithmetic rather than against itself: the denominator is
    # the CENTRED sum of squares, and using the raw one is close enough to look
    # right on returns whose mean is near zero
    result = run()
    values = metrics.returns(result)
    centred = values - values.mean()
    for lag in (1, 2, 5):
        expected = float(centred[:-lag].dot(centred[lag:]) / centred.dot(centred))
        assert metrics.autocorrelation(result, lag) == pytest.approx(expected, rel=1e-12)
    assert centred.dot(centred) != pytest.approx(float(values.dot(values)), rel=1e-12)


def test_the_probability_really_counts_the_effective_sample():
    # the correction is only worth having if it is applied, and on a run with no
    # persistence the adjusted and unadjusted answers agree, which hides it
    sticky = persistent()
    bars = len(metrics.returns(sticky))
    assert metrics.autocorrelation(sticky) > 0.5
    assert metrics.effective_sample(sticky) < 0.5 * bars
    adjusted = metrics.probabilistic_sharpe(sticky)
    naive = metrics.probabilistic_sharpe(sticky, independent=True)
    assert adjusted != pytest.approx(naive)
    assert adjusted < naive                        # fewer bets, less certainty
    band = metrics.sharpe_interval(sticky)
    tight = metrics.sharpe_interval(sticky, independent=True)
    assert band["high"] - band["low"] > tight["high"] - tight["low"]
    assert band["effective_bars"] < bars


def test_a_persistent_series_is_worth_fewer_bets_than_it_has_bars():
    curve = np.cumprod(1.0 + np.concatenate([
        np.full(60, 0.01), np.full(60, -0.005), np.full(60, 0.01)
    ])) * 100.0
    sticky = BacktestResult(stats={}, equity_curve=curve, trades=[], initial=100.0,
                            periods_per_year=365.0)
    assert metrics.autocorrelation(sticky) > 0.5
    assert metrics.effective_sample(sticky) < 0.5 * len(curve)
    # and negative autocorrelation is floored away rather than credited
    zigzag = np.cumprod(1.0 + np.tile([0.02, -0.019], 90)) * 100.0
    choppy = BacktestResult(stats={}, equity_curve=zigzag, trades=[], initial=100.0,
                            periods_per_year=365.0)
    assert metrics.autocorrelation(choppy) < 0.0
    assert metrics.effective_sample(choppy) == float(len(zigzag))


def test_autocorrelation_at_lag_zero_is_one_and_a_negative_lag_is_refused():
    result = run()
    assert metrics.autocorrelation(result, lag=0) == 1.0
    with pytest.raises(ValueError):
        metrics.autocorrelation(result, lag=-1)


def test_the_track_record_length_keeps_its_leading_term():
    # it needs a sharpe above the benchmark to answer at all, so a rising series
    result = Backtester(trending(), periods_per_year=365.0).run(Alternate())
    need = metrics.min_track_record_length(result, independent=True)
    observed = metrics.sharpe(result) / math.sqrt(result.periods_per_year)
    variance = 1.0 - metrics.skew(result) * observed + (
        metrics.kurtosis(result) - 1.0) / 4.0 * observed ** 2
    expected = 1.0 + variance * (metrics._normal_ppf(0.95) / observed) ** 2
    assert need["bars"] == pytest.approx(expected, rel=1e-12)
    # and counting bets rather than bars can only ask for more of them
    assert metrics.min_track_record_length(result)["bars"] >= need["bars"] - 1e-9


def test_a_sharpe_interval_brackets_the_sharpe():
    result = run()
    band = metrics.sharpe_interval(result)
    assert band["low"] < band["sharpe"] < band["high"]
    assert band["sharpe"] == pytest.approx(result.stats["sharpe"])
    tighter = metrics.sharpe_interval(result, confidence=0.68)
    assert tighter["high"] - tighter["low"] < band["high"] - band["low"]


def blocked():
    # 120 returns taking exactly two values, 1/128 + 1/64 and 1/128 - 1/64, laid
    # out in blocks of four. Two values put every centred return at plus or minus
    # 1/64, so the skew is zero and the kurtosis is exactly 1, which leaves the
    # estimator variance at exactly 1.0 and every figure below hand computable.
    # The blocks leave 90 same-signed neighbours against 29 sign changes, so the
    # first autocorrelation is exactly (90 - 29) / 120
    signs = np.where((np.arange(120) // 4) % 2 == 0, 1.0, -1.0)
    values = 1.0 / 128.0 + signs / 64.0
    return BacktestResult(
        stats={}, equity_curve=100.0 * np.cumprod(1.0 + values), trades=[],
        initial=100.0, periods_per_year=365.0,
    )


def test_the_effective_sample_counts_n_times_one_minus_r_over_one_plus_r():
    # pinned until now by inequalities alone, below the bar count and above two,
    # which n(1 - r), n(1 - r^2) and n / (1 + 2r) all satisfy on a persistent
    # series. This one has a first autocorrelation of exactly 61/120 by
    # construction, so the correction has exactly one answer, 120 * 59 / 181 or
    # 39.116, where those three wrong readings give 59, 88.99 and 59.50 (ADR 0061)
    result = blocked()
    assert metrics.autocorrelation(result, lag=1) == pytest.approx(61.0 / 120.0)
    assert metrics.effective_sample(result) == pytest.approx(7080.0 / 181.0)


def test_the_sharpe_interval_is_the_sharpe_plus_and_minus_a_hand_computed_half():
    # the half width was never checked against a number, so a one-sided quantile,
    # a missing square root or a variance read off the wrong estimator all passed
    # "low < sharpe < high". Here the skew is 0 and the kurtosis is 1, so the
    # estimator variance is exactly 1 and the half width is the normal quantile
    # over the root of size - 1 with nothing else in it (ADR 0061)
    result = blocked()
    assert metrics.skew(result) == pytest.approx(0.0, abs=1e-9)
    assert metrics.kurtosis(result) == pytest.approx(1.0)
    spread = 1.0 / 64.0 * math.sqrt(120.0 / 119.0)     # the sample deviation
    observed = 1.0 / 128.0 / spread                    # the per period sharpe
    root = math.sqrt(365.0)
    half = 1.959963984540054 * math.sqrt(1.0 / 119.0)  # two-sided 95%, 120 bars
    band = metrics.sharpe_interval(result, independent=True)
    assert band["bars"] == 120 and band["effective_bars"] == 120.0
    assert band["sharpe"] == pytest.approx(observed * root)
    assert band["low"] == pytest.approx((observed - half) * root)
    assert band["high"] == pytest.approx((observed + half) * root)
    # which is the sentence a reader quotes: sharpe 9.51, 95% interval 6.08 to 12.95
    assert band["sharpe"] == pytest.approx(9.5126013)
    assert band["low"] == pytest.approx(6.0800183)
    assert band["high"] == pytest.approx(12.9451843)
    # the confidence is two-sided, so 0.90 uses 1.6449 rather than 1.2816
    ninety = metrics.sharpe_interval(result, confidence=0.90, independent=True)
    assert ninety["high"] - ninety["low"] == pytest.approx(
        2.0 * 1.6448536269514722 * math.sqrt(1.0 / 119.0) * root
    )
    # and counting bets rather than bars widens it by exactly this much
    size = 7080.0 / 181.0
    wide = metrics.sharpe_interval(result)
    assert wide["effective_bars"] == pytest.approx(size)
    stretched = 1.959963984540054 * math.sqrt(1.0 / (size - 1.0))
    assert wide["low"] == pytest.approx((observed - stretched) * root)
    assert wide["high"] == pytest.approx((observed + stretched) * root)


def skewed():
    # 120 returns taking two values, 1/128 + 1/64 ninety times and 1/128 - 3/64
    # thirty times. Centred those are +u and -3u, so the population moments come
    # out at 3u^2, -6u^3 and 21u^4 and the shape is exact: skew -2/sqrt(3),
    # kurtosis 7/3. Unlike blocked() the estimator variance is then well away from
    # 1, which is what makes the non-normality correction visible
    step = np.arange(120)
    values = np.where(
        step % 4 == 3, 1.0 / 128.0 - 3.0 / 64.0, 1.0 / 128.0 + 1.0 / 64.0
    )
    return BacktestResult(
        stats={}, equity_curve=100.0 * np.cumprod(1.0 + values), trades=[],
        initial=100.0, periods_per_year=365.0,
    )


def test_the_sharpe_interval_carries_the_non_normality_correction():
    # blocked() is symmetric on purpose, which is what makes every figure above
    # hand computable and is also a hole: a two-valued symmetric sample has skew
    # exactly 0 and kurtosis exactly 1, so the estimator variance collapses to 1.0
    # and the whole Bailey and Lopez de Prado correction can be deleted, or have
    # its skew term flipped in sign, without moving a number. This sample is
    # skewed, so the correction is a real factor and each wrong reading of it
    # gives a different half width (ADR 0061)
    result = skewed()
    assert metrics.skew(result) == pytest.approx(-2.0 / math.sqrt(3.0))
    assert metrics.kurtosis(result) == pytest.approx(7.0 / 3.0)
    spread = math.sqrt(3.0) / 64.0 * math.sqrt(120.0 / 119.0)
    observed = 1.0 / 128.0 / spread
    # 1 - skew * observed + (kurtosis - 1) / 4 * observed^2, with the skew negative
    variance = 1.0 + 2.0 / math.sqrt(3.0) * observed + observed ** 2 / 3.0
    assert variance == pytest.approx(1.3594866)
    root = math.sqrt(365.0)
    quantile = 1.959963984540054
    half = quantile * math.sqrt(variance / 119.0)
    band = metrics.sharpe_interval(result, independent=True)
    assert band["sharpe"] == pytest.approx(observed * root)
    assert band["low"] == pytest.approx((observed - half) * root)
    assert band["high"] == pytest.approx((observed + half) * root)
    # and the two readings that survive blocked() are numbers this one is not:
    # no correction at all, and the skew term entered with the wrong sign
    uncorrected = quantile * math.sqrt(1.0 / 119.0)
    flipped = quantile * math.sqrt(
        (1.0 - 2.0 / math.sqrt(3.0) * observed + observed ** 2 / 3.0) / 119.0
    )
    assert abs(half - uncorrected) > 1e-3
    assert abs(half - flipped) > 1e-3


def test_the_track_record_length_converts_bets_back_into_bars_exactly_once():
    # the conversion was pinned as "at least the independent answer, less 1e-9",
    # which a SQUARED factor also satisfies: on this series it asks for 112.12
    # bars where the truth is 36.55, and both clear that floor. The answer comes
    # out in bets and the caller asked in bars, so it is multiplied by the bars
    # spent per bet once and only once (ADR 0061)
    result = blocked()
    spread = 1.0 / 64.0 * math.sqrt(120.0 / 119.0)
    observed = 1.0 / 128.0 / spread
    plain = metrics.min_track_record_length(result, independent=True)
    # the estimator variance is 1 here, so the leading term is the whole of it
    assert plain["bars"] == pytest.approx(1.0 + (1.6448536269514722 / observed) ** 2)
    assert plain["bars"] == pytest.approx(11.9131165)
    assert plain["effective_bars"] == 120.0
    adjusted = metrics.min_track_record_length(result)
    assert adjusted["effective_bars"] == pytest.approx(7080.0 / 181.0)
    # 120 bars are worth 7080/181 bets, so one bet is 181/59 bars and no more
    assert adjusted["bars"] == pytest.approx(plain["bars"] * 181.0 / 59.0, rel=1e-9)
    assert adjusted["bars"] == pytest.approx(36.5470183)
    assert adjusted["years"] == pytest.approx(adjusted["bars"] / 365.0)
    assert adjusted["have_bars"] == 120 and adjusted["enough"]


def test_the_risk_measures_read_the_tail_they_name():
    values = np.array([-0.05, -0.04, -0.03, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.05])
    curve = 100.0 * np.cumprod(1.0 + values)
    result = BacktestResult(stats={}, equity_curve=curve, trades=[], initial=100.0,
                            periods_per_year=365.0)
    # a loss is reported positive, so dropping the sign would report a gain
    assert metrics.value_at_risk(result, alpha=0.9) > 0.0
    # and the conditional figure averages the tail, not the whole series
    assert (metrics.conditional_value_at_risk(result, alpha=0.9)
            > metrics.value_at_risk(result, alpha=0.9))
    assert metrics.conditional_value_at_risk(result, alpha=0.9) == pytest.approx(
        -np.mean(values[values <= np.quantile(values, 0.1)]) * 100.0
    )


def test_a_tail_probability_passed_as_a_confidence_is_refused():
    result = run()
    for call in (metrics.value_at_risk, metrics.conditional_value_at_risk,
                 metrics.tail_ratio):
        with pytest.raises(ValueError) as excinfo:
            call(result, alpha=0.05)
        assert "0.95, not 0.05" in str(excinfo.value)


def test_the_drawdown_cannot_report_worse_than_a_total_loss():
    # bad debt takes equity below zero, and an uncapped fall gave a deeper bust a
    # larger number than a wipeout (ADR 0029)
    dead = BacktestResult(stats={}, equity_curve=np.array([100.0, 50.0, -80.0]),
                          trades=[], initial=100.0, periods_per_year=365.0)
    falls = metrics.drawdown(dead)
    assert falls.min() == pytest.approx(-100.0)
    assert metrics.ulcer_index(dead) <= 100.0


def test_a_drawdown_episode_counts_the_bars_it_actually_covers():
    curve = np.array([100.0, 90.0, 80.0, 95.0, 105.0, 110.0])
    result = BacktestResult(stats={}, equity_curve=curve, trades=[], initial=100.0,
                            periods_per_year=365.0)
    episode = metrics.drawdown_table(result)[0]
    falls = metrics.drawdown(result)
    assert episode["depth_pct"] == pytest.approx(falls.min())
    assert falls[episode["trough_bar"]] == pytest.approx(falls.min())
    assert episode["bars_under"] == episode["recovered_bar"] - episode["start_bar"]
    assert all(falls[b] < 0.0 for b in range(episode["start_bar"],
                                             episode["recovered_bar"]))


def test_the_side_split_counts_a_win_after_its_fee_like_everything_else():
    # on gross PnL a scalper whose edge is smaller than its costs reported a
    # perfect win rate beside a negative return, per side (ADR 0029)
    data = series()
    dear = Backtester(data, fee_taker=0.02, periods_per_year=365.0).run(Alternate())
    sides = metrics.long_short_split(dear)
    for block, side in ((sides["long"], "buy"), (sides["short"], "sell")):
        rows = [t for t in dear.trades if t["side"] == side]
        net_wins = [t for t in rows if t["net_pnl"] > 0.0]
        gross_wins = [t for t in rows if t["pnl"] > 0.0]
        if rows:
            assert block["win_rate"] == pytest.approx(len(net_wins) / len(rows))
        if len(gross_wins) != len(net_wins):
            assert block["win_rate"] < len(gross_wins) / len(rows)


# ------------------------------------------------------------ the trade shape


def test_the_trade_distribution_says_what_a_win_rate_cannot():
    result = run()
    shape = metrics.trade_stats(result)
    nets = [t["net_pnl"] for t in result.trades]
    assert shape["trades"] == len(nets)
    assert shape["wins"] + shape["losses"] <= shape["trades"]
    assert shape["expectancy"] == pytest.approx(float(np.mean(nets)))
    assert shape["largest_win"] == pytest.approx(max(nets))
    assert shape["largest_loss"] == pytest.approx(min(nets))
    if shape["losses"]:
        assert shape["payoff"] == pytest.approx(shape["avg_win"] / shape["avg_loss"])


def test_a_high_win_rate_at_a_terrible_payoff_is_visible():
    # the pair is the answer and either one alone describes two opposite rules
    made = [BacktestResult(stats={}, equity_curve=np.array([100.0]), trades=[],
                           initial=100.0, periods_per_year=365.0)]
    made[0].trades = [{"net_pnl": 1.0, "pnl": 1.0, "fees": 0.0, "side": "buy",
                       "bars_held": 1, "size": 1.0, "entry_price": 100.0,
                       "exit_price": 101.0, "entry_tick": 0, "exit_tick": 1}] * 9
    made[0].trades = made[0].trades + [dict(made[0].trades[0], net_pnl=-20.0)]
    shape = metrics.trade_stats(made[0])
    assert shape["wins"] / shape["trades"] == pytest.approx(0.9)
    assert shape["payoff"] == pytest.approx(1.0 / 20.0)
    assert shape["expectancy"] < 0.0                       # nine wins, still losing
    assert shape["max_consecutive_losses"] == 1


def test_the_costs_are_reported_as_a_share_of_the_edge():
    data = series()
    dear = Backtester(data, fee_taker=0.01, periods_per_year=365.0).run(Alternate())
    money = metrics.decompose(dear)
    assert money["fee_share"] == pytest.approx(
        money["fees"] / abs(money["gross_pnl"])
    )
    assert money["turnover"] > 0.0
    cheap = Backtester(data, fee_taker=0.0, periods_per_year=365.0).run(Alternate())
    assert metrics.decompose(cheap)["fee_share"] == 0.0


def test_turnover_is_both_sides_of_every_trade_over_the_opening_balance():
    # "> 0.0" is equally true of the entry notional alone, of the exit notional
    # alone, and of either of them over the closing balance, and a reader quotes
    # whichever comes back as the same sentence. Two trades, hand priced: 2 at 50
    # out at 55, and 1 at 20 out at 15, so both sides move 245 against an opening
    # 100. The entry side alone reads 1.20, the exit side 1.25, and both sides
    # over the closing balance 2.35 (ADR 0062)
    trades = [
        {"net_pnl": 9.5, "pnl": 10.0, "fees": 0.5, "side": "buy", "bars_held": 3,
         "size": 2.0, "entry_price": 50.0, "exit_price": 55.0, "entry_tick": 0,
         "exit_tick": 3},
        {"net_pnl": -5.2, "pnl": -5.0, "fees": 0.2, "side": "buy", "bars_held": 2,
         "size": 1.0, "entry_price": 20.0, "exit_price": 15.0, "entry_tick": 4,
         "exit_tick": 6},
    ]
    hand = BacktestResult(stats={}, equity_curve=np.array([104.3]), trades=trades,
                          initial=100.0, periods_per_year=365.0)
    money = metrics.decompose(hand)
    assert money["turnover"] == pytest.approx(2.45)
    # and the fee share is the same reading of the same two trades: 0.70 of a
    # gross 5.00, which is the sentence about having a cost problem
    assert money["fee_share"] == pytest.approx(0.14)
    assert money["net"] == pytest.approx(4.3)
    assert money["unrealized"] == pytest.approx(0.0, abs=1e-9)


def test_omega_and_the_tail_ratio_read_the_whole_distribution():
    result = run()
    assert metrics.omega(result) >= 0.0
    assert metrics.tail_ratio(result) >= 0.0
    # omega above one and a positive total return say the same thing
    assert (metrics.omega(result) > 1.0) == (result.stats["total_return_pct"] > 0.0)


def test_the_report_does_not_stutter_its_own_prefix():
    data = series()
    result = Backtester(data, periods_per_year=365.0).run(Alternate())
    row = metrics.report(result, data)
    assert "hold_return_pct" in row and "hold_sharpe" in row
    assert not any(key.startswith("hold_hold") for key in row)
    for key in ("trade_payoff", "trade_expectancy", "effective_bars", "omega",
                "tail_ratio", "ulcer_index", "return_per_exposure", "sharpe_low"):
        assert key in row


def test_the_return_per_exposure_is_the_return_over_the_time_at_risk():
    # twenty percent made while in the market a quarter of the time and twenty
    # percent made while in it four fifths of the time were the same row before
    # this, and only presence of the key was ever checked. The ratio the other way
    # up reads 125 and 400, which ranks the two backwards: it would call the run
    # that sat in the market four times as long the better use of the risk
    # (ADR 0062)
    curve = np.array([105.0, 110.0, 115.0, 120.0])
    lazy = BacktestResult(
        stats={"total_return_pct": 20.0, "exposure_pct": 25.0}, equity_curve=curve,
        trades=[], initial=100.0, periods_per_year=365.0,
    )
    busy = BacktestResult(
        stats={"total_return_pct": 20.0, "exposure_pct": 80.0}, equity_curve=curve,
        trades=[], initial=100.0, periods_per_year=365.0,
    )
    assert metrics.report(lazy)["return_per_exposure"] == pytest.approx(80.0)
    assert metrics.report(busy)["return_per_exposure"] == pytest.approx(25.0)


# ------------------------------------------------- what the windows add up to


def test_the_decay_is_the_mean_gap_and_a_flat_window_never_counts_as_a_win():
    # the two numbers a walk-forward is read on are computed from the per-window
    # scores metrics.segment produces, and both were pinned as "a float" and
    # "between 0 and 1", which the gap the other way round, the median of the gaps
    # and a >= 0.0 comparison all satisfy. A window scoring exactly 0.0 is the one
    # ADR 0060 exists for, a stretch nobody could trade, and it must not be
    # counted as a window that cleared zero; a window with no score at all, which
    # is what a callable objective leaves behind, is skipped by both rather than
    # counted as a failure
    windows = [
        {"fitted": 2.0, "traded": 1.5},
        {"fitted": 1.0, "traded": -0.5},
        {"fitted": 3.0, "traded": 0.0},
        {"fitted": 0.5, "traded": 1.0},
        {"fitted": 9.0, "traded": None},
    ]
    forward = WalkForward(run(), windows, (0, 120))
    # the four gaps are 0.5, 1.5, 3.0 and -0.5: a mean of 1.125, a median of 1.0,
    # and -1.125 with the subtraction the other way round
    assert forward.decay == pytest.approx(1.125)
    # two of the four scored windows are above zero. Counting the flat one as a
    # win gives 0.75, dropping it from the denominator gives 0.667
    assert forward.consistency == pytest.approx(0.5)


class Tunable(Strategy):
    # the shape every documented sweep example uses: the tunables arrive as
    # constructor arguments, so the class cannot be built with none of them
    def __init__(self, every):
        self.every = int(every)

    def next(self, state, engine):
        if state["tick_index"] % self.every == 0:
            engine.market_buy(1.0)
        elif state["position"] > 0.0:
            engine.close()


def test_a_sweep_names_the_strategy_it_cannot_build():
    # a sweep rebuilds the strategy per run, so it needs one it can build with no
    # arguments. Python's own message names neither the sweep nor the fix, and it
    # is the failure every documented example walked into (ADR 0092)
    with pytest.raises(TypeError) as excinfo:
        metrics.breakeven_bps(Tunable, series())
    said = str(excinfo.value)
    assert "Tunable" in said
    assert "configured instance" in said
    # and the instance form the message points at is accepted
    assert metrics.breakeven_bps(Tunable(4), series(), ceiling=20.0) is not None


def climbed(curve, ppy):
    # a short steep stretch, which is the only shape that puts the annualization
    # out of the float range: the exponent is ppy over the number of advances, so
    # few advances and a real gain is what reaches it
    return BacktestResult(stats={}, equity_curve=np.array(curve), trades=[],
                          initial=100.0, periods_per_year=ppy)


def test_a_growth_rate_saturates_in_python_where_the_engine_saturates_it():
    # a float power RAISES where the f64 it mirrors returns an infinity, so the
    # ceiling on the next line was unreachable in the one case it exists for.
    # 1.2 ** 8760 leaves the float range; the threshold at that exponent is a
    # ratio of about 1.0843 (ADR 0093)
    hourly = metrics.segment(climbed([120.0], 8760.0))
    assert math.isfinite(hourly["cagr_pct"])
    assert hourly["cagr_pct"] == pytest.approx(metrics._CAGR_CEILING * 100.0)

    minute = metrics.segment(climbed([120.0, 130.0], 525_600.0))
    assert math.isfinite(minute["cagr_pct"])
    assert minute["cagr_pct"] == pytest.approx(metrics._CAGR_CEILING * 100.0)


def test_a_growth_rate_that_fits_is_left_alone():
    # the ceiling must not swallow an ordinary number on its way past
    gentle = metrics.segment(climbed(list(np.linspace(100.5, 105.0, 100)), 8760.0))
    assert gentle["cagr_pct"] < metrics._CAGR_CEILING * 100.0
    assert gentle["cagr_pct"] == pytest.approx(7081.0, rel=1e-3)


def test_every_calendar_period_of_hourly_candles_reports_a_growth_rate():
    # ADR 0081 keeps a one-bar calendar period rather than dropping it, and one
    # bar puts the whole annualization in the exponent. A year of hourly candles
    # ending a bar into the next month is the shape that crashed period_returns
    # the account is small enough that the one position dominates it, because the
    # exponent only leaves the float range on a real gain: at 8760 the threshold
    # is a ratio of about 1.0843, and a unit against ten thousand cannot reach it
    pd = pytest.importorskip("pandas")
    close = np.full(745, 100.0)
    close[-1] = 130.0
    data = np.column_stack(
        [close, close + 1.0, close - 1.0, close, np.full(745, 1000.0)]
    )
    frame = pd.DataFrame(data, columns=["open", "high", "low", "close", "volume"],
                         index=pd.date_range("2024-01-01", periods=745, freq="1h"))
    result = Backtester(frame, quote=200.0, fee_taker=0.0, fee_maker=0.0).run(Hold())
    rows = metrics.period_returns(result, frame)
    assert len(rows) == 2
    assert rows[-1]["bars"] == 1
    assert all(math.isfinite(row["cagr_pct"]) for row in rows)


def test_an_excursion_does_not_read_the_bar_the_trade_left_on_whole():
    # exit_tick is the bar the closing fill resolved on, so everything it printed
    # after the exit was folded in. The engine already disagreed: bars_held is
    # exit_tick minus entry_tick, one fewer bar than this used to span (ADR 0095)
    data = np.array([
        [100.0, 100.0, 100.0, 100.0, 1e6],
        [100.0, 100.0, 100.0, 100.0, 1e6],
        [100.0, 101.0, 99.0, 100.0, 1e6],
        [100.0, 180.0, 20.0, 100.0, 1e6],
        [100.0, 100.0, 100.0, 100.0, 1e6],
    ])

    class InThenOut(Strategy):
        def next(self, state, engine):
            i = state["tick_index"]
            if i == 0:
                engine.market_buy(1.0)
            elif i == 2:
                engine.close()

    result = Backtester(data, periods_per_year=365.0).run(InThenOut())
    trade = result.trades[0]
    row = metrics.excursions(result, data)[0]
    # it exits at the open of bar 3, so the 180 and the 20 that bar went on to
    # print belong to nobody
    assert trade["exit_tick"] == 3
    assert row["best_pct"] == pytest.approx(1.0)
    assert row["worst_pct"] == pytest.approx(-1.0)


def test_an_excursion_does_not_credit_a_limit_entry_with_the_bar_before_it_filled():
    # a maker bid rests BELOW the market, so the bar has to come down to it and
    # everything it printed on the way is above the entry. Reading the entry bar
    # whole put that into what a target could have caught, which is the flattering
    # direction and it happens by construction rather than by accident (ADR 0095)
    data = np.array([
        [100.0, 100.0, 100.0, 100.0, 1e6],
        [130.0, 130.0, 100.0, 101.0, 1e6],
        [101.0, 101.0, 101.0, 101.0, 1e6],
        [101.0, 101.0, 101.0, 101.0, 1e6],
    ])

    class BidThenLeave(Strategy):
        def next(self, state, engine):
            i = state["tick_index"]
            if i == 0:
                engine.limit_buy(1.0, 100.0)
            elif i == 1:
                engine.close()

    result = Backtester(data, periods_per_year=365.0,
                        fee_taker=0.0, fee_maker=0.0).run(BidThenLeave())
    trade = result.trades[0]
    row = metrics.excursions(result, data)[0]
    assert trade["entry_tick"] == 1
    assert trade["entry_price"] == pytest.approx(100.0)
    # bar 1 opened at 130, its own high, and only then fell to the bid. Nothing
    # from the fill onward ever printed above 101, so the most a target could
    # have caught is one percent, not thirty
    assert row["best_pct"] == pytest.approx(1.0)


def test_an_excursion_still_holds_every_bar_a_trade_was_open_across():
    # the bars strictly between the two ends are lived through whole, and the
    # bound must not quietly shrink those away too
    data = np.array([
        [100.0, 100.0, 100.0, 100.0, 1e6],
        [100.0, 100.0, 100.0, 100.0, 1e6],
        [100.0, 130.0, 70.0, 100.0, 1e6],
        [100.0, 100.0, 100.0, 100.0, 1e6],
        [100.0, 100.0, 100.0, 100.0, 1e6],
    ])

    class Across(Strategy):
        def next(self, state, engine):
            i = state["tick_index"]
            if i == 0:
                engine.market_buy(1.0)
            elif i == 3:
                engine.close()

    result = Backtester(data, periods_per_year=365.0).run(Across())
    row = metrics.excursions(result, data)[0]
    assert row["best_pct"] == pytest.approx(30.0)
    assert row["worst_pct"] == pytest.approx(-30.0)
