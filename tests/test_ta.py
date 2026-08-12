"""Tests for emsl.ta: every function returns one value per bar with the warm-up
as a gap, and each number is checked against one worked by hand rather than
against another library, which would only prove the two agree.
"""

import math

import numpy as np
import pytest

import emsl
from emsl import ta

# 1, 2, 3, ... 20: every rolling answer over it can be worked out on paper
RAMP = np.arange(1.0, 21.0)

EVERY = {
    "sma": lambda: ta.sma(RAMP, 5),
    "ema": lambda: ta.ema(RAMP, 5),
    "wma": lambda: ta.wma(RAMP, 5),
    "rsi": lambda: ta.rsi(RAMP, 5),
    "roc": lambda: ta.roc(RAMP, 5),
    "stdev": lambda: ta.stdev(RAMP, 5),
    "zscore": lambda: ta.zscore(RAMP, 5),
    "atr": lambda: ta.atr(RAMP + 1.0, RAMP - 1.0, RAMP, 5),
    "true_range": lambda: ta.true_range(RAMP + 1.0, RAMP - 1.0, RAMP),
    "vwap": lambda: ta.vwap(RAMP + 1.0, RAMP - 1.0, RAMP, np.full(20, 10.0), 5),
}


def test_every_function_returns_one_value_per_bar():
    # the alignment contract: never a shorter array, so the same object goes into
    # a rule and into a chart with no padding decision in between
    for name, call in EVERY.items():
        out = call()
        assert out.shape == (20,), f"{name} returned {out.shape}"
        assert out.dtype == np.float64, f"{name} returned {out.dtype}"


def test_the_multi_line_functions_return_named_lines_not_tuples():
    bands = ta.bbands(RAMP, 5)
    assert bands.upper.shape == bands.middle.shape == bands.lower.shape == (20,)
    lines = ta.macd(RAMP, 3, 6, 3)
    assert lines.line.shape == lines.signal.shape == lines.histogram.shape == (20,)
    swing = ta.stoch(RAMP + 1.0, RAMP - 1.0, RAMP, 5, 3)
    assert swing.k.shape == swing.d.shape == (20,)
    channel = ta.donchian(RAMP + 1.0, RAMP - 1.0, 5)
    assert channel.upper.shape == channel.lower.shape == (20,)


def test_the_warm_up_is_a_gap_and_the_rest_is_not():
    for name, call in EVERY.items():
        out = call()
        if name == "true_range":
            continue  # the first bar is its own range, so nothing is missing
        assert np.isnan(out[0]), f"{name} produced a value on bar 0"
        assert np.isfinite(out[-1]), f"{name} never warmed up"


# ------------------------------------------------------------ worked by hand


def test_sma_is_the_mean_of_the_window():
    out = ta.sma(RAMP, 5)
    assert np.isnan(out[3])
    assert out[4] == pytest.approx(3.0)      # (1+2+3+4+5)/5
    assert out[19] == pytest.approx(18.0)    # (16+17+18+19+20)/5


def test_ema_seeds_from_the_simple_average_not_the_first_value():
    # seeded at bar 4 with (1+2+3+4+5)/5 = 3, then alpha = 2/6 = 1/3
    out = ta.ema(RAMP, 5)
    assert np.isnan(out[3])
    assert out[4] == pytest.approx(3.0)
    assert out[5] == pytest.approx(3.0 + (6.0 - 3.0) / 3.0)          # 4.0
    assert out[6] == pytest.approx(4.0 + (7.0 - 4.0) / 3.0)          # 5.0


def test_wma_leans_on_the_newest_bar():
    # weights 1..5 over values 1..5: (1*1 + 2*2 + 3*3 + 4*4 + 5*5) / 15
    out = ta.wma(RAMP, 5)
    assert out[4] == pytest.approx(55.0 / 15.0)
    # and it sits above the simple average on a rising series
    assert out[10] > ta.sma(RAMP, 5)[10]


def test_rsi_is_a_hundred_when_nothing_ever_falls():
    # a series that only rises has no losses, so the strength is infinite
    out = ta.rsi(RAMP, 5)
    assert out[-1] == pytest.approx(100.0)
    # and zero when nothing ever rises
    assert ta.rsi(RAMP[::-1].copy(), 5)[-1] == pytest.approx(0.0)


def test_rsi_uses_wilders_smoothing():
    # up 1 every bar except one fall of 5 at bar 10, checked against Wilder by hand
    values = RAMP.copy()
    values[10:] -= 6.0
    out = ta.rsi(values, 5)
    # seeded at bar 5 from the first five changes, all +1, so gain 1 loss 0
    assert out[5] == pytest.approx(100.0)
    # the fall of 5 enters at bar 10: gain 1*(4/5) = 0.8, loss 5/5 = 1.0
    gain, loss = 1.0 * 0.8, 5.0 / 5.0
    assert out[10] == pytest.approx(100.0 - 100.0 / (1.0 + gain / loss))


def test_true_range_takes_the_widest_of_the_three_spans():
    high = np.array([10.0, 12.0, 20.0])
    low = np.array([8.0, 11.0, 19.0])
    close = np.array([9.0, 11.5, 19.5])
    out = ta.true_range(high, low, close)
    assert out[0] == pytest.approx(2.0)   # 10 - 8, no previous close
    assert out[1] == pytest.approx(3.0)   # |12 - 9| beats 12 - 11
    assert out[2] == pytest.approx(8.5)   # |20 - 11.5| beats 20 - 19


def test_stdev_is_population_so_bbands_match_every_chart():
    # 1..5 has a population deviation of sqrt(2), a sample one of sqrt(2.5)
    assert ta.stdev(RAMP, 5)[4] == pytest.approx(math.sqrt(2.0))
    bands = ta.bbands(RAMP, 5, deviations=2.0)
    assert bands.middle[4] == pytest.approx(3.0)
    assert bands.upper[4] == pytest.approx(3.0 + 2.0 * math.sqrt(2.0))
    assert bands.lower[4] == pytest.approx(3.0 - 2.0 * math.sqrt(2.0))


def test_roc_is_a_percentage_of_the_bar_it_looks_back_to():
    out = ta.roc(RAMP, 5)
    assert np.isnan(out[4])
    assert out[5] == pytest.approx((6.0 / 1.0 - 1.0) * 100.0)


def test_donchian_brackets_the_window_and_sits_between():
    channel = ta.donchian(RAMP + 1.0, RAMP - 1.0, 5)
    assert channel.upper[4] == pytest.approx(6.0)    # highest high of 2..6
    assert channel.lower[4] == pytest.approx(0.0)    # lowest low of 0..4
    assert channel.middle[4] == pytest.approx(3.0)


def test_stoch_is_where_the_close_sits_in_the_range():
    swing = ta.stoch(RAMP + 1.0, RAMP - 1.0, RAMP, 5, 3)
    # close 5, range 0 to 6, so (5 - 0) / 6
    assert swing.k[4] == pytest.approx(5.0 / 6.0 * 100.0)
    assert 0.0 <= swing.k[-1] <= 100.0


def test_vwap_weights_by_volume():
    close = np.array([10.0, 20.0, 30.0, 40.0])
    volume = np.array([1.0, 1.0, 1.0, 97.0])
    out = ta.vwap(close, close, close, volume, 4)
    assert out[3] == pytest.approx((10.0 + 20.0 + 30.0 + 40.0 * 97.0) / 100.0)
    # a window that traded nothing has no weighted price
    assert np.isnan(ta.vwap(close, close, close, np.zeros(4), 4)[3])


def test_zscore_is_zero_on_a_flat_series_and_a_gap_with_no_deviation():
    flat = np.full(20, 7.0)
    assert np.isnan(ta.zscore(flat, 5)[-1])   # no deviation, so no position in it
    assert ta.zscore(RAMP, 5)[-1] == pytest.approx(
        (20.0 - 18.0) / math.sqrt(2.0)
    )


def test_macd_is_the_gap_between_two_averages_and_its_own_average():
    lines = ta.macd(RAMP, 3, 6, 3)
    assert np.allclose(lines.line, ta.ema(RAMP, 3) - ta.ema(RAMP, 6),
                       equal_nan=True)
    assert np.allclose(lines.histogram, lines.line - lines.signal, equal_nan=True)
    # the signal warms up after the line it is an average of
    assert np.isnan(lines.signal[np.flatnonzero(np.isfinite(lines.line))[0]])


# ------------------------------------------------------------ the awkward inputs


def test_a_gap_in_the_input_stays_a_gap_in_the_output():
    holed = RAMP.copy()
    holed[10] = np.nan
    out = ta.sma(holed, 5)
    assert np.isnan(out[10:15]).all()   # every window touching the hole
    assert np.isfinite(out[15])         # and the first window past it is fine


def test_a_length_longer_than_the_series_says_both_numbers():
    with pytest.raises(ValueError) as excinfo:
        ta.sma(RAMP, 50)
    assert "50" in str(excinfo.value) and "20" in str(excinfo.value)


def test_a_length_below_one_is_refused():
    with pytest.raises(ValueError):
        ta.sma(RAMP, 0)


def test_mismatched_series_lengths_are_refused_by_name():
    with pytest.raises(ValueError) as excinfo:
        ta.stoch(RAMP, RAMP[:5], RAMP, 3)
    assert "one length" in str(excinfo.value)


def test_macd_needs_the_fast_leg_to_be_faster():
    with pytest.raises(ValueError) as excinfo:
        ta.macd(RAMP, 26, 12)
    assert "fast below slow" in str(excinfo.value)


def test_an_empty_series_is_refused():
    with pytest.raises(ValueError):
        ta.sma(np.array([]), 1)


def test_a_pandas_series_works_without_pandas_being_a_dependency():
    pd = pytest.importorskip("pandas")
    assert np.allclose(ta.sma(pd.Series(RAMP), 5), ta.sma(RAMP, 5), equal_nan=True)


# ------------------------------------------------------------ it all fits together


def test_an_indicator_drives_a_strategy_and_draws_on_the_chart():
    pd = pytest.importorskip("pandas")

    class Cross(emsl.Strategy):
        def init(self, engine):
            self.fast = ta.ema(engine.closes, 5)
            self.slow = ta.ema(engine.closes, 15)
            self.warmup = 15

        def next(self, state, engine):
            i = state["tick_index"]
            if state["position"] == 0.0 and self.fast[i] > self.slow[i]:
                engine.market_buy(1.0)
            elif state["position"] > 0.0 and self.fast[i] < self.slow[i]:
                engine.close()

    rng = np.random.default_rng(1)
    close = 100.0 + np.cumsum(rng.normal(0.1, 1.0, 120))
    frame = pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0,
         "close": close, "volume": rng.uniform(500.0, 1500.0, 120)},
        index=pd.date_range("2025-01-01", periods=120, freq="1h", tz="UTC"),
    )
    strategy = Cross()
    result = emsl.backtest.Backtester(frame).run(strategy)
    # the same array the rule traded on goes straight onto the chart, no padding
    chart = emsl.chart(frame=frame, marks=emsl.plot.Line(values=strategy.fast,
                                                         name="EMA 5"), run=result)
    assert len(strategy.fast) == len(frame)
    assert chart.spec()["series"][0]["kind"] == "line"


def test_ta_is_exported_from_the_package():
    assert emsl.ta is ta
    assert "ta" in emsl.__all__
    assert len(ta.__all__) == 17  # fourteen functions and three result types
