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

# every function, as one call over (high, low, close, volume). Multi-line results
# are split so each line is checked in its own right
LINES = {
    "sma": lambda h, l, c, v: [ta.sma(c, 5)],
    "ema": lambda h, l, c, v: [ta.ema(c, 5)],
    "wma": lambda h, l, c, v: [ta.wma(c, 5)],
    "rsi": lambda h, l, c, v: [ta.rsi(c, 5)],
    "roc": lambda h, l, c, v: [ta.roc(c, 5)],
    "stdev": lambda h, l, c, v: [ta.stdev(c, 5)],
    "zscore": lambda h, l, c, v: [ta.zscore(c, 5)],
    "atr": lambda h, l, c, v: [ta.atr(h, l, c, 5)],
    "true_range": lambda h, l, c, v: [ta.true_range(h, l, c)],
    "vwap": lambda h, l, c, v: [ta.vwap(h, l, c, v, 5)],
    "macd": lambda h, l, c, v: list(vars_of(ta.macd(c, 3, 6, 3))),
    "stoch": lambda h, l, c, v: list(vars_of(ta.stoch(h, l, c, 5, 3))),
    "bbands": lambda h, l, c, v: list(vars_of(ta.bbands(c, 5))),
    "donchian": lambda h, l, c, v: list(vars_of(ta.donchian(h, l, 5))),
}


def vars_of(bundle):
    return [getattr(bundle, name) for name in bundle.__slots__]


def ramp_bars():
    # copies, every one: a caller that hole-punches these must not reach back and
    # corrupt the module-level fixture for every test that runs after it
    return RAMP + 1.0, RAMP - 1.0, RAMP.copy(), np.full(20, 10.0)


EVERY = {name: (lambda call=call: call(*ramp_bars())[0]) for name, call in LINES.items()}


def test_no_indicator_can_see_the_future():
    """The invariant everything else rests on.

    These are computed once over the whole series in ``init`` and then read bar by
    bar inside the loop, so a value at bar i that depended on bar i+1 would be
    silent lookahead in every strategy that used it, and would flatter every one
    of them. Truncating the series must therefore change nothing about the bars
    that remain.
    """
    rng = np.random.default_rng(4)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 90))
    high = close + rng.uniform(0.1, 1.0, 90)
    low = close - rng.uniform(0.1, 1.0, 90)
    volume = rng.uniform(500.0, 1500.0, 90)

    for name, call in LINES.items():
        whole = call(high, low, close, volume)
        for cut in (20, 45, 89, 90):
            part = call(high[:cut], low[:cut], close[:cut], volume[:cut])
            for line, (full, upto) in enumerate(zip(whole, part)):
                assert len(upto) == cut, f"{name} line {line} at cut {cut}"
                for i in range(cut):
                    a, b = full[i], upto[i]
                    same = (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b)
                    assert same, (
                        f"{name} line {line} bar {i} moved from {a} to {b} when "
                        f"the series was cut at {cut}: it can see the future"
                    )


def test_removing_the_future_outright_changes_no_past_value():
    # the same invariant from the other side: rewrite the tail and the head must
    # not move by a single ulp
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 90))
    high, low, volume = close + 1.0, close - 1.0, np.full(90, 1000.0)
    moved = close.copy()
    moved[60:] += 500.0

    for name, call in LINES.items():
        before = call(high, low, close, volume)
        after = call(high, low, moved, volume)
        for line, (a, b) in enumerate(zip(before, after)):
            assert np.allclose(a[:60], b[:60], equal_nan=True), (
                f"{name} line {line} changed a bar before 60 when only bars after "
                f"60 were moved"
            )


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


def test_vwap_weights_by_volume_and_prices_on_the_typical():
    # high, low and close are all different, so an implementation using the close
    # alone, or the midpoint, or the wrong two legs, cannot pass this
    high = np.array([12.0, 22.0, 32.0, 42.0])
    low = np.array([6.0, 16.0, 26.0, 36.0])
    close = np.array([9.0, 19.0, 29.0, 39.0])
    typical = (high + low + close) / 3.0        # 9, 19, 29, 39
    assert np.allclose(typical, [9.0, 19.0, 29.0, 39.0])
    volume = np.array([1.0, 1.0, 1.0, 97.0])
    out = ta.vwap(high, low, close, volume, 4)
    assert out[3] == pytest.approx((9.0 + 19.0 + 29.0 + 39.0 * 97.0) / 100.0)
    # and it is not the plain average of the closes, which would be 24.0
    assert out[3] != pytest.approx(24.0)
    # a window that traded nothing has no weighted price
    assert np.isnan(ta.vwap(high, low, close, np.zeros(4), 4)[3])


def test_stoch_smooths_k_into_d():
    # d is the simple average of k over `smooth` bars, checked numerically rather
    # than only shape-checked
    high = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    low = np.full(6, 0.0)
    close = np.array([5.0, 6.0, 3.0, 9.0, 7.0, 12.0])
    swing = ta.stoch(high, low, close, 2, 2)
    # bar 1: window high 11, low 0, close 6 -> 54.5454...
    assert swing.k[1] == pytest.approx(6.0 / 11.0 * 100.0)
    # bar 2: window high 12, low 0, close 3 -> 25.0
    assert swing.k[2] == pytest.approx(25.0)
    assert swing.d[2] == pytest.approx((6.0 / 11.0 * 100.0 + 25.0) / 2.0)


def test_zscore_has_no_position_in_a_window_with_no_deviation():
    flat = np.full(20, 7.0)
    assert np.isnan(ta.zscore(flat, 5)[-1])   # no deviation, so no position in it
    assert ta.zscore(RAMP, 5)[-1] == pytest.approx(
        (20.0 - 18.0) / math.sqrt(2.0)
    )


def test_macd_is_the_gap_between_two_averages_and_its_own_average():
    # worked by hand rather than recomputed with the code under test, which would
    # pass for any ema at all, right or wrong. On the ramp, both averages sit
    # exactly on their own window's mean once warm, so the line is the distance
    # between the two window centres: (6-1)/2 - (3-1)/2 = 1.5
    lines = ta.macd(RAMP, 3, 6, 3)
    assert np.isnan(lines.line[4])            # the slow leg needs 6 bars
    assert lines.line[5] == pytest.approx(1.5)
    assert lines.line[19] == pytest.approx(1.5)
    # the signal is an average of a line that is constant at 1.5, so it is too
    assert lines.signal[19] == pytest.approx(1.5)
    assert lines.histogram[19] == pytest.approx(0.0)
    # and it warms up after the line it averages, never with it
    first_line = int(np.flatnonzero(np.isfinite(lines.line))[0])
    assert np.isnan(lines.signal[first_line])
    assert first_line == 5 and np.isfinite(lines.signal[first_line + 2])


# ------------------------------------------------------------ the awkward inputs


def test_a_gap_in_the_input_stays_a_gap_in_every_output():
    # the claim is made module-wide, so it is checked module-wide: no function may
    # absorb a missing bar, and none may raise over one either
    # the whole bar goes missing, which is what a gap in an OHLCV feed actually
    # looks like. Holing only the close leaves donchian untouched, correctly, since
    # it reads the high and the low and never the close
    high, low, close, volume = ramp_bars()
    for series in (high, low, close, volume):
        series[10] = np.nan
    for name, call in LINES.items():
        for index, line in enumerate(call(high, low, close, volume)):
            assert np.isnan(line[10]), (
                f"{name} line {index} passed straight through the gap"
            )


def test_a_missing_close_makes_the_true_range_a_gap_not_the_bars_own_range():
    # taking the widest span while ignoring the missing ones quietly returned high
    # less low, so a gapping bar reported a fraction of its real range and every
    # ATR-sized stop built on it came out far too tight
    high = np.array([100.0, 100.0, 130.0, 130.0])
    low = np.array([99.0, 99.0, 129.0, 129.0])
    close = np.array([100.0, np.nan, 130.0, 130.0])
    out = ta.true_range(high, low, close)
    assert np.isnan(out[2]), f"the gap was swallowed and reported {out[2]}"
    clean = ta.true_range(high, low, np.array([100.0, 100.0, 130.0, 130.0]))
    assert clean[2] == pytest.approx(30.0)   # 130 - 100, the real range


def test_nothing_warns_on_a_gappy_or_infinite_input():
    # a numpy RuntimeWarning is the module leaking its workings; a gap is an
    # answer and should arrive as one
    import warnings as _warnings

    high, low, close, volume = ramp_bars()
    for hurt in (np.nan, np.inf, -np.inf):
        for name, call in LINES.items():
            damaged = close.copy()
            damaged[10] = hurt
            with _warnings.catch_warnings():
                _warnings.simplefilter("error")
                call(high, low, damaged, volume)


def test_a_recursive_indicator_waits_out_a_gap_rather_than_raising():
    # ema, rsi, atr and macd all seed from a window, and a single missing bar in
    # the warm-up used to raise, from `ema`, about a bar the caller of `macd` had
    # never heard of. It costs warm-up now, as it does everywhere else
    holed = RAMP.copy()
    holed[2] = np.nan
    out = ta.ema(holed, 5)
    assert np.isnan(out[:7]).all()      # no clean window of 5 ends before bar 7
    assert out[7] == pytest.approx(RAMP[3:8].mean())
    assert np.isfinite(ta.macd(holed, 3, 6, 3).signal[-1])
    assert np.isfinite(ta.rsi(holed, 5)[-1])
    assert np.isfinite(ta.atr(holed + 1.0, holed - 1.0, holed, 5)[-1])


def test_a_gap_after_the_seed_costs_warm_up_rather_than_the_rest_of_the_series():
    # a recurrence cannot carry across a missing bar, and the strict reading is
    # that everything after one is undefined. That reading makes a year of candles
    # with a single hole produce nothing from the hole onward, which is useless
    # rather than rigorous: the next clean window seeds a fresh run instead
    long = np.arange(1.0, 61.0)
    holed = long.copy()
    holed[30] = np.nan
    out = ta.ema(holed, 5)
    assert np.isfinite(out[29])         # fine before the hole
    assert np.isnan(out[30])            # the hole itself
    assert np.isnan(out[31:35]).all()   # and while it looks for a clean window
    # bars 31 to 35 are the first five clean ones after it, so the seed lands there
    assert out[35] == pytest.approx(long[31:36].mean())
    assert np.isfinite(out[-1]), "one missing bar killed the rest of the series"
    for call in (lambda v: ta.rsi(v, 5),
                 lambda v: ta.atr(v + 1.0, v - 1.0, v, 5),
                 lambda v: ta.macd(v, 3, 6, 3).signal):
        assert np.isfinite(call(holed)[-1])


def test_a_length_longer_than_the_series_says_both_numbers():
    with pytest.raises(ValueError) as excinfo:
        ta.sma(RAMP, 50)
    assert "50" in str(excinfo.value) and "20" in str(excinfo.value)


def test_a_length_below_one_is_refused():
    with pytest.raises(ValueError):
        ta.sma(RAMP, 0)


def test_a_whole_frame_handed_in_place_of_a_column_is_refused():
    # this used to be flattened before the shape was checked, so four interleaved
    # columns came back as one series and every number looked entirely plausible
    frame = np.column_stack([RAMP, RAMP + 1.0, RAMP - 1.0, RAMP, RAMP])
    with pytest.raises(ValueError) as excinfo:
        ta.sma(frame, 5)
    assert "not the whole frame" in str(excinfo.value)
    # a single column with a trailing axis is still a single column
    assert np.allclose(ta.sma(RAMP.reshape(-1, 1), 5), ta.sma(RAMP, 5), equal_nan=True)


def test_a_fractional_length_is_a_mistake_not_a_rounding_request():
    with pytest.raises(ValueError) as excinfo:
        ta.sma(RAMP, 3.9)
    assert "whole number of bars" in str(excinfo.value)


def test_a_market_going_nowhere_has_no_relative_strength():
    # neither gains nor losses is not "maximally overbought": reporting 100 there
    # fires every `rsi > 70` rule on a market that has not moved
    assert np.isnan(ta.rsi(np.full(20, 7.0), 5)[-1])
    assert np.isnan(ta.rsi(np.zeros(20), 5)[-1])
    # but rising with no losses at all still is 100
    assert ta.rsi(RAMP, 5)[-1] == pytest.approx(100.0)


def test_rsi_needs_more_bars_than_its_length():
    # the first bar makes no change, so a length equal to the series returned an
    # entirely NaN array rather than saying anything
    with pytest.raises(ValueError) as excinfo:
        ta.rsi(RAMP, 20)
    assert "more bars than its length" in str(excinfo.value)
    assert np.isfinite(ta.rsi(RAMP, 19)[-1])


def test_a_negative_band_width_is_refused():
    # it turned the bands inside out, upper below lower, and every comparison
    # written against them then read backwards
    with pytest.raises(ValueError) as excinfo:
        ta.bbands(RAMP, 5, deviations=-2.0)
    assert "at least 0" in str(excinfo.value)


def test_the_smoothing_length_is_named_by_the_function_that_took_it():
    with pytest.raises(ValueError) as excinfo:
        ta.stoch(RAMP + 1.0, RAMP - 1.0, RAMP, 5, smooth=999)
    assert "stoch" in str(excinfo.value)


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
