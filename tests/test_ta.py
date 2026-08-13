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
    "hma": lambda h, l, c, v: [ta.hma(c, 6)],
    "vwma": lambda h, l, c, v: [ta.vwma(c, v, 5)],
    "willr": lambda h, l, c, v: [ta.willr(h, l, c, 5)],
    "cci": lambda h, l, c, v: [ta.cci(h, l, c, 5)],
    "natr": lambda h, l, c, v: [ta.natr(h, l, c, 5)],
    "keltner": lambda h, l, c, v: list(vars_of(ta.keltner(h, l, c, 5, 5))),
    "highest": lambda h, l, c, v: [ta.highest(c, 5)],
    "lowest": lambda h, l, c, v: [ta.lowest(c, 5)],
    "shift": lambda h, l, c, v: [ta.shift(c, 3)],
    "change": lambda h, l, c, v: [ta.change(c, 3)],
    "supertrend": lambda h, l, c, v: list(vars_of(ta.supertrend(h, l, c, 5))),
    "adx": lambda h, l, c, v: list(vars_of(ta.adx(h, l, c, 5))),
    "obv": lambda h, l, c, v: [ta.obv(c, v)],
    "mfi": lambda h, l, c, v: [ta.mfi(h, l, c, v, 5)],
}

# the crossings are boolean by design, so they take the same causality checks and
# not the NaN ones (ADR 0063)
FLAGS = {
    "crossover": lambda h, l, c, v: [ta.crossover(ta.sma(c, 3), ta.sma(c, 5))],
    "crossunder": lambda h, l, c, v: [ta.crossunder(ta.sma(c, 3), 10.0)],
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
    high, low, close, _volume = ramp_bars()
    belt = ta.keltner(high, low, close, 5, 5)
    assert belt.upper.shape == belt.middle.shape == belt.lower.shape == (20,)
    trend = ta.adx(high, low, close, 5)
    assert trend.adx.shape == trend.plus.shape == trend.minus.shape == (20,)
    rails = ta.supertrend(high, low, close, 5)
    assert rails.direction.shape == rails.middle.shape == (20,)


def test_the_crossings_are_boolean_and_their_warm_up_is_false():
    # the one deliberate exception to the NaN rule. bool(nan) is True, so a float
    # warm-up would fire a rule on exactly the bars where nothing is known
    for name, call in FLAGS.items():
        out = call(*ramp_bars())[0]
        assert out.shape == (20,), name
        assert out.dtype == bool, f"{name} returned {out.dtype}"
        assert not bool(out[0]), f"{name} fired on bar 0"
    # and the trap itself, stated so nobody restores the float version
    assert bool(np.float64("nan"))


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
    # the fixture that used to stand here had high, low and close spaced so that
    # (h + l + c) / 3, the close alone and the midpoint were all the SAME array,
    # so it could not rule out the two things its comment said it ruled out.
    # These are chosen so all three differ on every bar
    high = np.array([12.0, 30.0, 34.0, 60.0])
    low = np.array([6.0, 16.0, 26.0, 36.0])
    close = np.array([11.0, 17.0, 33.0, 37.0])
    typical = (high + low + close) / 3.0
    midpoint = (high + low) / 2.0
    assert not np.allclose(typical, close) and not np.allclose(typical, midpoint)
    volume = np.array([1.0, 1.0, 1.0, 97.0])
    out = ta.vwap(high, low, close, volume, 4)
    expected = float((typical * volume).sum() / volume.sum())
    assert out[3] == pytest.approx(expected)
    # and it is neither of the two implementations the typical price is not
    assert out[3] != pytest.approx(float((close * volume).sum() / volume.sum()))
    assert out[3] != pytest.approx(float((midpoint * volume).sum() / volume.sum()))
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
        if name == "shift":
            # a lag MOVES its values, so it moves the gap with them. Checked
            # below rather than exempted, since a shift that lost the gap would
            # be the same defect as one that bridged it
            continue
        for index, line in enumerate(call(high, low, close, volume)):
            assert np.isnan(line[10]), (
                f"{name} line {index} passed straight through the gap"
            )
    lagged = ta.shift(close, 3)
    assert np.isnan(lagged[13]), "the shift lost the gap it was carrying"
    assert np.isfinite(lagged[10]), "the shift held the gap where it was not"


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
    kinds = [n for n in ta.__all__ if isinstance(getattr(ta, n), type)]
    assert len(ta.__all__) - len(kinds) == 30
    assert set(kinds) == {"Bands", "Channel", "Macd", "Stochastic", "Trend"}


# ------------------------------------------------- the comparison group


def test_shift_pads_rather_than_wrapping():
    # the whole reason it exists: values[i - n] on an early bar is a negative
    # index, which numpy resolves from the END of the array (ADR 0063)
    out = ta.shift(RAMP, 3)
    assert np.isnan(out[:3]).all()
    assert np.array_equal(out[3:], RAMP[:-3])
    # written by hand this reads the last bar of the series instead
    assert RAMP[0 - 3] == RAMP[17] == 18.0
    assert ta.shift(RAMP, 0).tolist() == RAMP.tolist()
    # a shift past the end is all warm-up, not an error
    assert np.isnan(ta.shift(RAMP, 20)).all()


def test_a_shift_cannot_be_asked_to_read_the_future():
    with pytest.raises(ValueError) as excinfo:
        ta.shift(RAMP, -1)
    assert "read the future" in str(excinfo.value)
    with pytest.raises(ValueError):
        ta.shift(RAMP, 1.5)


def test_change_is_the_move_over_the_lookback():
    out = ta.change(RAMP, 4)
    assert np.isnan(out[:4]).all()
    assert out[10] == pytest.approx(4.0)          # a ramp of one per bar


def test_highest_and_lowest_include_the_bar_they_are_read_on():
    # the same rule donchian states, so a breakout of `highest` cannot happen on
    # the bar that set it
    high = ta.highest(RAMP, 5)
    low = ta.lowest(RAMP, 5)
    assert np.isnan(high[:4]).all() and np.isnan(low[:4]).all()
    assert high[4] == 5.0 and low[4] == 1.0
    assert high[19] == 20.0 and low[19] == 16.0


def test_a_cross_fires_on_the_bar_it_happens_and_only_then():
    values = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0])
    up = ta.crossover(values, 2.0)
    down = ta.crossunder(values, 2.0)
    assert up.tolist() == [False, False, True, False, False, False, True]
    assert down.tolist() == [False, False, False, False, True, False, False]
    # touching the level and coming back is a cross; touching and staying is not
    assert ta.crossover(np.array([1.0, 2.0, 2.0]), 2.0).tolist() == [False, False, False]


def test_a_cross_against_a_missing_value_is_not_a_cross():
    a = np.array([1.0, np.nan, 3.0, 4.0])
    assert ta.crossover(a, 2.0).tolist() == [False, False, False, False]
    assert ta.crossover(np.array([1.0, 1.5, 3.0, 4.0]), 2.0).tolist() == [
        False, False, True, False
    ]


def test_a_cross_takes_a_level_or_a_second_series():
    fast = ta.sma(RAMP, 3)
    with pytest.raises(ValueError):
        ta.crossover(fast, np.arange(5.0))          # a series of the wrong length
    with pytest.raises(ValueError):
        ta.crossover(fast, np.nan)                  # a level that is not one


# ------------------------------------------------- the new families, by hand


def jagged(n=60, seed=5):
    # a varying true range, because on a straight ramp every smoothing constant
    # converges to the same number and a test on one cannot tell them apart
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.5, n))
    spread = rng.uniform(0.2, 4.0, n)
    return close + spread, close - spread, close


def test_atr_smooths_at_wilders_rate_and_not_the_exponential_one():
    # the difference is 1/n against 2/(n+1), which on a jagged series is a
    # different number every bar and on a smooth one is the same number
    high, low, close = jagged()
    ranges = ta.true_range(high, low, close)
    out = ta.atr(high, low, close, 5)
    seed = float(ranges[:5].mean())

    def rolled(alpha):
        running = seed
        for i in range(5, len(close)):
            running = alpha * ranges[i] + (1.0 - alpha) * running
        return running

    assert out[-1] == pytest.approx(rolled(1.0 / 5.0), rel=1e-12)
    assert out[-1] != pytest.approx(rolled(2.0 / 6.0), rel=1e-6)
    assert out[4] == pytest.approx(seed)          # the seed is the window mean


def test_rsi_smooths_at_wilders_rate_too():
    _h, _l, close = jagged()
    out = ta.rsi(close, 5)
    change = np.diff(close, prepend=np.nan)
    gains, losses = np.maximum(change, 0.0), -np.minimum(change, 0.0)

    def rolled(values, alpha):
        running = float(values[1:6].mean())
        for i in range(6, len(close)):
            running = alpha * values[i] + (1.0 - alpha) * running
        return running

    up, down = rolled(gains, 1.0 / 5.0), rolled(losses, 1.0 / 5.0)
    assert out[-1] == pytest.approx(100.0 - 100.0 / (1.0 + up / down), rel=1e-12)


def test_willr_is_the_stochastic_read_from_the_top():
    rng = np.random.default_rng(11)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 80))
    high, low = close + 1.0, close - 1.0
    percent_r = ta.willr(high, low, close, 14)
    raw_k = ta.stoch(high, low, close, 14, 1).k
    both = np.isfinite(percent_r) & np.isfinite(raw_k)
    assert both.any()
    assert np.allclose(percent_r[both], raw_k[both] - 100.0)
    assert (percent_r[both] <= 0.0).all() and (percent_r[both] >= -100.0).all()


def test_cci_divides_by_the_mean_absolute_deviation_not_the_standard_one():
    # the definition, and the thing that makes it behave differently from zscore
    values = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    high = low = close = values
    out = ta.cci(high, low, close, 5)
    window = values                                     # typical == values here
    mad = float(np.abs(window - window.mean()).mean())
    expected = (values[-1] - window.mean()) / (0.015 * mad)
    assert out[4] == pytest.approx(expected)
    assert mad != pytest.approx(float(window.std()))    # the two really differ


def test_natr_is_the_range_as_a_share_of_price():
    high, low, close, _v = ramp_bars()
    ranges = ta.atr(high, low, close, 5)
    out = ta.natr(high, low, close, 5)
    both = np.isfinite(out)
    assert np.allclose(out[both], ranges[both] / close[both] * 100.0)


def test_keltner_bands_the_bars_where_bbands_bands_the_closes():
    # a session of wide ranges closing in one place widens keltner and not bbands,
    # which is the reason both exist
    close = np.full(60, 100.0)
    high = close + np.where(np.arange(60) >= 30, 10.0, 1.0)
    low = close - np.where(np.arange(60) >= 30, 10.0, 1.0)
    belt = ta.keltner(high, low, close, 10, 10, 2.0)
    bands = ta.bbands(close, 10, 2.0)
    assert belt.upper[-1] - belt.lower[-1] > belt.upper[25] - belt.lower[25]
    assert bands.upper[-1] - bands.lower[-1] == pytest.approx(0.0)
    assert belt.middle[-1] == pytest.approx(ta.ema(close, 10)[-1])


def test_adx_reads_a_clean_trend_as_a_trend_and_a_range_as_none():
    trending = np.arange(1.0, 81.0)
    strong = ta.adx(trending + 1.0, trending - 1.0, trending, 14)
    assert strong.adx[-1] > 90.0
    assert strong.plus[-1] > strong.minus[-1]
    chop = 100.0 + np.tile([0.0, 1.0], 40)
    weak = ta.adx(chop + 1.0, chop - 1.0, chop, 14)
    assert weak.adx[-1] < strong.adx[-1]
    finite = np.isfinite(strong.adx)
    # a perfect one-way ramp reaches the ceiling exactly, so allow the last ulp
    assert (strong.adx[finite] >= 0.0).all()
    assert (strong.adx[finite] <= 100.0 + 1e-9).all()


def test_a_falling_market_puts_the_strength_on_the_other_side():
    falling = np.arange(80.0, 0.0, -1.0)
    trend = ta.adx(falling + 1.0, falling - 1.0, falling, 14)
    assert trend.minus[-1] > trend.plus[-1]


def test_supertrend_flips_when_price_closes_through_the_far_rail():
    up = np.arange(1.0, 61.0)
    close = np.concatenate([up, up[-1] - np.arange(1.0, 41.0) * 3.0])
    high, low = close + 1.0, close - 1.0
    rails = ta.supertrend(high, low, close, 10, 3.0)
    finite = np.isfinite(rails.direction)
    assert set(np.unique(rails.direction[finite])) <= {1.0, -1.0}
    assert rails.direction[59] == 1.0                    # still long at the top
    assert rails.direction[-1] == -1.0                   # short by the end
    # the line in force is the near rail, never the far one
    long_bars = finite & (rails.direction == 1.0)
    assert np.allclose(rails.middle[long_bars], rails.lower[long_bars])


def test_obv_signs_the_volume_by_the_direction_of_the_close():
    close = np.array([10.0, 11.0, 11.0, 9.0, 12.0])
    volume = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    out = ta.obv(close, volume)
    assert np.isnan(out[0])                              # no previous close
    assert out[1] == pytest.approx(200.0)                # up, add
    assert out[2] == pytest.approx(200.0)                # unchanged, hold
    assert out[3] == pytest.approx(-200.0)               # down, subtract
    assert out[4] == pytest.approx(300.0)                # up, add


def test_obv_restarts_after_a_gap_rather_than_absorbing_it():
    close = np.array([10.0, 11.0, np.nan, 13.0, 14.0])
    volume = np.full(5, 100.0)
    out = ta.obv(close, volume)
    assert np.isnan(out[2]) and np.isnan(out[3])         # the gap and its successor
    assert out[4] == pytest.approx(100.0)                # a fresh run, starting flat


def test_mfi_weights_the_move_by_what_traded():
    rng = np.random.default_rng(3)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 60))
    high, low = close + 1.0, close - 1.0
    quiet = ta.mfi(high, low, close, np.full(60, 1.0), 14)
    loud = ta.mfi(high, low, close, np.where(close > 100.0, 1000.0, 1.0), 14)
    finite = np.isfinite(quiet) & np.isfinite(loud)
    assert finite.any()
    assert not np.allclose(quiet[finite], loud[finite])
    assert (loud[finite] >= 0.0).all() and (loud[finite] <= 100.0).all()
    # a market that only rises is all inflow
    rising = np.arange(1.0, 41.0)
    assert ta.mfi(rising + 1.0, rising - 1.0, rising, np.full(40, 10.0), 14)[-1] == 100.0


def test_hma_turns_faster_than_the_average_it_is_built_from():
    close = np.concatenate([np.full(40, 100.0), np.full(40, 120.0)])
    hull = ta.hma(close, 16)
    plain = ta.sma(close, 16)
    # ten bars after the step, the hull is nearer the new level
    assert abs(hull[50] - 120.0) < abs(plain[50] - 120.0)
    assert np.isnan(hull[0])


def test_vwma_leans_on_the_bars_that_traded():
    values = np.array([10.0, 20.0, 30.0, 40.0])
    heavy = ta.vwma(values, np.array([1.0, 1.0, 1.0, 97.0]), 4)
    assert heavy[3] == pytest.approx((10.0 + 20.0 + 30.0 + 40.0 * 97.0) / 100.0)
    assert heavy[3] != pytest.approx(float(values.mean()))
    assert np.isnan(ta.vwma(values, np.zeros(4), 4)[3])


# ------------------------------------------- the blocked exponential pass


def block_size(alpha):
    # where the seams fall. Computed the way `_decay` computes it, from alpha, so a
    # test crosses the block hand-off on purpose rather than by being long enough by
    # luck. Every test below is worthless the moment its series stops crossing one
    return max(1, int(150.0 / -math.log10(1.0 - alpha)))


def rolled_ema(values, length):
    # the recurrence written the slow, obvious way, one bar at a time: the mean of
    # the first clean window seeds it, one multiply-add carries it, and a gap ends
    # the run so the next clean window seeds a fresh one. Written out here rather
    # than taken from the module, since a reference computed by the code under test
    # agrees with it however wrong both are (ADR 0064)
    alpha = 2.0 / (length + 1.0)
    keep = 1.0 - alpha
    plain = values.tolist()
    out = [float("nan")] * len(plain)
    running = None
    clean = 0
    for i, value in enumerate(plain):
        if not math.isfinite(value):
            running, clean = None, 0
            continue
        clean += 1
        if running is None:
            if clean == length:
                # the same numpy call the module makes, so the seed is bit for bit
                running = float(values[i - length + 1:i + 1].mean())
                out[i] = running
        else:
            running = alpha * value + keep * running
            out[i] = running
    return np.array(out)


def test_the_exponential_pass_matches_the_plain_recurrence_across_many_blocks():
    # the blocked loop in `_decay` has never gone round twice in this suite. The
    # block is 851 bars for `ema(5)` and no series tested reached 300, so the
    # hand-off between blocks, `carried`, has never been taken once: drop it, seed
    # every block from zero, or from the original seed, and every bar of a real 100k
    # frame after the first seam is wrong, worst at the seam and fading over the
    # next hundred bars, and no test in this file would move (ADR 0064)
    length = 5
    alpha = 2.0 / (length + 1.0)
    keep = 1.0 - alpha
    block = block_size(alpha)
    assert block == 851, "the seam moved; this series may no longer cross one"
    size = 4 * block + length + 7
    step = np.arange(float(size))
    close = 100.0 + 10.0 * np.sin(step / 17.0) + 0.5 * np.cos(step / 2.0)
    out = ta.ema(close, length)
    expected = rolled_ema(close, length)

    # ADR 0064 measured the reassociation at 2.1e-15 relative at worst and ADR 0065
    # declines to promise the last bit, so the tolerance sits three orders above the
    # noise and twelve below anything a mishandled carry produces
    assert np.isfinite(out[length - 1:]).all()
    assert np.allclose(out[length - 1:], expected[length - 1:],
                       rtol=1e-12, atol=0.0), "the blocked pass and the loop parted"

    # and the seams one at a time, where a lost carry shows before it decays away.
    # out[seam - 1] is the module's own, so this asks only that the one step across
    # the block boundary is the step the recurrence says it is
    seams = [length + m * block for m in range(1, 5)]
    assert seams[-1] < size
    for seam in seams:
        assert out[seam] == pytest.approx(
            alpha * close[seam] + keep * out[seam - 1], rel=1e-12
        ), f"the carry into the block starting at bar {seam} was dropped"


def test_a_gap_between_two_multi_block_stretches_reseeds_and_carries_correctly():
    # the blocking is per clean stretch, so the second stretch counts its blocks
    # from the bar the gap ended on and its seams land at a different phase from the
    # first stretch's. No test has ever run a stretch past one block, let alone two
    # stretches at two phases, so a `_decay` blocking the whole array rather than
    # each stretch, or carrying the value from before the gap across it, passes
    # everything else in this file (ADR 0064)
    length = 5
    alpha = 2.0 / (length + 1.0)
    keep = 1.0 - alpha
    block = block_size(alpha)
    head = 2 * block + length + 20
    restart = head + 3
    size = restart + 2 * block + length + 40
    step = np.arange(float(size))
    close = 100.0 + 10.0 * np.sin(step / 23.0) + 0.5 * np.cos(step / 3.0)
    close[head:restart] = np.nan

    out = ta.ema(close, length)
    expected = rolled_ema(close, length)
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0, equal_nan=True)

    # the hole, the warm-up behind it, and the fresh seed from the next clean window
    assert np.isfinite(out[head - 1])
    assert np.isnan(out[head:restart + length - 1]).all()
    assert out[restart + length - 1] == pytest.approx(
        float(close[restart:restart + length].mean())
    )
    assert np.isfinite(out[-1]), "one hole killed the rest of the series"

    # both stretches cross two seams, and the second stretch's sit at a phase the
    # first's do not, so blocking from bar zero would put them somewhere else
    first = [length + block, length + 2 * block]
    second = [restart + length + block, restart + length + 2 * block]
    assert first[-1] < head and second[-1] < size
    assert (second[0] - first[0]) % block != 0
    for seam in first + second:
        assert out[seam] == pytest.approx(
            alpha * close[seam] + keep * out[seam - 1], rel=1e-12
        ), f"the carry into the block starting at bar {seam} was dropped"


def test_a_short_length_over_a_hundred_thousand_bars_stays_finite_throughout():
    # the whole reason `_decay` is not a one-liner. Written as a single cumulative
    # sum the recurrence needs (1 - alpha)**-k, which passes 1e308 within a few
    # thousand bars and takes the rest of the output with it: ADR 0064 records
    # length 20 over 100k bars returning 7,045 finite values out of 99,981, in
    # silence. That is the size this suite has never run, so the overflow the
    # blocking exists to prevent has never actually been put to it (ADR 0064)
    length = 20
    size = 100000
    alpha = 2.0 / (length + 1.0)
    block = block_size(alpha)
    assert size > 25 * block, "tens of seams is the point; one is not"
    step = np.arange(float(size))
    close = 100.0 + 10.0 * np.sin(step / 500.0) + 0.5 * np.cos(step / 7.0)
    out = ta.ema(close, length)

    assert np.isnan(out[:length - 1]).all()
    assert np.isfinite(out[length - 1:]).all(), "the weights overflowed"
    # size less the warm-up, and the number ADR 0064 quotes against the one-liner's
    assert int(np.isfinite(out).sum()) == 99981
    # an exponential average of a bounded series is a convex combination of it, its
    # weights summing to one, so a weight multiplied back wrong leaves the series
    # rather than landing somewhere plausible inside it
    assert out[length - 1:].min() >= close.min() - 1e-9
    assert out[length - 1:].max() <= close.max() + 1e-9
    # and the numbers themselves, against the recurrence taken one bar at a time
    expected = rolled_ema(close, length)
    assert np.allclose(out[length - 1:], expected[length - 1:],
                       rtol=1e-12, atol=0.0)
    assert out[-1] == pytest.approx(expected[-1], rel=1e-12)
