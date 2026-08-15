"""Tests for emsl.chart: it builds one compact document from a frame, your arrays
and a run, holds the alignment contract, and refuses the inputs that would draw a
lie.

Everything here reads ``Chart.spec()`` rather than pixels. The gate has no browser,
and a spec assertion is the sharper test anyway: it says what is drawn, where a
screenshot only says it looked the same.
"""

import json
import pathlib
import re
import subprocess
import sys
import warnings

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

import emsl
from emsl.plot import (
    Background, Band, Histogram, Level, Line, Marker, Markers, Panel, ramp,
)


def frame(n=8):
    close = 100.0 + np.arange(n, dtype=np.float64)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            # varied, so the default frame does not trip the dead-feed warning on
            # every test in the file
            "volume": 1000.0 + np.arange(n, dtype=np.float64),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )


class BuyThenClose(emsl.Strategy):
    def next(self, state, engine):
        if state["tick_index"] == 0:
            engine.market_buy(1.0)
        elif state["tick_index"] == 4:
            engine.close()


class BuyAtOpen(emsl.Strategy):
    def next(self, state, engine):
        if state["tick_index"] == 0:
            engine.market_buy(50.0)


def sank(n=6):
    # a frame that falls hard on the bar the first order fills on, then recovers
    return pd.DataFrame(
        {
            "open": [100.0, 110.0, 104.0, 106.0, 108.0, 110.0][:n],
            "high": [112.0, 112.0, 108.0, 110.0, 112.0, 114.0][:n],
            "low": [98.0, 78.0, 98.0, 100.0, 102.0, 104.0][:n],
            "close": [110.0, 80.0, 106.0, 108.0, 110.0, 112.0][:n],
            "volume": 1000.0 + np.arange(n, dtype=np.float64),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )


def run(data=None):
    data = frame(8) if data is None else data
    return emsl.backtest.Backtester(
        data, market="spot", fee_taker=0.0, fee_maker=0.0
    ).run(BuyThenClose())


def everything():
    """One chart carrying every mark kind, so the golden covers the whole union."""
    data = frame(40)
    result = run(data)
    close = data.close.to_numpy()
    fast = pd.Series(close).rolling(5).mean().to_numpy()
    held = np.zeros(len(data), dtype=bool)
    for trade in result.trades:
        held[trade["entry_tick"]:trade["exit_tick"] + 1] = True
    return emsl.chart(
        data,
        [
            # the ramp and the fill are mutually exclusive: an area carries one
            # line colour and would drop the ramp while the legend went on
            # reporting it, so the pair is refused (ADR 0076). Both halves still
            # appear in the golden, on two marks
            Line(fast, "sma 5", color=ramp(close, "#000000", "#ffffff")),
            Line(fast - 2.0, "sma 5 band", fill="#11223344"),
            Histogram(close - 100.0, "delta", panel="flow"),
            Band(close + 1.0, close - 1.0, "channel", fill="#11223344"),
            Band(close, 120.0, only="above", panel="flow", fill=("#00000000", "#000000ff")),
            Level(110.0, "mid"),
            Marker(3, text="here", shape="arrow_up", offset=20, value=105.0,
                   color="#ff5470"),
            Markers(mask=held, shape="arrow_down", offset=-16, value=fast),
            Background(held, fill="#11223344"),
        ],
        result,
        candle_color=np.where(held, "#4d9fff", None),
        panels=[Panel("flow", weight=1.5, range=(-5, 50))],
        title="everything",
        focus=(2, 30),
    )


# ------------------------------------------------------- the alignment contract

def test_the_shared_time_axis_has_one_entry_per_bar():
    spec = emsl.chart(frame(8)).spec()
    assert spec["n"] == 8
    assert len(spec["t"]) == 8


def test_the_time_axis_is_utc_seconds_and_ignores_the_local_zone():
    # a tz-naive frame is read as UTC. Timestamp.timestamp() would read it as
    # local time and put every bar an hour off for most of the world
    naive = frame(4)
    aware = naive.tz_localize("UTC")
    assert emsl.chart(naive).spec()["t"] == emsl.chart(aware).spec()["t"]
    assert emsl.chart(naive).spec()["t"][0] == 1704067200


def test_a_full_length_series_starts_at_bar_zero():
    spec = emsl.chart(frame(8), Line(np.arange(8, dtype=float), "f")).spec()
    assert spec["series"][0]["i0"] == 0


def test_a_series_one_short_is_drawn_from_bar_one():
    spec = emsl.chart(frame(8), Line(np.arange(7, dtype=float), "f")).spec()
    assert spec["series"][0]["i0"] == 1
    assert len(spec["series"][0]["v"]) == 7


def test_a_series_of_any_other_length_names_both_numbers():
    with pytest.raises(ValueError) as excinfo:
        emsl.chart(frame(8), Line(np.arange(4, dtype=float), "f"))
    message = str(excinfo.value)
    assert "4" in message and "8" in message and "7" in message


def test_a_leading_nan_run_becomes_an_offset_rather_than_a_run_of_nulls():
    values = np.arange(8, dtype=float)
    values[:3] = np.nan
    spec = emsl.chart(frame(8), Line(values, "f")).spec()
    assert spec["series"][0]["i0"] == 3
    assert len(spec["series"][0]["v"]) == 5


def test_an_interior_nan_becomes_a_gap_rather_than_a_dropped_row():
    values = np.arange(8, dtype=float)
    values[4] = np.nan
    spec = emsl.chart(frame(8), Line(values, "f")).spec()
    assert spec["series"][0]["i0"] == 0
    assert spec["series"][0]["v"][4] is None
    assert len(spec["series"][0]["v"]) == 8


def test_a_short_series_that_also_has_a_gap_still_starts_at_bar_one():
    # the two halves of the contract had never met. Every gap test was full
    # length, so the start offset was zero and invisible, and both short-series
    # tests were gapless, so they took the fast path and never reached the line
    # that adds the two together. A short array drawn one bar early is the
    # lookahead signature ADR 0037 exists to prevent
    values = np.arange(7, dtype=float)
    values[:2] = np.nan
    spec = emsl.chart(frame(8), Line(values, "f")).spec()
    # 2 leading gaps on an array that already starts at bar 1
    assert spec["series"][0]["i0"] == 3
    assert len(spec["series"][0]["v"]) == 5
    assert spec["series"][0]["v"][0] == 2.0


def test_a_short_series_with_an_interior_gap_keeps_its_offset():
    values = np.arange(7, dtype=float)
    values[3] = np.nan
    spec = emsl.chart(frame(8), Line(values, "f")).spec()
    assert spec["series"][0]["i0"] == 1
    assert spec["series"][0]["v"][3] is None
    assert len(spec["series"][0]["v"]) == 7


def test_an_all_nan_series_draws_nothing_rather_than_raising():
    spec = emsl.chart(frame(8), Line(np.full(8, np.nan), "f")).spec()
    assert spec["series"][0]["v"] == []


def test_the_equity_panel_is_drawn_from_the_balance_the_run_opened_with():
    # this asserted i0 == 1 and a length of T-1, which was the curve's own shape
    # read straight onto the chart. The curve holds a point per advance, so its
    # first entry is the account AFTER the first bar, and everything that divided
    # by it was reporting a return from the wrong base (ADR 0098)
    result = run()
    spec = emsl.chart(frame(8), result).spec()
    assert spec["equity"]["i0"] == 0
    assert len(spec["equity"]["v"]) == 8
    assert spec["equity"]["v"][0] == pytest.approx(result.initial)
    assert spec["equity"]["v"][1:] == pytest.approx(list(result.equity_curve))
    # the drawdown panel starts on the same bar, so the crosshair reads both
    assert spec["drawdown"]["i0"] == 0
    assert spec["drawdown"]["v"][0] == pytest.approx(0.0)


def test_a_run_that_ends_where_it_started_reads_zero_on_the_equity_panel():
    # the number the legend prints on hover and the axis a percent panel asks for
    # are both taken from the track's first value, so both said +25% on a run the
    # headline correctly called +0.0%
    # the position has to be held THROUGH the first advance, or the curve's first
    # entry already equals the opening balance and there is no gap to expose: the
    # order decided on bar 0 fills at bar 1's open of 100, and bar 1 then closes
    # at 80, so the account is down a tenth on the very point the legend used to
    # divide by. It recovers to exactly where it began
    close = np.array([100.0, 80.0, 85.0, 90.0, 95.0, 100.0, 100.0, 100.0])
    opens = np.concatenate(([100.0], close[:-1]))
    data = pd.DataFrame(
        {"open": opens,
         "high": np.maximum(opens, close) + 1.0,
         "low": np.minimum(opens, close) - 1.0,
         "close": close,
         "volume": np.full(close.size, 1000.0)},
        index=pd.date_range("2024-01-01", periods=close.size, freq="1h"),
    )
    result = emsl.backtest.Backtester(
        data, fee_taker=0.0, fee_maker=0.0
    ).run(BuyAtOpen())
    spec = emsl.chart(data, run=result).spec()
    base, last = spec["equity"]["v"][0], spec["equity"]["v"][-1]
    assert base == pytest.approx(result.initial)
    assert (last / base - 1.0) * 100.0 == pytest.approx(
        result.stats["total_return_pct"], abs=1e-9
    )


def test_a_result_with_no_opening_balance_still_draws_from_bar_one():
    # a hand-built result carries no initial, and there is nothing honest to seed
    # with, so the track keeps the shape the curve has
    bare = emsl.backtest.BacktestResult(
        stats={}, equity_curve=np.arange(7, dtype=float) + 100.0, trades=[],
    )
    spec = emsl.chart(frame(8), bare).spec()
    assert spec["equity"]["i0"] == 1
    assert len(spec["equity"]["v"]) == 7


def test_a_colour_array_of_the_wrong_length_names_both_numbers():
    with pytest.raises(ValueError) as excinfo:
        emsl.chart(frame(8), candle_color=["#ffffff"] * 5)
    assert "candle_color" in str(excinfo.value)


def test_a_short_colour_array_is_drawn_from_bar_one_like_any_other_series():
    spec = emsl.chart(frame(8), candle_color=["#ffffff"] * 7).spec()
    assert spec["candles"]["color"]["i0"] == 1


def test_no_non_finite_value_reaches_the_document():
    values = np.arange(8, dtype=float)
    values[2] = np.inf
    values[5] = np.nan
    spec = emsl.chart(frame(8), Line(values, "f"), run()).spec()
    json.dumps(spec, allow_nan=False)


def test_an_infinite_stat_becomes_null_rather_than_breaking_the_document():
    # profit_factor with no losing trade is infinite and sharpe over a flat curve
    # is NaN; JSON carries neither
    spec = emsl.chart(frame(8), run()).spec()
    assert spec["stats"]
    json.dumps(spec["stats"], allow_nan=False)


def test_a_frame_with_a_non_finite_price_is_refused():
    data = frame(8)
    data.loc[data.index[3], "high"] = np.nan
    with pytest.raises(ValueError) as excinfo:
        emsl.chart(data)
    assert "bar 3" in str(excinfo.value)


# --------------------------------------------------------- panels and placement

def test_the_overlay_threshold_is_pinned_on_both_sides():
    # ADR 0039 calls the half-panel threshold the guarantee rather than a tuning
    # parameter, and the suite only ever exercised ratios of 1.0, 0.08 and 0.002,
    # so every value from 0.09 to 1.0 would have passed. frame(8) spans 99 to 108,
    # so a flat series at V merges to a span of V - 99 and the ratio is 9 / that
    overlays = emsl.chart(frame(8), Line(np.full(8, 116.0), "just inside")).spec()
    assert [p["name"] for p in overlays["panels"]] == ["price", "volume"]

    apart = emsl.chart(frame(8), Line(np.full(8, 118.0), "just outside")).spec()
    assert "just outside" in [p["name"] for p in apart["panels"]]


def test_an_accepted_overlay_widens_the_price_extent_for_the_next_one():
    # the placement rule is order dependent on purpose: an early overlay widens
    # the price extent, so a later array is judged against the panel as it now
    # stands and one that would have been pushed out can be let in. The comment
    # on that line claimed a deliberate, documented behaviour that nothing
    # demonstrated, and deleting the widening passed the whole suite
    near, far = Line(np.full(8, 116.0)), Line(np.full(8, 124.0))

    # near overlays and takes price to 99..116, after which far fits too
    together = emsl.chart(frame(8), [near, far]).spec()
    assert [p["name"] for p in together["panels"]] == ["price", "volume"]

    # judged first, against 99..108, far does not fit and takes its own panel
    apart = emsl.chart(frame(8), [Line(np.full(8, 124.0)), Line(np.full(8, 116.0))]).spec()
    assert len(apart["panels"]) == 3


def test_an_unnamed_band_is_placed_by_its_upper_edge():
    # one edge has to decide it, and the choice is not arbitrary: the upper edge
    # is what a band shares with the candles it might sit on. Swapping it for the
    # lower edge passed every test in the file
    band = Band(np.full(8, 116.0), np.full(8, 60.0), fill="#11223344")
    spec = emsl.chart(frame(8), band).spec()
    # by the upper edge this overlays; by the lower it would take its own panel
    assert [p["name"] for p in spec["panels"]] == ["price", "volume"]


def test_price_is_always_the_first_panel():
    spec = emsl.chart(frame(8), Line(np.arange(8, dtype=float), "f")).spec()
    assert spec["panels"][0]["name"] == "price"
    assert spec["panels"][0]["candles"] is True


def test_a_bare_array_on_the_price_scale_overlays_the_candles():
    spec = emsl.chart(frame(8), np.full(8, 104.0)).spec()
    assert spec["series"][0]["panel"] == "price"


def test_a_bare_array_that_would_flatten_the_candles_gets_its_own_panel():
    # the frame lives around 100 and this lives around 5000, so overlaying would
    # leave the candles a sliver
    spec = emsl.chart(frame(8), np.full(8, 5000.0)).spec()
    assert spec["series"][0]["panel"] != "price"


def test_naming_a_panel_overrides_the_placement_rule():
    spec = emsl.chart(frame(8), Line(np.full(8, 5000.0), "f", panel="price")).spec()
    assert spec["series"][0]["panel"] == "price"


def test_configuring_one_panel_leaves_every_other_panel_where_it_was():
    marks = [Line(np.full(8, 5000.0), "a", panel="one"),
             Line(np.full(8, 6000.0), "b", panel="two")]
    plain = [p["name"] for p in emsl.chart(frame(8), marks, run()).spec()["panels"]]
    tuned = emsl.chart(frame(8), marks, run(),
                       panels=[Panel("two", weight=3)]).spec()["panels"]
    assert [p["name"] for p in tuned] == plain


def test_listing_two_panels_fixes_their_order_relative_to_each_other():
    marks = [Line(np.full(8, 5000.0), "a", panel="one"),
             Line(np.full(8, 6000.0), "b", panel="two")]
    names = [p["name"] for p in emsl.chart(
        frame(8), marks, panels=[Panel("two"), Panel("one")]
    ).spec()["panels"]]
    assert names.index("two") < names.index("one")


def test_a_volume_column_draws_a_volume_panel():
    data = frame(8)
    data["volume"] = np.arange(8, dtype=float) + 1.0
    spec = emsl.chart(data).spec()
    assert any(p.get("volume") for p in spec["panels"])
    assert "vol" in spec


def test_a_hidden_panel_and_everything_on_it_leave_the_document():
    data = frame(8)
    data["volume"] = np.arange(8, dtype=float) + 1.0
    spec = emsl.chart(data, panels=[Panel("volume", show=False)]).spec()
    assert not any(p["name"] == "volume" for p in spec["panels"])
    assert "vol" not in spec


def test_the_price_panel_cannot_be_hidden():
    with pytest.raises(ValueError):
        emsl.chart(frame(8), panels=[Panel("price", show=False)])


def test_a_pinned_range_reaches_the_panel():
    spec = emsl.chart(frame(8), Line(np.arange(8, dtype=float), "f", panel="x"),
                      panels=[Panel("x", range=(0, 100))]).spec()
    panel = [p for p in spec["panels"] if p["name"] == "x"][0]
    assert panel["range"] == [0.0, 100.0]


def test_a_result_adds_an_equity_and_a_drawdown_panel():
    names = [p["name"] for p in emsl.chart(frame(8), run()).spec()["panels"]]
    assert "equity" in names and "drawdown" in names


def test_a_frame_whose_clock_is_a_column_is_told_the_line_that_fixes_it():
    # how a stored candle file nearly always arrives, and the shape every
    # published dataset uses. Refusing it correctly is not enough on its own
    data = frame(8).reset_index(names="timestamp")
    # spelled through datetime64[ms] rather than by dividing: pandas builds an
    # index in microseconds on some versions and nanoseconds on others, so a
    # divisor that reads as milliseconds on one reads as seconds on the next
    data["timestamp"] = data["timestamp"].astype("datetime64[ms]").astype("int64")
    with pytest.raises(TypeError) as excinfo:
        emsl.chart(frame=data)
    message = str(excinfo.value)
    assert "set_index" in message and "'timestamp'" in message
    assert "unit='ms'" in message


def test_a_clock_column_already_holding_datetimes_needs_no_conversion():
    data = frame(8).reset_index(names="date")
    with pytest.raises(TypeError) as excinfo:
        emsl.chart(frame=data)
    assert "frame.set_index('date')" in str(excinfo.value)


def test_the_clock_hint_never_becomes_the_error_it_decorates():
    # the hint reads the column to guess its unit, and a column that defeats the
    # guess must fall back to the plain refusal rather than raising something
    # else from inside the error path
    for clock in ([np.nan] * 4, [np.inf] * 4, ["a", "b", "c", "d"], [None] * 4):
        data = frame(4).reset_index(drop=True)
        data["timestamp"] = clock
        with pytest.raises(TypeError) as excinfo:
            emsl.chart(frame=data)
        assert "set_index" not in str(excinfo.value)


def test_a_frame_with_no_clock_anywhere_says_only_what_is_needed():
    data = frame(8).reset_index(drop=True)
    with pytest.raises(TypeError) as excinfo:
        emsl.chart(frame=data)
    assert "set_index" not in str(excinfo.value)


def test_the_whole_call_can_be_written_with_named_arguments():
    values = np.arange(8, dtype=float) + 100.0
    named = emsl.chart(
        frame=frame(8),
        marks=[Line(values=values, name="f")],
        run=run(),
        title="named",
    ).spec()
    positional = emsl.chart(
        frame(8), Line(values, "f"), run(), title="named"
    ).spec()
    assert named == positional


def test_marks_takes_one_mark_as_well_as_a_list():
    values = np.arange(8, dtype=float) + 100.0
    one = emsl.chart(frame=frame(8), marks=Line(values=values, name="f")).spec()
    assert [s["name"] for s in one["series"] if s.get("name")] == ["f"]


def test_naming_run_refuses_anything_that_is_not_a_run():
    with pytest.raises(TypeError) as excinfo:
        emsl.chart(frame=frame(8), run=np.arange(8, dtype=float))
    assert "marks=" in str(excinfo.value)


def test_a_chart_with_no_frame_names_the_argument():
    with pytest.raises(TypeError) as excinfo:
        emsl.chart(marks=[Line(np.arange(8, dtype=float))])
    assert "frame=" in str(excinfo.value)


def test_a_candle_ships_the_price_that_was_in_the_frame():
    # the payload is rounded by numpy, which rounds by scaling and can therefore
    # break a tie the other way from round(). A price is a whole number of ticks,
    # which is nowhere near a tie, and this pins that: a candle a cent away from
    # the frame it was drawn from is the one thing this layer must never do
    data = pd.DataFrame(
        {
            "open": [111.035, 104.695, 116.185, 0.00012345],
            "high": [112.045, 112.005, 118.015, 0.00012355],
            "low": [98.005, 78.045, 98.025, 0.00012335],
            "close": [110.025, 80.015, 106.035, 0.00012345],
            "volume": [1000.0, 1001.0, 1002.0, 1003.0],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="1h"),
    )
    spec = emsl.chart(data).spec()
    candles = data[["open", "high", "low", "close"]].to_numpy()
    from emsl._chart import _digits, _extent

    wanted = [
        [round(float(x), _digits(_extent(candles)) + 2) for x in row]
        for row in candles
    ]
    assert spec["ohlc"] == wanted


def test_what_else_shares_the_price_panel_cannot_change_the_candles():
    # the panel's digits are a display setting that every mark on it feeds. The
    # payload used to read them, so one Level on the wrong panel rounded a coin
    # priced in millionths to a flat line of zeros
    close = np.linspace(0.000012, 0.000019, 40)
    data = pd.DataFrame(
        {
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": 1000.0 + np.arange(40, dtype=np.float64),
        },
        index=pd.date_range("2024-01-01", periods=40, freq="1h"),
    )
    alone = emsl.chart(data).spec()["ohlc"]
    with pytest.warns(RuntimeWarning):
        beside = emsl.chart(data, Level(70.0)).spec()["ohlc"]
    assert alone == beside
    assert len({row[3] for row in alone}) == 40


def test_a_level_that_cannot_be_seen_on_the_price_panel_says_so():
    with pytest.warns(RuntimeWarning) as caught:
        emsl.chart(frame(8), Line(np.linspace(0.0, 100.0, 8), "rsi"), Level(70.0))
    assert "panel=" in str(caught[0].message)


def test_a_level_near_the_candles_is_not_second_guessed():
    # a target just outside the range is an ordinary thing to draw
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        emsl.chart(frame(8), Level(112.0))


def test_the_drawdown_panel_agrees_with_the_engine():
    # the fill lands on bar 1's open and the bar closes far below it, so the first
    # recorded equity point is already under the opening balance and a peak taken
    # from the curve alone reports no drawdown at all
    data = sank()
    result = emsl.backtest.Backtester(
        data, market="spot", fee_taker=0.0, fee_maker=0.0
    ).run(BuyAtOpen())
    values = emsl.chart(data, result).spec()["drawdown"]["v"]
    worst = min(v for v in values if v is not None)
    assert result.stats["max_drawdown_pct"] > 0.0
    assert abs(worst + result.stats["max_drawdown_pct"]) < 1e-9


def test_a_run_that_does_not_report_its_opening_balance_still_charts():
    result = run()
    result.initial = None
    values = emsl.chart(frame(8), result).spec()["drawdown"]["v"]
    assert len(values) == len(result.equity_curve)


def test_a_drawdown_is_capped_at_a_total_loss():
    # bad debt takes equity below zero (ADR 0003), and the engine caps there for a
    # reason: uncapped, a deeper bust reports a larger fall than a wipeout
    result = emsl.backtest.BacktestResult(
        stats={}, equity_curve=np.array([10.0, -50.0]), trades=[], initial=100.0
    )
    values = emsl.chart(frame(3), result, stats=[]).spec()["drawdown"]["v"]
    assert min(values) == -100.0


# ---------------------------------------------------------------- the guards

def test_a_result_from_a_different_number_of_bars_is_refused():
    result = run(frame(8))
    with pytest.raises(ValueError) as excinfo:
        emsl.chart(frame(6), result)
    message = str(excinfo.value)
    assert "8" in message and "6" in message and "focus" in message


def test_a_logarithmic_panel_whose_data_reaches_zero_names_the_bar():
    values = np.arange(8, dtype=float)          # values[0] is 0.0
    with pytest.raises(ValueError) as excinfo:
        emsl.chart(frame(8), Line(values, "f", panel="x"),
                   panels=[Panel("x", scale="log")])
    assert "bar 0" in str(excinfo.value)


def test_a_dead_volume_column_draws_no_panel_and_warns():
    data = frame(8)
    data["volume"] = 0.0
    with pytest.warns(RuntimeWarning):
        spec = emsl.chart(data).spec()
    assert not any(p.get("volume") for p in spec["panels"])


def test_a_frame_without_a_datetime_index_is_refused():
    data = frame(8).reset_index(drop=True)
    with pytest.raises(TypeError):
        emsl.chart(data)


def test_a_frame_whose_bars_share_a_timestamp_is_refused():
    data = frame(4)
    data.index = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00",
                                 "2024-01-01 01:00", "2024-01-01 02:00"])
    with pytest.raises(ValueError):
        emsl.chart(data)


def test_a_marker_outside_the_frame_is_refused():
    with pytest.raises(ValueError):
        emsl.chart(frame(8), Marker(99))


def test_a_second_result_warns_and_the_first_is_used():
    first, second = run(), run()
    with pytest.warns(RuntimeWarning):
        spec = emsl.chart(frame(8), first, second).spec()
    assert len(spec["trades"]) == len(first.trades)


# ----------------------------------------------------------------- encoding

def test_a_per_bar_colour_becomes_a_palette_and_indices():
    colors = ["#ff0000"] * 4 + ["#00ff00"] * 4
    spec = emsl.chart(frame(8), candle_color=colors).spec()
    tint = spec["candles"]["color"]
    assert tint["p"] == [None, "#ff0000", "#00ff00"]
    assert tint["k"] == [1, 1, 1, 1, 2, 2, 2, 2]


def test_the_first_palette_entry_is_always_the_absent_colour():
    spec = emsl.chart(frame(8), candle_color=["#ff0000"] * 8).spec()
    assert spec["candles"]["color"]["p"][0] is None


def test_a_colour_array_of_object_dtype_with_none_entries_encodes():
    # numpy.where(mask, ramp(...), None) is the ordinary way to write one of
    # these and it lands as an object array holding str and None
    mask = np.array([True, False] * 4)
    spec = emsl.chart(frame(8), candle_color=np.where(mask, "#ff0000", None)).spec()
    assert spec["candles"]["color"]["k"] == [1, 0] * 4


def test_a_background_becomes_spans_rather_than_one_entry_per_bar():
    mask = np.array([False, True, True, True, False, False, True, False])
    spec = emsl.chart(frame(8), Background(mask, fill="#112233")).spec()
    entry = [s for s in spec["series"] if s["kind"] == "background"][0]
    assert entry["spans"] == [[1, 4, 0], [6, 7, 0]]


def test_a_background_label_with_no_fill_shades_nothing():
    labels = np.array(["calm", "wild", "calm", "wild",
                       "calm", "calm", "calm", "calm"], dtype=object)
    spec = emsl.chart(frame(8),
                      Background(labels, fill={"wild": "#ff0000"})).spec()
    entry = [s for s in spec["series"] if s["kind"] == "background"][0]
    assert entry["spans"] == [[1, 2, 0], [3, 4, 0]]


def test_a_nan_in_a_background_mask_shades_nothing():
    # numpy.array([nan]).astype(bool) is True, so a mask padded with NaN would
    # otherwise report the condition it was looking for on its last bar
    mask = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan], dtype=object)
    spec = emsl.chart(frame(8), Background(mask, fill="#112233")).spec()
    entry = [s for s in spec["series"] if s["kind"] == "background"][0]
    assert entry["spans"] == [[1, 2, 0]]


def test_a_band_with_a_scalar_edge_carries_a_level_rather_than_a_second_array():
    spec = emsl.chart(frame(8),
                      Band(np.arange(8, dtype=float), 3.0, panel="x")).spec()
    entry = [s for s in spec["series"] if s["kind"] == "band"][0]
    assert entry["level"] == 3.0
    assert "v2" not in entry


def test_a_band_with_two_edges_ships_them_at_one_start_and_one_length():
    upper = np.arange(8, dtype=float)
    lower = np.arange(8, dtype=float) - 5.0
    lower[:3] = np.nan
    spec = emsl.chart(frame(8), Band(upper, lower, panel="x")).spec()
    entry = [s for s in spec["series"] if s["kind"] == "band"][0]
    assert entry["i0"] == 3
    assert len(entry["v"]) == len(entry["v2"]) == 5


def test_a_below_band_ships_its_fill_flipped_so_stop_zero_is_the_reference_edge():
    stops = ("#00000000", "#000000ff")
    above = emsl.chart(frame(8), Band(np.arange(8, dtype=float), 3.0,
                                      only="above", panel="x", fill=stops)).spec()
    below = emsl.chart(frame(8), Band(np.arange(8, dtype=float), 3.0,
                                      only="below", panel="x", fill=stops)).spec()
    up = [s for s in above["series"] if s["kind"] == "band"][0]
    down = [s for s in below["series"] if s["kind"] == "band"][0]
    assert up["fill"] == list(stops)
    assert down["fill"] == list(reversed(stops))


def test_a_trade_row_carries_no_numpy_datetime():
    spec = emsl.chart(frame(8), run()).spec()
    assert spec["trades"]
    for trade in spec["trades"]:
        assert "entry_time" not in trade and "exit_time" not in trade
        assert set(trade) == {"i", "in", "out", "side", "size", "px_in", "px_out",
                              "fees", "net", "bars"}


def test_the_interval_is_read_off_the_axis():
    assert emsl.chart(frame(8)).spec()["interval"] == "1h"


def test_the_interval_survives_a_structural_gap_in_the_series():
    # weekdays only, so one gap in five is three days rather than one. The median
    # is what makes that a 1d chart rather than a 1d-and-a-bit one
    index = pd.bdate_range("2024-01-01", periods=40)
    data = frame(40).set_index(index)
    assert emsl.chart(data).spec()["interval"] == "1d"


def test_every_calendar_timeframe_is_named_rather_than_counted_in_days():
    # no calendar unit is a fixed number of seconds, so the raw gap makes the
    # same series read 30d over one window and 31d over the next, and a yearly
    # one 365d or 366d depending on the leap years it spans
    for freq, expected in (("7D", "1w"), ("MS", "1M"), ("ME", "1M"),
                           ("2MS", "2M"), ("QS", "3M"), ("QE", "3M"),
                           ("2QS", "6M"), ("YS", "1y"), ("YE", "1y")):
        index = pd.date_range("2020-01-01", periods=24, freq=freq)
        data = frame(24).set_index(index)
        assert emsl.chart(data).spec()["interval"] == expected, freq


def test_a_leap_span_and_a_common_span_name_the_same_year():
    leap = frame(8).set_index(pd.date_range("2019-01-01", periods=8, freq="YS"))
    common = frame(8).set_index(pd.date_range("2021-01-01", periods=8, freq="YS"))
    assert emsl.chart(leap).spec()["interval"] == "1y"
    assert emsl.chart(common).spec()["interval"] == "1y"


def test_a_gap_with_no_calendar_name_is_reported_in_days():
    # 45 days is not a unit, and inventing a name for it would be worse than
    # saying what it is
    data = frame(20).set_index(pd.date_range("2024-01-01", periods=20, freq="45D"))
    assert emsl.chart(data).spec()["interval"] == "45d"


def test_an_interval_that_is_not_a_round_unit_is_reported_in_seconds():
    data = frame(30).set_index(pd.date_range("2024-01-01", periods=30, freq="90s"))
    assert emsl.chart(data).spec()["interval"] == "90s"


# ----------------------------------------------------------------- document

def test_focus_moves_the_viewport_and_leaves_every_bar_in_the_data():
    spec = emsl.chart(frame(8), focus=(2, 5)).spec()
    assert spec["focus"] == [2.0, 5.0]
    assert spec["n"] == 8 and len(spec["t"]) == 8


def test_focus_bounds_are_clamped_rather_than_refused():
    spec = emsl.chart(frame(8), focus=(2, 900)).spec()
    assert spec["focus"][1] <= 7.0


def test_focus_of_an_integer_means_the_last_n_bars():
    spec = emsl.chart(frame(8), focus=3).spec()
    assert spec["focus"] == [5.0, 7.0]


def test_focus_of_a_trade_frames_that_trade():
    result = run()
    spec = emsl.chart(frame(8), result, focus=result.trades[0]).spec()
    assert spec["focus"][0] <= result.trades[0]["entry_tick"]


def test_two_charts_of_the_same_inputs_produce_the_same_document():
    def once():
        return emsl.chart(frame(8), Line(np.arange(8, dtype=float), "f"), run())._html()

    assert once() == once()


def test_a_series_name_containing_a_closing_script_tag_cannot_break_the_document():
    document = emsl.chart(
        frame(8), Line(np.arange(8, dtype=float), "</script><b>x")
    )._html()
    assert "</script><b>x" not in document
    assert document.count("</script>") == 2


def test_a_title_containing_markup_is_escaped_in_the_title_element():
    document = emsl.chart(frame(8), title="<b>hi</b>")._html()
    assert "<title>&lt;b&gt;hi&lt;/b&gt;</title>" in document


def test_the_saved_file_carries_its_renderer_and_references_no_url(tmp_path):
    path = emsl.chart(frame(8), run()).save(tmp_path / "run.html")
    document = pathlib.Path(path).read_text(encoding="utf-8")
    assert "TradingView" in document or "Lightweight Charts" in document
    assert 'src="http' not in document
    assert "src='http" not in document
    assert 'href="http' not in document


def test_save_returns_the_path_it_was_given(tmp_path):
    target = tmp_path / "nested" / "run.html"
    assert emsl.chart(frame(8)).save(target) == target


def test_show_outside_a_notebook_points_at_save():
    with pytest.raises(RuntimeError) as excinfo:
        emsl.chart(frame(8)).show()
    assert "save" in str(excinfo.value)


def test_chart_defaults_returns_the_previous_settings_and_takes_only_known_keywords():
    previous = emsl.chart_defaults(height=700)
    try:
        assert emsl.chart(frame(8)).spec()["height"] == 700
        assert emsl.chart(frame(8), height=300).spec()["height"] == 300
    finally:
        emsl.chart_defaults(**{k: v for k, v in previous.items() if v is not None})
    with pytest.raises(TypeError):
        emsl.chart_defaults(them="dark")


def test_both_palettes_travel_so_the_theme_toggle_works_offline():
    theme = emsl.chart(frame(8)).spec()["theme"]
    assert set(theme["dark"]) == set(theme["light"])
    assert theme["mode"] == "dark"


# ------------------------------------------------- packaging and the js bundle

def test_every_vendored_asset_ships_inside_the_wheel():
    # the gate does COPY tests /tests and nothing else, so the source tree does
    # not exist here and this resolves each asset out of the installed package.
    # maturin ships python-source with no include key, so a non-.py file rides
    # along by convention, and a packaging regression is invisible until a user
    # calls save()
    from emsl import _chart

    names = ("chart.html", "chart.css", "lightweight-charts.js") + _chart._ASSETS
    for name in names:
        assert _chart._asset(name).strip()


def test_the_vendored_renderer_keeps_its_licence():
    from emsl import _chart

    assert "Apache" in _chart._asset("lightweight-charts.js")[:2000]
    assert "Apache" in _chart._asset("LICENSE-lightweight-charts")


def test_the_manifest_lists_every_script_in_the_static_directory():
    # the bundle is built from an explicit tuple so a stray file cannot ship;
    # this is the other half of that bargain, catching a file added to _static
    # and never wired in
    from emsl import _chart

    on_disk = sorted(
        str(p.relative_to(_chart._STATIC)).replace("\\", "/")
        for p in _chart._STATIC.rglob("*.js")
        if p.name != "lightweight-charts.js"
    )
    assert on_disk == sorted(_chart._ASSETS)


def test_no_top_level_javascript_name_is_declared_twice():
    # the assets are concatenated into one scope, so a repeated top-level const
    # throws at load and takes the whole chart with it. There is no bundler and
    # no javascript engine in the gate, so the check is done on the text
    from emsl import _chart

    pattern = re.compile(r"^(?:const|let|var|function|class)\s+(\w+)", re.M)
    seen = {}
    for name in _chart._ASSETS:
        for declared in pattern.findall(_chart._asset(name)):
            assert declared not in seen, f"{declared} in {seen.get(declared)} and {name}"
            seen[declared] = name


def test_every_asset_opts_into_strict_mode():
    # in development the assets load as separate script tags and in the wheel
    # they load inside one iife; a per-file directive is what makes the two agree
    from emsl import _chart

    for name in _chart._ASSETS:
        assert _chart._asset(name).lstrip().startswith('"use strict";')


def test_no_colour_is_written_into_the_javascript():
    # python decides what is drawn and javascript decides only how it is painted,
    # so a colour on that side is a decision in the wrong half. Full transparency
    # is not a colour, it is a way of drawing nothing
    from emsl import _chart

    pattern = re.compile(r"#[0-9a-fA-F]{3,8}\b|rgba?\(")
    for name in _chart._ASSETS:
        for hit in pattern.findall(_chart._asset(name)):
            assert hit == "rgba(", f"{name} carries a colour: {hit}"
    for text in (_chart._asset(n) for n in _chart._ASSETS):
        for hit in re.findall(r"rgba\([^)]*\)", text):
            assert hit == "rgba(0,0,0,0)", f"a javascript asset carries {hit}"


def test_importing_emsl_pulls_in_no_optional_dependency():
    # pandas is duck-typed and IPython is imported inside show(); a stray
    # module-level import would make emsl unimportable for anyone without them
    code = ("import sys, emsl; "
            "print(int('pandas' in sys.modules), int('IPython' in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["0", "0"]


# ------------------------------------------------------------ the headline

def test_a_run_puts_four_numbers_above_the_chart():
    spec = emsl.chart(frame(8), run(), title="a run").spec()
    assert spec["title"] == "a run"
    labels = [row[0] for row in spec["headline"]]
    assert labels == ["return", "sharpe", "max dd", "trades"]


def test_the_headline_is_preformatted_so_the_renderer_decides_nothing():
    spec = emsl.chart(frame(8), run()).spec()
    for label, value in spec["headline"]:
        assert isinstance(label, str) and isinstance(value, str)


def test_a_non_finite_stat_reads_as_not_available_rather_than_breaking():
    spec = emsl.chart(frame(8), run(), stats=["profit_factor"]).spec()
    assert spec["headline"][0][0] == "profit factor"
    json.dumps(spec, allow_nan=False)


def test_stats_chooses_which_numbers_are_shown():
    spec = emsl.chart(frame(8), run(), stats=["sharpe", "exposure_pct"]).spec()
    assert [row[0] for row in spec["headline"]] == ["sharpe", "exposure"]


def test_an_empty_stats_list_shows_none():
    assert "headline" not in emsl.chart(frame(8), run(), stats=[]).spec()


def test_stats_naming_a_key_the_run_does_not_report_is_refused():
    with pytest.raises(KeyError):
        emsl.chart(frame(8), run(), stats=["alpha"])


def test_stats_without_a_run_is_refused():
    with pytest.raises(ValueError):
        emsl.chart(frame(8), stats=["sharpe"])


def test_a_chart_with_no_title_and_no_run_has_no_headline():
    spec = emsl.chart(frame(8)).spec()
    assert "headline" not in spec and "title" not in spec


def test_the_headline_is_written_as_text_rather_than_as_markup():
    # a title carrying angle brackets is a title, not a decision anyone has to
    # think about
    from emsl import _chart

    controls = _chart._asset("controls.js")
    assert "textContent" in controls
    document = emsl.chart(frame(8), title="<b>hi</b>")._html()
    assert "<b>hi</b>" not in document


# ------------------------------------------------------ room past the candles

def test_future_lengthens_the_axis_and_leaves_the_candles_alone():
    spec = emsl.chart(frame(8), future=3).spec()
    assert spec["n"] == 8
    assert len(spec["t"]) == 11
    assert len(spec["ohlc"]) == 8


def test_the_projected_bars_continue_the_interval():
    spec = emsl.chart(frame(8), future=3).spec()
    gaps = np.diff(spec["t"])
    assert len(set(gaps.tolist())) == 1


def test_a_series_running_past_the_last_candle_is_accepted_only_with_future():
    ahead = np.arange(11, dtype=float)
    with pytest.raises(ValueError):
        emsl.chart(frame(8), Line(ahead, "cloud"))
    spec = emsl.chart(frame(8), Line(ahead, "cloud"), future=3).spec()
    assert spec["series"][0]["i0"] == 0
    assert len(spec["series"][0]["v"]) == 11


def test_the_length_error_names_the_projected_length_when_future_is_set():
    with pytest.raises(ValueError) as excinfo:
        emsl.chart(frame(8), Line(np.arange(5, dtype=float), "f"), future=3)
    message = str(excinfo.value)
    assert "11" in message and "past the last candle" in message


def test_the_two_ordinary_lengths_still_work_with_future_set():
    spec = emsl.chart(frame(8), Line(np.arange(8, dtype=float), "a"),
                      Line(np.arange(7, dtype=float), "b"), future=3).spec()
    assert spec["series"][0]["i0"] == 0
    assert spec["series"][1]["i0"] == 1


def test_a_band_can_be_projected_which_is_the_whole_point_of_a_cloud():
    upper = np.arange(11, dtype=float) + 10
    lower = np.arange(11, dtype=float)
    spec = emsl.chart(frame(8), Band(upper, lower, "cloud", panel="x"),
                      future=3).spec()
    entry = [s for s in spec["series"] if s["kind"] == "band"][0]
    assert entry["i0"] == 0
    assert len(entry["v"]) == len(entry["v2"]) == 11


def test_a_run_can_be_drawn_without_its_trades():
    with_them = emsl.chart(frame(8), run()).spec()
    without = emsl.chart(frame(8), run(), trades=False).spec()
    assert with_them["trades"]
    assert without["trades"] == []
    assert without["equity"] == with_them["equity"]


def test_hiding_a_panel_drops_its_data_and_not_just_its_pixels():
    shown = emsl.chart(frame(8), run()).spec()
    hidden = emsl.chart(
        frame(8), run(), panels=[Panel("drawdown", show=False)]
    ).spec()
    assert "drawdown" in shown
    assert "drawdown" not in hidden
    assert [p["name"] for p in hidden["panels"]] == ["price", "volume", "equity"]


def test_markers_draws_one_entry_holding_every_bar_the_mask_is_true_on():
    mask = np.array([False, True, False, True, True, False, False, False])
    spec = emsl.chart(frame(8), Markers(mask=mask)).spec()
    entries = [s for s in spec["series"] if s["kind"] == "marker"]
    assert len(entries) == 1
    assert entries[0]["bars"] == [1, 3, 4]


def test_markers_one_bar_short_lands_a_bar_later_like_every_other_mark():
    mask = np.array([False, True, False, False, False, False, False])
    spec = emsl.chart(frame(8), Markers(mask=mask)).spec()
    assert [s for s in spec["series"] if s["kind"] == "marker"][0]["bars"] == [2]


def test_markers_anchors_each_glyph_to_its_own_bar_when_given_an_array():
    mask = np.array([False, True, False, True, False, False, False, False])
    values = np.arange(8, dtype=float) + 100.0
    spec = emsl.chart(frame(8), Markers(mask=mask, value=values)).spec()
    entry = [s for s in spec["series"] if s["kind"] == "marker"][0]
    assert entry["bars"] == [1, 3]
    assert entry["values"] == [101.0, 103.0]


def test_markers_refuses_a_float_mask_because_nan_is_truthy():
    # how a comparison arrives once a warm-up NaN has been through it, and
    # astype(bool) would put a glyph on the one bar nothing could be decided on
    with pytest.raises(TypeError) as excinfo:
        Markers(mask=np.array([0.0, float("nan"), 1.0]))
    assert "NaN is truthy" in str(excinfo.value)


def test_a_marker_may_sit_in_the_projected_room():
    spec = emsl.chart(frame(8), Marker(9), future=3).spec()
    assert [s for s in spec["series"] if s["kind"] == "marker"][0]["bars"] == [9]
    with pytest.raises(ValueError):
        emsl.chart(frame(8), Marker(11), future=3)


def test_focus_reaches_the_projected_room():
    spec = emsl.chart(frame(8), future=3).spec()
    assert spec["focus"] is None if "focus" in spec else True
    windowed = emsl.chart(frame(8), future=3, focus=4).spec()
    assert windowed["focus"] == [4.0, 10.0]


def test_future_is_refused_when_it_cannot_know_the_interval():
    with pytest.raises(ValueError):
        emsl.chart(frame(1), future=3)


def test_future_must_be_zero_or_positive():
    with pytest.raises(ValueError):
        emsl.chart(frame(8), future=-1)


def test_the_renderer_lets_the_cursor_reach_the_projected_room():
    # spec.n counts candles and spec.t counts the axis; the crosshair must
    # travel the axis or the projection cannot be read
    from emsl import _chart

    assert "spec.t.length - 1" in _chart._asset("core.js")
    assert "i < SPEC.n" in _chart._asset("legend.js")


# ------------------------------------------------ regressions from the audit

def test_a_non_string_title_does_not_crash_inside_save(tmp_path):
    # the title used to be coerced for the payload and handed to Chart raw, so
    # html.escape met an int long after chart() had returned
    emsl.chart(frame(8), title=20260803).save(tmp_path / "run.html")


def test_a_background_label_mapped_to_nothing_ships_no_fill():
    # a null in fills breaks the renderer's gradient cache on every repaint, even
    # when no bar carries the label
    labels = np.array(["calm"] * 7 + ["wild"], dtype=object)
    spec = emsl.chart(
        frame(8), Background(labels, fill={"wild": "#ff0000", "calm": None})
    ).spec()
    entry = [s for s in spec["series"] if s["kind"] == "background"][0]
    assert entry["fills"] == [["#ff0000"]]
    assert entry["spans"] == [[7, 8, 0]]


def test_an_empty_frame_is_refused_at_a_guard():
    with pytest.raises(ValueError) as excinfo:
        emsl.chart(frame(8).iloc[:0])
    assert "no bars" in str(excinfo.value)


def test_a_pandas_nullable_mask_does_not_crash_a_background():
    # bool(pandas.NA) raises rather than answering, and a value that cannot answer
    # the question is not a condition being met
    mask = pd.array([True, False, True, None, True, False, True, False],
                    dtype="boolean")
    spec = emsl.chart(frame(8), Background(mask, fill="#112233")).spec()
    entry = [s for s in spec["series"] if s["kind"] == "background"][0]
    assert [span[2] for span in entry["spans"]] == [0, 0, 0, 0]


def test_a_nan_in_candle_color_is_read_as_no_colour_rather_than_crashing():
    tint = np.array(["#2fe0a8"] * 7 + [np.nan], dtype=object)
    spec = emsl.chart(frame(8), candle_color=tint).spec()
    assert spec["candles"]["color"]["k"][-1] == 0
    json.dumps(spec, allow_nan=False)


def test_a_per_bar_band_colour_becomes_a_palette_like_every_other_mark():
    colors = ["#ff0000"] * 4 + ["#00ff00"] * 4
    spec = emsl.chart(
        frame(8), Band(np.arange(8, dtype=float), 3.0, panel="x", color=colors)
    ).spec()
    entry = [s for s in spec["series"] if s["kind"] == "band"][0]
    assert entry["color"]["p"] == [None, "#ff0000", "#00ff00"]


def test_a_line_with_a_fill_becomes_an_area_that_is_actually_drawn():
    # the fill reached the spec and nothing read it, so it was accepted, shipped
    # and never painted
    from emsl import _chart

    spec = emsl.chart(
        frame(8), Line(np.arange(8, dtype=float), "f", fill=("#00000000", "#000000ff"))
    ).spec()
    assert spec["series"][0]["fill"] == ["#00000000", "#000000ff"]
    core = _chart._asset("core.js")
    assert "spec.fill" in core, "the renderer must read the fill it is given"


def test_a_line_refuses_a_per_bar_fill_and_names_the_substitute():
    with pytest.raises(TypeError) as excinfo:
        Line(np.arange(8, dtype=float), fill=["#ff0000"] * 8)
    assert "Band" in str(excinfo.value)


def test_a_per_bar_band_fill_becomes_a_palette():
    colors = ["#ff0000"] * 4 + ["#00ff00"] * 4
    spec = emsl.chart(
        frame(8), Band(np.arange(8, dtype=float), 3.0, panel="x", fill=colors)
    ).spec()
    entry = [s for s in spec["series"] if s["kind"] == "band"][0]
    assert entry["fill"]["p"] == [None, "#ff0000", "#00ff00"]
    assert entry["fill"]["k"] == [1, 1, 1, 1, 2, 2, 2, 2]


def test_a_per_bar_band_fill_of_the_wrong_length_raises():
    with pytest.raises(ValueError):
        emsl.chart(frame(8), Band(np.arange(8, dtype=float), 3.0, panel="x",
                                  fill=["#ff0000"] * 5))


def test_a_gradient_of_three_stops_is_still_a_gradient():
    stops = ("#000000", "#111111", "#222222")
    spec = emsl.chart(frame(8), Band(np.arange(8, dtype=float), 3.0, panel="x",
                                     fill=stops)).spec()
    entry = [s for s in spec["series"] if s["kind"] == "band"][0]
    assert entry["fill"] == list(stops)


def test_the_band_primitive_cuts_its_shape_where_a_per_bar_fill_changes():
    # one canvas path carries one fillStyle, so a two coloured region is two
    # shapes that have to meet rather than one gradient
    from emsl import _chart

    band = _chart._asset("primitives/band.js")
    assert "perBar" in band
    assert "colorAt(spec.fill" in band


def test_the_band_primitive_reads_the_palette_it_is_handed():
    # a palette object assigned to strokeStyle is a canvas no-op, so a per-bar
    # band colour used to be accepted, encoded correctly, and then painted in
    # the default colour with nothing anywhere saying so
    from emsl import _chart

    assert "colorAt(" in _chart._asset("primitives/band.js")


def test_a_per_bar_band_colour_of_the_wrong_length_raises():
    with pytest.raises(ValueError):
        emsl.chart(
            frame(8),
            Band(np.arange(8, dtype=float), 3.0, panel="x", color=["#ff0000"] * 3),
        )


def test_volume_keeps_the_precision_its_own_panel_advertises():
    data = frame(8)
    data["volume"] = np.linspace(0.00432, 0.00711, 8)
    spec = emsl.chart(data).spec()
    panel = [p for p in spec["panels"] if p.get("volume")][0]
    assert len(set(spec["vol"]["v"])) == 8, "rounding flattened the column"
    assert panel["digits"] >= 4


def test_a_panel_name_carrying_markup_is_refused():
    # controls.js writes the name into a DOM attribute, which payload escaping
    # cannot reach because JSON.parse restores the original
    with pytest.raises(ValueError):
        emsl.chart(frame(8), Line(np.arange(8, dtype=float), panel='"><script>'))


def test_focus_never_names_a_bar_outside_the_frame():
    assert emsl.chart(frame(8), focus=1).spec()["focus"] == [6.0, 7.0]
    assert emsl.chart(frame(8), focus=(7, 7)).spec()["focus"][1] <= 7.0


def test_naming_a_panel_twice_in_panels_is_refused():
    with pytest.raises(ValueError) as excinfo:
        emsl.chart(frame(8), panels=[Panel("equity", weight=2),
                                     Panel("equity", scale="log")])
    assert "twice" in str(excinfo.value)


def test_chart_and_chart_defaults_enforce_the_same_height_floor():
    with pytest.raises(ValueError):
        emsl.chart(frame(8), height=40)


def test_height_sizes_a_notebook_cell_and_a_saved_file_uses_the_window():
    # one file has to be right on a laptop and on a wide monitor, and another
    # page embedding it should size it from the outside
    tall = emsl.chart(frame(8), height=900)
    short = emsl.chart(frame(8), height=200)
    assert tall.spec()["height"] == 900 and short.spec()["height"] == 200
    assert "height: 100%" in tall._html()
    # the number reaches the document only as a value show() reads back out
    assert '"height":900' in tall._html()
    assert len(tall._html()) == len(short._html())


def test_a_list_valued_palette_entry_is_validated_rather_than_stringified():
    # str(["#a", "#b"]) contains nothing _UNSAFE looks for, so the repr used to
    # pass the check and ship where the renderer needs three entries
    previous = emsl.chart_defaults()
    try:
        emsl.chart_defaults(palette={"dark": {"selected": ["#000000", "#111111"]}})
        theme = emsl.chart(frame(8)).spec()["theme"]
        assert theme["dark"]["selected"] == ["#000000", "#111111"]
        with pytest.raises(ValueError):
            emsl.chart_defaults(palette={"dark": {"selected": ["</style>"]}})
    finally:
        emsl.chart_defaults(theme=previous["theme"], height=previous["height"])
        emsl._chart._DEFAULTS["palette"] = previous["palette"]


def test_chart_defaults_changes_nothing_when_one_argument_is_bad():
    previous = emsl.chart_defaults()
    with pytest.raises(ValueError):
        emsl.chart_defaults(theme="light", height=40)
    assert emsl.chart_defaults() == previous


def test_an_integer_index_is_refused_rather_than_read_as_epoch_seconds():
    data = frame(8)
    data.index = np.arange(1000, 1008)
    with pytest.raises(TypeError):
        emsl.chart(data)


def test_a_tz_aware_frame_still_charts():
    assert emsl.chart(frame(8).tz_localize("UTC")).spec()["n"] == 8


def test_a_tz_aware_frame_warns_about_nothing():
    # the conversion used to go through an object array of Timestamp, which numpy
    # warns about because it has no representation for a zone
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        emsl.chart(frame(8).tz_localize("America/New_York")).spec()


def test_an_offset_zone_names_the_same_instant_as_its_utc_equivalent():
    aware = frame(8).tz_localize("UTC").tz_convert("America/New_York")
    assert emsl.chart(aware).spec()["t"] == emsl.chart(frame(8)).spec()["t"]


def test_the_log_guard_names_the_bar_a_short_series_is_drawn_on():
    # a T-1 series starts at bar 1, so the array index is one less than the bar
    values = np.arange(1, 8, dtype=float)
    values[2] = 0.0
    with pytest.raises(ValueError) as excinfo:
        emsl.chart(frame(8), Line(values, "f", panel="x"),
                   panels=[Panel("x", scale="log")])
    assert "bar 3" in str(excinfo.value)


def test_the_log_guard_sees_a_level_and_a_bands_scalar_edge():
    with pytest.raises(ValueError):
        emsl.chart(frame(8), Line(np.arange(1, 9, dtype=float), "f", panel="x"),
                   Level(0.0, panel="x"), panels=[Panel("x", scale="log")])
    with pytest.raises(ValueError):
        emsl.chart(frame(8),
                   Band(np.arange(1, 9, dtype=float), 0.0, panel="y"),
                   panels=[Panel("y", scale="log")])


def test_a_bands_lower_edge_widens_the_panel_extent():
    upper = np.full(8, 10.0)
    lower = np.full(8, -500.0)
    spec = emsl.chart(frame(8), Band(upper, lower, panel="x")).spec()
    panel = [p for p in spec["panels"] if p["name"] == "x"][0]
    assert panel["digits"] <= 1


def test_panels_configures_and_never_creates():
    names = [p["name"] for p in
             emsl.chart(frame(8), panels=[Panel("momentom", weight=3)]).spec()["panels"]]
    assert "momentom" not in names


def test_a_mark_named_after_an_automatic_panel_does_not_land_on_it():
    # _place has already refused this from the candles; letting the label become
    # a registry key would put it back on them
    spec = emsl.chart(frame(8), Line(np.full(8, 5000.0), "price")).spec()
    assert spec["series"][0]["panel"] != "price"


def test_a_label_carrying_markup_is_stripped_when_it_becomes_a_panel_name():
    # the caller wrote a legend label, not a panel name, so the derived key is
    # cleaned rather than refused; the label itself still travels intact
    spec = emsl.chart(frame(8), Line(np.full(8, 5000.0), "</script>squeeze")).spec()
    assert spec["series"][0]["name"] == "</script>squeeze"
    panel = spec["series"][0]["panel"]
    assert not set(panel) & set("<>\";{}")
    assert "squeeze" in panel


def test_a_wrong_length_bare_array_says_which_one_it_was():
    with pytest.raises(ValueError) as excinfo:
        emsl.chart(frame(8), np.arange(8, dtype=float), np.arange(3, dtype=float))
    assert "series 1" in str(excinfo.value)


# --------------------------------------------------------------- the golden

def key_paths(node, prefix=""):
    if isinstance(node, dict):
        out = []
        for key in node:
            out += key_paths(node[key], f"{prefix}.{key}" if prefix else key)
        return out
    if isinstance(node, list):
        if not node:
            return [prefix + "[]"]
        # every element, not the first: series is a union of six mark kinds, and
        # reading element zero recorded the keys of a Line and nothing else, so
        # renaming a key on any of the other five passed this test untouched
        out = []
        for item in node:
            out += key_paths(item, prefix + "[]")
        return out
    return [prefix]


def test_the_document_carries_exactly_the_keys_the_renderer_reads():
    # the javascript half cannot be run in this gate and its whole contract is
    # the key set, so a renamed key would pass every other test here and draw
    # nothing. Regenerate it with dev/golden.py, in the same commit that bumps
    # the spec's schema number
    path = pathlib.Path(__file__).parent / "chart_schema.json"
    golden = json.loads(path.read_text(encoding="utf-8"))
    assert sorted(set(key_paths(everything().spec()))) == golden


def test_a_marker_and_a_markers_carry_the_same_keywords_into_the_document():
    # one primitive draws both, so a keyword one branch writes and the other drops
    # is a difference with no cause. The golden could not catch this: it records
    # the keys that were there, so an absent one reads as the contract (ADR 0089)
    mask = np.zeros(8, dtype=bool)
    mask[3] = True
    spec = emsl.chart(frame(8), [
        Marker(3, text="label", color="#ff0000"),
        Markers(mask, text="label", color="#ff0000"),
    ]).spec()
    single, plural = spec["series"][0], spec["series"][1]
    assert single["text"] == plural["text"] == "label"
    assert single["color"] == plural["color"] == "#ff0000"
    assert set(single) == set(plural)


def test_a_marker_given_neither_keyword_spends_no_bytes_on_them():
    spec = emsl.chart(frame(8), Marker(3)).spec()
    entry = spec["series"][0]
    assert "text" not in entry and "color" not in entry


def test_a_forced_close_is_marked_in_the_trade_payload():
    # the golden covers everything() and everything() never dies, so this key is
    # conditional and has to be asserted here instead. A forced close reads as an
    # ordinary exit otherwise, and the leverage that caused it goes unread (0091)
    data = sank()
    dead = emsl.backtest.Backtester(
        data, market="perp", quote=100.0, leverage=10.0,
        fee_taker=0.0, fee_maker=0.0,
    ).run(BuyAtOpen())
    assert dead.bust
    rows = emsl.chart(data, run=dead).spec()["trades"]
    assert any(row.get("liq") for row in rows)

    ordinary = emsl.chart(frame(8), run=run()).spec()["trades"]
    assert ordinary
    assert not any("liq" in row for row in ordinary)


def test_the_drawdown_panel_falls_from_the_running_peak_not_from_the_balance():
    # ADR 0042 seeds the peak from the opening balance and then RUNS it, the way
    # the engine does. The fixture behind that never makes a new high before its
    # worst bar, so the peak IS the opening balance throughout and a plain anchor
    # on `initial` returns the identical number. A run that doubles and then gives
    # it all back is the shape that separates them (ADR 0108)
    close = np.array([100.0, 100.0, 200.0, 100.0])
    data = pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0, "close": close,
         "volume": np.full(4, 1000.0)},
        index=pd.date_range("2024-01-01", periods=4, freq="1h"),
    )
    result = emsl.backtest.Backtester(
        data, market="spot", fee_taker=0.0, fee_maker=0.0
    ).run(BuyAtOpen())
    # 50 units bought at 100, so the account rides to 15,000 and back to 10,000
    assert list(result.equity_curve) == pytest.approx([10_000.0, 15_000.0, 10_000.0])
    fallen = emsl.chart(data, run=result).spec()["drawdown"]["v"]
    # a third given back from the peak, where an anchor on the opening balance
    # reports a run that never fell at all. The tolerance is the track's own
    # display rounding and nothing looser: the error being hunted is 33 points
    assert min(fallen) == pytest.approx(-100.0 / 3.0, abs=1e-3)
    assert result.stats["max_drawdown_pct"] == pytest.approx(100.0 / 3.0, abs=1e-9)


def test_a_line_ships_the_values_it_was_handed_not_the_panels_rounding():
    # the module already records this for the candles: a Level(70) beside a coin
    # priced 0.000012 dragged the panel from eight digits to two and rounded every
    # candle to 0.0. A line rounded to the PANEL's digits instead of its own extent
    # is that same defect one mark over, and across the whole file the only value
    # ever asserted on a series' v is a single 2.0, an integer that survives
    # rounding to any number of places (ADR 0108)
    feature = np.full(8, 1.2345e-05)
    spec = emsl.chart(frame(8), Line(feature, "tiny", panel="price")).spec()
    line = next(s for s in spec["series"] if s.get("name") == "tiny")
    assert line["v"][0] == pytest.approx(1.2345e-05, rel=1e-9)
    assert line["v"][0] != 0.0
