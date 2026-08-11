"""Tests for emsl.plot: the marks are value objects that copy their arrays, check
what they can check without knowing the frame, and compute nothing.

No pandas here, so this file always runs. Anything needing a frame lives in
test_chart.py, because the length of a series is the one thing a mark cannot know
about itself.
"""

import numpy as np
import pytest

from emsl.plot import (
    Background, Band, Histogram, Level, Line, Marker, Panel, Recorder, at_bar,
    at_next, ramp,
)


def values():
    return np.array([10.0, 20.0, 30.0])


def test_a_line_copies_its_values_so_a_later_mutation_does_not_change_it():
    source = values()
    line = Line(source, "sma")
    source[0] = 999.0
    assert line.values[0] == 10.0


def test_a_line_rejects_a_two_dimensional_array():
    with pytest.raises(ValueError):
        Line(np.zeros((3, 2)))


def test_a_line_rejects_an_empty_array():
    with pytest.raises(ValueError):
        Line(np.array([]))


def test_a_line_rejects_values_that_are_not_numbers():
    with pytest.raises(TypeError):
        Line(["a", "b", "c"])


def test_a_line_rejects_an_unknown_style_and_names_what_it_was_given():
    with pytest.raises(ValueError) as excinfo:
        Line(values(), style="dotdash")
    assert "dotdash" in str(excinfo.value)


def test_a_line_rejects_a_width_below_one():
    with pytest.raises(ValueError):
        Line(values(), width=0)


def test_a_line_keeps_a_per_bar_colour_array_as_strings_and_nones():
    line = Line(values(), color=np.where([True, False, True], "#ff0000", None))
    assert line.color == ["#ff0000", None, "#ff0000"]


def test_a_line_reads_a_numpy_string_array_of_colours():
    # numpy.where with two colours gives a <U7 array of numpy.str_, which
    # subclasses str, and the encoder has to accept it
    line = Line(values(), color=np.where([True, False, True], "#ff0000", "#00ff00"))
    assert line.color == ["#ff0000", "#00ff00", "#ff0000"]


def test_a_line_rejects_a_colour_array_holding_something_that_is_not_a_colour():
    with pytest.raises(TypeError):
        Line(values(), color=[1, 2, 3])


def test_level_refuses_a_nan_value():
    with pytest.raises(ValueError):
        Level(float("nan"))


def test_level_refuses_an_infinite_value():
    with pytest.raises(ValueError):
        Level(float("inf"))


def test_level_accepts_a_numpy_float():
    assert Level(np.float64(70.0)).value == 70.0


def test_a_band_keeps_a_scalar_lower_edge_as_a_level():
    band = Band(values(), 70)
    assert band.level == 70.0
    assert band.lower is None


def test_a_band_keeps_a_numpy_scalar_lower_edge_as_a_level():
    band = Band(values(), np.float64(70.0))
    assert band.level == 70.0
    assert band.lower is None


def test_a_band_refuses_two_edges_of_different_lengths():
    with pytest.raises(ValueError) as excinfo:
        Band(values(), np.array([1.0, 2.0]))
    assert "3" in str(excinfo.value) and "2" in str(excinfo.value)


def test_a_band_rejects_an_unknown_only():
    with pytest.raises(ValueError):
        Band(values(), 70, only="over")


def test_a_band_normalises_a_single_fill_colour_into_a_list():
    assert Band(values(), 70, fill="#112233").fill == ["#112233"]


def test_a_band_fill_of_more_than_three_entries_is_read_as_one_per_bar():
    # a chart of three bars or fewer is not a chart, so a gradient and a per-bar
    # fill cannot collide in practice. The two forms differ in what they accept:
    # a per-bar fill may hold None, meaning shade nothing there
    band = Band(values(), 70, fill=["#ff0000", None, "#ff0000", None])
    assert band.fill == ["#ff0000", None, "#ff0000", None]

    with pytest.raises(TypeError):
        Band(values(), 70, fill=("#ff0000", None))


def test_a_band_rejects_an_empty_fill():
    with pytest.raises(ValueError):
        Band(values(), 70, fill=[])


def test_a_panel_refuses_a_range_whose_low_is_not_below_its_high():
    with pytest.raises(ValueError):
        Panel("momentum", range=(100, 0))


def test_a_panel_refuses_a_negative_weight():
    with pytest.raises(ValueError):
        Panel("momentum", weight=-1)


def test_a_panel_is_hidden_by_a_word_rather_than_by_a_number():
    assert Panel("volume", show=False).show is False
    assert Panel("volume").show is True


def test_a_zero_weight_names_the_word_that_replaced_it():
    # it used to mean hidden, and somebody's old code still says it
    with pytest.raises(ValueError) as excinfo:
        Panel("volume", weight=0)
    assert "show=False" in str(excinfo.value)


def test_a_panel_rejects_an_unknown_scale():
    with pytest.raises(ValueError):
        Panel("equity", scale="logarithmic")


def test_a_marker_takes_its_bar_positionally():
    assert Marker(1204).bar == 1204


def test_a_marker_accepts_a_numpy_integer_bar():
    assert Marker(np.int64(1204)).bar == 1204


def test_a_marker_rejects_a_negative_bar():
    with pytest.raises(ValueError):
        Marker(-1)


def test_a_marker_rejects_an_unknown_shape_and_names_the_legal_set():
    with pytest.raises(ValueError) as excinfo:
        Marker(10, shape="triangleup")
    message = str(excinfo.value)
    assert "triangleup" in message
    assert "arrow_up" in message and "circle" in message


def test_a_histogram_refuses_a_non_finite_base():
    with pytest.raises(ValueError):
        Histogram(values(), base=float("nan"))


def test_background_accepts_a_boolean_mask():
    assert Background(np.array([True, False, True])).values == [True, False, True]


def test_background_accepts_a_label_array_and_a_fill_map():
    background = Background(
        np.array(["calm", "wild", "calm"], dtype=object),
        fill={"wild": "#ff0000"},
    )
    assert background.fill == {"wild": ["#ff0000"]}


def test_ramp_returns_the_first_stop_at_the_low_and_the_last_at_the_high():
    out = ramp([0.0, 1.0], "#000000", "#ffffff")
    assert out == ["#000000", "#ffffff"]


def test_ramp_takes_its_stops_as_a_named_list():
    loose = ramp([0.0, 0.5, 1.0], "#000000", "#888888", "#ffffff")
    named = ramp(values=[0.0, 0.5, 1.0], colors=["#000000", "#888888", "#ffffff"])
    assert named == loose


def test_ramp_refuses_stops_given_both_ways():
    with pytest.raises(TypeError):
        ramp([0.0, 1.0], "#000000", "#ffffff", colors=["#000000", "#ffffff"])


def test_ramp_ignores_a_warmup_when_it_computes_its_own_domain():
    # a plain min over a rolling feature is NaN, and every colour would collapse
    # to one with nothing on screen saying so
    nan = float("nan")
    out = ramp([nan, nan, nan, 0.0, 1.0], "#000000", "#ffffff")
    assert out[:3] == [None, None, None]
    assert out[3] == "#000000" and out[4] == "#ffffff"


def test_ramp_leaves_a_non_finite_value_uncoloured():
    out = ramp([0.0, float("nan"), 1.0, float("inf")], "#000000", "#ffffff")
    assert out[1] is None and out[3] is None


def test_ramp_pins_its_domain_when_one_is_given():
    # without the domain the two values would be the two ends of the ramp
    out = ramp([0.0, 1.0], "#000000", "#ffffff", domain=(-1.0, 2.0))
    assert out[0] != "#000000" and out[1] != "#ffffff"


def test_ramp_clamps_a_value_outside_a_pinned_domain():
    out = ramp([-5.0, 5.0], "#000000", "#ffffff", domain=(0.0, 1.0))
    assert out == ["#000000", "#ffffff"]


def test_ramp_needs_at_least_two_stops():
    with pytest.raises(ValueError):
        ramp([0.0, 1.0], "#000000")


def test_ramp_rejects_a_stop_that_is_not_hex():
    with pytest.raises(ValueError):
        ramp([0.0, 1.0], "black", "#ffffff")


def test_ramp_carries_alpha_through_when_a_stop_has_it():
    out = ramp([0.0, 1.0], "#00000000", "#ffffffff")
    assert out[0] == "#00000000"
    assert out[1] == "#ffffff"


def test_ramp_produces_no_more_colours_than_it_has_steps():
    # the palette encoder depends on this bound: a continuous ramp over a long
    # frame must not produce one palette entry per bar
    out = ramp(np.linspace(0.0, 1.0, 5000), "#000000", "#ffffff")
    assert len(set(out)) <= 128


def test_ramp_over_a_flat_series_gives_one_colour_rather_than_raising():
    out = ramp([2.0, 2.0, 2.0], "#000000", "#ffffff")
    assert len(set(out)) == 1


def test_ramp_over_an_all_nan_series_colours_nothing():
    nan = float("nan")
    assert ramp([nan, nan], "#000000", "#ffffff") == [None, None]


def test_ramp_rejects_a_domain_whose_low_is_not_below_its_high():
    with pytest.raises(ValueError):
        ramp([0.0, 1.0], "#000000", "#ffffff", domain=(1.0, 0.0))


def test_a_mark_repr_names_its_type_and_panel():
    assert "SMA 20" in repr(Line(values(), "SMA 20", panel="price"))


def test_a_fill_map_is_refused_on_a_mark_that_is_not_a_background():
    # list(dict) is the keys, every one of them a string, so this would otherwise
    # pass every check and shade the panel in a colour named "trend"
    with pytest.raises(TypeError):
        Line(values(), fill={"trend": "#4d9fff1f", "chop": "#ff54701f"})


def test_an_empty_panel_name_is_refused():
    # Panel("") raises, so accepting one here builds a panel that can never be
    # weighted, scaled or hidden
    with pytest.raises(ValueError):
        Line(values(), panel="")


def test_level_refuses_a_zero_width_the_way_line_does():
    with pytest.raises(ValueError):
        Level(70.0, width=0)


def test_level_refuses_a_per_bar_colour_and_names_the_substitute():
    # str() on an array gives its repr, numpy's truncating ellipsis and all,
    # which nothing downstream can tell apart from a colour
    with pytest.raises(TypeError) as excinfo:
        Level(70.0, color=np.array(["#ff0000", "#00ff00", "#ff0000"]))
    assert "Line" in str(excinfo.value)


def test_marker_refuses_a_per_bar_colour():
    with pytest.raises(TypeError):
        Marker(3, color=["#ff0000", "#00ff00"])


def test_level_refuses_a_colour_that_is_not_a_colour():
    with pytest.raises(TypeError):
        Level(70.0, color=object())


def test_marker_refuses_an_array_of_bars_and_names_the_idiom():
    # the first thing a plotshape() port reaches for; numpy's own message names
    # neither Marker nor the repair
    with pytest.raises(TypeError) as excinfo:
        Marker(np.flatnonzero(np.array([False, True, True])))
    message = str(excinfo.value)
    assert "Marker takes one bar" in message
    assert "Markers(mask=" in message


def test_at_bar_pads_the_tail_and_at_next_pads_the_head():
    recorded = [1.0, 2.0, 3.0]
    assert at_bar(recorded)[:3].tolist() == recorded
    assert np.isnan(at_bar(recorded)[3])
    assert np.isnan(at_next(recorded)[0])
    assert at_next(recorded)[1:].tolist() == recorded


def test_at_bar_pads_a_mask_with_false_rather_than_nan():
    # numpy.array([nan]).astype(bool) is True, so a float pad makes a mask
    # report the condition it was looking for on the one bar it knows nothing of
    mask = np.array([True, False, True])
    padded = at_bar(mask)
    assert padded.dtype == bool
    assert padded.tolist() == [True, False, True, False]
    assert int(np.append(mask.astype(float), np.nan).astype(bool).sum()) == 3
    assert int(padded.sum()) == 2


def test_at_next_pads_a_mask_with_false_too():
    padded = at_next(np.array([True, True]))
    assert padded.dtype == bool
    assert padded.tolist() == [False, True, True]


def test_a_recorder_places_at_bar_values_on_the_bar_they_were_recorded_on():
    log = Recorder(5)
    for bar in range(4):                      # next runs on every bar but the last
        log.at_bar({"tick_index": bar}, lev=float(bar))
    assert log["lev"].tolist()[:4] == [0.0, 1.0, 2.0, 3.0]
    assert np.isnan(log["lev"][4])


def test_a_recorder_places_at_next_values_one_bar_later():
    log = Recorder(5)
    for bar in range(4):
        log.at_next({"tick_index": bar}, liq=float(bar))
    assert np.isnan(log["liq"][0])
    assert log["liq"].tolist()[1:] == [0.0, 1.0, 2.0, 3.0]


def test_a_recorder_leaves_a_bar_it_never_reached_as_a_gap():
    # an early return during a warm-up is the ordinary case, and it must not
    # shift everything recorded after it
    log = Recorder(6)
    for bar in range(5):
        if bar < 2:
            continue
        log.at_bar(bar, value=float(bar))
    assert np.isnan(log["value"][0]) and np.isnan(log["value"][1])
    assert log["value"].tolist()[2:5] == [2.0, 3.0, 4.0]


def test_a_recorder_gives_one_value_per_bar_whatever_was_recorded():
    log = Recorder(9)
    for bar in range(8):
        log.at_bar(bar, a=1.0)
        if bar % 2 == 0:
            log.at_bar(bar, b=2.0)
    assert len(log["a"]) == 9 and len(log["b"]) == 9


def test_a_recorder_fills_an_unreached_boolean_bar_with_false():
    log = Recorder(5)
    for bar in range(3):
        log.at_bar(bar, flat=True)
    assert log["flat"].dtype == bool
    assert log["flat"].tolist() == [True, True, True, False, False]


def test_a_recorder_refuses_to_change_a_keys_alignment():
    log = Recorder(5)
    log.at_bar(0, lev=1.0)
    with pytest.raises(ValueError) as excinfo:
        log.at_next(1, lev=2.0)
    assert "at_bar" in str(excinfo.value) and "at_next" in str(excinfo.value)


def test_a_recorder_refuses_to_change_a_keys_kind():
    log = Recorder(5)
    log.at_bar(0, flag=True)
    with pytest.raises(TypeError):
        log.at_bar(1, flag=0.5)


def test_a_recorder_names_what_it_has_when_a_key_is_missing():
    log = Recorder(5)
    log.at_bar(0, lev=1.0)
    with pytest.raises(KeyError) as excinfo:
        log["leverage"]
    assert "lev" in str(excinfo.value)
    assert "lev" in log and "leverage" not in log
    assert log.keys() == ["lev"] and len(log) == 1


def test_a_recorder_hands_back_a_copy():
    log = Recorder(4)
    log.at_bar(0, lev=1.0)
    log["lev"][0] = 99.0
    assert log["lev"][0] == 1.0


def test_a_recorder_takes_an_engine_or_a_bar_count():
    class FakeEngine:
        data = np.zeros((7, 5))

        def reset(self):
            pass

    from_engine = Recorder(FakeEngine())
    from_engine.at_bar(0, v=1.0)
    assert len(from_engine["v"]) == 7

    from_count = Recorder(7)
    from_count.at_bar(0, v=1.0)
    assert len(from_count["v"]) == 7


def test_a_recorder_needs_at_least_one_bar():
    with pytest.raises(ValueError):
        Recorder(0)
