"""Turn a frame, your arrays and a run into one self-contained HTML document.

Three steps and no fourth: build a compact spec dict, substitute it and the
vendored renderer into a template, and hand the string to a notebook cell or a
file. A cell's output is stored in the notebook, so a chart drawn this way is
still there, and still interactive, when the notebook is reopened with no kernel
behind it. There is no server, no build step and no bundler (ADR 0041).

This module computes nothing about your data. The one number it derives is the
drawdown of the engine's own equity curve, which is a second view of a number the
engine already returned rather than a statistic about your series. It does not
import pandas, pyarrow or IPython at module level.
"""

from __future__ import annotations

import copy
import functools
import html
import json
import math
import pathlib
import re
import warnings

import numpy as np

from ._data import prepare
from .plot import Line, Panel, _Mark, _color

__all__ = ["chart", "chart_defaults", "Chart"]

_HERE = pathlib.Path(__file__).parent
_STATIC = _HERE / "_static"

# an explicit ordered tuple rather than a directory glob: nothing here executes at
# load, so the order is documentation, but a glob would silently ship any file
# that happened to land in _static (ADR 0043)
_ASSETS = (
    "core.js",
    "primitives/band.js",
    "primitives/background.js",
    "primitives/callout.js",
    "legend.js",
    "trades.js",
    "controls.js",
)

_PLACEHOLDER = re.compile(r"__[A-Z_]+__")

# a value that lands inside a <style> block or a canvas call rather than in the
# JSON payload, so escaping cannot save it and it is refused instead. A title is
# not in here: that one is escaped on both paths it travels
_UNSAFE = re.compile(r"[<>\";{}]")

_WEIGHTS = {"price": 5.0, "volume": 1.0, "equity": 2.0, "drawdown": 1.2}
_AUTO = ("price", "volume", "equity", "drawdown")

# a notebook stores a cell's output verbatim, so a chart's data is paid for once
# in the .ipynb and again in every copy of it. Measured rather than guessed from
# the bar count, because the width of a row depends on how many series are on it
_BIG_BYTES = 8_000_000

_PALETTES = {
    "dark": {
        "surface": "#070d13", "plane": "#05090e", "grid": "#131c26",
        "axis": "#1f2a36", "ink": "#e9ebee", "ink2": "#aab1bb",
        "muted": "#717a85", "hairline": "rgba(255,255,255,0.16)",
        "up": "#cdd5df", "down": "#39434f", "win": "#2fe0a8", "loss": "#ff5470",
        "s1": "#4d9fff", "s4": "#8b97a5",
        "selected": ["rgba(47,224,168,0.24)", "rgba(47,224,168,0.10)",
                     "rgba(47,224,168,0.02)"],
    },
    "light": {
        "surface": "#ffffff", "plane": "#eef1f5", "grid": "#e3e8ee",
        "axis": "#c8d0d9", "ink": "#070d13", "ink2": "#48525e",
        "muted": "#78828e", "hairline": "rgba(7,13,19,0.14)",
        "up": "#2b3441", "down": "#97a2af", "win": "#00996b", "loss": "#e0243f",
        "s1": "#1a73e8", "s4": "#6b7684",
        "selected": ["rgba(0,153,107,0.20)", "rgba(0,153,107,0.09)",
                     "rgba(0,153,107,0.02)"],
    },
}

_DEFAULTS = {"theme": "dark", "height": 660, "palette": None}

# the four that decide whether a strategy is worth keeping. Formatting lives here
# rather than in the renderer, so the seam holds: Python says what the numbers
# are and what they read as, and the JavaScript only draws the pairs
_HEADLINE = ("total_return_pct", "sharpe", "max_drawdown_pct", "num_trades")

_STAT_LABELS = {
    "total_return_pct": ("return", "{:+.1f}%"),
    "net_profit_pct": ("net", "{:+.1f}%"),
    "cagr_pct": ("cagr", "{:+.1f}%"),
    "sharpe": ("sharpe", "{:.2f}"),
    "sortino": ("sortino", "{:.2f}"),
    "calmar": ("calmar", "{:.2f}"),
    "max_drawdown_pct": ("max dd", "{:.1f}%"),
    "volatility_pct": ("vol", "{:.1f}%"),
    "exposure_pct": ("exposure", "{:.1f}%"),
    "win_rate": ("win rate", "{:.0%}"),
    "profit_factor": ("profit factor", "{:.2f}"),
    "avg_trade_pct": ("avg trade", "{:+.2f}%"),
    "num_trades": ("trades", "{:.0f}"),
    "num_fills": ("fills", "{:.0f}"),
}

_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# a calendar unit is not a fixed number of seconds, so each is a band of days
# rather than a value. Without them the same monthly series reads 30d over one
# window and 31d over the next, and a yearly one 365d or 366d depending on which
# leap years it happens to span: true, unstable, and useless. Longest first, and
# every band is narrow enough that nothing but the unit it names falls inside.
# A gap between two bands, 45 days say, is reported in days, because there is no
# name for it and inventing one would be worse than saying what it is
_CALENDAR = (
    (365, 366, "1y"),
    (181, 184, "6M"),
    (89, 92, "3M"),
    (59, 62, "2M"),
    (28, 31, "1M"),
    (7, 7, "1w"),
)

_NEEDS_TIME = (
    "chart needs timestamps, so it takes a DataFrame with a DatetimeIndex or a "
    "parquet path; the other entry points run fine on a bare array and leave "
    "trades tick-only, but a chart cannot omit its x axis"
)

# a stored candle file usually carries its time as a column rather than as an
# index, so the frame that raises above is nearly always one set_index away from
# working. Naming the column and writing the line out is the difference between a
# correct refusal and a useful one
_TIME_COLUMNS = ("timestamp", "time", "datetime", "date", "open_time", "ts")

_EPOCH_UNITS = (
    (1e18, "ns"),
    (1e15, "us"),
    (1e12, "ms"),
    (0.0, "s"),
)

_bundled = None


def _time_hint(frame):
    # returns the set_index line that would fix this frame, or None if there is no
    # column that could plausibly be its clock
    columns = getattr(frame, "columns", None)
    if columns is None:
        return None
    found = next((c for c in _TIME_COLUMNS if c in columns), None)
    if found is None:
        return None
    column = frame[found]
    if getattr(column, "ndim", 1) != 1:
        return None
    if column.dtype.kind == "M":
        return f"frame.set_index({found!r})"
    if column.dtype.kind not in "iuf" or not len(column):
        return None
    values = np.abs(np.asarray(column, dtype=np.float64))
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    biggest = float(values.max())
    unit = next(u for threshold, u in _EPOCH_UNITS if biggest >= threshold)
    return (
        f"frame.set_index(pandas.to_datetime(frame[{found!r}], unit={unit!r}))"
    )


def _needs_time(frame):
    # a hint is a courtesy wrapped around the real error, so it is never allowed
    # to become the error: anything it trips over is dropped and the plain
    # refusal goes out instead
    try:
        hint = _time_hint(frame)
    except Exception:
        hint = None
    if hint is None:
        return TypeError(_NEEDS_TIME)
    return TypeError(
        f"{_NEEDS_TIME}. This frame keeps its clock in a column, so it is one "
        f"line away: {hint}"
    )


@functools.lru_cache(maxsize=None)
def _asset(name):
    # cached for the same reason _bundle is: the assets are static for the life of
    # the process, and the template, the stylesheet and the 196 KB renderer were
    # being re-read off disk on every single render
    return (_STATIC / name).read_text(encoding="utf-8")


def _bundle():
    # read once: the assets are static for the life of the process and a chart
    # drawn in a loop would otherwise re-read every file per call
    global _bundled
    if _bundled is None:
        _bundled = "\n".join(
            "// ---- " + name + "\n" + _asset(name) for name in _ASSETS
        )
    return _bundled


def _ipython():
    try:
        from IPython.display import HTML, display
    except ImportError as exc:
        raise ImportError(
            "IPython is required for Chart.show; use Chart.save(path) to write a "
            "file instead, which needs nothing"
        ) from exc
    return display, HTML


def _in_notebook():
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and hasattr(shell, "kernel")


def _safe(value, where):
    text = str(value)
    if _UNSAFE.search(text):
        raise ValueError(
            f"{where} cannot contain <, >, quotes, semicolons or braces; "
            f"got {value!r}"
        )
    return text


def _safe_value(value, where):
    # 'selected' is a list of gradient stops rather than one colour, and str() on
    # a list is a repr carrying no character _UNSAFE looks for, so it would pass
    # the check and ship as one long string where the renderer needs three entries
    if isinstance(value, (list, tuple)):
        return [_safe(item, where) for item in value]
    return _safe(value, where)


def _height(value):
    # one floor for both entry points: chart(height=) and chart_defaults(height=)
    # are the same setting, and one refusing what the other accepts is a
    # difference no caller can see
    value = int(value)
    if value < 120:
        raise ValueError(
            f"height must be at least 120 pixels, got {value}; a shorter chart "
            f"has no room for an axis"
        )
    return value


def _start_for(size, num_bars, label, ahead=0):
    # the bar index a series of this length begins at (ADR 0037)
    if size == num_bars:
        return 0
    if size == num_bars - 1:
        return 1
    if ahead and size == num_bars + ahead:
        return 0
    lengths = (
        f"{num_bars}, one per bar, or {num_bars - 1}, drawn from bar 1 like an "
        f"equity curve or a diff"
    )
    if ahead:
        lengths += f", or {num_bars + ahead}, running {ahead} bars past the last candle"
    raise ValueError(f"{label} has {size} values, frame has {num_bars}; expected {lengths}")


def _align(values, num_bars, label, places=8, ahead=0):
    # a length T array starts at bar 0 and a length T-1 array at bar 1. That
    # second case is not a claim about when a value could first be known: a number
    # recorded inside Strategy.next was known on bar i, and the right repair for
    # one of those is to pad it to T. It is that drawing a short array early is
    # the visual signature of lookahead, and that the engine records an equity
    # point only on a real advance, so its entry i is the account at tick i+1
    # (ADR 0037). The leading non-finite run folds into i0, so a rolling warm-up
    # costs no bytes and the alignment stays exact; an interior gap stays in place
    # as None, because a hole in the middle is information (ADR 0038)
    start = _start_for(len(values), num_bars, label, ahead)
    finite = np.isfinite(values)
    if not finite.any():
        return start, []
    lead = int(np.argmax(finite))
    kept = finite[lead:]
    # rounding through numpy and letting tolist build the Python floats is the whole
    # of the work when nothing is missing, which is the ordinary case: an equity
    # curve, a volume column and any rolling feature are gapless once warmed up.
    # Only a series with a hole in it pays for the per-bar None substitution.
    # Unlike a price, a volume or a mean is not tick-aligned, so a value sitting on
    # a tie can round the other way from round(); places is two past what the panel
    # displays, so the disagreement is a unit of something already below reading
    rounded = np.round(values[lead:], places)
    if kept.all():
        return start + lead, rounded.tolist()
    tail = [
        None if not ok else v
        for v, ok in zip(rounded.tolist(), kept.tolist())
    ]
    return start + lead, tail


def _track(values, num_bars, label, places=8, ahead=0):
    # rounded to two places past what the panel displays. A price is stored as a
    # double and a naive round(x, 8) keeps every one of its thirteen significant
    # digits, which on a 100k bar frame is megabytes of notebook nobody can read
    i0, v = _align(np.asarray(values, dtype=np.float64), num_bars, label, places, ahead)
    return {"i0": i0, "v": v}


def _encode_color(value, num_bars, label, ahead=0):
    # a scalar colour passes through; a per-bar one becomes a palette and indices,
    # which also puts it through the length contract every other array obeys
    if value is None or isinstance(value, str):
        return value
    start = _start_for(len(value), num_bars, label, ahead)
    palette = [None]
    seen = {}
    keys = []
    for item in value:
        if item is None:
            keys.append(0)
            continue
        at = seen.get(item)
        if at is None:
            at = len(palette)
            palette.append(item)
            seen[item] = at
        keys.append(at)
    return {"i0": start, "p": palette, "k": keys}


def _truthy(value):
    # a NaN is not a condition being met. numpy.array([nan]).astype(bool) is True,
    # so a mask padded with NaN reports the event it was looking for, and the doc
    # that teaches the padding is the thing that creates the bug. pandas.NA
    # answers neither way and raises on bool(), and a value that cannot answer the
    # question is by definition not a condition being met
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    try:
        return bool(value)
    except TypeError:
        return False


def _spans(values, fill, num_bars, label, ahead=0):
    # run-length a label or mask array into spans plus the fill table they index,
    # so a background costs one entry per region rather than one per bar
    start = _start_for(len(values), num_bars, label, ahead)
    if isinstance(fill, dict):
        table = {}
        fills = []
        for key, stops in fill.items():
            # a label mapped to nothing shades nothing, which is what an absent
            # key already means. It must not reach fills: the renderer builds its
            # gradient cache by mapping over every entry
            if stops is None:
                continue
            table[key] = len(fills)
            fills.append(stops)

        def at(value):
            try:
                return table.get(value)
            except TypeError:
                return None
    else:
        fills = [fill if fill else ["rgba(139,151,165,0.14)"]]

        def at(value):
            return 0 if _truthy(value) else None

    spans = []
    open_at = -1
    key = None
    for offset in range(len(values) + 1):
        here = at(values[offset]) if offset < len(values) else None
        if here != key:
            if key is not None and open_at >= 0:
                spans.append([start + open_at, start + offset, key])
            key = here
            open_at = offset if here is not None else -1
    return spans, fills


def _extent(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return None
    return float(finite.min()), float(finite.max())


def _digits(extent):
    if extent is None:
        return 2
    span = extent[1] - extent[0]
    if span <= 0:
        span = abs(extent[1]) or 1.0
    places = int(math.ceil(-math.log10(span / 1000.0)))
    return max(0, min(8, places))


def _edges(mark, num_bars, index=0, ahead=0):
    # every value a mark puts on its panel's axis, as (label, values, start),
    # where start is the bar the first entry is drawn on or None for a value that
    # is not per-bar. One accessor, because the panel extent and the log guard
    # both need this answer and used to disagree: a band's lower edge moved
    # neither, and a level moved the extent but not the guard
    label = _label(mark, index)
    if mark.kind in ("line", "histogram"):
        start = _start_for(len(mark.values), num_bars, label, ahead)
        out = [(label, mark.values, start)]
        if mark.kind == "histogram":
            out.append((f"{label} base", np.full(1, mark.base), None))
        return out
    if mark.kind == "band":
        start = _start_for(len(mark.upper), num_bars, label, ahead)
        out = [(f"{label} upper", mark.upper, start)]
        if mark.lower is None:
            out.append((f"{label} lower", np.full(1, mark.level), None))
        else:
            below = _start_for(len(mark.lower), num_bars, label, ahead)
            out.append((f"{label} lower", mark.lower, below))
        return out
    if mark.kind == "level":
        return [(label, np.full(1, mark.value), None)]
    return []


def _place(extent, price):
    # overlay unless doing so costs the candles more than half the price panel.
    # Range overlap was the obvious rule and it carries no information about
    # units: an oscillator bounded 0 to 100 overlaps a coin priced 20 to 90, rides
    # the candles and stretches the axis. This rule cannot flatten them, because
    # the threshold is the guarantee (ADR 0039)
    if extent is None or price is None or price[1] <= price[0]:
        return True
    span = max(price[1], extent[1]) - min(price[0], extent[0])
    return span <= 0 or (price[1] - price[0]) / span >= 0.5


def _dead(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return "every value is NaN"
    if float(finite.min()) == float(finite.max()):
        return f"every value is {float(finite.min())}"
    return None


def _panel_key(name, index):
    # a panel created from a mark's label is an identity as well as a header, and
    # controls.js writes it into a DOM attribute. The derived form is stripped
    # rather than refused, because the caller wrote a legend label and not a panel
    # name; a panel= they typed themselves still raises. A label that collides
    # with an automatic panel falls back too, so a series named "price" cannot
    # land on the candles _place just refused it from
    key = _UNSAFE.sub("", name or "").strip()
    return key if key and key not in _AUTO else f"series {index}"


def _label(mark, index):
    # an unnamed mark reports its position, because every bare array is wrapped as
    # a Line and "Line" is the same word for all four of the ones just passed
    if mark.name:
        return f"{type(mark).__name__} {mark.name!r}"
    return f"series {index}"


def _flatten(arg):
    # the one ambiguous case: a list whose every element is a number is one
    # series, and anything else is a list of series
    if isinstance(arg, (list, tuple)):
        if arg and all(isinstance(x, (int, float, np.number)) for x in arg):
            return [arg]
        out = []
        for item in arg:
            out.extend(_flatten(item))
        return out
    return [arg]


def _is_result(item):
    return (
        hasattr(item, "stats")
        and hasattr(item, "equity_curve")
        and hasattr(item, "trades")
    )


def _collect(args):
    marks = []
    result = None
    spare = 0
    for arg in args:
        for item in _flatten(arg):
            if item is None:
                continue
            if isinstance(item, _Mark):
                marks.append(item)
            elif _is_result(item):
                if result is None:
                    result = item
                else:
                    spare += 1
            else:
                marks.append(Line(item))
    if spare:
        warnings.warn(
            f"chart draws one run and {spare + 1} were given; the first is used. "
            f"To compare runs, pass the others' equity_curve as Line(..., "
            f"panel='equity')",
            RuntimeWarning,
            stacklevel=3,
        )
    return marks, result


def chart_defaults(theme=None, height=None, palette=None):
    """Set how every later chart looks, and return the previous settings.

    Explicit keywords rather than ``**options``, so ``chart_defaults(them="dark")``
    is a ``TypeError`` at the call rather than a key stored and never read. Only
    appearance lives here, never what gets drawn. A document is built when
    ``chart`` is called, so changing a default afterwards changes no chart that
    already exists and no file already saved.

    ``height`` is the height of a notebook cell, in pixels. A saved file sizes
    itself to the window instead, so one file is right on any screen. There is no
    width anywhere: a chart fills whatever contains it.
    """
    previous = dict(_DEFAULTS)
    pending = {}
    if theme is not None:
        if theme not in _PALETTES:
            raise ValueError(f"theme must be 'dark' or 'light', got {theme!r}")
        pending["theme"] = theme
    if height is not None:
        pending["height"] = _height(height)
    if palette is not None:
        merged = {}
        for mode, values in palette.items():
            if mode not in _PALETTES:
                raise ValueError(
                    f"palette keys must be 'dark' or 'light', got {mode!r}"
                )
            merged[mode] = {k: _safe_value(v, "palette") for k, v in values.items()}
        pending["palette"] = merged
    # applied only once every argument has passed. A call that raised halfway
    # through used to leave the theme set and the height not, and the caller never
    # got the previous values back to restore from
    _DEFAULTS.update(pending)
    return previous


def _assign(marks, ohlc):
    # which panel every mark lands on, in argument order
    price = (float(np.nanmin(ohlc[:, 2])), float(np.nanmax(ohlc[:, 1])))
    names = []
    for index, mark in enumerate(marks):
        if mark.panel is not None:
            names.append(mark.panel)
            continue
        if mark.kind in ("line", "histogram", "band"):
            values = mark.upper if mark.kind == "band" else mark.values
            extent = _extent(values)
            if _place(extent, price):
                names.append("price")
                if extent is not None:
                    # the price extent widens as each overlay is accepted, so an
                    # early one can push a later one out. Order dependent, and
                    # deterministic for a given argument order
                    price = (min(price[0], extent[0]), max(price[1], extent[1]))
            else:
                names.append(_panel_key(mark.name, index))
        else:
            # a level, a marker or a background with no panel= goes on the price
            # panel, and a level is the one that can vanish there: the renderer
            # draws it with createPriceLine, which the autoscale never reads, so
            # Level(70) beside a coin at 61,000 is off screen with nothing said.
            # The oscillator idiom that puts a line and its thresholds together
            # is the most common chart in the world and it failed in silence
            if mark.kind == "level" and not _within(mark.value, price):
                warnings.warn(
                    f"Level({mark.value:g}) has no panel=, so it went on the "
                    f"price panel, where the candles run {price[0]:g} to "
                    f"{price[1]:g} and it cannot be seen; pass the panel= of "
                    f"the series it belongs with",
                    RuntimeWarning,
                    stacklevel=3,
                )
            names.append("price")
    return names


def _within(value, extent):
    # generous, because a level just outside the range is a target and belongs on
    # the chart; a level orders of magnitude away is on the wrong panel
    low, high = extent
    room = (high - low) or abs(high) or 1.0
    return low - room <= value <= high + room


def _order(names, config, has_volume, has_result):
    order = ["price"]
    if has_volume:
        order.append("volume")
    for name in names:
        if name not in order:
            order.append(name)
    if has_result:
        for name in ("equity", "drawdown"):
            if name not in order:
                order.append(name)
    # panels= configures and never creates: a typo would otherwise manufacture a
    # blank pane with an axis, a legend and nothing on it, while the panel the
    # caller meant stayed unconfigured

    # panels= fixes the relative order of the panels it names without moving them
    # relative to the panels it does not: take the slots those panels already
    # occupy and write them back in the listed order. Naming one panel therefore
    # provably changes no order at all
    listed = [p.name for p in config if p.name in order]
    if len(listed) > 1:
        slots = sorted(order.index(name) for name in listed)
        for slot, name in zip(slots, listed):
            order[slot] = name
    return order


def _build_panels(order, config, marks, names, num_bars, ohlc, vol, equity, dd,
                  ahead=0):
    by_name = {p.name: p for p in config}
    if by_name.get("price") is not None and not by_name["price"].show:
        raise ValueError(
            "panel 'price' carries the candles, so it cannot be hidden; drop the "
            "frame's other panels instead or pass a shorter height"
        )

    hidden = {p.name for p in config if not p.show}
    panels = []
    for name in order:
        if name in hidden:
            continue
        cfg = by_name.get(name)
        pool = []
        if name == "price":
            pool.append((float(np.nanmin(ohlc[:, 2])), float(np.nanmax(ohlc[:, 1]))))
        if name == "volume" and vol is not None:
            pool.append(_extent(vol))
        if name == "equity" and equity is not None:
            pool.append(_extent(equity))
        if name == "drawdown" and dd is not None:
            pool.append(_extent(dd))
        for index, (mark, on) in enumerate(zip(marks, names)):
            if on != name:
                continue
            pool.extend(
                _extent(values)
                for _, values, _ in _edges(mark, num_bars, index, ahead)
            )
        pool = [p for p in pool if p is not None]
        extent = (min(p[0] for p in pool), max(p[1] for p in pool)) if pool else None

        panel = {
            # controls.js writes this into a DOM attribute, which the payload
            # escaping cannot reach because JSON.parse restores the original
            "name": _safe(name, "panel name"),
            "weight": float(cfg.weight) if cfg else _WEIGHTS.get(name, 1.5),
            "scale": cfg.scale if cfg else "linear",
            "digits": _digits(extent),
        }
        if cfg is not None and cfg.range is not None:
            panel["range"] = [cfg.range[0], cfg.range[1]]
        if name == "price":
            panel["candles"] = True
        if name == "volume" and vol is not None:
            panel["volume"] = True
        panels.append(panel)
    return panels, hidden


def _check_log(panels, marks, names, ohlc, tracks, ahead=0):
    num_bars = len(ohlc)
    for panel in panels:
        if panel["scale"] != "log":
            continue
        pools = []
        if panel.get("candles"):
            pools.append(("the candles", ohlc[:, 2], 0))
        for key, values in tracks.items():
            if key == panel["name"] and values is not None:
                start = _start_for(len(values), num_bars, key, ahead)
                pools.append((key, values, start))
        for index, (mark, on) in enumerate(zip(marks, names)):
            if on == panel["name"]:
                pools.extend(_edges(mark, num_bars, index, ahead))
        for label, values, start in pools:
            arr = np.asarray(values, dtype=np.float64)
            bad = np.flatnonzero(np.isfinite(arr) & (arr <= 0))
            if not bad.size:
                continue
            # the index into the array is not the bar it is drawn on: a T-1 series
            # starts at bar 1, and naming a bar one early is exactly the confusion
            # the alignment contract exists to prevent (ADR 0037)
            at = int(bad[0])
            where = "" if start is None else f" at bar {start + at}"
            raise ValueError(
                f"panel {panel['name']!r} is logarithmic but {label} reaches "
                f"{float(arr[at])}{where}; use scale='linear' or shift the "
                f"series above zero"
            )


def _focus(value, num_bars, result, ahead=0):
    last = num_bars + ahead - 1.0
    if value is None:
        return None
    if isinstance(value, dict) and "entry_tick" in value:
        pad = max(18.0, max(1, value["exit_tick"] - value["entry_tick"]) * 0.7)
        low, high = value["entry_tick"] - pad, value["exit_tick"] + pad
    elif isinstance(value, (int, np.integer)):
        # the last N bars, plus whatever was projected past them: asking for a
        # window and being shown less than the chart holds is never what was meant
        low, high = num_bars - int(value), last
    else:
        low, high = float(value[0]), float(value[1])
    # clamped rather than sliced, so every marker stays on its own bar whatever
    # was asked for. The cap goes last: widening to a one bar minimum first and
    # capping second used to emit a range one bar past the end of the axis
    high = min(max(float(high), float(low) + 1.0), last)
    low = max(0.0, min(float(low), high - 1.0))
    return [low, high]


def _trades(result, num_bars):
    if result is None or not result.trades:
        return []
    out = []
    for index, trade in enumerate(result.trades):
        # an explicit key list, never dict(trade): Backtester._stamp_times adds
        # entry_time and exit_time as numpy.datetime64, which json.dumps refuses,
        # and the renderer reads its timestamps off the shared axis anyway
        out.append({
            "i": index + 1,
            "in": int(trade["entry_tick"]),
            "out": int(trade["exit_tick"]),
            "side": str(trade["side"]),
            "size": round(float(trade["size"]), 8),
            "px_in": round(float(trade["entry_price"]), 8),
            "px_out": round(float(trade["exit_price"]), 8),
            "fees": round(float(trade["fees"]), 8),
            "net": round(float(trade["net_pnl"]), 8),
            "bars": int(trade["bars_held"]),
        })
    return out


def _headline(stats, chosen):
    keys = _HEADLINE if chosen is None else list(chosen)
    out = []
    for key in keys:
        if key not in stats:
            raise KeyError(
                f"stats names {key!r}, which the run does not report; choose "
                f"from {sorted(stats)}"
            )
        value = float(stats[key])
        label, form = _STAT_LABELS.get(key, (key, "{:.2f}"))
        out.append([label, form.format(value) if math.isfinite(value) else "n/a"])
    return out


def _theme(mode, palette):
    out = {"mode": mode, "font": _FONT}
    for key in ("dark", "light"):
        merged = dict(_PALETTES[key])
        if palette and key in palette:
            merged.update(palette[key])
        out[key] = merged
    return out


def _render(spec, title):
    # one re.sub pass rather than chained .replace(): the vendored renderer is
    # 196 KB of minified JavaScript, and a chained replace would happily expand a
    # placeholder that appeared inside text already substituted
    payload = json.dumps(spec, separators=(",", ":"), allow_nan=False)
    if len(payload) > _BIG_BYTES:
        # gzip was measured before writing this: 3.1x raw, and 2.3x once base64
        # armours it back into a document, which turns 45 MB into 17 MB and solves
        # nothing while costing an async bootstrap and a modern-browser floor.
        # Fewer bars is the only real answer, so the warning names one
        bars = spec.get("n", 0)
        keep = max(1_000, 10 ** (len(str(bars // 20)) - 1) * 2)
        warnings.warn(
            f"this chart carries {len(payload) // 1_000_000} MB across "
            f"{bars:,} bars, and a notebook stores that verbatim in every copy "
            f"of itself. It will draw, but it will be slow to open and awkward "
            f"to send: chart a slice with frame.tail({keep:_}), or resample to a "
            f"coarser bar and chart that",
            RuntimeWarning,
            stacklevel=4,
        )
    # json.dumps does not escape <, so a series named "</script>" would close the
    # script element and take the document with it
    payload = (
        payload.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    values = {
        "__TITLE__": html.escape(title or "emsl chart"),
        "__CSS__": _asset("chart.css"),
        "__LIBRARY__": _asset("lightweight-charts.js"),
        "__BUNDLE__": _bundle(),
        "__SPEC__": payload,
    }
    return _PLACEHOLDER.sub(lambda m: values[m.group(0)], _asset("chart.html"))


class Chart:
    """One built chart: the document, and the spec it was built from."""

    def __init__(self, spec, title):
        self._spec = spec
        self._title = title
        self._document = None

    def __repr__(self):
        return f"Chart(bars={self._spec['n']}, panels={len(self._spec['panels'])})"

    def spec(self):
        """The underlying document as a plain dict.

        A fresh copy each call, so a caller can pull it apart without changing
        what gets drawn. This is what the tests assert against, because a spec
        says what is drawn where a screenshot only says it looked the same.
        """
        return copy.deepcopy(self._spec)

    def save(self, path):
        """Write one self-contained HTML file and return ``path``.

        For when somebody wants the picture without the notebook. The renderer,
        the styles and the data are all inside the one file, so it opens in a
        browser on a machine that has no Python. Missing parent directories are
        created.

        The document sizes itself to the window rather than to ``height``, which
        is a notebook cell setting. That is what makes one file right on a laptop
        and on a wide monitor, and it is what lets another page embed it in an
        iframe of the page's own choosing.
        """
        target = pathlib.Path(path)
        if target.parent != pathlib.Path(""):
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._html(), encoding="utf-8")
        return path

    def show(self, height=None):
        """Draw the chart in the current notebook cell.

        ``height`` overrides the cell height for this one display, without
        rebuilding anything: the document is already built and only the frame
        around it reads a height. Width is always the width of the cell, because
        a cell is a container of unknown size and a pinned width is how a chart
        ends up cut off on one screen and short of the edge on another.

        An iframe with ``srcdoc`` rather than a bare ``HTML()``: JupyterLab strips
        script tags out of raw HTML output, while an iframe is its own document
        and always runs, and a cell's output is stored inside the ``.ipynb``, so
        the chart survives the notebook being saved, closed and reopened with no
        kernel and no network.

        Displays rather than returning, so a chart drawn inside a loop draws every
        time instead of only on the last expression, and returns ``None``. Raises
        ``RuntimeError`` outside a notebook, where there is nothing to draw into
        and a silent no-op is the expensive kind of nothing.
        """
        if not _in_notebook():
            raise RuntimeError(
                "show() draws into a notebook cell and there is no notebook here; "
                "use save(path) to write a file you can open in a browser"
            )
        display, HTML = _ipython()
        srcdoc = html.escape(self._html(), quote=True)
        height = self._spec["height"] if height is None else _height(height)
        # wrapped, because IPython warns on output that opens with a bare iframe
        # and points at IPython.display.IFrame, which takes a src url and so
        # cannot carry its document inline, which is the whole point of this one
        display(HTML(
            f'<div class="emsl-chart">'
            f'<iframe srcdoc="{srcdoc}" style="width:100%;height:{height}px;'
            f'border:0"></iframe></div>'
        ))

    def _html(self):
        if self._document is None:
            self._document = _render(self._spec, self._title)
        return self._document


def chart(
    frame=None, *args, marks=None, run=None, panels=None, focus=None,
    candle_color=None, theme=None, height=None, title=None, future=0, stats=None,
    trades=True,
):
    """Draw ``frame`` as candles, with your arrays and a run on top of it.

    ``frame`` is a pandas DataFrame with ``open``, ``high``, ``low``, ``close``,
    ``volume`` and a DatetimeIndex, or a parquet path to one. The positional
    arguments are matched by type: a ``BacktestResult`` becomes the run, a mark
    from ``emsl.plot`` becomes itself, a bare array becomes a ``Line``, and a list
    is flattened. Returns a ``Chart``; call ``show()`` or ``save(path)`` on it.

    Every one of those can be named instead, and the whole call written with no
    positional arguments at all, which is worth doing in anything anyone else will
    read: ``marks`` takes one mark or a list of them, ``run`` takes the
    ``BacktestResult``, and both say at the call site what type dispatch otherwise
    leaves you to infer. Naming ``run`` also means a mistyped argument is refused
    here rather than drawn as a line.

    ``title`` and ``stats`` are drawn above the plot, which is what turns a saved
    file into a document rather than a picture: the title on the left and a few
    numbers on the right. ``stats`` names the keys, defaulting to the four that
    decide whether a strategy is worth keeping, and ``stats=[]`` shows none.

    Everything a run brings can be turned off. ``trades=False`` drops the arrows
    and the table under the chart. The equity and drawdown panels are ordinary
    panels, so ``Panel(name="drawdown", weight=0)`` removes one, and a hidden
    panel ships no data at all rather than merely going unpainted. Equity as a
    percentage is ``Panel(name="equity", scale="percent")``, which is a different
    question from the drawdown panel: percent rescales the same curve, while
    drawdown measures the fall from the running peak and sits at zero on every
    new high.

    ``future`` extends the time axis by that many intervals past the last candle,
    so a series computed from real bars but belonging ahead of them has somewhere
    to go: an Ichimoku cloud, a projected pivot, a regression carried forward.
    The candles are untouched, because everything that paints one loops the candle
    count, so no bar that never traded can appear. With it, a third length is
    accepted, ``T + future``, drawn from bar 0.

    Nothing here computes anything about your data, and nothing is fetched at any
    point: the result is one HTML document carrying its own renderer (ADR 0041).
    """
    if frame is None:
        raise TypeError(
            "chart needs a frame to draw; pass the candles as the first argument "
            "or as frame="
        )
    if run is not None and not _is_result(run):
        raise TypeError(
            f"run must be the BacktestResult a Backtester returned, got "
            f"{type(run).__name__}; a series belongs in marks="
        )
    args = list(args)
    if marks is not None:
        args.append(marks)
    if run is not None:
        args.append(run)
    candles, index = prepare(frame)
    # index_of returns None only for a RangeIndex, so an integer or a float index
    # arrives here as a plain array and a cast would read bar ids as epoch
    # seconds. The numeric kinds are rejected by name rather than by requiring
    # "M", because a tz-aware index is neither
    stamps = None if index is None else np.asarray(index)
    if stamps is None or stamps.dtype.kind in "biufc":
        raise _needs_time(frame)
    num_bars = int(candles.shape[0])
    if num_bars == 0:
        raise ValueError(
            "frame has no bars; a chart needs at least one row, so check the "
            "filter or the slice that produced it"
        )
    if stamps.dtype == object:
        # index_of hands back frame.index.to_numpy(), and a tz-aware index comes
        # out of that as an object array of Timestamp. Casting one warns that
        # numpy has no representation for a zone, and every Timestamp already
        # carries its own epoch nanoseconds in UTC, which is the number wanted
        try:
            times = np.array([t.value for t in stamps], dtype=np.int64)
        except AttributeError:
            raise TypeError(_NEEDS_TIME) from None
        times = times // 1_000_000_000
    else:
        times = stamps.astype("datetime64[s]").astype("int64")
    if num_bars > 1 and int(np.diff(times).min()) <= 0:
        at = int(np.argmin(np.diff(times)))
        raise ValueError(
            f"frame has two bars at the same timestamp, at bars {at} and {at + 1}; "
            f"the renderer keys on time, so one of them would be dropped silently"
        )

    future = int(future)
    if future < 0:
        raise ValueError(f"future must be zero or positive, got {future}")
    if future:
        if num_bars < 2:
            raise ValueError(
                "future needs at least two bars, because the interval it extends "
                "the axis by is read off the gaps between them"
            )
        # the axis grows and the candles do not. Everything that paints a candle
        # loops the candle count, so nothing can invent a bar that never traded
        step = int(np.median(np.diff(times)))
        times = np.concatenate(
            [times, times[-1] + step * np.arange(1, future + 1, dtype=np.int64)]
        )

    marks, result = _collect(args)
    config = list(panels or [])
    seen = set()
    for panel in config:
        if not isinstance(panel, Panel):
            raise TypeError(f"panels= takes emsl.plot.Panel objects, got {panel!r}")
        if panel.name in seen:
            # list.index returns the first match, so a repeated name yields the
            # same slot twice and the second write clobbers the first, deleting a
            # panel and duplicating another
            raise ValueError(
                f"panels= names {panel.name!r} twice; configure a panel once, "
                f"with one Panel, or the second wins on some settings and the "
                f"first on others"
            )
        seen.add(panel.name)

    if result is not None:
        ran_on = len(result.equity_curve) + 1 if len(result.equity_curve) else None
        if ran_on is None and result.trades:
            ran_on = max(int(t["exit_tick"]) for t in result.trades) + 1
        if ran_on is not None and ran_on != num_bars:
            raise ValueError(
                f"result was produced on {ran_on} bars but {num_bars} were given; "
                f"trade rows carry bar indices into the original series, so every "
                f"marker would land on the wrong bar. To zoom, keep the full frame "
                f"and pass focus={num_bars}, or focus=(start, end). To chart a "
                f"slice for real, re-run the backtest on it"
            )

    ohlc = candles[:, :4]
    broken = np.flatnonzero(~np.isfinite(ohlc).all(axis=1))
    if broken.size:
        at = int(broken[0])
        raise ValueError(
            f"frame has a non-finite price at bar {at}; a candle with no value "
            f"cannot be drawn, so drop the row or fill it before charting"
        )
    volume = candles[:, 4]
    dead = _dead(volume)
    if dead is not None:
        warnings.warn(
            f"the volume column is not usable ({dead}), so no volume panel is "
            f"drawn; a dead feed renders as a perfectly plausible flat chart",
            RuntimeWarning,
            stacklevel=2,
        )
        volume = None

    equity = None
    drawdown = None
    if result is not None and len(result.equity_curve):
        equity = np.asarray(result.equity_curve, dtype=np.float64)
        # the curve holds a point per advance, so the balance the run opened with is
        # not in it. Taking the peak from the curve alone loses every fall from that
        # balance, and the engine takes it from the balance (stats.rs:64-66), so a
        # first bar that lost money drew a flat zero panel under a headline reading
        # the engine's own number. Capped for the same reason the engine caps it: a
        # bust past zero is 100% down, not more (ADR 0042)
        start = getattr(result, "initial", None)
        opened = equity if start is None else np.concatenate(([float(start)], equity))
        peak = np.maximum.accumulate(opened)[-len(equity):]
        with np.errstate(divide="ignore", invalid="ignore"):
            fallen = np.maximum((equity / peak - 1.0) * 100.0, -100.0)
            drawdown = np.where(peak > 0, fallen, np.nan)

    names = _assign(marks, ohlc)
    order = _order(names, config, volume is not None, result is not None)
    panels_out, hidden = _build_panels(
        order, config, marks, names, num_bars, ohlc, volume, equity, drawdown,
        future,
    )
    _check_log(panels_out, marks, names, ohlc, {
        "equity": equity, "drawdown": drawdown, "volume": volume,
    }, future)

    shown = {p["name"]: p["digits"] for p in panels_out}

    def places(name, fallback=4):
        return shown.get(name, fallback) + 2

    series = []
    for index, (mark, on) in enumerate(zip(marks, names)):
        if on in hidden:
            continue
        label = _label(mark, index)
        entry = {"kind": mark.kind, "panel": on}
        if mark.name:
            entry["name"] = mark.name
        if mark.kind in ("line", "histogram"):
            digits = _digits(_extent(mark.values))
            entry.update(
                _track(mark.values, num_bars, label, digits + 2, future)
            )
            entry["color"] = _encode_color(
                mark.color, num_bars, f"{label} color", future
            )
            entry["digits"] = digits
            if mark.kind == "line":
                entry["width"] = mark.width
                entry["style"] = mark.style
                entry["step"] = mark.step
                if mark.fill:
                    entry["fill"] = mark.fill
            else:
                entry["base"] = mark.base
        elif mark.kind == "band":
            entry.update(
                _track(mark.upper, num_bars, f"{label} upper", places(on), future)
            )
            if mark.lower is None:
                entry["level"] = mark.level
            else:
                lower = _track(
                    mark.lower, num_bars, f"{label} lower", places(on), future
                )
                # both edges are one shape, so they share one start
                start = max(entry["i0"], lower["i0"])
                entry["v"] = entry["v"][start - entry["i0"]:]
                entry["v2"] = lower["v"][start - lower["i0"]:]
                entry["i0"] = start
            stops = mark.fill or ["rgba(139,151,165,0.14)"]
            if len(stops) > 3:
                # one colour per bar, which the renderer draws by splitting the
                # shape where the colour changes rather than by one gradient
                entry["fill"] = _encode_color(
                    stops, num_bars, f"{label} fill", future
                )
            else:
                # only="below" runs downward on screen, and flipping it here is
                # what lets the renderer always paint stop 0 on the reference edge
                if mark.only == "below" and len(stops) > 1:
                    stops = list(reversed(stops))
                entry["fill"] = stops
            entry["only"] = mark.only
            entry["color"] = _encode_color(
                mark.color, num_bars, f"{label} color", future
            )
        elif mark.kind == "level":
            entry["value"] = mark.value
            entry["width"] = mark.width
            entry["style"] = mark.style
            if mark.color:
                entry["color"] = mark.color
        elif mark.kind == "marker":
            if not 0 <= mark.bar < num_bars + future:
                where = (f"{num_bars} bars" if not future else
                         f"{num_bars} bars and {future} projected past them")
                raise ValueError(
                    f"Marker bar {mark.bar} is outside the chart, which has {where}"
                )
            # both kinds ship a list, so the renderer holds one primitive and one
            # loop rather than one attached primitive per glyph
            entry["kind"] = "marker"
            entry["bars"] = [mark.bar]
            entry["shape"] = mark.shape
            entry["offset"] = mark.offset
            if mark.value is not None:
                entry["values"] = [mark.value]
        elif mark.kind == "markers":
            # one flatnonzero, used for both the bars and their anchors, so the
            # two lists cannot come out of step
            start = _start_for(mark.mask.size, num_bars, label, future)
            where = np.flatnonzero(mark.mask)
            entry["kind"] = "marker"
            entry["bars"] = (where + start).tolist()
            entry["shape"] = mark.shape
            entry["offset"] = mark.offset
            if mark.value is not None:
                if np.ndim(mark.value) == 0:
                    entry["values"] = [float(mark.value)] * where.size
                else:
                    # a null anchor falls back to the bar's own extreme, which is
                    # what an absent one does, so a glyph on a bar its anchor had
                    # not warmed up for lands on the candle rather than vanishing
                    picked = np.asarray(mark.value)[where]
                    entry["values"] = [
                        None if not np.isfinite(v) else round(float(v), 8)
                        for v in picked.tolist()
                    ]
            if mark.text:
                entry["text"] = mark.text
            if mark.color:
                entry["color"] = mark.color
        elif mark.kind == "background":
            spans, fills = _spans(mark.values, mark.fill, num_bars, label, future)
            entry["spans"] = spans
            entry["fills"] = fills
        entry = {k: v for k, v in entry.items() if v is not None}
        series.append(entry)

    mode = theme if theme is not None else _DEFAULTS["theme"]
    if mode not in _PALETTES:
        raise ValueError(f"theme must be 'dark' or 'light', got {mode!r}")

    spec = {
        # 2 adds stats.funding_paid, which arrives because the spec mirrors the
        # whole stats dict and the engine now reports the funding a perp run paid
        "schema": 2,
        "n": num_bars,
        # both built by numpy rather than by a comprehension. The candles are the
        # largest thing in the document by far, and rounding each of the four
        # values in Python was two thirds of the cost of a chart: tolist() already
        # materialises T lists, and the comprehension built T more on top of them.
        # numpy rounds by scaling, so it can break a tie the other way from
        # round(); a price is a whole number of ticks, which puts it nowhere near
        # a tie, and 2.48M real prices came out identical. Dropping the rounding
        # was tried and rejected: it costs nothing on exchange data, whose reprs
        # are already short, and 2.6x on a frame somebody computed.
        # The precision comes from the candles' own range and NOT from the price
        # panel's digits, which every mark sharing that panel contributes to. A
        # Level(70) beside a coin priced 0.000012 dragged the panel from eight
        # digits to two and rounded every candle to 0.0, and one Line reaching a
        # million took XRP from 4990 distinct candles to 52. What is displayed is
        # a panel setting; what is stored must depend on the data alone
        "t": times.tolist(),
        "ohlc": np.round(ohlc, _digits(_extent(ohlc)) + 2).tolist(),
        "candles": {"panel": "price"},
        "panels": panels_out,
        "series": series,
        # an empty list is what the renderer already reads as nothing to draw, so
        # trades=False costs no branch there and no bytes here
        "trades": _trades(result, num_bars) if trades else [],
        "theme": _theme(mode, _DEFAULTS["palette"]),
        "height": _height(height) if height is not None else _DEFAULTS["height"],
    }
    if title is not None:
        spec["title"] = str(title)
    if num_bars > 1:
        gap = int(np.median(np.diff(times)))
        spec["interval"] = _interval(gap)
    if volume is not None and "volume" not in hidden:
        spec["vol"] = _track(volume, num_bars, "volume", places("volume"))
    if candle_color is not None:
        # the one colour that arrives as raw user input rather than through a
        # mark, so it needs the same coercion every other colour already gets: a
        # NaN entry would otherwise become a palette entry and fail at save()
        spec["candles"]["color"] = _encode_color(
            _color(candle_color, "candle_color"), num_bars, "candle_color", future
        )
    if equity is not None and "equity" not in hidden:
        spec["equity"] = _track(equity, num_bars, "equity_curve", places("equity"))
    if drawdown is not None and "drawdown" not in hidden:
        spec["drawdown"] = _track(drawdown, num_bars, "drawdown", places("drawdown"))
    if result is not None and result.stats:
        # a stat is legitimately infinite (profit_factor with no losing trade) or
        # NaN (sharpe over a flat curve), and JSON carries neither, so those
        # become null rather than a crash on the way out
        spec["stats"] = {
            key: (float(v) if math.isfinite(float(v)) else None)
            for key, v in result.stats.items()
        }
        headline = _headline(result.stats, stats)
        if headline:
            spec["headline"] = headline
    elif stats:
        raise ValueError(
            "stats names the numbers to show above the chart, and there is no "
            "run to take them from; pass a BacktestResult too"
        )
    window = _focus(focus, num_bars, result, future)
    if window is not None:
        spec["focus"] = window

    # the coerced title, not the argument: html.escape wants a string and a
    # non-string one used to reach it from inside save(), long after this returned
    return Chart(spec, spec.get("title"))


def _interval(seconds):
    # the caption the legend shows, read off the axis rather than declared, so it
    # cannot disagree with the bars it labels. The median gap rather than the
    # first: a weekend, an overnight session break and a hole in the data all
    # leave it alone
    days = seconds / 86400.0
    for low, high, name in _CALENDAR:
        if low <= days <= high:
            return name
    for size, suffix in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds >= size and seconds % size == 0:
            return f"{seconds // size}{suffix}"
    return f"{seconds}s"
