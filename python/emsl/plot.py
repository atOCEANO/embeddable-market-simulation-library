"""The marks a chart can carry, and the colour helper that feeds them.

Every class here is a plain value object: it copies its arrays so a later mutation
cannot change the picture, checks what it can check without knowing the frame, and
holds no reference to a chart. Nothing in this module computes anything. There is
no ``sma``, no ``rsi``, no ``zscore`` and no indicator registry, because a chart
that computes is a chart that can disagree with the run it is drawing (ADR 0042).
It also does not know how many bars the frame has, so the length of a series is
checked by ``chart`` rather than here (ADR 0037).

``ramp`` is the one function, and it turns numbers into colours so a line can
carry a second variable in its own colour.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "Panel", "Line", "Histogram", "Band", "Level", "Marker", "Markers",
    "Background", "Recorder", "ramp", "at_bar", "at_next",
]

_STYLES = ("solid", "dashed", "dotted")
_SHAPES = ("circle", "square", "arrow_up", "arrow_down")
_SCALES = ("linear", "log", "percent")
_SIDES = ("above", "below")

# past the point a screen can tell two shades apart, and it keeps the palette a
# fixed size no matter how many distinct values the input holds
_RAMP_STEPS = 128

# a fill is one colour, a (bottom, top) pair, or three stops. Anything longer is
# one colour per bar
_MAX_STOPS = 3


def _as_array(values, where):
    # pandas is duck-typed by module name before anything is imported, so an
    # unsupported argument on a pandas-free install is not sent after a
    # dependency that would not have helped (_data.py does the same)
    if type(values).__module__.split(".")[0] == "pandas":
        values = values.to_numpy()
    try:
        arr = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{where} must be numbers; pass a numpy array, a pandas Series, "
            f"or a list of floats"
        ) from exc
    if arr.ndim != 1:
        raise ValueError(f"{where} must be one dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{where} is empty")
    return arr


def _as_colors(values, where):
    # never coerced to float: numpy.where(mask, ramp(...), None) is the ordinary
    # way to write one of these and it lands as an object array holding str and
    # None, which a float cast would destroy
    if type(values).__module__.split(".")[0] == "pandas":
        values = values.to_numpy()
    arr = np.asarray(values, dtype=object)
    if arr.ndim != 1:
        raise ValueError(f"{where} must be one dimensional, got shape {arr.shape}")
    out = []
    for value in arr.tolist():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            out.append(None)
        elif isinstance(value, str):
            out.append(value)
        else:
            raise TypeError(
                f"{where} must hold colour strings or None, got {value!r}"
            )
    return out


def _color(value, where):
    if value is None or isinstance(value, str):
        return value
    return _as_colors(value, where)


def _one_color(value, where):
    # a mark that draws one shape takes one colour. str() on an array gives its
    # repr, numpy's truncating ellipsis and all, which is not a colour and which
    # nothing downstream can tell apart from one
    if value is None or isinstance(value, str):
        return value
    raise TypeError(
        f"{where} takes one colour, not one per bar; for a line whose colour "
        f"changes along it use Line(numpy.full(len(frame), value), color=...)"
    )


def _stops(value, where):
    # a fill becomes a list of colour strings, bottom of the shape first, so no
    # colour arithmetic and no notion of direction ever crosses into the renderer
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        # list(dict) is the keys, every one of them a string, so this would pass
        # every check below and shade the panel in a colour named "calm"
        raise TypeError(
            f"{where} takes a colour or a gradient of stops; a "
            f"{{label: colour}} map is only meaningful on Background"
        )
    stops = list(value)
    if not stops:
        raise ValueError(f"{where} is empty")
    if len(stops) > _MAX_STOPS:
        # more entries than a gradient can have is one per bar. A chart of three
        # bars or fewer is not a chart, so the two forms cannot collide in practice
        return _as_colors(value, where)
    for stop in stops:
        if not isinstance(stop, str):
            raise TypeError(f"{where} stops must be colour strings, got {stop!r}")
    return stops


def _finite(value, where):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{where} needs a finite value, got {value}")
    return value


def _one_of(value, allowed, where):
    if value not in allowed:
        raise ValueError(
            f"{where} must be one of {', '.join(repr(a) for a in allowed)}, "
            f"got {value!r}"
        )
    return value


def _hex_to_rgba(value, where):
    ok = isinstance(value, str) and value.startswith("#") and len(value) in (7, 9)
    if not ok:
        raise ValueError(
            f"{where} stops must be #rrggbb or #rrggbbaa hex strings, got {value!r}"
        )
    try:
        parts = tuple(int(value[i:i + 2], 16) for i in range(1, len(value), 2))
    except ValueError as exc:
        raise ValueError(f"{where} stop {value!r} is not valid hex") from exc
    return parts if len(parts) == 4 else parts + (255,)


def _rgba_to_hex(rgba):
    body = "".join(f"{c:02x}" for c in rgba[:3])
    return f"#{body}" if rgba[3] == 255 else f"#{body}{rgba[3]:02x}"


def _mix(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(4))


class _Mark:
    """Base for every mark: the panel it lives on, and an optional name."""

    kind = None

    def __init__(self, name, panel):
        self.name = None if name is None else str(name)
        if panel is None:
            self.panel = None
        else:
            self.panel = str(panel)
            if not self.panel.strip():
                # Panel('') raises, so accepting one here builds a panel that can
                # never be weighted, scaled or hidden, because the object needed
                # to name it cannot be constructed
                raise ValueError(
                    f"{type(self).__name__} panel needs a name; leave it out to "
                    f"place the mark automatically"
                )

    def __repr__(self):
        label = "" if self.name is None else f" {self.name!r}"
        panel = "" if self.panel is None else f" on {self.panel!r}"
        return f"<{type(self).__name__}{label}{panel}>"


class Panel:
    """Configuration for one panel: how tall, what scale, whether pinned.

    ``show=False`` removes the panel and everything drawn on it, and its data is
    left out of the document rather than merely going unpainted. ``weight`` is a
    stretch factor rather than pixels, so a layout holds at any size; ``height``
    means pixels and lives on ``chart``. ``scale`` is ``"linear"``, ``"log"`` or
    ``"percent"``, matching the A / L / % buttons the panel draws in its own axis,
    and ``range`` pins the axis to a fixed window.

    Naming a panel here configures it; it never creates or removes one, and it
    never moves it relative to a panel you did not name.
    """

    def __init__(self, name, *, show=True, weight=1, scale="linear", range=None):
        self.name = str(name)
        if not self.name:
            raise ValueError("a panel needs a name")
        self.show = bool(show)
        self.weight = float(weight)
        # hiding used to be weight=0, which is a puzzle rather than a word:
        # nobody looking for how to drop a panel searches for its stretch factor,
        # and zero is a legitimate value in that vocabulary overloaded to mean
        # something categorical. One spelling, and the old one names the new one
        if self.weight == 0.0:
            raise ValueError(
                f"panel {self.name!r} weight is zero, which used to mean hidden; "
                f"say so instead, with Panel(name={self.name!r}, show=False)"
            )
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError(
                f"panel {self.name!r} weight must be a positive number, "
                f"got {weight}"
            )
        self.scale = _one_of(scale, _SCALES, f"panel {self.name!r} scale")
        if range is None:
            self.range = None
        else:
            low, high = (float(x) for x in range)
            if not (math.isfinite(low) and math.isfinite(high)):
                raise ValueError(
                    f"panel {self.name!r} range must be finite, got {range}"
                )
            if low >= high:
                raise ValueError(
                    f"panel {self.name!r} range low {low} must be below high {high}"
                )
            self.range = (low, high)

    def __repr__(self):
        shown = "" if self.show else " hidden"
        return f"<Panel {self.name!r} weight={self.weight} scale={self.scale!r}{shown}>"


class Line(_Mark):
    """A line through your values, one point per bar.

    ``color`` is one colour or one per bar, so a line can carry a second variable
    in its own colour; ``fill`` shades from the line to the panel edge and takes a
    colour or a ``(bottom, top)`` pair. ``step`` holds each value until the next
    one, which is right for a position or a discrete action. A ``NaN`` is a gap,
    never a dropped row (ADR 0038).
    """

    kind = "line"

    def __init__(
        self, values, name=None, *, panel=None, color=None, fill=None,
        width=2, style="solid", step=False,
    ):
        super().__init__(name, panel)
        where = f"Line {name!r}" if name else "Line"
        self.values = _as_array(values, where)
        self.color = _color(color, f"{where} color")
        self.fill = _stops(fill, f"{where} fill")
        if self.fill is not None and len(self.fill) > _MAX_STOPS:
            # a line's fill is one area under one line, and an area carries one
            # gradient. Two colours means two regions, which is two marks
            raise TypeError(
                f"{where} fill takes a colour or a gradient, not one per bar; "
                f"for a region that changes colour use two conditional Bands"
            )
        if self.fill is not None and not isinstance(self.color, (str, type(None))):
            # a filled line is drawn as an area, and an area carries ONE line
            # colour: the renderer ignores a per-bar colour on it entirely. The
            # two together were accepted and drew a flat line while the legend
            # swatch reported a different colour at every bar, which is the one
            # thing the legend exists not to do. Neither half can be picked
            # without lying about the other, so the pair is refused (ADR 0076)
            raise TypeError(
                f"{where} takes a per-bar colour or a fill, not both: an area "
                f"carries one line colour and the ramp would be dropped while "
                f"the legend went on reporting it; drop fill, or pass a plain "
                f"colour, or draw the fill as a Band underneath"
            )
        self.width = int(width)
        if self.width < 1:
            raise ValueError(f"{where} width must be at least 1, got {width}")
        self.style = _one_of(style, _STYLES, f"{where} style")
        self.step = bool(step)


class Histogram(_Mark):
    """Columns from ``base`` to your values, one per bar."""

    kind = "histogram"

    def __init__(self, values, name=None, *, panel=None, color=None, base=0):
        super().__init__(name, panel)
        where = f"Histogram {name!r}" if name else "Histogram"
        self.values = _as_array(values, where)
        self.color = _color(color, f"{where} color")
        self.base = _finite(base, f"{where} base")


class Band(_Mark):
    """A shaded region between two edges.

    ``lower`` is an array or a single number, so a channel and a threshold are the
    same mark. ``only="above"`` or ``"below"`` makes the shading conditional: it
    exists solely where ``upper`` is past ``lower``, and its gradient runs from
    that edge toward each excursion's own extreme, which is what an overbought
    shading actually is. ``color`` strokes the edges and ``fill`` shades between.
    """

    kind = "band"

    def __init__(
        self, upper, lower, name=None, *, panel=None, fill=None, color=None,
        only=None,
    ):
        super().__init__(name, panel)
        where = f"Band {name!r}" if name else "Band"
        self.upper = _as_array(upper, f"{where} upper")
        if np.isscalar(lower) or isinstance(lower, (int, float)):
            self.lower = None
            self.level = _finite(lower, f"{where} lower")
        else:
            self.lower = _as_array(lower, f"{where} lower")
            self.level = None
            if self.lower.size != self.upper.size:
                raise ValueError(
                    f"{where} edges must be the same length; upper has "
                    f"{self.upper.size} values and lower has {self.lower.size}"
                )
        self.fill = _stops(fill, f"{where} fill")
        self.color = _color(color, f"{where} color")
        self.only = None if only is None else _one_of(only, _SIDES, f"{where} only")


class Level(_Mark):
    """A horizontal line at one value, drawn across the whole panel."""

    kind = "level"

    def __init__(
        self, value, name=None, *, panel=None, color=None, width=1,
        style="solid",
    ):
        super().__init__(name, panel)
        self.value = _finite(value, "Level")
        self.color = _one_color(color, "Level color")
        self.width = int(width)
        if self.width < 1:
            raise ValueError(f"Level width must be at least 1, got {width}")
        self.style = _one_of(style, _STYLES, "Level style")


class Marker(_Mark):
    """One annotation at one bar.

    ``value`` is where on the panel's own scale it anchors, and it is a price only
    on a price panel; leave it out and the marker sits at the bar's own extreme.
    ``offset`` is a signed distance in **pixels**, positive upward, so the glyph
    holds its distance from the bar at every zoom level instead of drifting as the
    scale changes.
    """

    kind = "marker"

    def __init__(
        self, bar, *, panel=None, value=None, text=None, shape="circle",
        offset=0, color=None,
    ):
        super().__init__(None, panel)
        try:
            self.bar = int(bar)
        except (TypeError, ValueError) as exc:
            # the first thing a plotshape() port reaches for is an array, and
            # numpy's own message names neither Marker nor the repair
            raise TypeError(
                f"Marker takes one bar, not an array; for a shape on every bar "
                f"meeting a condition use Markers(mask=...), which is the same "
                f"thing plotshape does. Got {bar!r}"
            ) from exc
        if self.bar < 0:
            raise ValueError(f"Marker bar must be zero or positive, got {bar}")
        self.value = None if value is None else _finite(value, "Marker value")
        self.text = None if text is None else str(text)
        self.shape = _one_of(shape, _SHAPES, "Marker shape")
        self.offset = int(offset)
        self.color = _one_color(color, "Marker color")


class Markers(_Mark):
    """A glyph on every bar a condition holds, which is Pine's ``plotshape``.

    ``mask`` is a boolean array under the same length rule as any other series,
    so ``T`` marks bar ``i`` and ``T - 1`` marks bar ``i + 1``. ``value`` is
    where on the panel's own scale each glyph anchors: leave it out and each one
    sits at its own bar's extreme, pass a number and they all sit there, pass an
    array as long as the mask and each reads its own bar, where a non-finite
    entry falls back to that bar's extreme rather than dropping the glyph.
    Everything else means what it means on ``Marker``, and ``text`` is one label
    repeated rather than one per bar, because a glyph on four hundred bars
    carrying four hundred captions is not a chart.

    This is one mark however many bars match. Building a ``Marker`` each, from a
    comprehension over ``flatnonzero``, draws the same picture and costs the
    renderer a primitive and a draw call for every single glyph.
    """

    kind = "markers"

    def __init__(
        self, mask, *, panel=None, value=None, text=None, shape="circle",
        offset=0, color=None,
    ):
        super().__init__(None, panel)
        if type(mask).__module__.split(".")[0] == "pandas":
            mask = mask.to_numpy()
        arr = np.asarray(mask)
        if arr.ndim != 1:
            raise ValueError(
                f"Markers mask must be one dimensional, got shape {arr.shape}"
            )
        if arr.size == 0:
            raise ValueError("Markers mask is empty")
        if arr.dtype != bool:
            # a float mask is how a comparison arrives after passing through NaN,
            # and astype(bool) would read that NaN as True, putting a glyph on the
            # one bar the condition could not be evaluated on
            if arr.dtype.kind not in "iu":
                raise TypeError(
                    f"Markers mask must be boolean, got {arr.dtype}; compare "
                    f"first, as in Markers(mask=fast > slow), and repair any NaN "
                    f"before it gets here, because NaN is truthy"
                )
            arr = arr != 0
        self.mask = arr.copy()
        self.value = _marker_values(value, arr.size)
        self.text = None if text is None else str(text)
        self.shape = _one_of(shape, _SHAPES, "Markers shape")
        self.offset = int(offset)
        self.color = _one_color(color, "Markers color")


def _marker_values(value, size):
    # None anchors every glyph to its own bar, a number puts them all on one
    # line, and an array lets each read its own bar
    if value is None:
        return None
    if np.ndim(value) == 0:
        return _finite(value, "Markers value")
    arr = _as_array(value, "Markers value")
    if arr.size != size:
        raise ValueError(
            f"Markers value has {arr.size} entries and the mask has {size}; one "
            f"anchor per bar, or a single number for all of them"
        )
    return arr


class Background(_Mark):
    """Shading behind the bars, on a condition of your choosing.

    ``values`` is a boolean mask for one region, or an array of labels for as many
    as you like. ``fill`` takes a colour, a ``(bottom, top)`` pair, or a
    ``{label: colour}`` map in which an absent label shades nothing, which is how
    a three-state background stays readable.
    """

    kind = "background"

    def __init__(self, values, *, panel=None, fill=None):
        super().__init__(None, panel)
        if type(values).__module__.split(".")[0] == "pandas":
            values = values.to_numpy()
        arr = np.asarray(values, dtype=object)
        if arr.ndim != 1:
            raise ValueError(
                f"Background values must be one dimensional, got shape {arr.shape}"
            )
        if arr.size == 0:
            raise ValueError("Background values is empty")
        self.values = arr.tolist()
        if isinstance(fill, dict):
            self.fill = {
                key: _stops(value, f"Background fill {key!r}")
                for key, value in fill.items()
            }
        else:
            self.fill = _stops(fill, "Background fill")


def ramp(values, *stops, colors=None, domain=None):
    """Turn numbers into colours, one per value, for a ``color`` or a ``fill``.

    ``stops`` are two or more ``#rrggbb`` or ``#rrggbbaa`` colours, spread evenly
    across the domain, given either loose or as a list under ``colors`` so the
    whole call can be written with named arguments. The domain is the data's own
    finite range unless you pin it, so a ramp is comparable inside one chart but
    not across two slices, and pinning it is what makes two charts speak the same
    language.

    A non-finite value returns ``None``, which every colour argument reads as
    "leave this bar alone". The implicit domain ignores non-finite values too: a
    plain minimum over a rolling feature with a warmup head is ``NaN``, and every
    colour would collapse to one with nothing on screen saying so. Quantised to
    128 steps, so the result is reproducible and the palette stays small.
    """
    arr = _as_array(values, "ramp values")
    if colors is not None:
        if stops:
            raise TypeError(
                f"ramp was given {len(stops)} loose colour stops and a colors= "
                f"list; pass them one way or the other"
            )
        stops = tuple(colors)
    if len(stops) < 2:
        raise ValueError(f"ramp needs at least two colour stops, got {len(stops)}")
    rgba = [_hex_to_rgba(stop, "ramp") for stop in stops]
    finite = np.isfinite(arr)

    if domain is None:
        if not finite.any():
            return [None] * arr.size
        low = float(np.nanmin(arr[finite]))
        high = float(np.nanmax(arr[finite]))
    else:
        low, high = (float(x) for x in domain)
        if not (math.isfinite(low) and math.isfinite(high)):
            raise ValueError(f"ramp domain must be finite, got {domain}")
        if low >= high:
            raise ValueError(f"ramp domain low {low} must be below high {high}")

    if high <= low:
        first = _rgba_to_hex(rgba[0])
        return [first if ok else None for ok in finite.tolist()]

    # the quantisation is what makes this cheap: there are only _RAMP_STEPS
    # distinct colours no matter how long the series, so mix them once and index.
    # Mixing per bar meant a 100k series paid 100k tuple builds and 100k string
    # joins to produce 128 different answers
    span = len(rgba) - 1
    palette = []
    for step in range(_RAMP_STEPS):
        at = (step / (_RAMP_STEPS - 1)) * span
        k = min(span - 1, int(at))
        palette.append(_rgba_to_hex(_mix(rgba[k], rgba[k + 1], at - k)))

    # a non-finite entry is scored as zero so the cast has something to bite on;
    # its index is never read, because finite decides the output
    scaled = np.where(finite, (arr - low) / (high - low), 0.0)
    steps = np.rint(np.clip(scaled, 0.0, 1.0) * (_RAMP_STEPS - 1)).astype(np.int64)
    return [
        palette[step] if ok else None
        for step, ok in zip(steps.tolist(), finite.tolist())
    ]


class Recorder:
    """Collect values inside ``Strategy.next`` with their alignment declared.

    ``next`` runs on every bar but the last, so anything gathered there is
    ``T - 1`` long, and both of the legal lengths mean something. The length
    contract catches a wrong count and can never catch a wrong phase, because
    both wrong forms are legal lengths. Worse, a series passed one bar late lands
    exactly on the engine's own fill arrows, so the wrong chart looks more
    internally consistent than the right one (ADR 0037).

    This moves the choice to where the meaning is obvious. ``at_bar`` is for a
    number that was already true on the bar you were handed. ``at_next`` is for
    one that describes the bar after it, like a level the next bar is tested
    against. Reading a key back gives one value per bar, already placed:

        class Carry(emsl.Strategy):
            def init(self, engine):
                self.log = emsl.plot.Recorder(engine)

            def next(self, state, engine):
                self.log.at_bar(state, lev=..., flat=state["position"] == 0.0)

        emsl.chart(frame, Line(strategy.log["lev"], "leverage"), result).show()

    A bar you never reached is a gap, and a boolean one is ``False``, so an early
    return during a warm-up costs nothing and needs no padding. One key holds one
    kind and one alignment; changing either raises rather than converting.

    Recording costs one array write per value per bar, so leave it out of a
    strategy you mean to sweep across hundreds of trials.
    """

    def __init__(self, bars):
        # an engine, or the number of bars. Knowing the length up front is what
        # lets a bar that was never reached stay a gap instead of shifting
        # everything after it
        if hasattr(bars, "reset") and hasattr(bars, "data"):
            bars = len(bars.data)
        self._n = int(bars)
        if self._n < 1:
            raise ValueError(f"Recorder needs at least one bar, got {bars}")
        self._series = {}
        self._shift = {}

    def at_bar(self, at, **values):
        """Record values that were already true on the bar you were handed."""
        self._write(0, at, values)

    def at_next(self, at, **values):
        """Record values that describe the bar after the one you were handed."""
        self._write(1, at, values)

    def _write(self, shift, at, values):
        bar = at["tick_index"] if isinstance(at, dict) else int(at)
        for key, value in values.items():
            store = self._series.get(key)
            if store is None:
                store = self._open(key, value, shift)
            elif self._shift[key] != shift:
                was = "at_bar" if self._shift[key] == 0 else "at_next"
                now = "at_bar" if shift == 0 else "at_next"
                raise ValueError(
                    f"{key!r} was recorded with {was} and is now {now}; one key "
                    f"belongs to one bar, so record the two under two names"
                )
            elif store.dtype == bool and not isinstance(value, (bool, np.bool_)):
                raise TypeError(
                    f"{key!r} was first recorded as a boolean and is now "
                    f"{value!r}; one key holds one kind"
                )
            target = bar + shift
            if 0 <= target < self._n:
                store[target] = value

    def _open(self, key, value, shift):
        # a boolean fills with False rather than NaN, because
        # numpy.array([nan]).astype(bool) is True and a bar nothing was recorded
        # on is not a bar the condition was met on
        blank = (np.zeros(self._n, dtype=bool)
                 if isinstance(value, (bool, np.bool_))
                 else np.full(self._n, np.nan))
        self._series[key] = blank
        self._shift[key] = shift
        return blank

    def __getitem__(self, key):
        try:
            return self._series[key].copy()
        except KeyError:
            raise KeyError(
                f"nothing was recorded under {key!r}; recorded so far: "
                f"{sorted(self._series)}"
            ) from None

    def __contains__(self, key):
        return key in self._series

    def __len__(self):
        return len(self._series)

    def keys(self):
        return list(self._series)

    def __repr__(self):
        return f"<Recorder {self._n} bars, {sorted(self._series)}>"


def at_bar(values):
    """Pad a series recorded at each bar out to one value per bar.

    ``Strategy.next`` runs on every bar but the last, so anything accumulated
    inside it is ``T - 1`` long. Those numbers were known **on** bar ``i``, so
    they belong at bar ``i``, and passing one short instead draws the whole
    series one bar right of the candles it is meant to explain. Neither raises,
    because both are legal lengths, which is why this exists (ADR 0037).

    A boolean array pads with ``False``. ``numpy.array([nan]).astype(bool)`` is
    ``True``, so padding a mask the way a float is padded makes it report the
    condition it was looking for on the one bar it knows nothing about.
    """
    arr = np.asarray(values)
    if arr.dtype == bool:
        return np.append(arr, False)
    return np.append(np.array(arr, dtype=np.float64), np.nan)


def at_next(values):
    """Pad a series describing the following bar out to one value per bar.

    The mirror of ``at_bar``, for a value that belongs one bar later than it was
    computed: a level the next bar is tested against, a diff, or the engine's own
    ``equity_curve``. Passing one of those short is already correct, so this is
    for when you would rather every array on a chart be the same length and read
    the padding off the call instead of off the length.
    """
    arr = np.asarray(values)
    if arr.dtype == bool:
        return np.insert(arr, 0, False)
    return np.insert(np.array(arr, dtype=np.float64), 0, np.nan)
