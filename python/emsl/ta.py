"""A small, pinned set of indicators.

Thirty functions, chosen to cover what a bar-level strategy usually reaches for
and stopped there. This is not an attempt to be ta-lib: an indicator library is
mostly somebody's opinion about warm-up and smoothing, and a hundred of those
opinions is a hundred chances to disagree with the chart you have been reading
for years.

Three rules hold for every function here, and they are the reason it lives in the
library rather than in your notebook:

**One length, one alignment, and gaps stay gaps.** Every function returns a
float64 array of length ``T``, aligned so that entry ``i`` is bar ``i``, with the
warm-up as ``NaN``. Never a shorter array. A missing bar is a gap in the output
too, never bridged and never absorbed; an infinity in the input counts as missing,
because it is not a price. Nothing here raises over a gap, it costs warm-up. The
chart's length contract is the sharpest idea in this library and a second
convention beside it would undo that, so the same array goes straight into a rule
and into ``emsl.chart`` with no padding decision in between. The two functions
that answer a yes-or-no question, ``crossover`` and ``crossunder``, return
booleans and pad with ``False`` rather than ``NaN``, because ``bool(nan)`` is
``True`` and a warm-up that reads as a signal is the one shape of this rule that
would be dangerous (ADR 0063).

**The convention is written down.** Where implementations differ, and they differ
most on how an exponential average is seeded and which smoothing an oscillator
uses, this module says which one it picked in the function's own docstring. A
curve that quietly disagrees with the chart you trust is worse than no curve.

**Nothing here knows about the engine.** These are functions of arrays. They hold
no state, take no engine, need no pandas, and the chart still computes nothing
(ADR 0042): you compute once in ``init``, trade on that array, and hand the same
object to the chart, so the line on screen cannot drift from the line that made
the decision.

    class Cross(emsl.Strategy):
        def init(self, engine):
            self.fast = emsl.ta.ema(engine.closes, 20)
            self.slow = emsl.ta.ema(engine.closes, 60)
            self.warmup = 60
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "sma",
    "ema",
    "wma",
    "hma",
    "vwma",
    "vwap",
    "rsi",
    "macd",
    "stoch",
    "willr",
    "cci",
    "roc",
    "atr",
    "natr",
    "true_range",
    "bbands",
    "keltner",
    "stdev",
    "donchian",
    "supertrend",
    "adx",
    "obv",
    "mfi",
    "zscore",
    "shift",
    "change",
    "highest",
    "lowest",
    "crossover",
    "crossunder",
    "Bands",
    "Channel",
    "Macd",
    "Stochastic",
    "Trend",
]


class Bands:
    """Three lines around a middle: an upper, a middle and a lower."""

    __slots__ = ("upper", "middle", "lower")

    def __init__(self, upper, middle, lower):
        self.upper = upper
        self.middle = middle
        self.lower = lower

    def __repr__(self):
        return f"Bands({len(self.middle)} bars)"


class Channel:
    """A band around a middle, plus which side of it the price sits on."""

    __slots__ = ("upper", "middle", "lower", "direction")

    def __init__(self, upper, middle, lower, direction):
        self.upper = upper
        self.middle = middle
        self.lower = lower
        self.direction = direction

    def __repr__(self):
        return f"Channel({len(self.middle)} bars)"


class Trend:
    """A directional reading and the two strengths behind it."""

    __slots__ = ("adx", "plus", "minus")

    def __init__(self, adx, plus, minus):
        self.adx = adx
        self.plus = plus
        self.minus = minus

    def __repr__(self):
        return f"Trend({len(self.adx)} bars)"


class Macd:
    """The convergence line, its signal, and the gap between them."""

    __slots__ = ("line", "signal", "histogram")

    def __init__(self, line, signal, histogram):
        self.line = line
        self.signal = signal
        self.histogram = histogram

    def __repr__(self):
        return f"Macd({len(self.line)} bars)"


class Stochastic:
    """The raw oscillator and its smoothing."""

    __slots__ = ("k", "d")

    def __init__(self, k, d):
        self.k = k
        self.d = d

    def __repr__(self):
        return f"Stochastic({len(self.k)} bars)"


def _series(values, where):
    # the shape is checked BEFORE any flattening. Ravelling first made the check
    # unreachable, so a whole OHLCV frame handed in place of one of its columns
    # was silently interleaved into a single series and every number that came
    # back was computed from four columns at once, and looked entirely plausible
    out = np.asarray(values, dtype=np.float64)
    if out.ndim == 2 and out.shape[1] == 1:
        out = out.ravel()
    if out.ndim != 1:
        raise ValueError(
            f"{where} needs one series, but was given an array of shape "
            f"{out.shape}; pass a single column, not the whole frame"
        )
    if out.size == 0:
        raise ValueError(f"{where} needs a non-empty series")
    # an infinity is not a price, so it is a gap for the same reason a NaN is.
    # Handled once here rather than in fourteen places: left alone it survived
    # into arithmetic that produced NaN anyway, but by way of `inf - inf` and
    # `inf / inf`, which numpy reports as a warning from the middle of whichever
    # function happened to hit it first
    return np.where(np.isfinite(out), out, np.nan)


def _length(length, size, where):
    # a fractional length is a mistake, not a request to round one way or the
    # other, and silently flooring it answered a question nobody asked
    if length != int(length):
        raise ValueError(f"{where} needs a whole number of bars, got {length}")
    length = int(length)
    if length < 1:
        raise ValueError(f"{where} needs a length of at least 1, got {length}")
    if length > size:
        raise ValueError(
            f"{where} needs a length no longer than the series, but {length} was "
            f"asked of {size} bars"
        )
    return length


def _blank(size):
    return np.full(size, np.nan, dtype=np.float64)


def _windows(values, length):
    # every full window of `length`, as a (size - length + 1, length) view. A NaN
    # anywhere inside a window carries through to that window's result, which is
    # what makes a gap in the input stay a gap in the output
    return np.lib.stride_tricks.sliding_window_view(values, length)


def _first_finite(values):
    finite = np.flatnonzero(np.isfinite(values))
    return int(finite[0]) if finite.size else None


def _aligned(where, **named):
    # every series a multi-input indicator reads has to describe the same bars,
    # and the error names which one is the odd size rather than reporting a shape
    series = {name: _series(values, where) for name, values in named.items()}
    sizes = {name: value.size for name, value in series.items()}
    if len(set(sizes.values())) > 1:
        shown = ", ".join(f"{name} {size}" for name, size in sizes.items())
        raise ValueError(
            f"{where} needs its series to be one length, got {shown}"
        )
    return tuple(series.values())


def _against(other, size, where):
    # the second argument of a comparison: another series, or a level, which is a
    # constant series and is by far the more common thing to cross
    if np.ndim(other) == 0:
        value = float(other)
        if not np.isfinite(value):
            raise ValueError(f"{where} needs a finite level, got {value}")
        return np.full(size, value, dtype=np.float64)
    out = _series(other, where)
    if out.size != size:
        raise ValueError(
            f"{where} needs two series of one length, got {size} and {out.size}"
        )
    return out


def _smooth(values, alpha, length, where):
    # one exponential pass, seeded from the simple average of the first full
    # window rather than from the first value. Seeding from the first value lets a
    # single opening print steer the curve for hundreds of bars, and the simple
    # average is what TradingView and Wilder both use.
    #
    # The seed is the first CLEAN window, and a gap restarts the search for one.
    # A recurrence cannot carry across a missing bar, so the strict reading is
    # that everything after one is undefined; that reading makes a year of
    # candles with a single hole in it produce nothing at all from the hole
    # onward, which is useless rather than rigorous. A gap ends the run and the
    # next clean window seeds a fresh one, so a hole costs the same warm-up here
    # as it does to a rolling window and the module behaves one way throughout.
    #
    # The loop is over the gaps rather than over the bars. A per-bar Python loop
    # measured 191 ms for one EMA over 100k candles against 18 ms for the simple
    # average of the same length, which made two EMAs computed in `init` 42% of an
    # entire backtest trial, paid again on every trial of every search. Each clean
    # stretch is a plain geometric decay, so it runs as one vectorized pass and
    # only the seeding and the gaps stay in Python (ADR 0064).
    size = values.size
    out = _blank(size)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    # the half-open bounds of every maximal run of finite values
    edges = np.flatnonzero(np.diff(finite.astype(np.int8)))
    starts = np.concatenate((
        np.array([0], dtype=np.int64) if finite[0] else np.empty(0, dtype=np.int64),
        edges[finite[edges + 1]] + 1,
    ))
    stops = np.concatenate((
        edges[~finite[edges + 1]] + 1,
        np.array([size], dtype=np.int64) if finite[size - 1]
        else np.empty(0, dtype=np.int64),
    ))
    for first, last in zip(starts, stops):
        first, last = int(first), int(last)
        if last - first < length:
            continue
        seed = first + length - 1
        out[seed] = float(values[first:seed + 1].mean())
        if last > seed + 1:
            out[seed + 1:last] = _decay(values[seed + 1:last], out[seed], alpha)
    return out


def _decay(values, seed, alpha):
    """One exponential pass over a stretch that has no gaps in it.

    ``out[k] = alpha * values[k] + (1 - alpha) * out[k - 1]`` unrolled is
    ``keep**(k + 1) * seed`` plus a decaying sum of the values, and written as one
    cumulative sum that needs ``keep**-k``. Which overflows to infinity within a
    few thousand bars at any ordinary length, taking most of the output with it
    and saying nothing: at ``length`` 20 over 100k bars the one-line version
    returns 7,045 finite values out of 99,981. So the sum is taken in blocks short
    enough that the largest weight stays well inside the float range, each seeded
    from the last value of the block before it. That is the whole reason this is
    not a one-liner, and the block size has to come from ``alpha`` rather than
    being picked.
    """
    keep = 1.0 - alpha
    if keep <= 0.0:
        # alpha of exactly 1 keeps nothing, so every value is its own average
        return values * alpha
    # keep**-block held under about 1e150, ten orders of magnitude inside the
    # largest finite double
    block = max(1, min(values.size, int(150.0 / -math.log10(keep))))
    out = np.empty(values.size, dtype=np.float64)
    carried = seed
    for start in range(0, values.size, block):
        chunk = values[start:start + block]
        powers = keep ** np.arange(1.0, chunk.size + 1.0)
        # sum_{j <= k} alpha * values[j] * keep**(k - j), by dividing the weights
        # out of a cumulative sum and multiplying them back in
        out[start:start + chunk.size] = (
            np.cumsum(alpha * chunk / powers) * powers + carried * powers
        )
        carried = out[start + chunk.size - 1]
    return out


def sma(values, length):
    """The simple moving average over ``length`` bars."""
    values = _series(values, "sma")
    length = _length(length, values.size, "sma")
    out = _blank(values.size)
    out[length - 1:] = _windows(values, length).mean(axis=-1)
    return out


def ema(values, length):
    """The exponential moving average over ``length`` bars.

    Smoothed at ``2 / (length + 1)`` and seeded from the simple average of the
    first ``length`` values, which is TradingView's convention. Seeding from the
    first value instead, as some implementations do, lets one opening print steer
    the curve for hundreds of bars.
    """
    values = _series(values, "ema")
    length = _length(length, values.size, "ema")
    return _smooth(values, 2.0 / (length + 1.0), length, "ema")


def wma(values, length):
    """The linearly weighted moving average: the newest bar counts ``length``
    times as much as the oldest.
    """
    values = _series(values, "wma")
    length = _length(length, values.size, "wma")
    weights = np.arange(1.0, length + 1.0)
    out = _blank(values.size)
    out[length - 1:] = _windows(values, length) @ (weights / weights.sum())
    return out


def vwap(high, low, close, volume, length):
    """The volume-weighted average price over a rolling window of ``length`` bars.

    Rolling, not anchored to a session, because the engine has bars and no notion
    of a trading day. Priced on the typical price ``(high + low + close) / 3``.
    A window whose volume is zero has no weighted price, so it is a gap.
    """
    high = _series(high, "vwap")
    low = _series(low, "vwap")
    close = _series(close, "vwap")
    volume = _series(volume, "vwap")
    if not high.size == low.size == close.size == volume.size:
        raise ValueError(
            f"vwap needs four series of one length, got {high.size}, {low.size}, "
            f"{close.size} and {volume.size}"
        )
    length = _length(length, close.size, "vwap")
    typical = (high + low + close) / 3.0
    paid = _windows(typical * volume, length).sum(axis=-1)
    traded = _windows(volume, length).sum(axis=-1)
    out = _blank(close.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[length - 1:] = np.where(traded > 0.0, paid / traded, np.nan)
    return out


def rsi(values, length=14):
    """The relative strength index over ``length`` bars, 0 to 100.

    Wilder's smoothing, ``1 / length`` rather than ``2 / (length + 1)``, which is
    what TradingView's ``ta.rsi`` uses and what the indicator was defined with.

    Rising with no losses at all is 100. Going **nowhere**, with neither gains nor
    losses, is a gap: there is no ratio of strength to weakness when both are
    zero, and the alternative conventions each pick a side of a coin that was
    never tossed. Reporting 100 for a dead flat market, as some do, is a live trap
    for the rule this exists to be traded on, since ``rsi > 70`` then fires on a
    market that has not moved at all.
    """
    values = _series(values, "rsi")
    if int(length) >= values.size:
        raise ValueError(
            f"rsi needs more bars than its length, since the first bar makes no "
            f"change to measure; {length} was asked of {values.size} bars"
        )
    length = _length(length, values.size, "rsi")
    change = np.diff(values, prepend=np.nan)
    gain = _smooth(np.maximum(change, 0.0), 1.0 / length, length, "rsi")
    loss = _smooth(-np.minimum(change, 0.0), 1.0 / length, length, "rsi")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(loss > 0.0, 100.0 - 100.0 / (1.0 + gain / loss), 100.0)
    moved = (gain > 0.0) | (loss > 0.0)
    return np.where(np.isfinite(gain) & np.isfinite(loss) & moved, out, np.nan)


def macd(values, fast=12, slow=26, signal=9):
    """Moving average convergence divergence, as ``.line``, ``.signal`` and
    ``.histogram``.

    The line is the fast exponential average less the slow one; the signal is an
    exponential average of the line; the histogram is the gap. Each is length
    ``T`` with its own warm-up, and the signal warms up after the line does.
    """
    values = _series(values, "macd")
    if not int(fast) < int(slow):
        raise ValueError(f"macd needs fast below slow, got {fast} and {slow}")
    line = ema(values, fast) - ema(values, slow)
    smoothed = ema(line, signal)
    return Macd(line, smoothed, line - smoothed)


def stoch(high, low, close, length=14, smooth=3):
    """The stochastic oscillator, as ``.k`` and ``.d``, both 0 to 100.

    ``k`` is where the close sits inside the last ``length`` bars' range, and
    ``d`` is its simple average over ``smooth`` bars. A window with no range at
    all has no position inside it, so it is a gap rather than a fifty.

    Both run 0 to 100 for well-formed bars. A close outside its own bar's high
    and low is not well-formed, and produces a value outside that range rather
    than being clamped, because a clamp would hide the only evidence that the
    data is wrong.
    """
    high = _series(high, "stoch")
    low = _series(low, "stoch")
    close = _series(close, "stoch")
    if not high.size == low.size == close.size:
        raise ValueError(
            f"stoch needs three series of one length, got {high.size}, "
            f"{low.size} and {close.size}"
        )
    length = _length(length, close.size, "stoch")
    _length(smooth, close.size, "stoch")  # named here, not from inside sma
    top = _blank(close.size)
    bottom = _blank(close.size)
    top[length - 1:] = _windows(high, length).max(axis=-1)
    bottom[length - 1:] = _windows(low, length).min(axis=-1)
    span = top - bottom
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(span > 0.0, (close - bottom) / span * 100.0, np.nan)
    return Stochastic(k, sma(k, smooth))


def roc(values, length):
    """The rate of change over ``length`` bars, in percent."""
    values = _series(values, "roc")
    length = _length(length, values.size, "roc")
    out = _blank(values.size)
    past = values[:-length]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[length:] = np.where(past != 0.0, (values[length:] / past - 1.0) * 100.0,
                                np.nan)
    return out


def true_range(high, low, close):
    """The true range of each bar: its own range, or the gap from the previous
    close if that is wider.

    The first bar has no previous close, so it is its own high less its low.

    A missing bar anywhere else is a gap in the output, not a fallback to the
    bar's own range. Taking the widest span while ignoring the missing ones was
    the first version, and it is worse than useless: with the previous close gone
    it quietly returned high less low, which on a gapping bar understated the true
    range by whatever the gap was, so an ATR-sized stop came out far too tight on
    exactly the data where the gap mattered.
    """
    high = _series(high, "true_range")
    low = _series(low, "true_range")
    close = _series(close, "true_range")
    if not high.size == low.size == close.size:
        raise ValueError(
            f"true_range needs three series of one length, got {high.size}, "
            f"{low.size} and {close.size}"
        )
    previous = np.roll(close, 1)
    previous[0] = np.nan
    spans = np.vstack([high - low, np.abs(high - previous), np.abs(low - previous)])
    # propagating, so any missing leg makes the bar a gap
    out = spans.max(axis=0)
    # except the first, which has no previous close by construction rather than by
    # accident, and so is its own range
    out[0] = high[0] - low[0]
    return out


def atr(high, low, close, length=14):
    """The average true range over ``length`` bars.

    Wilder's smoothing of the true range, ``1 / length``, matching ``rsi`` and
    TradingView's ``ta.atr``.
    """
    ranges = true_range(high, low, close)
    length = _length(length, ranges.size, "atr")
    return _smooth(ranges, 1.0 / length, length, "atr")


def stdev(values, length):
    """The rolling standard deviation over ``length`` bars.

    Population, dividing by ``length`` rather than ``length - 1``, because that is
    what ``bbands`` is defined against and a band that disagrees with every chart
    by a factor of ``sqrt(n / (n - 1))`` is worse than no band.
    """
    values = _series(values, "stdev")
    length = _length(length, values.size, "stdev")
    out = _blank(values.size)
    # an infinity in the window makes the deviation undefined, and numpy says so
    # through a warning on the way. The gap is the answer; the warning is noise
    with np.errstate(invalid="ignore"):
        out[length - 1:] = _windows(values, length).std(axis=-1)
    return out


def bbands(values, length=20, deviations=2.0):
    """Bollinger bands, as ``.upper``, ``.middle`` and ``.lower``.

    A simple moving average, with a band ``deviations`` population standard
    deviations either side of it.
    """
    values = _series(values, "bbands")
    length = _length(length, values.size, "bbands")
    deviations = float(deviations)
    if not np.isfinite(deviations) or deviations < 0.0:
        # a negative width turns the bands inside out, upper below lower, and
        # every comparison written against them then reads backwards
        raise ValueError(
            f"bbands needs a width of at least 0, got {deviations}"
        )
    middle = sma(values, length)
    spread = stdev(values, length) * deviations
    return Bands(middle + spread, middle, middle - spread)


def donchian(high, low, length=20):
    """The Donchian channel, as ``.upper``, ``.middle`` and ``.lower``: the
    highest high and the lowest low of the last ``length`` bars, and their midpoint.

    Both edges include the current bar, so a breakout of ``upper`` cannot happen
    on the bar that set it. Compare against the previous bar's edge if that is
    what you mean.
    """
    high = _series(high, "donchian")
    low = _series(low, "donchian")
    if high.size != low.size:
        raise ValueError(
            f"donchian needs two series of one length, got {high.size} and {low.size}"
        )
    length = _length(length, high.size, "donchian")
    top = _blank(high.size)
    bottom = _blank(low.size)
    top[length - 1:] = _windows(high, length).max(axis=-1)
    bottom[length - 1:] = _windows(low, length).min(axis=-1)
    return Bands(top, (top + bottom) / 2.0, bottom)


def zscore(values, length):
    """How many rolling standard deviations the current value sits from its own
    rolling mean. A window with no deviation at all is a gap, not a zero.
    """
    values = _series(values, "zscore")
    length = _length(length, values.size, "zscore")
    spread = stdev(values, length)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(spread > 0.0, (values - sma(values, length)) / spread, np.nan)


def shift(values, bars=1):
    """``values`` moved ``bars`` bars into the future, so entry ``i`` carries what
    entry ``i - bars`` held. The first ``bars`` entries are the warm-up.

    The safe way to look back. Written by hand, ``values[i - bars]`` on an early
    bar is a NEGATIVE index, which numpy resolves from the END of the array, so
    the rule reads a price from the far future, produces a spectacular result and
    raises nothing at all. ``Strategy.warmup`` (ADR 0050) closes that only when
    the declared warm-up is at least as deep as the lookback, and the natural
    error is declaring the indicator's warm-up and then reaching further back
    than it inside ``next``. Computed once here, the array is padded rather than
    wrapped, and there is no index to get wrong (ADR 0063).

    A negative ``bars`` is refused rather than read as a lead: shifting the other
    way is lookahead by construction, and this module has no honest use for it.
    """
    values = _series(values, "shift")
    if bars != int(bars):
        raise ValueError(f"shift needs a whole number of bars, got {bars}")
    bars = int(bars)
    if bars < 0:
        raise ValueError(
            f"shift moves values forward in time, so bars is zero or more, got "
            f"{bars}; a negative shift would read the future into the past"
        )
    if bars == 0:
        return values.copy()
    out = _blank(values.size)
    if bars < values.size:
        out[bars:] = values[:-bars]
    return out


def change(values, bars=1):
    """How much ``values`` moved over the last ``bars`` bars, in its own units."""
    values = _series(values, "change")
    return values - shift(values, bars)


def highest(values, length):
    """The highest value of the last ``length`` bars, the current one included."""
    values = _series(values, "highest")
    length = _length(length, values.size, "highest")
    out = _blank(values.size)
    out[length - 1:] = _windows(values, length).max(axis=-1)
    return out


def lowest(values, length):
    """The lowest value of the last ``length`` bars, the current one included."""
    values = _series(values, "lowest")
    length = _length(length, values.size, "lowest")
    out = _blank(values.size)
    out[length - 1:] = _windows(values, length).min(axis=-1)
    return out


def crossover(values, other):
    """``True`` on each bar where ``values`` crosses above ``other``: at or below
    it on the previous bar, above it on this one.

    ``other`` is a second series or a plain number, since crossing a level is the
    commoner case. Returns **bool**, not float, and a bar whose comparison cannot
    be made, either end of it missing or the very first bar, is ``False``. That is
    the one deliberate exception to this module's NaN warm-up: ``bool(nan)`` is
    ``True`` in Python, so a float warm-up would fire the rule on exactly the bars
    where nothing is known.
    """
    values = _series(values, "crossover")
    other = _against(other, values.size, "crossover")
    out = np.zeros(values.size, dtype=bool)
    # a comparison against a missing value is False in numpy, which is the answer
    # wanted here: no evidence of a cross is not a cross
    out[1:] = (values[1:] > other[1:]) & (values[:-1] <= other[:-1])
    return out


def crossunder(values, other):
    """``True`` on each bar where ``values`` crosses below ``other``. The mirror
    of ``crossover``, and boolean for the same reason.
    """
    values = _series(values, "crossunder")
    other = _against(other, values.size, "crossunder")
    out = np.zeros(values.size, dtype=bool)
    out[1:] = (values[1:] < other[1:]) & (values[:-1] >= other[:-1])
    return out


def hma(values, length):
    """Hull's moving average: two weighted averages differenced and smoothed again,
    which turns far faster than a simple average of the same length for the same
    amount of noise.

    ``wma(2 * wma(values, length / 2) - wma(values, length), sqrt(length))``, with
    both inner lengths floored to whole bars, which is the published definition.
    """
    values = _series(values, "hma")
    length = _length(length, values.size, "hma")
    half = max(1, length // 2)
    root = max(1, int(math.sqrt(length)))
    raw = 2.0 * wma(values, half) - wma(values, length)
    return wma(raw, root)


def vwma(values, volume, length):
    """The volume-weighted moving average of ``values`` over ``length`` bars.

    ``sma`` treats a bar nobody traded as equal to one everybody did. A window
    whose volume is zero has no weighted average, so it is a gap.
    """
    values, volume = _aligned("vwma", values=values, volume=volume)
    length = _length(length, values.size, "vwma")
    paid = _windows(values * volume, length).sum(axis=-1)
    traded = _windows(volume, length).sum(axis=-1)
    out = _blank(values.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[length - 1:] = np.where(traded > 0.0, paid / traded, np.nan)
    return out


def willr(high, low, close, length=14):
    """Williams %R over ``length`` bars, -100 to 0.

    Where the close sits inside the recent range, read from the top: 0 is the
    high of the window and -100 the low. The same measurement ``stoch`` makes and
    the opposite sign convention, which is why both exist rather than one. A
    window with no range at all has no position inside it, so it is a gap.
    """
    high, low, close = _aligned("willr", high=high, low=low, close=close)
    length = _length(length, close.size, "willr")
    top, bottom = highest(high, length), lowest(low, length)
    span = top - bottom
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(span > 0.0, (close - top) / span * 100.0, np.nan)


def cci(high, low, close, length=20):
    """The commodity channel index over ``length`` bars.

    How far the typical price ``(high + low + close) / 3`` sits from its own
    average, in units of its mean absolute deviation, scaled by the conventional
    0.015 so that most readings land inside -100 to 100. Mean ABSOLUTE deviation,
    not standard deviation, which is the definition and is what makes it react
    differently from ``zscore`` on a fat-tailed series. A window with no deviation
    at all is a gap.
    """
    high, low, close = _aligned("cci", high=high, low=low, close=close)
    length = _length(length, close.size, "cci")
    typical = (high + low + close) / 3.0
    frames = _windows(typical, length)
    with np.errstate(invalid="ignore"):
        spread = np.abs(frames - frames.mean(axis=-1, keepdims=True)).mean(axis=-1)
    deviation = _blank(close.size)
    deviation[length - 1:] = spread
    middle = sma(typical, length)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(deviation > 0.0,
                        (typical - middle) / (0.015 * deviation), np.nan)


def natr(high, low, close, length=14):
    """The average true range as a percent of the close.

    The number to size a stop with across instruments and across time: an ATR of
    400 means one thing on a coin at 90,000 and another on one at 800, and the
    same strategy run over two years of a trending market is comparing two
    different sizes without this.
    """
    high, low, close = _aligned("natr", high=high, low=low, close=close)
    ranges = atr(high, low, close, length)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(close > 0.0, ranges / close * 100.0, np.nan)


def keltner(high, low, close, length=20, atr_length=10, multiplier=2.0):
    """Keltner channels, as ``.upper``, ``.middle`` and ``.lower``.

    An exponential average with bands ``multiplier`` average true ranges either
    side. Where ``bbands`` measures the spread of the closes, this measures the
    spread of the BARS, so a session of wide ranges closing in one place widens
    this and not that. It is also the band an ATR-sized stop belongs on, which is
    the question ``atr`` and ``metrics.excursions`` both point at and neither
    could answer.
    """
    high, low, close = _aligned("keltner", high=high, low=low, close=close)
    multiplier = float(multiplier)
    if not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError(
            f"keltner needs a width of at least 0, got {multiplier}"
        )
    middle = ema(close, length)
    spread = atr(high, low, close, atr_length) * multiplier
    return Bands(middle + spread, middle, middle - spread)


def supertrend(high, low, close, length=10, multiplier=3.0):
    """The supertrend, as ``.upper``, ``.lower``, ``.middle`` (the line in force)
    and ``.direction`` (``1.0`` while long, ``-1.0`` while short).

    An ATR band that ratchets: each edge may only tighten toward price while the
    trend holds, and the line flips when the close closes through the far one.
    Read ``direction`` rather than comparing the close against ``middle``, since
    the line is the edge in force and the comparison is what set it.

    A gap ends the ratchet and the next clean stretch starts a fresh one, on the
    same rule the exponential averages use: a recurrence cannot carry across a
    bar that is not there.
    """
    high, low, close = _aligned("supertrend", high=high, low=low, close=close)
    multiplier = float(multiplier)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(
            f"supertrend needs a width above 0, got {multiplier}"
        )
    ranges = atr(high, low, close, length)
    middle_price = (high + low) / 2.0
    raw_upper = middle_price + multiplier * ranges
    raw_lower = middle_price - multiplier * ranges
    size = close.size
    upper, lower, line = _blank(size), _blank(size), _blank(size)
    direction = _blank(size)
    running = False
    for i in range(size):
        if not (np.isfinite(raw_upper[i]) and np.isfinite(raw_lower[i])
                and np.isfinite(close[i])):
            running = False
            continue
        if not running:
            upper[i], lower[i], direction[i] = raw_upper[i], raw_lower[i], 1.0
            running = True
        else:
            upper[i] = (raw_upper[i] if raw_upper[i] < upper[i - 1]
                        or close[i - 1] > upper[i - 1] else upper[i - 1])
            lower[i] = (raw_lower[i] if raw_lower[i] > lower[i - 1]
                        or close[i - 1] < lower[i - 1] else lower[i - 1])
            if direction[i - 1] > 0.0:
                direction[i] = -1.0 if close[i] < lower[i] else 1.0
            else:
                direction[i] = 1.0 if close[i] > upper[i] else -1.0
        line[i] = lower[i] if direction[i] > 0.0 else upper[i]
    return Channel(upper, line, lower, direction)


def adx(high, low, close, length=14):
    """The average directional index, as ``.adx``, ``.plus`` and ``.minus``.

    How strongly the market is trending, without saying which way: ``plus`` and
    ``minus`` are the two directional strengths and ``adx`` is how far apart they
    are, smoothed. Above 20 to 25 is the usual reading for "there is a trend
    here", and it is the filter a moving-average cross is conventionally paired
    with, since a cross in a range is the most reliable way to lose money slowly.

    Wilder's smoothing throughout, matching ``rsi`` and ``atr``.
    """
    high, low, close = _aligned("adx", high=high, low=low, close=close)
    length = _length(length, close.size, "adx")
    up = high - shift(high, 1)
    down = shift(low, 1) - low
    plus_move = np.where((up > down) & (up > 0.0), up, 0.0)
    minus_move = np.where((down > up) & (down > 0.0), down, 0.0)
    # a bar with no previous is not a zero move, it is an unknown one
    missing = ~(np.isfinite(up) & np.isfinite(down))
    plus_move = np.where(missing, np.nan, plus_move)
    minus_move = np.where(missing, np.nan, minus_move)
    alpha = 1.0 / length
    ranges = _smooth(true_range(high, low, close), alpha, length, "adx")
    plus_smoothed = _smooth(plus_move, alpha, length, "adx")
    minus_smoothed = _smooth(minus_move, alpha, length, "adx")
    with np.errstate(divide="ignore", invalid="ignore"):
        plus = np.where(ranges > 0.0, plus_smoothed / ranges * 100.0, np.nan)
        minus = np.where(ranges > 0.0, minus_smoothed / ranges * 100.0, np.nan)
        total = plus + minus
        dx = np.where(total > 0.0, np.abs(plus - minus) / total * 100.0, np.nan)
    return Trend(_smooth(dx, alpha, length, "adx"), plus, minus)


def obv(close, volume):
    """On-balance volume: the running total of each bar's volume, signed by the
    direction the close moved.

    The one thing in this module that reads participation rather than price. A
    price making new highs on falling on-balance volume is the divergence it
    exists to show.

    The total restarts from zero after a gap rather than carrying across it, on
    the same rule the exponential averages use, so a hole costs the warm-up of one
    bar and never silently absorbs a missing day's volume into the running sum.
    """
    close, volume = _aligned("obv", close=close, volume=volume)
    size = close.size
    out = _blank(size)
    running = None
    for i in range(1, size):
        if not (np.isfinite(close[i]) and np.isfinite(volume[i])
                and np.isfinite(close[i - 1])):
            running = None
            continue
        # the first bar of a fresh run has a direction but no history, so the
        # total starts flat there. Bar zero has no previous close at all, which is
        # a warm-up rather than a zero
        if running is None:
            running = 0.0
        if close[i] > close[i - 1]:
            running += volume[i]
        elif close[i] < close[i - 1]:
            running -= volume[i]
        out[i] = running
    return out


def mfi(high, low, close, volume, length=14):
    """The money flow index over ``length`` bars, 0 to 100.

    ``rsi`` weighted by what actually traded: the same up-against-down ratio, on
    the typical price times the volume rather than on the price change alone. A
    move on no volume moves this hardly at all, which is the whole point of
    reading it beside ``rsi``.

    A window with flows in one direction only is 100 or 0. A window with no flow
    at all is a gap, for the reason ``rsi`` gives: there is no ratio of strength
    to weakness when both are zero.
    """
    high, low, close, volume = _aligned(
        "mfi", high=high, low=low, close=close, volume=volume
    )
    length = _length(length, close.size, "mfi")
    typical = (high + low + close) / 3.0
    flow = typical * volume
    rising = typical > shift(typical, 1)
    falling = typical < shift(typical, 1)
    known = np.isfinite(flow) & np.isfinite(shift(typical, 1))
    positive = np.where(known, np.where(rising, flow, 0.0), np.nan)
    negative = np.where(known, np.where(falling, flow, 0.0), np.nan)
    up = _blank(close.size)
    down = _blank(close.size)
    up[length - 1:] = _windows(positive, length).sum(axis=-1)
    down[length - 1:] = _windows(negative, length).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(down > 0.0, 100.0 - 100.0 / (1.0 + up / down), 100.0)
    moved = (up > 0.0) | (down > 0.0)
    return np.where(np.isfinite(up) & np.isfinite(down) & moved, out, np.nan)
