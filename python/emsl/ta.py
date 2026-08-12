"""A small, pinned set of indicators.

Fourteen functions, chosen to cover what a bar-level strategy usually reaches for
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
because it is not a price. Nothing here raises over a gap, it costs warm-up. The chart's length contract is the sharpest idea in this
library and a second convention beside it would undo that, so the same array goes
straight into a rule and into ``emsl.chart`` with no padding decision in between.

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

import numpy as np

__all__ = [
    "sma",
    "ema",
    "wma",
    "vwap",
    "rsi",
    "macd",
    "stoch",
    "roc",
    "atr",
    "true_range",
    "bbands",
    "stdev",
    "donchian",
    "zscore",
    "Bands",
    "Macd",
    "Stochastic",
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


def _smooth(values, alpha, length, where):
    # one exponential pass, seeded from the simple average of the first full
    # window rather than from the first value. Seeding from the first value lets a
    # single opening print steer the curve for hundreds of bars, and the simple
    # average is what TradingView and Wilder both use.
    #
    # The seed is the first CLEAN window, not the first window. Insisting on the
    # one at the first finite value meant a single missing bar anywhere in the
    # warm-up raised, and raised from `ema` about a bar index the caller had never
    # heard of when they had called `macd`. Waiting instead makes the whole module
    # behave one way: a gap costs warm-up, never an exception
    size = values.size
    out = _blank(size)
    start = _seed_at(values, length)
    if start is None:
        return out
    running = float(values[start - length + 1:start + 1].mean())
    out[start] = running
    for i in range(start + 1, size):
        running = alpha * values[i] + (1.0 - alpha) * running
        out[i] = running
    return out


def _seed_at(values, length):
    # the index of the last bar of the earliest run of `length` finite values
    run = 0
    for i in range(values.size):
        run = run + 1 if np.isfinite(values[i]) else 0
        if run == length:
            return i
    return None


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
