"""Post-hoc evaluation of a finished run: where the money went, and whether the
number means anything.

The fourteen statistics on a ``BacktestResult`` are computed in Rust once per
trial, so they have to be cheap and they stay there. Everything here runs once,
on a result you already have, which is why it is Python: it is read far more
often than it is executed. Nothing here simulates anything.

Every function takes the ``BacktestResult`` itself rather than an equity curve, so
it can read the annualization and the opening balance the run recorded rather
than being told them again. Two readings of one number are how two numbers come
to disagree, which is the whole reason ``initial`` and ``periods_per_year`` are
carried on the result at all (ADR 0048).

The return series is derived by the identical rule the engine uses: seeded from
the opening balance, one return per interval, and a non-positive previous equity
contributing zero rather than a meaningless ratio. ``metrics.sharpe`` reproduces
``result.stats["sharpe"]`` exactly, and a test pins that.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "returns",
    "cost_curve",
    "breakeven_bps",
    "excursions",
    "session_buckets",
    "sharpe",
    "skew",
    "kurtosis",
    "drawdown",
    "drawdown_table",
    "time_under_water",
    "decompose",
    "long_short_split",
    "buy_and_hold",
    "value_at_risk",
    "conditional_value_at_risk",
    "probabilistic_sharpe",
    "min_track_record_length",
    "autocorrelation",
    "compare",
    "report",
    "summary",
]


def _curve(result):
    # the equity path including the point it started from. The engine records a
    # point only on a real advance, so the opening balance is not in the curve and
    # anything reading the curve alone cannot see the first bar's move
    equity = np.asarray(result.equity_curve, dtype=np.float64)
    start = getattr(result, "initial", None)
    if start is None:
        raise ValueError(
            "this result carries no opening balance, so its returns cannot be "
            "seeded; it was built by hand rather than by a Backtester"
        )
    return np.concatenate(([float(start)], equity))


def _annualization(result):
    ppy = getattr(result, "periods_per_year", None)
    if ppy is None or not math.isfinite(float(ppy)) or float(ppy) <= 0.0:
        raise ValueError(
            "this result carries no annualization, so nothing here can be scaled "
            "to a year; run it through a Backtester, or set periods_per_year on it"
        )
    return float(ppy)


def returns(result):
    """The per-period simple returns the statistics were computed from.

    A bar following a non-positive equity contributes zero rather than a ratio
    through zero, matching the engine exactly, which is what lets everything here
    reproduce the numbers the result already reports.
    """
    series = _curve(result)
    previous = series[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(previous > 0.0, series[1:] / previous - 1.0, 0.0)
    return out


def sharpe(result, risk_free=None):
    """Annualized Sharpe, recomputed here rather than read off the result.

    It exists to be compared against ``result.stats["sharpe"]``: they are two
    paths to one number, and a test asserts they agree. Everything else in this
    module rests on the return series being the same one the engine used.
    """
    rate = _risk_free(result, risk_free)
    values = returns(result)
    if values.size < 2:
        return 0.0
    ppy = _annualization(result)
    excess = float(values.mean()) - rate / ppy
    spread = float(values.std(ddof=1))
    if spread > 0.0:
        return excess / spread * math.sqrt(ppy)
    return math.inf if excess > 0.0 else 0.0


def _risk_free(result, given):
    if given is not None:
        return float(given)
    return float(getattr(result, "risk_free", 0.0) or 0.0)


def skew(result):
    """Sample skewness of the returns. Negative means the losses are the fat tail."""
    values = returns(result)
    if values.size < 3:
        return 0.0
    centred = values - values.mean()
    spread = float(np.sqrt((centred ** 2).mean()))
    if spread == 0.0:
        return 0.0
    return float((centred ** 3).mean()) / spread ** 3


def kurtosis(result, excess=False):
    """Sample kurtosis of the returns, NOT excess by default.

    A normal distribution scores 3.0 here. The probabilistic Sharpe below wants
    this convention, and passing an excess figure to it is the common way to get
    a confidently wrong answer, so this is the default and the other is a keyword.
    """
    values = returns(result)
    if values.size < 4:
        return 0.0 if excess else 3.0
    centred = values - values.mean()
    spread = float(np.sqrt((centred ** 2).mean()))
    if spread == 0.0:
        return 0.0 if excess else 3.0
    full = float((centred ** 4).mean()) / spread ** 4
    return full - 3.0 if excess else full


def autocorrelation(result, lag=1):
    """Autocorrelation of the returns at ``lag``.

    Reported beside the probabilistic Sharpe because that statistic assumes the
    returns are independent, and a strategy holding a position for two days on
    hourly bars has nothing of the kind. A high value here means the effective
    number of independent bets is far below the number of bars, and every
    confidence figure computed from the bar count is overstated.
    """
    values = returns(result)
    if values.size <= lag + 1:
        return 0.0
    a = values[:-lag] - values.mean()
    b = values[lag:] - values.mean()
    bottom = float((values - values.mean()).dot(values - values.mean()))
    if bottom == 0.0:
        return 0.0
    return float(a.dot(b)) / bottom


def drawdown(result):
    """The fall from the running peak at each bar, in percent, zero or negative.

    Seeded from the opening balance, so a strategy that loses on its first bar
    shows a drawdown on its first bar. Reading a peak off the equity curve alone
    cannot see that, which is what made the chart's drawdown panel disagree with
    the engine's own maximum.
    """
    series = _curve(result)
    peak = np.maximum.accumulate(series)
    with np.errstate(divide="ignore", invalid="ignore"):
        fall = np.where(peak > 0.0, (series / peak - 1.0) * 100.0, 0.0)
    return np.maximum(fall, -100.0)[1:]


def drawdown_table(result, top=5):
    """The worst ``top`` drawdowns, each with where it began, where it bottomed,
    where it recovered, and how long it took.

    One maximum flattens a run into a single number and hides the shape: five
    shallow falls and one long one are the same ``max_drawdown_pct`` and not
    remotely the same strategy to hold.
    """
    falls = drawdown(result)
    episodes = []
    start = None
    for i, value in enumerate(falls):
        if value < 0.0 and start is None:
            start = i
        elif value >= 0.0 and start is not None:
            episodes.append(_episode(falls, start, i))
            start = None
    if start is not None:
        episodes.append(_episode(falls, start, None))
    episodes.sort(key=lambda e: e["depth_pct"])
    return episodes[:top]


def _episode(falls, start, recovered):
    stop = len(falls) if recovered is None else recovered
    window = falls[start:stop]
    trough = start + int(np.argmin(window))
    return {
        "start_bar": start,
        "trough_bar": trough,
        "recovered_bar": recovered,
        "depth_pct": float(window.min()),
        "bars_to_trough": trough - start,
        "bars_under": stop - start,
    }


def time_under_water(result):
    """How long the account spent below a previous high: the longest run and the
    average, in bars, plus the share of the run spent there.

    A strategy that makes its money in two weeks and spends the other fifty below
    the old high is not the one the return figure describes.
    """
    falls = drawdown(result)
    under = falls < 0.0
    runs = []
    length = 0
    for flag in under:
        if flag:
            length += 1
        elif length:
            runs.append(length)
            length = 0
    if length:
        runs.append(length)
    return {
        "longest_bars": max(runs) if runs else 0,
        "average_bars": float(np.mean(runs)) if runs else 0.0,
        "episodes": len(runs),
        "share_pct": float(under.mean() * 100.0) if under.size else 0.0,
    }


def decompose(result):
    """Where the money went: gross price PnL, fees, funding, and what is still open.

    The four add up to the change in equity, by construction, so the identity is
    the check. On a perp this is the first thing to look at, because a large share
    of what looks like alpha in crypto is a funding carry wearing a costume, and a
    larger share of dead strategies died on the fee line rather than on the idea.

    ``unrealized`` is whatever the other three do not account for: a position
    still open at the end, marked, net of the entry fee it already paid. It is
    zero on a run that ends flat, and a test pins that.
    """
    trades = result.trades or []
    gross = float(sum(t["pnl"] for t in trades))
    fees = float(sum(t["fees"] for t in trades))
    funding = float((result.stats or {}).get("funding_paid", 0.0))
    series = _curve(result)
    net = float(series[-1] - series[0])
    return {
        "gross_pnl": gross,
        "fees": fees,
        "funding": funding,
        "unrealized": net - (gross - fees - funding),
        "net": net,
        "net_pct": net / series[0] * 100.0 if series[0] else 0.0,
    }


def long_short_split(result):
    """The trade statistics split by the side the position was on.

    In crypto almost every naive rule earns on the long side and bleeds on the
    short, and a single win rate averages the two into something that describes
    neither.
    """
    out = {}
    for side in ("buy", "sell"):
        rows = [t for t in (result.trades or []) if t["side"] == side]
        wins = [t for t in rows if t["net_pnl"] > 0.0]
        out["long" if side == "buy" else "short"] = {
            "trades": len(rows),
            "net_pnl": float(sum(t["net_pnl"] for t in rows)),
            "fees": float(sum(t["fees"] for t in rows)),
            "win_rate": len(wins) / len(rows) if rows else 0.0,
            "avg_bars_held": float(np.mean([t["bars_held"] for t in rows])) if rows else 0.0,
        }
    return out


def buy_and_hold(result, frame):
    """What holding the thing would have done over the same bars, and how the run
    compares: the excess return, the beta against it, and the information ratio.

    The first question anyone asks of a crypto strategy is whether it beat holding
    the coin, and until now nothing here could answer it.
    """
    close = _closes(frame)
    if close.size < 2:
        raise ValueError(f"buy_and_hold needs at least 2 bars, got {close.size}")
    held = close[1:] / close[:-1] - 1.0
    mine = returns(result)
    if held.size != mine.size:
        held = held[-mine.size:] if held.size > mine.size else held
        mine = mine[-held.size:]
    ppy = _annualization(result)
    spread = mine - held
    tracking = float(spread.std(ddof=1)) if spread.size > 1 else 0.0
    variance = float(held.var(ddof=1)) if held.size > 1 else 0.0
    return {
        "hold_return_pct": float(close[-1] / close[0] - 1.0) * 100.0,
        "hold_sharpe": _ratio(float(held.mean()), float(held.std(ddof=1)), ppy),
        "excess_return_pct": (result.stats or {}).get("total_return_pct", 0.0)
        - float(close[-1] / close[0] - 1.0) * 100.0,
        "beta": float(np.cov(mine, held, ddof=1)[0, 1] / variance) if variance else 0.0,
        "information_ratio": _ratio(float(spread.mean()), tracking, ppy),
    }


def _closes(frame):
    if hasattr(frame, "close"):
        return np.asarray(frame.close, dtype=np.float64)
    data = np.asarray(frame, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 5:
        raise TypeError(
            "frame must be a DataFrame with a close column, or a (T, 5) OHLCV array"
        )
    return data[:, 3]


def _ratio(mean, spread, ppy):
    if spread > 0.0:
        return mean / spread * math.sqrt(ppy)
    return math.inf if mean > 0.0 else 0.0


def value_at_risk(result, alpha=0.95):
    """The per-period loss that only ``1 - alpha`` of bars are worse than, as a
    positive percent. Historical, not a normal assumption.
    """
    values = returns(result)
    if values.size == 0:
        return 0.0
    return float(-np.quantile(values, 1.0 - alpha) * 100.0)


def conditional_value_at_risk(result, alpha=0.95):
    """The mean loss across the bars worse than the value at risk, positive percent.

    The number to look at rather than the value at risk itself: it says how bad
    the bad days are, where the quantile only says where they start.
    """
    values = returns(result)
    if values.size == 0:
        return 0.0
    cutoff = np.quantile(values, 1.0 - alpha)
    tail = values[values <= cutoff]
    return float(-tail.mean() * 100.0) if tail.size else 0.0


def probabilistic_sharpe(result, benchmark=0.0):
    """The probability the true Sharpe is above ``benchmark``, given the sample
    length and how far the returns are from normal (Bailey and Lopez de Prado).

    ``benchmark`` is an ANNUALIZED Sharpe, like everything else the library
    reports, and is de-annualized here along with the observed one. Handing this
    the annualized figure by hand is the standard way to get a confidently wrong
    answer, so it is not possible to: this takes the result, not a number.

    Assumes the returns are independent. They are usually not, so read
    ``autocorrelation`` beside it: a strategy holding a position for two days on
    hourly candles has a few hundred independent bets, not a few thousand, and
    this figure will happily read 0.999 on noise. Raises rather than returning
    NaN when the sample cannot support the estimate (ADR 0007).
    """
    values = returns(result)
    if values.size < 4:
        raise ValueError(
            f"probabilistic_sharpe needs at least 4 returns, got {values.size}"
        )
    if float(values.std(ddof=1)) == 0.0:
        # a dead-flat account has a sharpe of exactly zero and the arithmetic runs
        # right through to a tidy 0.5, which reads as a coin flip on whether the
        # strategy is good. There is no variation here to infer anything from
        raise ValueError(
            "every return is identical, so this run carries no variation to "
            "estimate from and no probability can be attached to its sharpe"
        )
    ppy = _annualization(result)
    observed = sharpe(result) / math.sqrt(ppy)
    target = float(benchmark) / math.sqrt(ppy)
    variance = 1.0 - skew(result) * observed + (kurtosis(result) - 1.0) / 4.0 * observed ** 2
    if variance <= 0.0:
        raise ValueError(
            f"the sharpe estimator's variance came out at {variance:.4g}, which is "
            f"not positive, so this sample cannot support a probability; it means "
            f"the returns are far enough from normal that the approximation breaks"
        )
    z = (observed - target) * math.sqrt(values.size - 1) / math.sqrt(variance)
    return _normal_cdf(z)


def min_track_record_length(result, benchmark=0.0, confidence=0.95):
    """How many bars it would take before the Sharpe is distinguishable from
    ``benchmark`` at ``confidence``, in bars and in wall time.

    Answers "is this backtest long enough" with a number. Read ``num_trades``
    beside it: three thousand bars carrying forty round trips is forty bets, and
    it is the bets that carry the information.
    """
    values = returns(result)
    ppy = _annualization(result)
    observed = sharpe(result) / math.sqrt(ppy)
    target = float(benchmark) / math.sqrt(ppy)
    if not math.isfinite(observed) or observed <= target:
        raise ValueError(
            f"the observed sharpe is not above the benchmark, so no sample length "
            f"makes it distinguishable; observed {observed * math.sqrt(ppy):.4g} "
            f"annualized against a benchmark of {benchmark:.4g}"
        )
    variance = 1.0 - skew(result) * observed + (kurtosis(result) - 1.0) / 4.0 * observed ** 2
    bars = 1.0 + variance * (_normal_ppf(confidence) / (observed - target)) ** 2
    return {
        "bars": bars,
        "years": bars / ppy,
        "have_bars": int(values.size),
        "enough": values.size >= bars,
        "num_trades": int((result.stats or {}).get("num_trades", 0)),
    }


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p):
    # Acklam's rational approximation, accurate to about 1.15e-9 across the whole
    # range, which is far past what a confidence level needs. Written out rather
    # than pulled from scipy, because scipy is not a dependency and will not
    # become one for one function
    if not 0.0 < p < 1.0:
        raise ValueError(f"confidence must be above 0 and below 1, got {p}")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    low, high = 0.02425, 1.0 - 0.02425
    if p < low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def cost_curve(strategy, data, costs=(0.0, 2.0, 5.0, 10.0, 20.0), **config):
    """Re-run ``strategy`` over ``data`` at each round-trip cost in ``costs``, in
    basis points, and report what survives.

    The shipped defaults are a frictionless venue: no slippage, no impact, and one
    order allowed to eat a whole bar. An edge found there is an edge nobody can
    trade, and the useful question is not whether a strategy makes money but how
    much friction it survives. Each cost is split evenly across the two sides of a
    round trip and applied to both fee rates, so 10 means five basis points in and
    five out. Anything else in ``config`` goes to the ``Backtester`` unchanged.

    ``strategy`` is a ``Strategy`` subclass, which is rebuilt for each run, or an
    instance, which is reused.
    """
    from .backtest import Backtester

    _own_the_fees(config, "cost_curve")
    out = []
    for bps in costs:
        rate = float(bps) / 2.0 / 10_000.0
        result = Backtester(
            data, fee_taker=rate, fee_maker=rate, **config
        ).run(_fresh(strategy))
        out.append({
            "round_trip_bps": float(bps),
            "total_return_pct": result.stats["total_return_pct"],
            "sharpe": result.stats["sharpe"],
            "num_trades": result.stats["num_trades"],
            "fees": float(sum(t["fees"] for t in result.trades)),
        })
    return out


def breakeven_bps(strategy, data, ceiling=100.0, tolerance=0.05, **config):
    """The round-trip cost, in basis points, at which ``strategy`` stops making
    money. ``None`` when it is already losing at zero cost, and ``ceiling`` when it
    survives all the way there.

    "This dies at 8 basis points round trip and you pay 6" is a sentence worth more
    in the first week than any probability, because it says how much of the edge is
    real and how much is the absence of friction.
    """
    from .backtest import Backtester

    _own_the_fees(config, "breakeven_bps")

    def earned(bps):
        rate = float(bps) / 2.0 / 10_000.0
        run = Backtester(data, fee_taker=rate, fee_maker=rate, **config)
        return run.run(_fresh(strategy)).stats["total_return_pct"]

    if earned(0.0) <= 0.0:
        return None
    if earned(ceiling) > 0.0:
        return float(ceiling)
    # bisect: the return is monotone in the cost for a fixed set of decisions, and
    # not quite monotone once the costs change which trades happen, so this finds
    # a crossing rather than the crossing. It is the honest kind of answer anyway
    low, high = 0.0, float(ceiling)
    while high - low > tolerance:
        middle = (low + high) / 2.0
        if earned(middle) > 0.0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _fresh(strategy):
    # a class is rebuilt per run so nothing carries over between them; an instance
    # is taken as given, since the caller may have configured it
    return strategy() if isinstance(strategy, type) else strategy


def _own_the_fees(config, where):
    # sweeping the fee IS the job, so one arriving in the configuration is a
    # contradiction rather than an override. Without this it surfaced as Python's
    # own "got multiple values for keyword argument", which says nothing about why
    clash = sorted(key for key in ("fee_taker", "fee_maker") if key in config)
    if clash:
        raise TypeError(
            f"{where} sets {clash} itself, because sweeping the cost is what it "
            f"does; pass the rest of the configuration and leave those out. To "
            f"vary a cost it does not set, slippage_bps and impact go through"
        )


def excursions(result, frame):
    """The worst and best each trade went while it was open, in percent of its entry.

    The maximum adverse excursion is where a stop would have been hit, and the
    maximum favourable is what a target would have caught. A trade log says what a
    rule earned; this says what it lived through, and it is the direct answer to
    "where does my stop go".
    """
    high, low = _highs_lows(frame)
    out = []
    for trade in result.trades or []:
        first, last = trade["entry_tick"], trade["exit_tick"]
        if not 0 <= first <= last < len(high):
            continue
        entry = trade["entry_price"]
        peak, trough = float(high[first:last + 1].max()), float(low[first:last + 1].min())
        if trade["side"] == "buy":
            best, worst = peak - entry, trough - entry
        else:
            best, worst = entry - trough, entry - peak
        out.append({
            "entry_tick": first,
            "exit_tick": last,
            "side": trade["side"],
            "net_pnl": trade["net_pnl"],
            "best_pct": best / entry * 100.0 if entry else 0.0,
            "worst_pct": worst / entry * 100.0 if entry else 0.0,
        })
    return out


def session_buckets(result, frame, by="hour"):
    """Trade PnL grouped by hour of day or day of week, booked at the exit bar.

    Crypto trades around the clock, and an hourly strategy routinely has its whole
    edge inside the few hours a day funding is stamped. A single number cannot show
    that, and a bucket that holds the entire result is a stronger sign of an overfit
    than any statistic.
    """
    if by not in ("hour", "weekday"):
        raise ValueError(f"by must be 'hour' or 'weekday', got {by!r}")
    stamps = _stamps(frame)
    unit = stamps.astype("datetime64[h]").astype("int64")
    keys = (unit % 24) if by == "hour" else ((unit // 24 + 4) % 7)
    buckets = {}
    for trade in result.trades or []:
        exit_tick = trade["exit_tick"]
        if not 0 <= exit_tick < keys.size:
            continue
        slot = buckets.setdefault(
            int(keys[exit_tick]), {"trades": 0, "net_pnl": 0.0, "wins": 0}
        )
        slot["trades"] += 1
        slot["net_pnl"] += trade["net_pnl"]
        slot["wins"] += 1 if trade["net_pnl"] > 0.0 else 0
    for slot in buckets.values():
        slot["win_rate"] = slot["wins"] / slot["trades"] if slot["trades"] else 0.0
    return dict(sorted(buckets.items()))


def _highs_lows(frame):
    if hasattr(frame, "high") and hasattr(frame, "low"):
        return (np.asarray(frame.high, dtype=np.float64),
                np.asarray(frame.low, dtype=np.float64))
    data = np.asarray(frame, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 5:
        raise TypeError(
            "frame must be a DataFrame with high and low columns, or a (T, 5) "
            "OHLCV array"
        )
    return data[:, 1], data[:, 2]


def _stamps(frame):
    # only a real clock can be bucketed by hour; an array of bars carries none, and
    # bucketing by bar index instead would look like an answer and be one about
    # nothing
    index = getattr(frame, "index", None)
    if index is None:
        raise TypeError(
            "session_buckets needs the timestamps, so pass the DataFrame the "
            "backtest ran on rather than the (T, 5) array"
        )
    from ._data import _epoch_seconds

    seconds = _epoch_seconds(np.asarray(index))
    if seconds is None:
        raise TypeError(
            "the frame's index carries no timestamps, so its bars cannot be "
            "grouped by hour or by weekday"
        )
    return seconds.astype("int64").astype("datetime64[s]")


def report(result, frame=None):
    """Everything in this module as one flat dict, for storing or comparing runs.

    ``frame`` is optional and only the benchmark needs it.
    """
    out = dict(result.stats or {})
    out.update({f"decompose_{k}": v for k, v in decompose(result).items()})
    out.update({f"under_water_{k}": v for k, v in time_under_water(result).items()})
    out["skew"] = skew(result)
    out["kurtosis"] = kurtosis(result)
    out["autocorr_1"] = autocorrelation(result)
    out["value_at_risk_pct"] = value_at_risk(result)
    out["conditional_value_at_risk_pct"] = conditional_value_at_risk(result)
    sides = long_short_split(result)
    for name, block in sides.items():
        out.update({f"{name}_{k}": v for k, v in block.items()})
    try:
        out["probabilistic_sharpe"] = probabilistic_sharpe(result)
    except ValueError:
        out["probabilistic_sharpe"] = None
    if frame is not None:
        out.update({f"hold_{k}": v for k, v in buy_and_hold(result, frame).items()})
    return out


def compare(results, keys=None):
    """Line several runs up against each other and print the rows, one per result.

    A notebook accumulates a great many backtests and nothing about a
    ``BacktestResult`` on its own says which data or which costs produced it. Each
    row therefore leads with the fingerprint of the bars and the strategy that ran
    on them, so two rows that differ only in a fee are visibly two rows that differ
    only in a fee. Returns the rows it printed.

    ``results`` may be a list or a dict keyed by whatever you want the rows called.
    """
    if hasattr(results, "items"):
        named = list(results.items())
    else:
        named = [(r.strategy or f"run {i}", r) for i, r in enumerate(results)]
    if not named:
        return []
    shown = list(keys) if keys else ["total_return_pct", "sharpe",
                                     "max_drawdown_pct", "num_trades"]
    width = max(len(str(name)) for name, _ in named)
    # one wider than the longest label, so a label that exactly fills the column
    # cannot run into its neighbour; "max drawdown %" is precisely 14 characters
    column = max(len(_short(key)) for key in shown) + 2
    header = f"  {'':<{width}}  {'data':<8}"
    for key in shown:
        header += f"{_short(key):>{column}}"
    print(header)
    rows = []
    for name, result in named:
        stats = result.stats or {}
        line = f"  {name:<{width}}  {result.data_hash or '':<8}"
        for key in shown:
            value = stats.get(key)
            line += (f"{value:>{column},.3f}" if isinstance(value, float)
                     else f"{value!s:>{column}}")
        print(line)
        row = {"name": name}
        row.update(result.to_dict())
        rows.append(row)
    return rows


def _short(key):
    return key.replace("_pct", " %").replace("_", " ")


def summary(result, frame=None):
    """Print the headline of a run: what it made, what it cost, and what that is
    worth believing. Returns the dict it printed from.
    """
    stats = result.stats or {}
    money = decompose(result)
    water = time_under_water(result)
    sides = long_short_split(result)
    rows = [
        ("return", f"{stats.get('total_return_pct', 0.0):>12,.2f} %"),
        ("sharpe", f"{stats.get('sharpe', 0.0):>12,.2f}"),
        ("max drawdown", f"{stats.get('max_drawdown_pct', 0.0):>12,.2f} %"),
        ("", ""),
        ("gross pnl", f"{money['gross_pnl']:>12,.2f}"),
        ("fees", f"{-money['fees']:>12,.2f}"),
        ("funding", f"{-money['funding']:>12,.2f}"),
        ("still open", f"{money['unrealized']:>12,.2f}"),
        ("net", f"{money['net']:>12,.2f}"),
        ("", ""),
        ("trades", f"{stats.get('num_trades', 0):>12,}"),
        ("long / short", f"{sides['long']['trades']:>6,} /{sides['short']['trades']:>5,}"),
        ("under water", f"{water['share_pct']:>12,.1f} % of bars"),
        ("longest", f"{water['longest_bars']:>12,} bars"),
    ]
    if frame is not None:
        hold = buy_and_hold(result, frame)
        rows += [("", ""), ("vs holding", f"{hold['excess_return_pct']:>12,.2f} %")]
    for label, value in rows:
        print(f"  {label:<14}{value}" if label else "")
    return report(result, frame)
