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
import warnings

import numpy as np

__all__ = [
    "returns",
    "segment",
    "cost_curve",
    "breakeven_bps",
    "excursions",
    "session_buckets",
    "period_returns",
    "sharpe",
    "rolling_sharpe",
    "sharpe_interval",
    "skew",
    "kurtosis",
    "omega",
    "tail_ratio",
    "drawdown",
    "drawdown_table",
    "time_under_water",
    "ulcer_index",
    "decompose",
    "trade_stats",
    "long_short_split",
    "buy_and_hold",
    "value_at_risk",
    "conditional_value_at_risk",
    "probabilistic_sharpe",
    "deflated_sharpe",
    "deflation_threshold",
    "min_track_record_length",
    "autocorrelation",
    "effective_sample",
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

    A strategy holding a position for two days on hourly bars has nothing like
    independent returns, and a high value here means the number of independent
    bets is far below the number of bars. ``effective_sample`` turns this into
    that count, and the two statistics that depend on a sample size use it.
    """
    lag = int(lag)
    if lag < 0:
        raise ValueError(f"lag must be zero or a positive number of bars, got {lag}")
    values = returns(result)
    if lag == 0:
        # a series correlates perfectly with itself, and the general branch below
        # divides the whole sum of squares by itself through an empty slice
        return 1.0 if values.size else 0.0
    if values.size <= lag + 1:
        return 0.0
    a = values[:-lag] - values.mean()
    b = values[lag:] - values.mean()
    bottom = float((values - values.mean()).dot(values - values.mean()))
    if bottom == 0.0:
        return 0.0
    return float(a.dot(b)) / bottom


def effective_sample(result):
    """How many independent bets the return series is worth, which is fewer than
    the number of bars whenever a position is held across them.

    A confidence figure computed from the bar count answers "how sure am I, given
    this many independent observations", and a strategy holding for two days on
    hourly candles does not have a few thousand of those, it has a few hundred.
    This is the standard effective sample size for a persistent series,
    ``n (1 - r) / (1 + r)`` at the first autocorrelation ``r``, and it is what
    ``probabilistic_sharpe`` and ``min_track_record_length`` count in (ADR 0061).

    Negative autocorrelation is floored away rather than credited. It would make
    this larger than the bar count, which reads as free confidence bought by
    mean reversion, and the correction exists to remove overstatement rather than
    to hand any back.
    """
    values = returns(result)
    size = values.size
    if size < 4:
        return float(size)
    rho = autocorrelation(result, lag=1)
    if not math.isfinite(rho) or rho <= 0.0:
        return float(size)
    rho = min(rho, 0.99)
    return max(2.0, float(size) * (1.0 - rho) / (1.0 + rho))


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
    # the notional both sides of every round trip moved, over the opening balance.
    # Read with `fee_share`: a strategy paying half its gross edge away in costs
    # has a cost problem rather than an idea problem, and neither number alone
    # says which one you have
    traded = float(sum(t["size"] * (t["entry_price"] + t["exit_price"]) for t in trades))
    return {
        "gross_pnl": gross,
        "fees": fees,
        "funding": funding,
        "unrealized": net - (gross - fees - funding),
        "net": net,
        "net_pct": net / series[0] * 100.0 if series[0] else 0.0,
        "fee_share": _ratio(fees, abs(gross), 1.0),
        "turnover": traded / series[0] if series[0] else 0.0,
    }


def trade_stats(result):
    """The shape of the trade distribution, which a win rate hides.

    ``win_rate`` and ``profit_factor`` are the same pair for a rule that grinds
    out small wins and for one that is short gamma waiting for the bar that ends
    it. A 70% win rate at a payoff of 0.3 is a losing rule, and until now the
    numbers to see that were sitting unread in ``result.trades``.

    ``expectancy`` is the average net PnL of a trade, in quote, and
    ``expectancy_pct`` the same as a percent of the opening balance.
    ``payoff`` is the average win over the average loss, both positive, and
    infinite when nothing lost. ``max_consecutive_losses`` is the run of them a
    live account would have had to sit through, which is what actually decides
    whether a rule gets switched off.
    """
    trades = result.trades or []
    nets = [float(t["net_pnl"]) for t in trades]
    wins = [v for v in nets if v > 0.0]
    losses = [-v for v in nets if v < 0.0]
    streak = worst = 0
    for value in nets:
        streak = streak + 1 if value < 0.0 else 0
        worst = max(worst, streak)
    average_win = float(np.mean(wins)) if wins else 0.0
    average_loss = float(np.mean(losses)) if losses else 0.0
    opening = getattr(result, "initial", None)
    expectancy = float(np.mean(nets)) if nets else 0.0
    return {
        "trades": len(nets),
        "wins": len(wins),
        "losses": len(losses),
        "avg_win": average_win,
        "avg_loss": average_loss,
        "payoff": (average_win / average_loss if average_loss > 0.0
                   else (math.inf if average_win > 0.0 else 0.0)),
        "expectancy": expectancy,
        "expectancy_pct": (expectancy / float(opening) * 100.0
                           if opening else 0.0),
        "largest_win": max(nets) if nets else 0.0,
        "largest_loss": min(nets) if nets else 0.0,
        "max_consecutive_losses": worst,
        "avg_bars_held": (float(np.mean([t["bars_held"] for t in trades]))
                          if trades else 0.0),
    }


def omega(result, threshold=0.0):
    """Total gain above ``threshold`` over total loss below it, per period.

    The whole return distribution in one number rather than its first two
    moments, so a fat left tail cannot hide behind a tidy standard deviation the
    way it does under Sharpe. ``threshold`` is an annualized return, de-annualized
    here like every other rate in this module. One means break even.
    """
    values = returns(result)
    if values.size == 0:
        return 0.0
    bar = float(threshold) / _annualization(result)
    above = float(np.maximum(values - bar, 0.0).sum())
    below = float(np.maximum(bar - values, 0.0).sum())
    return _ratio(above, below, 1.0)


def tail_ratio(result, alpha=0.95):
    """The size of the right tail over the size of the left, at ``1 - alpha``.

    Above one the good bars are bigger than the bad ones. Below one the run makes
    its money in small pieces and gives it back in large ones, which is the
    profile that survives a backtest and not a year.
    """
    values = returns(result)
    if values.size < 2:
        return 0.0
    alpha = _confidence(alpha, "tail_ratio")
    right = float(np.quantile(values, alpha))
    left = float(-np.quantile(values, 1.0 - alpha))
    return _ratio(right, left, 1.0)


def ulcer_index(result):
    """The root mean square of the drawdown, in percent.

    A maximum says how deep the worst fall was and nothing about how long the
    account sat in one. This charges for depth and duration together, so a run
    that spends a year down 15% scores worse than one that touches 30% for a
    week, which is the right way round for anybody who has to hold it.
    """
    falls = drawdown(result)
    if falls.size == 0:
        return 0.0
    return float(np.sqrt((falls ** 2).mean()))


def rolling_sharpe(result, window=252):
    """Annualized Sharpe over a rolling ``window`` of bars, as a length-``T`` array
    aligned to the run's bars, warm-up as ``NaN``.

    Decay is a shape, not a scalar, and a single Sharpe over three years cannot
    show that the whole edge was in the first six months. Aligned and padded on
    the same rule ``emsl.ta`` uses, so it goes straight into ``emsl.chart``
    beside the equity curve with no padding decision in between.
    """
    values = returns(result)
    window = int(window)
    if window < 2:
        raise ValueError(f"rolling_sharpe needs a window of at least 2, got {window}")
    bars = values.size + 1
    out = np.full(bars, np.nan, dtype=np.float64)
    if window > values.size:
        return out
    ppy = _annualization(result)
    per = _risk_free(result, None) / ppy
    frames = np.lib.stride_tricks.sliding_window_view(values, window)
    means = frames.mean(axis=-1) - per
    spreads = frames.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scored = np.where(spreads > 0.0, means / spreads * math.sqrt(ppy),
                          np.where(means > 0.0, np.inf, 0.0))
    # a window ending on return j ends on bar j + 1, and the returns start at bar 1
    out[window:] = scored
    return out


def period_returns(result, frame, by="month"):
    """The run broken down by calendar period: month, quarter or year.

    "Is the whole edge in one quarter" is the first stability question anybody
    asks of a crypto backtest, and a single number cannot answer it. Each period
    carries the statistics of that stretch alone, computed on the equity the
    account actually carried into it (``segment``), so the periods compound to the
    run rather than each starting from a fresh balance.
    """
    units = {"month": "M", "quarter": "M", "year": "Y"}
    if by not in units:
        raise ValueError(f"by must be 'month', 'quarter' or 'year', got {by!r}")
    stamps = _stamps(frame)
    _same_bars(result, frame, stamps.size, "period_returns")
    keys = stamps.astype(f"datetime64[{units[by]}]")
    if by == "quarter":
        months = keys.astype("int64")
        keys = (months - months % 3).astype("datetime64[M]")
    out = []
    edges = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
    for position, first in enumerate(edges):
        last = int(edges[position + 1]) if position + 1 < edges.size else keys.size
        if last - int(first) < 2:
            continue
        block = segment(result, int(first), last)
        block["period"] = str(keys[int(first)])
        block["start_bar"] = int(first)
        block["bars"] = last - int(first)
        out.append(block)
    return out


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

    ``frame`` needs one price per bar of the run, and that is the only thing
    checked. A **different asset** over the same bars is not a mistake here, it is
    the point: ``beta`` and ``information_ratio`` are benchmark statistics and are
    only interesting against something other than what you traded, so benchmarking
    an ETH strategy against holding BTC is a supported question. A different
    NUMBER of bars is a mistake, and used to be absorbed by trimming the two
    series to a common length, from opposite ends: the run's last returns against
    the frame's first (ADR 0059).
    """
    close = _closes(frame)
    if close.size < 2:
        raise ValueError(f"buy_and_hold needs at least 2 bars, got {close.size}")
    mine = returns(result)
    if close.size != mine.size + 1:
        raise ValueError(
            f"buy_and_hold needs one price per bar of the run, but the run covered "
            f"{mine.size + 1} bars and this frame carries {close.size}; a different "
            f"asset over the same bars is fine and is what beta is for, a different "
            f"number of bars is not"
        )
    held = close[1:] / close[:-1] - 1.0
    ppy = _annualization(result)
    rate = _risk_free(result, None)
    spread = mine - held
    tracking = float(spread.std(ddof=1)) if spread.size > 1 else 0.0
    variance = float(held.var(ddof=1)) if held.size > 1 else 0.0
    return {
        "hold_return_pct": float(close[-1] / close[0] - 1.0) * 100.0,
        # net of the same risk-free rate the run's own sharpe is net of, or the
        # two ratios printed side by side are different quantities
        "hold_sharpe": _ratio(float(held.mean()) - rate / ppy,
                              float(held.std(ddof=1)), ppy),
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


def segment(result, start=0, stop=None):
    """The run's statistics over bars ``[start, stop)`` alone.

    The same keys the engine reports, on the equity the account actually carried
    into that stretch rather than a fresh balance, so a bad first quarter shrinks
    the size available in the second exactly as it did in the run. ``start`` is
    inclusive and ``stop`` exclusive, both in bar indices, and a trade belongs to
    the stretch its EXIT falls in, which is the same rule ``session_buckets`` uses.

    Two keys the engine reports are absent, because a finished result does not
    carry what they are computed from: ``exposure_pct`` needs the per-bar record
    of whether a position was held, and ``num_fills`` counts fills rather than
    trades.

    This is a second path to numbers the engine already produces, which is the
    drift this codebase pays for elsewhere, so a test pins ``segment`` over the
    whole run against ``result.stats`` key by key (ADR 0060).
    """
    curve = np.asarray(result.equity_curve, dtype=np.float64)
    bars = curve.size + 1
    start = int(start)
    stop = bars if stop is None else int(stop)
    if not 0 <= start < stop <= bars:
        raise ValueError(
            f"segment needs 0 <= start < stop <= {bars}, got start={start} and "
            f"stop={stop}"
        )
    # the balance carried INTO bar `start`, so consecutive segments partition the
    # returns exactly: [a, b) owns the returns landing on bars a to b - 1, and the
    # one landing on bar b opens the next. Seeding from the balance AT bar `start`
    # instead loses one return per boundary, which compounds two halves of a run
    # to slightly less than the run
    opening = _curve(result)[max(0, start - 1)]
    trades = [t for t in (result.trades or []) if start <= t["exit_tick"] < stop]
    series = np.concatenate(([opening], curve[max(0, start - 1):stop - 1]))
    return _stats(series, trades, _annualization(result), _risk_free(result, None))


def _stats(series, trades, ppy, rate):
    # the engine's own arithmetic, from bar-engine/src/stats.rs, over an equity
    # path that already includes the balance it started from. Every convention is
    # ADR 0007's: sample deviation under sharpe, population under sortino's
    # downside, drawdown capped at a total loss, trade metrics net of the fee
    opening, final = float(series[0]), float(series[-1])
    n_returns = series.size - 1
    if opening > 0.0 and n_returns > 0:
        total = final / opening - 1.0
        cagr = (final / opening) ** (ppy / n_returns) - 1.0 if final > 0.0 else -1.0
    else:
        total, cagr = 0.0, 0.0
    previous = series[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(previous > 0.0, series[1:] / previous - 1.0, 0.0)
    per = rate / ppy
    if rets.size >= 2:
        excess = float(rets.mean()) - per
        spread = float(rets.std(ddof=1))
        downside = float(np.sqrt((np.minimum(rets - per, 0.0) ** 2).mean()))
        sharpe_, sortino = _ratio(excess, spread, ppy), _ratio(excess, downside, ppy)
        volatility = spread * math.sqrt(ppy)
    else:
        sharpe_, sortino, volatility = 0.0, 0.0, 0.0
    peak = np.maximum.accumulate(series)
    with np.errstate(divide="ignore", invalid="ignore"):
        falls = np.where(peak > 0.0, np.minimum((peak - series) / peak, 1.0), 0.0)
    max_dd = float(falls.max()) if falls.size else 0.0
    nets = [float(t["net_pnl"]) for t in trades]
    wins = [v for v in nets if v > 0.0]
    losses = [-v for v in nets if v < 0.0]
    return {
        "total_return_pct": total * 100.0,
        "net_profit_pct": total * 100.0,
        "cagr_pct": cagr * 100.0,
        "sharpe": sharpe_,
        "sortino": sortino,
        "calmar": _ratio(cagr, max_dd, 1.0),
        "max_drawdown_pct": max_dd * 100.0,
        "volatility_pct": volatility * 100.0,
        "win_rate": len(wins) / len(nets) if nets else 0.0,
        "profit_factor": _ratio(sum(wins), sum(losses), 1.0),
        "num_trades": len(nets),
        "avg_trade_pct": (sum(nets) / len(nets) / opening * 100.0
                          if nets and opening > 0.0 else 0.0),
    }


def _confidence(alpha, where):
    # alpha is the CONFIDENCE, so 0.95 is the loss exceeded one bar in twenty.
    # Half the industry writes the tail probability instead, and 0.05 read as a
    # confidence asks for the ninety-fifth percentile of the GAINS, which comes
    # back as a negative loss and looks like a number rather than a mistake
    alpha = float(alpha)
    if not 0.5 <= alpha < 1.0:
        raise ValueError(
            f"{where} takes a confidence, so alpha is at least 0.5 and below 1, "
            f"got {alpha}; for the worst one bar in twenty pass 0.95, not 0.05"
        )
    return alpha


def value_at_risk(result, alpha=0.95):
    """The per-period loss that only ``1 - alpha`` of bars are worse than, as a
    positive percent. Historical, not a normal assumption.

    ``alpha`` is the confidence, so the default asks about the worst bar in
    twenty. It comes back NEGATIVE when even that bar made money, which is the
    honest answer for a run whose whole return distribution sits above zero,
    rather than a floor at zero pretending there was a loss.
    """
    values = returns(result)
    if values.size == 0:
        return 0.0
    alpha = _confidence(alpha, "value_at_risk")
    return float(-np.quantile(values, 1.0 - alpha) * 100.0)


def conditional_value_at_risk(result, alpha=0.95):
    """The mean loss across the bars worse than the value at risk, positive percent.

    The number to look at rather than the value at risk itself: it says how bad
    the bad days are, where the quantile only says where they start.
    """
    values = returns(result)
    if values.size == 0:
        return 0.0
    alpha = _confidence(alpha, "conditional_value_at_risk")
    cutoff = np.quantile(values, 1.0 - alpha)
    tail = values[values <= cutoff]
    return float(-tail.mean() * 100.0) if tail.size else 0.0


def probabilistic_sharpe(result, benchmark=0.0, independent=False):
    """The probability the true Sharpe is above ``benchmark``, given the sample
    length and how far the returns are from normal (Bailey and Lopez de Prado).

    ``benchmark`` is an ANNUALIZED Sharpe, like everything else the library
    reports, and is de-annualized here along with the observed one. Handing this
    the annualized figure by hand is the standard way to get a confidently wrong
    answer, so it is not possible to: this takes the result, not a number.

    The published estimator counts bars and assumes they are independent bets.
    They are usually not, and this library is pointed at hourly crypto where a
    position is routinely held for days, so the sample size counted here is
    ``effective_sample`` rather than the bar count (ADR 0061). Pass
    ``independent=True`` for the published formula unadjusted, which is the right
    thing when the returns really are independent and the wrong thing by a wide
    margin when they are not. Raises rather than returning NaN when the sample
    cannot support the estimate (ADR 0007).
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
    variance = _psr_variance(result, observed)
    size = values.size if independent else effective_sample(result)
    z = (observed - target) * math.sqrt(size - 1) / math.sqrt(variance)
    return _normal_cdf(z)


def sharpe_interval(result, confidence=0.95, independent=False):
    """The annualized Sharpe with a confidence interval around it.

    "Sharpe 1.4, 95% interval 0.2 to 2.6" is the sentence a reader wants and the
    one nothing here could produce: ``probabilistic_sharpe`` answers a yes-or-no
    question about a benchmark, which is a different question. Same estimator,
    same non-normality correction, same effective sample size (ADR 0061), read as
    an interval instead of a tail.

    ``low`` at or below zero says the run does not distinguish itself from nothing
    at this confidence, however good the point estimate looks.
    """
    values = returns(result)
    if values.size < 4:
        raise ValueError(
            f"sharpe_interval needs at least 4 returns, got {values.size}"
        )
    ppy = _annualization(result)
    observed = sharpe(result) / math.sqrt(ppy)
    variance = _psr_variance(result, observed)
    size = values.size if independent else effective_sample(result)
    half = _normal_ppf(1.0 - (1.0 - float(confidence)) / 2.0) * math.sqrt(
        variance / (size - 1)
    )
    return {
        "sharpe": observed * math.sqrt(ppy),
        "low": (observed - half) * math.sqrt(ppy),
        "high": (observed + half) * math.sqrt(ppy),
        "confidence": float(confidence),
        "effective_bars": float(size),
        "bars": int(values.size),
    }


def _psr_variance(result, observed):
    # the variance of the sharpe estimator under non-normal returns: the term
    # Bailey and Lopez de Prado put under the root. Kurtosis is the plain kind,
    # not excess, which is why `kurtosis` defaults to that and says so
    variance = (
        1.0 - skew(result) * observed
        + (kurtosis(result) - 1.0) / 4.0 * observed ** 2
    )
    if variance <= 0.0:
        raise ValueError(
            f"the sharpe estimator's variance came out at {variance:.4g}, which is "
            f"not positive, so this sample cannot support a probability; it means "
            f"the returns are far enough from normal that the approximation breaks"
        )
    return variance


_EULER = 0.5772156649015329


def deflated_sharpe(study, null):
    """The probability the winner of ``study`` is real, given how many times you
    looked (Bailey and Lopez de Prado, 2014).

    A search returns the best of many noisy scores, so its winner is high partly
    because it is good and partly because you looked a lot. This computes the
    Sharpe you would expect the best of that many looks to reach **by luck alone**,
    and asks for the probability the winner beats it.

    ``null`` is a second ``TuneResult`` from a **random** search over the same
    space and the same bars::

        study = venue.tune(SmaCross, space, candles, n_trials=200, oos=0.3)
        null = venue.tune(SmaCross, space, candles, n_trials=200, oos=0.3,
                          sampler="random")
        metrics.deflated_sharpe(study, null)

    Both take the same ``oos``, because the fingerprint is checked and a holdout
    changes which bars a search actually ran on. A null fitted on the whole
    series is not a null for a search fitted on 70% of it.

    It is required, and that is the whole design. The threshold depends on how
    many independent looks were taken and how much the scores vary across them,
    and a TPE search supplies neither honestly: it concentrates its trials in the
    winning basin, so the trials are not independent draws and the spread between
    them **shrinks as the search converges**. Computed from the search's own
    trials, this number would grow more permissive the harder you overfit, which
    is exactly backwards. A random search over the same space is an honest "best
    of N looks at this space" and cannot do that (ADR 0054).

    The two arguments answer different halves of the threshold, and the null
    answers only one of them: how far a sharpe scatters over this space. How many
    times you looked is a property of **your** search and is read off it (ADR
    0058).
    """
    _trials, looks, spread = _null_shape(study, null)
    return probabilistic_sharpe(
        study.best_result, benchmark=deflation_threshold(spread, looks)
    )


def deflation_threshold(spread, looks):
    """The Sharpe the best of ``looks`` independent draws would reach by luck
    alone, when the draws are spread ``spread`` apart.

    The expected maximum of a sample from a normal, via its Gumbel limit. It rises
    with both arguments, which is the whole intuition: the more times you look and
    the more the scores scatter, the higher a winner has to be before it means
    anything.
    """
    if not looks >= 2:
        raise ValueError(f"looks must be at least 2, got {looks}")
    if not spread >= 0.0:
        raise ValueError(f"spread must be at least 0, got {spread}")
    return spread * (
        (1.0 - _EULER) * _normal_ppf(1.0 - 1.0 / looks)
        + _EULER * _normal_ppf(1.0 - 1.0 / (looks * math.e))
    )


def _null_shape(study, null):
    if getattr(study, "best_result", None) is None:
        raise TypeError(
            "deflated_sharpe needs the TuneResult a search returned, since it "
            "reads the winner's own returns"
        )
    if study.objective != "sharpe":
        raise ValueError(
            f"this search selected on {study.objective!r}, and the deflation is "
            f"about the quantity you selected on, so deflating its sharpe would "
            f"be answering a question nobody asked; tune with objective='sharpe'"
        )
    if getattr(null, "sampler", None) != "random":
        raise ValueError(
            "null must be a search run with sampler='random'. A TPE search "
            "concentrates its trials, so the spread between them shrinks as it "
            "converges and the threshold would fall the harder you overfit"
        )
    if null.data_hash != study.data_hash:
        raise ValueError(
            f"the null ran on different bars ({null.data_hash} against "
            f"{study.data_hash}), so it is not a null for this search; run it "
            f"over the same data and the same space, with the same oos"
        )
    # an activity floor fails the thin cells, and the thin cells are where the
    # extreme sharpes live, so the trials that survive one scatter LESS than the
    # space does. Reading the spread off them made the threshold fall the
    # stricter the floor, which is the same inversion this decision exists to
    # prevent, arriving through the other term (ADR 0058)
    if getattr(null, "min_trades", 0):
        raise ValueError(
            f"the null was searched with min_trades={null.min_trades}, and an "
            f"activity floor keeps the trials that traded most, whose sharpes "
            f"scatter less than the whole space does; run the null at "
            f"min_trades=0. The search being deflated may still use one"
        )
    scored = [t["stats"]["sharpe"] for t in null.trials if t["stats"]]
    scored = [s for s in scored if math.isfinite(s)]
    if len(scored) < 2:
        raise ValueError(
            f"the null produced {len(scored)} usable trials, and the spread "
            f"across them is what sets the threshold; it needs at least 2"
        )
    # every trial of the SEARCH is a look, including one that failed: a
    # configuration you evaluated and rejected was still one you evaluated.
    # Counting the null's trials instead answered about the wrong search
    # entirely, so a 5-trial null made a 120-trial winner 46% more convincing
    # than a 400-trial null did, and nothing said which number it described
    looks = len(study.trials)
    if looks < 2:
        raise ValueError(
            f"this search ran {looks} trial(s), and a deflation is a correction "
            f"for having looked more than once; it needs at least 2"
        )
    if len(null.trials) < looks:
        warnings.warn(
            f"the null drew {len(null.trials)} trials to estimate the spread that "
            f"deflates a search of {looks}, so the threshold rests on fewer draws "
            f"than the thing it is judging; run the null at the same n_trials",
            stacklevel=3,
        )
    return scored, looks, float(np.std(np.asarray(scored, dtype=np.float64), ddof=1))


def min_track_record_length(result, benchmark=0.0, confidence=0.95,
                            independent=False):
    """How many bars it would take before the Sharpe is distinguishable from
    ``benchmark`` at ``confidence``, in bars and in wall time.

    Answers "is this backtest long enough" with a number. It counts independent
    bets rather than bars (``effective_sample``, ADR 0061) and then converts back,
    so a strategy holding a position for days needs proportionally more hourly
    candles than the published formula would ask for, which is the truth of it.
    ``independent=True`` gives the unadjusted version.

    Read ``num_trades`` beside it either way: three thousand bars carrying forty
    round trips is forty bets, and it is the bets that carry the information.
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
    variance = _psr_variance(result, observed)
    needed = 1.0 + variance * (_normal_ppf(confidence) / (observed - target)) ** 2
    have = float(values.size)
    effective = have if independent else effective_sample(result)
    # the answer comes out in independent bets, and the caller asked in bars, so
    # scale by however many bars this run spends per bet
    bars = needed * (have / effective) if effective > 0.0 else needed
    return {
        "bars": bars,
        "years": bars / ppy,
        "have_bars": int(values.size),
        "effective_bars": effective,
        "enough": have >= bars,
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
    _same_bars(result, frame, len(high), "excursions")
    out = []
    for trade in result.trades or []:
        first, last = trade["entry_tick"], trade["exit_tick"]
        if not 0 <= first <= last < len(high):
            # unreachable on the right frame, since a position still open at the
            # end is never logged as a trade (ADR 0009). The guard above is what
            # makes that true rather than hoped for
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

    Hours are UTC, 0 to 23. Weekdays are **Monday 0 to Sunday 6**, which is what
    ``datetime.weekday`` and pandas' ``dayofweek`` use, so a key can be read
    against any other clock you are holding.
    """
    if by not in ("hour", "weekday"):
        raise ValueError(f"by must be 'hour' or 'weekday', got {by!r}")
    stamps = _stamps(frame)
    _same_bars(result, frame, stamps.size, "session_buckets")
    unit = stamps.astype("datetime64[h]").astype("int64")
    # epoch day zero is a Thursday, so +3 puts Monday at 0 and matches
    # datetime.weekday, pandas dayofweek and day_name. It used to be +4, which is
    # Sunday-first, stated in no docstring and on no page, so buckets[0] read as
    # Monday and was Sunday (ADR 0059)
    keys = (unit % 24) if by == "hour" else ((unit // 24 + 3) % 7)
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


def _ran_on(result):
    curve = getattr(result, "equity_curve", None)
    return None if curve is None else len(curve) + 1


def _fingerprint_of(frame):
    # the frame's own fingerprint, on the identical rule the Backtester stamps a
    # result with, or None when it carries too few columns to compute one
    from ._data import to_ohlcv
    from .backtest import _fingerprint

    try:
        return _fingerprint(to_ohlcv(frame))
    except (TypeError, ValueError, ImportError):
        return None


def _same_bars(result, frame, bars, where):
    """Refuse a frame that is not the one the run was produced on.

    These read the run's own ``entry_tick`` and ``exit_tick`` straight into the
    frame's bars, so a different frame is not a smaller answer, it is an answer
    about something else. Both halves earned their place. The length check catches
    the accident: given a 2,000-bar slice of its own 100,000-bar frame,
    ``excursions`` silently returned 35 rows out of 1,516 and said nothing. The
    fingerprint catches what length cannot: given another asset of the same
    length, a losing trade came back with both its excursions positive. The chart
    refuses the first of these and not the second (ADR 0059).
    """
    ran_on = _ran_on(result)
    if ran_on is not None and bars != ran_on:
        raise ValueError(
            f"{where} needs the bars the run was produced on, but the run covered "
            f"{ran_on} bars and this frame carries {bars}; trade rows hold indices "
            f"into the original series, so each one would be read against the "
            f"wrong bar. To look at a slice, re-run the backtest on it"
        )
    stamped = getattr(result, "data_hash", None)
    seen = _fingerprint_of(frame)
    if stamped and seen and seen != stamped:
        raise ValueError(
            f"{where} was given a different series ({seen}) from the one the run "
            f"saw ({stamped}); they are the same length, so nothing else would "
            f"have caught it, and every trade's ticks would have been read "
            f"against another asset's prices"
        )


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
    out.update({f"trade_{k}": v for k, v in trade_stats(result).items()})
    out["skew"] = skew(result)
    out["kurtosis"] = kurtosis(result)
    out["autocorr_1"] = autocorrelation(result)
    out["effective_bars"] = effective_sample(result)
    out["value_at_risk_pct"] = value_at_risk(result)
    out["conditional_value_at_risk_pct"] = conditional_value_at_risk(result)
    out["omega"] = omega(result)
    out["tail_ratio"] = tail_ratio(result)
    out["ulcer_index"] = ulcer_index(result)
    # what the return is worth per unit of time actually at risk. Twenty percent
    # made while in the market a twelfth of the time and twenty percent made fully
    # invested are the same row without it
    exposure = float(out.get("exposure_pct") or 0.0)
    out["return_per_exposure"] = (
        float(out.get("total_return_pct", 0.0)) / exposure * 100.0 if exposure else 0.0
    )
    sides = long_short_split(result)
    for name, block in sides.items():
        out.update({f"{name}_{k}": v for k, v in block.items()})
    try:
        out["probabilistic_sharpe"] = probabilistic_sharpe(result)
    except ValueError:
        out["probabilistic_sharpe"] = None
    try:
        out.update({f"sharpe_{k}": v for k, v in sharpe_interval(result).items()
                    if k in ("low", "high")})
    except ValueError:
        out["sharpe_low"], out["sharpe_high"] = None, None
    if frame is not None:
        # `hold_return_pct` and `hold_sharpe` already carry the prefix, and
        # prefixing them again produced `hold_hold_return_pct`
        for key, value in buy_and_hold(result, frame).items():
            out[key if key.startswith("hold_") else f"hold_{key}"] = value
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
    # a key nothing reports printed as the word None in every row, which reads as
    # a run that scored nothing rather than as a column that does not exist.
    # `tune` checks its objective against the same set before it searches
    available = set()
    for _name, result in named:
        available.update(result.stats or {})
    unknown = sorted(set(shown) - available)
    if unknown:
        raise KeyError(
            f"compare was asked for {unknown}, which no result reports; choose "
            f"from {sorted(available)}"
        )
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
    shape = trade_stats(result)
    try:
        band = sharpe_interval(result)
    except ValueError:
        band = None
    rows = [
        ("return", f"{stats.get('total_return_pct', 0.0):>12,.2f} %"),
        ("sharpe", f"{stats.get('sharpe', 0.0):>12,.2f}"),
        ("  95% interval",
         "         n/a" if band is None
         else f"{band['low']:>7,.2f} to {band['high']:<7,.2f}"),
        ("max drawdown", f"{stats.get('max_drawdown_pct', 0.0):>12,.2f} %"),
        ("", ""),
        ("gross pnl", f"{money['gross_pnl']:>12,.2f}"),
        ("fees", f"{-money['fees']:>12,.2f}"),
        ("funding", f"{-money['funding']:>12,.2f}"),
        ("still open", f"{money['unrealized']:>12,.2f}"),
        ("net", f"{money['net']:>12,.2f}"),
        ("fee share", f"{money['fee_share'] * 100.0:>12,.1f} % of gross"),
        ("", ""),
        ("trades", f"{stats.get('num_trades', 0):>12,}"),
        ("long / short", f"{sides['long']['trades']:>6,} /{sides['short']['trades']:>5,}"),
        ("win rate", f"{stats.get('win_rate', 0.0) * 100.0:>12,.1f} %"),
        # beside the win rate, never without it: the pair is the answer and
        # either one alone describes two opposite strategies equally well
        ("payoff", f"{shape['payoff']:>12,.2f}"),
        ("expectancy", f"{shape['expectancy']:>12,.2f} per trade"),
        ("worst streak", f"{shape['max_consecutive_losses']:>12,} losses"),
        ("under water", f"{water['share_pct']:>12,.1f} % of bars"),
        ("longest", f"{water['longest_bars']:>12,} bars"),
    ]
    if frame is not None:
        hold = buy_and_hold(result, frame)
        rows += [("", ""), ("vs holding", f"{hold['excess_return_pct']:>12,.2f} %")]
    for label, value in rows:
        print(f"  {label:<14}{value}" if label else "")
    return report(result, frame)
