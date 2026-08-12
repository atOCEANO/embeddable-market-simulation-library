"""Walk forward: refit periodically, and trade only what was fitted before.

A holdout asks whether one winner survives one tail. A walk-forward asks the
larger question, which is what the whole PROCEDURE would have earned: fit on what
you had, trade the next stretch, refit, repeat. That is what running a strategy
actually looks like, and it is the only construction here whose answer is a curve
rather than a number (ADR 0057).

The curve is produced by one real engine pass, not by stitching. A composite
strategy carries the schedule of fitted parameters and delegates each bar to the
window that owns it, so the run has real fills, real funding, a continuous
account across the seams, and full history for every warm-up. Every metric and
the chart take the result as they would any other, because it is one.

    forward = emsl.walk_forward(SmaCross, space, candles, windows=5, train=0.5)

    forward.stats["sharpe"]        # out of sample, by construction
    forward.decay                  # how much the fit flattered itself, per window
    emsl.chart(frame=candles, run=forward.result).show()
"""

from __future__ import annotations

import numpy as np

from ._data import annualization, prepare
from .backtest import Backtester, Strategy

__all__ = ["walk_forward", "WalkForward"]


class _Scheduled(Strategy):
    """One strategy per window, each trading only the bars its window owns.

    The windows share an account and a series, which is the point: refitting does
    not hand you a fresh 10,000, and a strategy entering the last bar of one
    window still holds that position in the next. Every part sees the whole
    series in ``init``, so a warm-up spans a seam instead of restarting at it.
    """

    def __init__(self, build, schedule):
        self._build = build
        self._schedule = schedule
        self.parts = []

    def init(self, engine):
        self.parts = [
            (start, stop, self._build(**params)) for start, stop, params in self._schedule
        ]
        for _, _, part in self.parts:
            part.init(engine)
        # the earliest bar any window may decide on is its own start, and a part
        # cannot decide before its own warm-up either
        self.warmup = min(
            (max(start, getattr(part, "warmup", 0)) for start, _, part in self.parts),
            default=0,
        )

    def next(self, state, engine):
        bar = state["tick_index"]
        for start, stop, part in self.parts:
            if start <= bar < stop:
                if bar >= getattr(part, "warmup", 0):
                    part.next(state, engine)
                return


class WalkForward:
    """What the procedure earned, and how much each fit flattered itself.

    ``result`` is an ordinary ``BacktestResult`` over the whole series, so every
    metric and the chart take it directly. It is flat until the first window,
    because before that there was nothing fitted to trade.

    ``windows`` is one record per refit: the bars it was fitted on, the bars it
    traded, the parameters it chose, the score it reached in the fit, and the
    score the same parameters reached on the bars it then traded.
    """

    def __init__(self, result, windows, span):
        self.result = result
        self.stats = result.stats
        self.windows = windows
        self.span = span

    @property
    def decay(self):
        """Mean fitted score less traded score, across the windows.

        How much the fit flattered itself, in the units of the objective. A large
        positive number says the search is finding things that do not survive the
        next stretch, which is the failure a single holdout can see once and this
        sees repeatedly.
        """
        gaps = [w["fitted"] - w["traded"] for w in self.windows
                if w["traded"] is not None and w["fitted"] is not None]
        return float(np.mean(gaps)) if gaps else float("nan")

    @property
    def consistency(self):
        """The share of windows whose traded score was above zero.

        Blunt on purpose. A procedure that works should clear zero more often than
        not, and one that clears it in one window out of five had one good quarter
        rather than an edge.
        """
        scored = [w["traded"] for w in self.windows if w["traded"] is not None]
        return float(np.mean([s > 0.0 for s in scored])) if scored else float("nan")

    def __repr__(self):
        return (
            f"WalkForward({len(self.windows)} windows, "
            f"sharpe={self.stats.get('sharpe', float('nan')):.3f} out of sample)"
        )


def walk_forward(
    strategy,
    space,
    data,
    windows=5,
    train=0.5,
    anchored=False,
    objective="sharpe",
    direction="maximize",
    n_trials=50,
    n_jobs=1,
    seed=None,
    sampler="tpe",
    min_trades=0,
    market="spot",
    quote=10_000.0,
    fee_taker=0.0006,
    fee_maker=0.0002,
    slippage_bps=0.0,
    max_fill_fraction=1.0,
    max_open_orders=8,
    leverage=10.0,
    impact=0.0,
    funding_rate=0.0,
    funding_interval=0,
    periods_per_year=None,
    risk_free=0.0,
):
    """Refit ``windows`` times and trade each stretch with what was fitted before it.

    ``train`` is the share of the series the first fit gets; the rest is divided
    into ``windows`` stretches, each traded with parameters chosen on the bars
    before it and never on itself. ``anchored`` fits on everything before each
    window rather than on a moving block of the same length as the first fit.

    The remaining arguments are ``tune``'s, and each window is an ordinary search:
    the objective, the trial count, the sampler and the engine's own knobs all
    mean what they mean there.

    **Window length and step are hyperparameters too.** Trying five layouts and
    reporting the best of them is the same mistake one level up, and nothing here
    can catch you doing it.
    """
    from ._tune import tune

    candles, index = prepare(data)
    periods_per_year = annualization(periods_per_year, index)
    config = dict(
        market=market, quote=quote, fee_taker=fee_taker, fee_maker=fee_maker,
        slippage_bps=slippage_bps, max_fill_fraction=max_fill_fraction,
        max_open_orders=max_open_orders, leverage=leverage, impact=impact,
        funding_rate=funding_rate, funding_interval=funding_interval,
    )
    layout = _layout(len(candles), int(windows), float(train), bool(anchored))

    records = []
    schedule = []
    for fit_from, fit_to, test_to in layout:
        study = tune(
            strategy, space, candles[fit_from:fit_to], objective=objective,
            direction=direction, n_trials=n_trials, n_jobs=n_jobs, seed=seed,
            oos=0, sampler=sampler, min_trades=min_trades,
            periods_per_year=periods_per_year, risk_free=risk_free, **config,
        )
        traded = Backtester(
            candles[fit_to:test_to], periods_per_year=periods_per_year,
            risk_free=risk_free, **config,
        ).run(strategy(**study.best_params))
        records.append({
            "fitted_on": (fit_from, fit_to),
            "traded_on": (fit_to, test_to),
            "params": dict(study.best_params),
            "fitted": study.best_value,
            "traded": _score(traded, objective),
        })
        schedule.append((fit_to, test_to, dict(study.best_params)))

    # one real pass over the whole series, so the account is continuous across the
    # seams and every warm-up has the history before its own window to use
    result = Backtester(
        candles, periods_per_year=periods_per_year, risk_free=risk_free, **config,
    ).run(_Scheduled(strategy, schedule))
    return WalkForward(result, records, (layout[0][1], layout[-1][2]))


def _score(result, objective):
    if callable(objective):
        try:
            return float(objective(result))
        except Exception:
            return None
    value = (result.stats or {}).get(objective)
    return None if value is None else float(value)


def _layout(size, windows, train, anchored):
    """The (fit_from, fit_to, test_to) of each window.

    The stretches are consecutive and never overlap, and no window is fitted on a
    bar it later trades: `fit_to` is both the end of the fit and the start of the
    trading, so the boundary belongs to the trading side.
    """
    if windows < 1:
        raise ValueError(f"walk_forward needs at least one window, got {windows}")
    if not 0.0 < train < 1.0:
        raise ValueError(f"train must be above 0 and below 1, got {train}")
    first = int(round(size * train))
    step = (size - first) // windows
    if first < 2 or step < 2:
        raise ValueError(
            f"walk_forward({windows} windows, train={train}) splits {size} bars "
            f"into a {first}-bar fit and {step}-bar stretches, and each side needs "
            f"at least 2; use fewer windows, a longer series, or a smaller train"
        )
    out = []
    for k in range(windows):
        fit_to = first + k * step
        test_to = size if k == windows - 1 else fit_to + step
        fit_from = 0 if anchored else max(0, fit_to - first)
        out.append((fit_from, fit_to, test_to))
    return out
