"""Parameter tuning over the ``backtest.Strategy`` spine.

``tune`` searches a strategy's parameters for the configuration that maximizes (or
minimizes) an objective, running each trial as a full backtest through the same
``emsl.backtest.Backtester`` a single run uses. The same ``Strategy`` subclass
feeds a backtest and a tune; the difference is that ``tune`` builds a fresh one per
trial from the sampled parameters, ``StrategyClass(**params)``, so a strategy
declares its tunables as constructor arguments and stores them as fields.

- **Search space**: a dict from parameter name to a range. ``(low, high)`` of ints
  is an integer axis, ``(low, high)`` of floats a continuous one, ``(low, high,
  "log")`` a log axis, and a ``list`` a categorical choice. ``tune.Int``,
  ``tune.Float``, and ``tune.Categorical`` give the same with ``step`` and ``log``
  control.
- **Objective**: a stats-key string (``"sharpe"``, ``"calmar"``, ...) or a callable
  taking the ``BacktestResult`` and returning a number; higher is better under
  ``direction="maximize"``. A trial whose strategy or objective raises, or whose
  objective is ``NaN``, is marked failed and the search continues; an infinite value
  (``profit_factor`` with no losing trades, say) is kept and ranks at the extreme.
- **Minimum activity**: ``min_trades`` fails any trial that closed fewer trades than
  that. A search ranking on a point estimate walks to the thinnest cell it can find,
  where a handful of samples produces the widest interval and the best-looking
  number, so a floor is the cheapest defence against a winner that is really noise
  (ADR 0034).
- **Parallelism**: ``n_jobs=1`` runs in this process; ``n_jobs>1`` (or ``-1`` for
  every core) runs trials in worker processes, rebuilding the engine in each worker
  rather than shipping a live one across the boundary (ADR 0021). Note that only
  ``n_jobs=1`` is reproducible from ``seed``: a parallel run asks for trials before
  earlier ones have reported, so the order results reach the sampler depends on
  which worker finishes first and the search follows a different path each time.
  Pin ``n_jobs=1`` when a result has to be reproducible (ADR 0036).

``optuna`` drives the search and ``cloudpickle`` carries the strategy and objective
to the workers; both install with ``pip install 'emsl[tune]'``.

```python
from emsl import tune
from emsl.backtest import Strategy


class SmaCross(Strategy):
    def __init__(self, fast, slow):
        self.fast = fast
        self.slow = slow

    def init(self, engine):
        self.close = engine.data[:, 3]

    def next(self, state, engine):
        i = state["tick_index"]
        if i < self.slow:
            return
        fast = self.close[i - self.fast:i].mean()
        slow = self.close[i - self.slow:i].mean()
        if state["position"] == 0 and fast > slow:
            engine.market_buy(1.0)
        elif state["position"] > 0 and fast < slow:
            engine.close()


result = tune(
    SmaCross,
    {"fast": (5, 40), "slow": (40, 200)},
    data,
    objective="sharpe",
    n_trials=200,
    n_jobs=-1,
)
print(result.best_params, result.best_value)
print(result.best_stats["max_drawdown_pct"])
```
"""

from __future__ import annotations

import inspect
import math
import os
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

from ._data import to_ohlcv
from .backtest import Backtester

__all__ = ["tune", "Int", "Float", "Categorical", "TuneResult"]

# the stats keys the engine reports, so a string objective can be checked before the
# search rather than failing every trial; kept in step with bar-engine stats.rs and
# ADR 0007, the one place the set is defined
_STAT_KEYS = frozenset(
    {
        "total_return_pct",
        "net_profit_pct",
        "cagr_pct",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown_pct",
        "volatility_pct",
        "exposure_pct",
        "win_rate",
        "profit_factor",
        "num_trades",
        "avg_trade_pct",
        "num_fills",
        "funding_paid",
    }
)


class _Spec:
    """Base for an explicit search-space entry; subclasses build an optuna
    distribution through ``_distribution``.
    """

    def _distribution(self):
        raise NotImplementedError


class Int(_Spec):
    """An integer parameter drawn from ``[low, high]``, with an optional ``step`` and
    ``log`` scale. The shorthand ``(low, high)`` of two ints builds the same.
    """

    def __init__(self, low, high, step=1, log=False):
        self.low = int(low)
        self.high = int(high)
        self.step = int(step)
        self.log = bool(log)

    def _distribution(self):
        from optuna.distributions import IntDistribution

        return IntDistribution(self.low, self.high, step=self.step, log=self.log)


class Float(_Spec):
    """A floating parameter drawn from ``[low, high]``, with an optional ``step`` and
    ``log`` scale. The shorthand ``(low, high)`` with a float endpoint builds the
    same, and ``(low, high, "log")`` sets ``log``.
    """

    def __init__(self, low, high, step=None, log=False):
        self.low = float(low)
        self.high = float(high)
        self.step = None if step is None else float(step)
        self.log = bool(log)

    def _distribution(self):
        from optuna.distributions import FloatDistribution

        return FloatDistribution(self.low, self.high, step=self.step, log=self.log)


class Categorical(_Spec):
    """A parameter drawn from a fixed list of choices. The shorthand is a plain
    ``list``.
    """

    def __init__(self, choices):
        self.choices = list(choices)
        if not self.choices:
            raise ValueError("a categorical space needs at least one choice")

    def _distribution(self):
        from optuna.distributions import CategoricalDistribution

        return CategoricalDistribution(self.choices)


def _load_cloudpickled(payload):
    # rebuild a wrapper from its cloudpickle bytes; the reconstruction target of
    # _CloudPickled.__reduce__, so it stays wrapped on both fork and spawn workers
    import cloudpickle

    return _CloudPickled(cloudpickle.loads(payload))


class _CloudPickled:
    """Wrap an object so a stdlib pickler routes it through cloudpickle.

    ``ProcessPoolExecutor`` pickles the initializer arguments with the stdlib
    pickler, which cannot handle a lambda, a closure, or a class defined in
    ``__main__`` or a notebook. A strategy or objective wrapped in this makes the
    stdlib pickler call ``__reduce__``, which serializes the payload with cloudpickle
    instead; the worker restores it the same way. A fork worker inherits the wrapper
    directly, so it stays wrapped either way and the worker reads ``.obj`` (ADR 0021).
    """

    def __init__(self, obj):
        self.obj = obj

    def __reduce__(self):
        import cloudpickle

        return (_load_cloudpickled, (cloudpickle.dumps(self.obj),))


# per-worker state, populated once by _worker_init and read by each _worker_eval, so
# the candles and the strategy cross the process boundary once per worker rather than
# once per trial (ADR 0021)
_WORKER = {}


def _worker_init(
    candles, config, periods_per_year, risk_free, strategy_wrapped, objective_wrapped, min_trades
):
    # runs once when a worker process starts: rebuild the Backtester from the candles
    # and config, and unwrap the cloudpickled strategy and objective
    _WORKER["backtester"] = Backtester(
        candles, periods_per_year=periods_per_year, risk_free=risk_free, **config
    )
    _WORKER["strategy"] = strategy_wrapped.obj
    _WORKER["objective"] = objective_wrapped.obj
    _WORKER["min_trades"] = min_trades


def _worker_eval(params):
    return _evaluate(
        _WORKER["backtester"],
        _WORKER["strategy"],
        _WORKER["objective"],
        params,
        _WORKER["min_trades"],
    )


def _evaluate(backtester, strategy, objective, params, min_trades=0):
    # one trial: build a fresh strategy from the sampled params, backtest it, and
    # score it. A NaN score is an undefined result the caller fails the trial on; an
    # infinite score (profit_factor with no losses, say) is kept so it can rank
    result = backtester.run(strategy(**params))
    # a configuration that barely traded is scored on a handful of samples, and the
    # best point estimate in a search walks straight to the thinnest cell. Failing it
    # is the same treatment a NaN objective gets (ADR 0034)
    if min_trades > 0 and result.stats["num_trades"] < min_trades:
        raise ValueError(
            f"only {result.stats['num_trades']} trades, below min_trades={min_trades}, "
            f"for params {params}"
        )
    value = float(objective(result))
    if math.isnan(value):
        raise ValueError(f"objective returned NaN for params {params}")
    return value, result.stats


def _to_distribution(name, spec):
    if isinstance(spec, _Spec):
        return spec._distribution()
    if isinstance(spec, list):
        return Categorical(spec)._distribution()
    if isinstance(spec, tuple) and len(spec) in (2, 3):
        log = len(spec) == 3
        if log and spec[2] != "log":
            raise ValueError(f"space['{name}'] third element must be 'log', got {spec[2]!r}")
        low, high = spec[0], spec[1]
        if isinstance(low, bool) or isinstance(high, bool):
            raise TypeError(f"space['{name}'] bounds must be numbers, not booleans")
        if isinstance(low, int) and isinstance(high, int):
            return Int(low, high, log=log)._distribution()
        return Float(low, high, log=log)._distribution()
    raise TypeError(
        f"space['{name}'] must be (low, high), (low, high, 'log'), a list of "
        f"choices, or a tune.Int/Float/Categorical, got {spec!r}"
    )


def _resolve_objective(objective):
    if callable(objective):
        return objective
    if isinstance(objective, str):
        if objective not in _STAT_KEYS:
            raise KeyError(
                f"objective '{objective}' is not a stats key; choose one of "
                f"{sorted(_STAT_KEYS)} or pass a callable"
            )
        key = objective

        def named(result):
            return float(result.stats[key])

        return named
    raise TypeError(
        "objective must be a stats-key string or a callable taking a BacktestResult"
    )


def _resolve_n_jobs(n_jobs):
    if n_jobs is None:
        return 1
    n_jobs = int(n_jobs)
    if n_jobs == -1:
        return max(1, os.cpu_count() or 1)
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be a positive integer or -1, got {n_jobs}")
    return n_jobs


def _validate_param_names(strategy, space):
    # check the space names against the strategy's constructor up front, so a typo
    # ('fst' for 'fast') is a clear error rather than every trial failing. Done by
    # signature, not by a probe run, so no particular parameter combination can abort
    # a valid search. A callable that takes **kwargs accepts any name, and anything
    # whose signature cannot be read is trusted
    try:
        sig = inspect.signature(strategy)
    except (TypeError, ValueError):
        return
    params = sig.parameters.values()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return
    accepted = {
        p.name
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    unknown = sorted(k for k in space if k not in accepted)
    if unknown:
        raise TypeError(
            f"space names the strategy does not accept: {unknown}; the strategy "
            f"takes {sorted(accepted)}"
        )


def _drive_sequential(
    study, distributions, n_trials, strategy, objective, backtester, min_trades
):
    import optuna

    last_error = None
    for _ in range(n_trials):
        trial = study.ask(distributions)
        try:
            value, stats = _evaluate(
                backtester, strategy, objective, trial.params, min_trades
            )
        except Exception as exc:
            last_error = exc
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            trial.set_user_attr("stats", stats)
            study.tell(trial, value)
    return last_error


def _drive_parallel(
    study, distributions, n_trials, n_jobs, candles, config,
    periods_per_year, risk_free, strategy, objective, min_trades,
):
    import optuna

    initargs = (
        candles,
        config,
        periods_per_year,
        risk_free,
        _CloudPickled(strategy),
        _CloudPickled(objective),
        min_trades,
    )
    inflight = {}
    submitted = 0
    last_error = None
    with ProcessPoolExecutor(
        max_workers=n_jobs, initializer=_worker_init, initargs=initargs
    ) as pool:
        # a broken pool (a worker died: OOM, a native crash) escapes here so the
        # completed trials are still returned rather than lost with the exception
        try:
            while submitted < min(n_jobs, n_trials):
                trial = study.ask(distributions)
                inflight[pool.submit(_worker_eval, trial.params)] = trial
                submitted += 1
            while inflight:
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for future in done:
                    trial = inflight.pop(future)
                    try:
                        value, stats = future.result()
                    except Exception as exc:
                        last_error = exc
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    else:
                        trial.set_user_attr("stats", stats)
                        study.tell(trial, value)
                    if submitted < n_trials:
                        nxt = study.ask(distributions)
                        inflight[pool.submit(_worker_eval, nxt.params)] = nxt
                        submitted += 1
        except BrokenProcessPool as exc:
            last_error = exc
    if isinstance(last_error, BrokenProcessPool):
        warnings.warn(
            "a tuning worker process died and the pool broke; returning the trials "
            "that completed before the failure",
            RuntimeWarning,
            stacklevel=2,
        )
    return last_error


class TuneResult:
    """The outcome of a search: the best parameters, their objective value, and the
    winning run's full stats, plus a record of every trial. ``study`` is the
    underlying optuna study for deeper inspection, and ``best_strategy()`` builds a
    fresh strategy from ``best_params``.
    """

    def __init__(self, study, strategy):
        self._study = study
        self._strategy = strategy
        best = study.best_trial
        self.best_params = dict(best.params)
        self.best_value = float(best.value)
        self.best_stats = dict(best.user_attrs.get("stats", {}))
        self.trials = [
            {
                "number": t.number,
                "params": dict(t.params),
                "value": t.value,
                "state": t.state.name,
                "stats": t.user_attrs.get("stats"),
            }
            for t in study.trials
        ]

    @property
    def study(self):
        return self._study

    def best_strategy(self):
        return self._strategy(**self.best_params)

    def __repr__(self):
        return f"TuneResult(best_value={self.best_value:.6g}, n_trials={len(self.trials)})"


def tune(
    strategy,
    space,
    data,
    objective="sharpe",
    direction="maximize",
    n_trials=100,
    n_jobs=1,
    seed=None,
    verbose=False,
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
    periods_per_year=365.0,
    risk_free=0.0,
):
    """Search ``strategy``'s parameters over ``space`` for the best ``objective``.

    ``strategy`` is a ``backtest.Strategy`` subclass, or any callable that builds one
    from the sampled parameters; each trial runs ``strategy(**params)`` through a
    ``Backtester`` on ``data`` (a numpy array, DataFrame, or parquet path) with the
    given engine configuration. ``space`` maps each parameter name to a range (see
    the module docstring). Returns a ``TuneResult``. Raises ``ImportError`` if optuna
    (or, for ``n_jobs>1``, cloudpickle) is not installed, and ``RuntimeError`` if
    every trial fails.
    """
    optuna = _import_optuna()

    if not callable(strategy):
        raise TypeError("strategy must be a Strategy subclass or a callable that builds one")
    if not isinstance(space, dict) or not space:
        raise ValueError("space must be a dict of name to range with at least one entry")
    if direction not in ("maximize", "minimize"):
        raise ValueError(f"direction must be 'maximize' or 'minimize', got {direction!r}")
    if int(n_trials) < 1:
        raise ValueError(f"n_trials must be a positive integer, got {n_trials}")
    if market not in ("spot", "perp"):
        raise ValueError(f"market must be 'spot' or 'perp', got {market!r}")
    if int(min_trades) < 0:
        raise ValueError(f"min_trades must be zero or a positive integer, got {min_trades}")
    n_trials = int(n_trials)
    min_trades = int(min_trades)
    n_jobs = _resolve_n_jobs(n_jobs)
    _validate_param_names(strategy, space)

    candles = to_ohlcv(data)
    config = dict(
        market=market,
        quote=quote,
        fee_taker=fee_taker,
        fee_maker=fee_maker,
        slippage_bps=slippage_bps,
        max_fill_fraction=max_fill_fraction,
        max_open_orders=max_open_orders,
        leverage=leverage,
        impact=impact,
        funding_rate=funding_rate,
        funding_interval=funding_interval,
    )
    distributions = {name: _to_distribution(name, spec) for name, spec in space.items()}
    objective_fn = _resolve_objective(objective)

    optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)

    if n_jobs == 1:
        backtester = Backtester(
            candles, periods_per_year=periods_per_year, risk_free=risk_free, **config
        )
        last_error = _drive_sequential(
            study, distributions, n_trials, strategy, objective_fn, backtester, min_trades
        )
    else:
        if n_jobs > n_trials:
            n_jobs = n_trials
        _require_cloudpickle()
        last_error = _drive_parallel(
            study, distributions, n_trials, n_jobs, candles, config,
            periods_per_year, risk_free, strategy, objective_fn, min_trades,
        )

    if not any(t.state.name == "COMPLETE" for t in study.trials):
        raise RuntimeError(
            "every trial failed; the last error was: " + repr(last_error)
        ) from last_error
    return TuneResult(study, strategy)


def _import_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "optuna is required for emsl.tune; pip install 'emsl[tune]' "
            "(installs optuna and cloudpickle)"
        ) from exc
    return optuna


def _require_cloudpickle():
    try:
        import cloudpickle  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "cloudpickle is required for n_jobs > 1; pip install 'emsl[tune]' "
            "(installs optuna and cloudpickle)"
        ) from exc


tune.Int = Int
tune.Float = Float
tune.Categorical = Categorical
tune.TuneResult = TuneResult
