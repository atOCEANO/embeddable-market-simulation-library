"""The venue and your account on it, as one object.

The engine takes eleven knobs, and a backtest, a parameter search and an RL
rollout each take the same eleven. Written out at three call sites they are three
chances to disagree, and the claim this library makes, that one fill model sits
behind every surface, was being upheld by whoever typed them retyping them
identically. ``Market`` holds them once and hands out the surfaces itself::

    binance = emsl.Market(kind="perp", fee_taker=0.0004, fee_maker=0.0002,
                          slippage_bps=2.0, funding_rate=0.0001,
                          funding_interval=8)

    result = binance.backtest(candles).run(MyStrategy())
    study = binance.tune(MyStrategy, space, candles, n_trials=200, oos=0.3)
    env = binance.env(candles, num_envs=4096, window=60)

Each method takes only the arguments that are not the venue, so there is nothing
to reconcile: a knob cannot be passed twice, and no rule about which copy wins is
needed because there is only ever one copy (ADR 0053).

The keyword form still works everywhere and is the right thing for a one-off.
This is for when the venue is fixed and the surfaces are many.
"""

from __future__ import annotations

_KINDS = ("spot", "perp")


class Market:
    """The fill model, the costs and the opening balance, as one value.

    ``kind`` is ``"spot"`` or ``"perp"``. The rest are the engine's own knobs and
    carry its defaults: see the Python API for what each one does. They are
    checked when an engine is built from them, not here, so the rules live in one
    place rather than two.
    """

    def __init__(self, kind="spot", quote=10_000.0, fee_taker=0.0006,
                 fee_maker=0.0002, slippage_bps=0.0, max_fill_fraction=1.0,
                 max_open_orders=8, leverage=10.0, impact=0.0, funding_rate=0.0,
                 funding_interval=0):
        if kind not in _KINDS:
            raise ValueError(f"kind must be 'spot' or 'perp', got {kind!r}")
        self.kind = kind
        self.quote = quote
        self.fee_taker = fee_taker
        self.fee_maker = fee_maker
        self.slippage_bps = slippage_bps
        self.max_fill_fraction = max_fill_fraction
        self.max_open_orders = max_open_orders
        self.leverage = leverage
        self.impact = impact
        self.funding_rate = funding_rate
        self.funding_interval = funding_interval

    def as_dict(self):
        """The knobs as the keyword arguments the engine takes.

        ``kind`` becomes ``market``, which is what the engine calls it; the name
        differs here only so that ``Market(market=...)`` does not read as though
        a market contains a market.
        """
        return {
            "market": self.kind,
            "quote": self.quote,
            "fee_taker": self.fee_taker,
            "fee_maker": self.fee_maker,
            "slippage_bps": self.slippage_bps,
            "max_fill_fraction": self.max_fill_fraction,
            "max_open_orders": self.max_open_orders,
            "leverage": self.leverage,
            "impact": self.impact,
            "funding_rate": self.funding_rate,
            "funding_interval": self.funding_interval,
        }

    def replace(self, **changes):
        """A copy with some knobs changed, for asking what a worse venue does.

        ``binance.replace(fee_taker=0.001)`` is the whole point: the two runs
        differ in exactly one number and are visibly the same venue otherwise.
        """
        settings = self.as_dict()
        settings["kind"] = settings.pop("market")
        unknown = sorted(set(changes) - set(settings))
        if unknown:
            raise TypeError(
                f"a market has no {unknown}; it carries {sorted(settings)}"
            )
        settings.update(changes)
        return Market(**settings)

    def engine(self, candles, report=False):
        """An ``Engine`` over ``candles`` at this venue."""
        from ._emsl import Engine

        return Engine(candles, report=report, **self.as_dict())

    def backtest(self, candles, periods_per_year=None, risk_free=0.0):
        """A ``Backtester`` over ``candles`` at this venue."""
        from .backtest import Backtester

        return Backtester(candles, periods_per_year=periods_per_year,
                          risk_free=risk_free, **self.as_dict())

    def tune(self, strategy, space, data, **search):
        """Run ``emsl.tune`` at this venue. ``search`` takes the search controls
        (``objective``, ``n_trials``, ``oos``, ``n_jobs``, ``seed``, ...) and not
        the venue, which this object already is.
        """
        from ._tune import tune

        self._only_the_search(search, "tune")
        return tune(strategy, space, data, **search, **self.as_dict())

    def walk_forward(self, strategy, space, data, **rest):
        """Run ``emsl.walk_forward`` at this venue. ``rest`` takes the layout and
        the search controls (``windows``, ``train``, ``anchored``, ``n_trials``,
        ...) and not the venue, which this object already is.
        """
        from ._walk import walk_forward

        self._only_the_search(rest, "walk_forward")
        return walk_forward(strategy, space, data, **rest, **self.as_dict())

    def env(self, data, **rest):
        """A ``VectorEnv`` at this venue. ``rest`` takes the RL arguments
        (``features``, ``num_envs``, ``window``, ``reward_fn``, ...).
        """
        from .rl import VectorEnv

        self._only_the_search(rest, "env")
        return VectorEnv(data, **rest, **self.as_dict())

    def _only_the_search(self, given, where):
        # the venue is this object, so a knob arriving beside it is not an
        # override to resolve, it is two answers to one question
        clash = sorted(set(given) & set(self.as_dict()))
        if clash:
            raise TypeError(
                f"{clash} belongs to the market, not to {where}; set it on the "
                f"Market, or use market.replace({clash[0]}=...) for a variant"
            )

    def __eq__(self, other):
        return isinstance(other, Market) and self.as_dict() == other.as_dict()

    def __hash__(self):
        return hash(tuple(sorted(self.as_dict().items())))

    def __repr__(self):
        # only what differs from the defaults, so a printed market reads as the
        # handful of choices somebody actually made
        default = Market().as_dict()
        mine = self.as_dict()
        shown = [f"kind={self.kind!r}"]
        shown += [f"{k}={v!r}" for k, v in mine.items()
                  if k != "market" and v != default[k]]
        return f"Market({', '.join(shown)})"
