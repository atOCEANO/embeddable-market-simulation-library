"""emsl: embeddable market simulation library.

The compiled Rust core is exposed as ``emsl._emsl``; this package re-exports its
public surface. ``Engine`` drives a single environment one bar at a time: build it
from a ``(T, 5)`` OHLCV numpy array, then ``reset`` and ``step``. ``Batch`` runs
many independent envs over one shared series and steps them in parallel with the
GIL released. The Python wrappers (rl, backtest) layer on top of these, and ``tune``
searches a strategy's parameters by running many backtests across cores. ``chart``
renders a frame, your own arrays and a run into one self-contained HTML document
for a notebook cell or a file, computing nothing itself.
"""

from ._emsl import Batch, Engine
from ._data import to_ohlcv
from . import backtest, metrics, plot, rl, ta
from .backtest import Strategy
from .market import Market
from ._tune import tune
from ._walk import walk_forward
from ._chart import chart, chart_defaults

__all__ = [
    "Batch", "Engine", "Market", "Strategy", "chart", "chart_defaults", "to_ohlcv",
    "tune", "walk_forward", "backtest", "metrics", "plot", "rl", "ta",
]

try:
    from importlib.metadata import version as _version

    __version__ = _version("emsl")
except Exception:
    __version__ = "0.0.0"
