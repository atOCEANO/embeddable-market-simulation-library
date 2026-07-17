"""emsl: embeddable market simulation library.

The compiled Rust core is exposed as ``emsl._emsl``; this package re-exports its
public surface. ``Engine`` drives a single environment one bar at a time: build it
from a ``(T, 5)`` OHLCV numpy array, then ``reset`` and ``step``. ``Batch`` runs
many independent envs over one shared series and steps them in parallel with the
GIL released. The Python wrappers (rl, backtest, batch) layer on top of these.
"""

from ._emsl import Batch, Engine
from ._data import to_ohlcv
from . import backtest, batch, rl

__all__ = ["Batch", "Engine", "to_ohlcv", "backtest", "batch", "rl"]

try:
    from importlib.metadata import version as _version

    __version__ = _version("emsl")
except Exception:
    __version__ = "0.0.0"
