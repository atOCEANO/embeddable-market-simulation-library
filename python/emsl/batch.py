"""Compiled-strategy parameter sweep.

``BatchRunner`` runs a built-in compiled strategy over a grid of parameter rows in
parallel, with the GIL released, and returns a dict of final-stat arrays aligned to
the rows. Use it for brute-force search over a built-in strategy; a Python
``Strategy`` subclass runs one at a time through ``emsl.backtest`` instead, at
Python speed.

```python
import numpy as np
from emsl.batch import BatchRunner

runner = BatchRunner(ohlcv, market="spot")       # numpy, DataFrame, or parquet path
params = np.random.randint(5, 100, size=(10_000, 2)).astype(float)   # [fast, slow]
results = runner.run_all("sma_cross", params)

best = results["sharpe"].argmax()
print(params[best], results["sharpe"][best])
```
"""

from __future__ import annotations

import numpy as np

from ._data import to_ohlcv
from ._emsl import BatchRunner as _BatchRunner

__all__ = ["BatchRunner"]


class BatchRunner:
    """Sweep a built-in compiled strategy over a parameter grid.

    ``data`` is a numpy array, a pandas DataFrame, or a parquet path; the rest are
    the engine's configuration.
    """

    def __init__(
        self,
        data,
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
    ):
        self._inner = _BatchRunner(
            to_ohlcv(data),
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

    def run_all(self, strategy, params, periods_per_year=365.0, risk_free=0.0):
        """Run ``strategy`` over each row of the ``(G, P)`` ``params`` grid, returning
        a dict of ``(G,)`` stat arrays aligned to the rows.
        """
        params = np.ascontiguousarray(np.asarray(params, dtype=np.float64))
        return self._inner.run_all(strategy, params, periods_per_year, risk_free)
