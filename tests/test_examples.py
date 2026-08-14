"""Run the plotting guide's examples instead of only reading them.

``test_docs.py`` parses every fenced block and checks that each keyword names a
real parameter, and says in its own docstring that it does not execute them,
because most stand on a frame and a strategy the prose describes. That line is
reasonable and it has a cost that came due twice: a flagship ``deflated_sharpe``
example composed two calls that could never work together (ADR 0066), and the
plotting guide's longest block read an attribute off a ``BacktestResult`` that
has never existed, so it raised on the first copy-paste and no amount of parsing
could see it.

So the page gets a prelude, and every block that stands on it is executed. The
names come from what the page itself binds where it binds them, so a block is
run against the values a reader would have in hand at that point rather than
against something convenient. ``show()`` needs a notebook, so it is swapped for
``spec()``, which builds the whole document and throws everything a real call
would; nothing here is asserted about pixels.
"""

import pathlib
import re

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

import emsl
from emsl.plot import (
    Background, Band, Histogram, Level, Line, Marker, Markers, Panel, Recorder,
    at_bar, at_next, ramp,
)

PAGE = pathlib.Path("/docs/Plotting.md")

# the gate stages mount the doc set; the browser stage copies only tests/, so
# this skips there rather than failing on a path that was never meant to exist
if not PAGE.exists():
    pytest.skip("the documentation is not mounted here", allow_module_level=True)

# A block whose point is prose rather than a call, keyed by a phrase it contains
# so the list survives the page being edited around it. each one says why
NOT_EXECUTABLE = {
    "class SmaCross": "declares the strategy the rest of the page stands on",
    "def marks(self)": "a method body, shown to explain where marks can live",
    "Traceback": "the error a wrong length produces, quoted rather than run",
}


def blocks():
    text = PAGE.read_text(encoding="utf-8")
    found, buffer, inside = [], [], False
    for line in text.splitlines():
        if line.startswith("```python"):
            inside, buffer = True, []
        elif line.startswith("```") and inside:
            found.append("\n".join(buffer))
            inside = False
        elif inside:
            buffer.append(line)
    return found


def frame(n=300):
    # the swing has to be big enough against the drift for a 20/50 cross to fire,
    # or half the page stands on a run with no trades in it
    step = np.sin(np.arange(n, dtype=np.float64) / 9.0) * 2_600.0
    close = 20_000.0 + np.arange(n, dtype=np.float64) * 12.0 + step
    return pd.DataFrame(
        {
            "open": close - 3.0,
            "high": close + 25.0,
            "low": close - 25.0,
            "close": close,
            "volume": 900.0 + np.arange(n, dtype=np.float64),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )


class SmaCross(emsl.Strategy):
    def __init__(self, fast=20, slow=50):
        self.fast_n, self.slow_n = int(fast), int(slow)

    def init(self, engine):
        close = pd.Series(engine.data[:, 3])
        self.fast = close.rolling(self.fast_n).mean().to_numpy()
        self.slow = close.rolling(self.slow_n).mean().to_numpy()
        self.warmup = self.slow_n
        self.log = Recorder(engine)

    def marks(self):
        return [
            Line(self.fast, f"SMA {self.fast_n}"),
            Line(self.slow, f"SMA {self.slow_n}", style="dashed"),
        ]

    def next(self, state, engine):
        i = state["tick_index"]
        self.log.at_bar(state, lev=state["position"])
        if self.fast[i] > self.slow[i] and state["position"] <= 0.0:
            engine.market_buy(0.1)
        elif self.fast[i] < self.slow[i] and state["position"] > 0.0:
            engine.close()


def prelude():
    """Every name the page uses, bound the way the page binds it."""
    data = frame()
    close = data.close
    strategy = SmaCross(fast=20, slow=50)
    result = emsl.backtest.Backtester(candles=data).run(strategy)
    fast, slow = close.rolling(20).mean(), close.rolling(50).mean()
    logret = np.log(close / close.shift(1))
    sd = close.rolling(20).std()
    held = np.zeros(len(data), dtype=bool)
    for trade in result.trades:
        held[trade["entry_tick"]:trade["exit_tick"] + 1] = True

    rows = np.asarray(logret.rolling(24).std())
    regime = np.where(rows < np.nanquantile(rows, 0.33), "calm", "wild").astype(object)

    runs = {"20/50": result, "20/60": result, "24/50": result}
    return {
        "numpy": np, "pandas": pd, "emsl": emsl,
        "Background": Background, "Band": Band, "Histogram": Histogram,
        "Level": Level, "Line": Line, "Marker": Marker, "Markers": Markers,
        "Panel": Panel, "Recorder": Recorder, "ramp": ramp,
        "at_bar": at_bar, "at_next": at_next, "SmaCross": SmaCross,
        "frame": data, "close": np.asarray(close, dtype=float),
        "strategy": strategy, "result": result, "win": result,
        "neighbours": runs, "runs": runs,
        "fast": fast, "slow": slow, "logret": logret,
        "rsi": emsl.ta.rsi(np.asarray(close, dtype=float), 14),
        "upper": np.asarray(fast + 2 * sd), "lower": np.asarray(fast - 2 * sd),
        "crossed": emsl.ta.crossover(fast, slow),
        "risk_off": np.asarray(rows > np.nanquantile(rows, 0.66)),
        "trending": np.asarray(close > close.rolling(50).mean()),
        "held": held,
        # a per-bar COLOUR, which is how the page uses it, not the mask behind it
        "shade": np.where(held, "#4d9fff", None),
        "regime": regime,
        "fast_z": np.asarray((close - fast) / sd),
        "slow_z": np.asarray((close - slow) / sd),
        "vam24": np.asarray(logret.rolling(24).mean()),
        "vam72": np.asarray(logret.rolling(72).mean()),
        "vam168": np.asarray(logret.rolling(168).mean()),
        "squeeze": np.asarray(sd / close),
        "forward": np.concatenate([
            np.asarray(close, dtype=float)[4:] / np.asarray(close, dtype=float)[:-4] - 1.0,
            np.full(4, np.nan),
        ]),
        "buy_and_hold": np.asarray(close / close.iloc[0] * 10_000.0),
        "tenkan": close.rolling(9).mean(), "kijun": close.rolling(26).mean(),
        "GREEN": "#2fe0a8", "RED": "#ff5470",
        "senkou_a": np.concatenate(
            [np.full(26, np.nan), np.asarray((close.rolling(9).mean() + close.rolling(26).mean()) / 2)]
        ),
        "senkou_b": np.concatenate(
            [np.full(26, np.nan), np.asarray(close.rolling(52).mean())]
        ),
        "marks": [Line(np.asarray(fast), "SMA 20")],
        "chart": emsl.chart(frame=data, run=result),
    }


def test_the_page_is_not_quietly_empty():
    assert PAGE.exists()
    assert len(blocks()) > 25


def test_every_example_that_stands_on_the_prelude_actually_runs():
    # show() wants a notebook and save() wants a path, so both become spec(),
    # which builds the entire document and raises everything a real call raises
    ran, skipped, failures = 0, [], []
    for number, source in enumerate(blocks()):
        why = next((r for k, r in NOT_EXECUTABLE.items() if k in source), None)
        if why is not None:
            skipped.append((number, why))
            continue
        # show() takes a height and save() a path, so both swallow their arguments
        code = re.sub(r"\.show\(\s*[^)]*\)", ".spec()", source)
        code = re.sub(r"\.save\(\s*[^)]*\)", ".spec()", code)
        namespace = prelude()
        try:
            exec(compile(code, f"Plotting.md block {number}", "exec"), namespace)
        except Exception as exc:
            failures.append(
                f"\n--- block {number}: {type(exc).__name__}: {exc}\n{source}"
            )
        else:
            ran += 1

    assert not failures, "\n".join(failures)
    assert ran >= 30, f"only {ran} blocks ran, {len(skipped)} skipped"
