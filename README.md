<h1>OCEΛNO <small><code>embeddable-market-simulation-library</code></small></h1>


<div style="padding-top: 0px;">
  <a href="https://github.com/atOCEANO/embeddable-market-simulation-library/releases"><img src="https://img.shields.io/github/v/release/atOCEANO/embeddable-market-simulation-library?label=release&color=2ea043" alt="Latest release" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+" /></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.88-orange.svg?logo=rust&logoColor=white" alt="Rust 1.88" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" /></a>
</div>

<sub>
  <b>Introduction</b> &nbsp;•&nbsp;
  <a href=".Documentation/Python_API.md">Python API</a> &nbsp;•&nbsp;
  <a href=".Documentation/RL_Guide.md">RL Guide</a> &nbsp;•&nbsp;
  <a href=".Documentation/Plotting.md">Plotting</a> &nbsp;•&nbsp;
  <a href=".Documentation/Architecture.md">Architecture</a> &nbsp;•&nbsp;
  <a href=".Documentation/Decisions.md">Decisions</a> &nbsp;•&nbsp;
  <a href=".Documentation/Contributor_Guide.md">Contributor Guide</a> &nbsp;•&nbsp;
  <a href=".Documentation/Validation_Guide.md">Validation Guide</a>
</sub>

<br>
<br>
<br>
<br>

## Introduction

`emsl` is **a bar-level crypto market simulator: a Rust core with a small Python surface.** Give it a table of candles and it stands in for the exchange: you step forward one bar at a time, place the order types a real venue offers, and read back your balance, position, and equity, with the real fees, funding, and slippage charged as you go.

The point is research speed. A single backtest, a parameter search over thousands of configurations, and a batched reinforcement-learning rollout all run on the same engine through one API, so the simulator is never the part you rebuild between experiments. You write a strategy or an agent once, against one fill model and one set of costs, and move between those jobs without reconciling separate simulators or second-guessing which costs each of them charged. The honest market stays a fixed point; only the question you are asking changes.

The whole engine is one small loop: load candles, place an order if you want to, call `step` to advance exactly one candle, read your new state, repeat until the data runs out. Everything else, the RL env and the backtester, is a thin wrapper that drives that same loop for you.

**Embeddable, not a framework.** You own the loop, so it drops into a system you already have instead of forcing you into its own. The core knows nothing about rewards or indicators; it advances a candle and returns a state. That is what lets one engine serve both a five-line backtest and a batched RL rollout, and it is why each word of the name is a fact rather than a brand. The chart layer sits outside that core and holds to the same rule: it computes nothing and knows no indicator names, it draws the arrays you hand it.

<br>

<div align="center">
  <img src=".Documentation/imgs/205310.png" alt="One engine behind every surface" width="80%" />
  <p style="margin: 0;"><i>One engine and one fill model behind a backtest, a Gymnasium RL env, and a parameter search</i></p>
</div>

<br>
<br>

## Scope and Fidelity

It **does**: event-driven bar simulation, thousands of parallel runs over one shared series, the order types a real venue offers, a cost model (maker/taker fees, perp funding, slippage, market impact, a per-order volume cap), a small Python API, and a chart layer that draws a run and your own arrays into one self-contained HTML file.

It **does not**: model the full order book (L2), latency, or queue position, and it places no live orders. This is the bar-level, approximate member of a planned family; a future sibling is a tick and L2 engine at higher resolution. At bar granularity the fill model cannot see the intrabar path, so it can be slightly more generous than a live book would be; the costs it charges are the real ones. The bar-level resolution limits are recorded in [ADR 0006](.Documentation/Decisions.md).

**Who it is for.** Quant and machine-learning researchers. Not market makers or high-frequency desks, who run C++ and FPGA next to the exchange and need the order book this tier does not model. One decision per candle is the resolution most research works at.

**How it differs from what you already use.** If you have driven a vectorized backtester, you know it cannot express a stop or a flip whose fill depends on the path the price took; emsl can, and runs thousands of such paths in parallel. If you have driven a general backtesting framework, you know it owns your loop; here you own it. If you have used an off-the-shelf RL trading env, you know the fills are a toy; here the backtest, the RL env, and the sweep share one fill model and the same costs.

<br>
<br>

## How It Works

You hand the engine a table of candles and walk it forward one bar at a time. On any bar you can place an order, a market buy or sell, a limit, or a stop, then call `step` to advance exactly one candle: it resolves whatever is pending and hands back your new account, the balances, position, and equity. Repeat to the end of the data. That loop is the whole engine; the backtester and the RL env are wrappers around it.

The rule that keeps you honest is one bar of delay: you decide on the close of a bar and the order fills on the **next** one, never the bar the decision saw. That guard against same-bar lookahead is what separates a believable backtest from a fooled one.

Point it at real candles with the sibling [`exchange-router-service`](https://github.com/atOCEANO/exchange-router-service), which hands back a ready-to-use DataFrame:

```python
from exchange_router_client import ExchangeRouterClient

with ExchangeRouterClient("http://localhost:8040") as client:
    candles = client.get_candles("binance", "spot", "BTCUSDT", interval="1h", limit=5000)
```

Then step through it. `emsl.to_ohlcv` turns that DataFrame (or a parquet path, or a numpy array you already hold) into the `(T, 5)` array the engine takes, ignoring the router's extra `volume_usd` column and handling its datetime index for you. Here a plain momentum rule: long above the last day's average, flat below it.

```python
from emsl import Engine, to_ohlcv

eng = Engine(to_ohlcv(candles), market="spot", quote=10_000)
close = eng.closes                             # zero-copy view of every close
state = eng.reset()
while not eng.done():
    i = state["tick_index"]
    if i >= 24:                                # a day of hourly history
        avg = close[i - 24:i].mean()
        if state["position"] == 0 and close[i] > avg:
            eng.market_buy(0.1)                # decided now, fills on the next bar
        elif state["position"] > 0 and close[i] < avg:
            eng.close()                        # flatten below it
    state = eng.step()

print(state["equity"], state["position"])      # account value and position at the end
```

<br>
<br>

## Ways to Drive It

### Raw engine

The candles are readable as named zero-copy views, `engine.opens`, `engine.highs`, `engine.lows`, `engine.closes` and `engine.volumes`, so nothing has to remember that the close is column 3. `engine.data` is still the whole `(T, 5)` array.

The order verbs mirror a real exchange, so a strategy is written against the same calls it would use live. `market_buy` / `market_sell` (taker, fills at the next open), `limit_buy` / `limit_sell` (maker, rests until a candle reaches it), `stop(side, size, trigger)`, `close()`, `cancel(order_id)`, and `cancel_all()`. The full `order(side, size, type=, price=, trigger=, reduce_only=, post_only=, tif=)` primitive sets the flags the shortcuts leave default (a reduce-only stop-loss, a post-only limit, an IOC or FOK fill), and `qty_from_weight` / `qty_from_quote` size an order from a fraction of equity or a cash amount. One netted position: net long, net short, or flat, never both, and never short on spot. Every `step` returns a state dict; see the [Python API](.Documentation/Python_API.md) for the full field list and order semantics.

### Backtest

A classic `Strategy` class with `init` and `next`, driven over the series, returning stats, the equity curve, and the trade log. Set `warmup` to the longest history a decision reads and `next` is not called until that many bars exist, which replaces the guard every strategy used to open with and closes what that guard was really for: on an early bar `close[i - 50]` is a negative index, numpy resolves it from the **end** of the array, and the rule reads prices from the future without raising anything.

```python
from emsl.backtest import Backtester, Strategy

class BuyDip(Strategy):
    warmup = 1                                # next() waits until a previous bar exists

    def init(self, engine):
        self.close = engine.closes            # zero-copy view of the closes

    def next(self, state, engine):
        i = state["tick_index"]
        if state["position"] == 0 and self.close[i] < self.close[i - 1]:
            engine.market_buy(1.0)            # buy after a down bar
        elif state["position"] > 0 and self.close[i] > self.close[i - 1]:
            engine.close()                    # close into the next up bar

result = Backtester(ohlcv, market="spot").run(BuyDip())
print(result.stats["sharpe"], result.stats["max_drawdown_pct"])
print(len(result.trades), "trades")
```

The full stats set (total return, net profit, CAGR, Sharpe, Sortino, Calmar, drawdown, volatility, exposure, win rate, profit factor, average trade, and the trade log) is in the [Python API](.Documentation/Python_API.md#backtesting).

### Reinforcement learning

A Gymnasium-compatible vectorized env. You control the observation (a window of your own features), the reward (a function over the batched state), and the action decoding. Thousands of envs step in parallel, each starting at a random offset so they walk decorrelated stretches of history, sharing one read-only copy of the data.

```python
import numpy as np
from emsl.rl import VectorEnv

def reward_fn(state, prev):                       # batched, one value per env
    return (state.equity - prev.equity) - 0.01 * np.abs(state.position)

env = VectorEnv(
    data=ohlcv,                                    # (T, 5) fills orders
    features=my_indicators,                        # (T, F) what the agent sees
    reward_fn=reward_fn,
    num_envs=4096, window=60, market="perp", slippage_bps=2.0,
)
obs, info = env.reset(seed=0)                      # (4096, 60, F) float32
for _ in range(1000):
    actions = policy(obs)                          # (4096,) Discrete(3)
    obs, rewards, terminations, truncations, infos = env.step(actions)
    # 4096 envs stepped in parallel, GIL released; finished ones auto-reset,
    # their final obs and equity in infos
```

Or skip the loop and hand it to Stable-Baselines3, which trains on the same env in three lines:

```python
from stable_baselines3 import PPO
from emsl.sb3 import EmslVecEnv

PPO("MlpPolicy", EmslVecEnv(env)).learn(total_timesteps=100_000)   # A2C, DQN, SAC swap in
```

The observation, reward, action, and autoreset contract, and the full training examples, are in the [RL Guide](.Documentation/RL_Guide.md).

### Parameter search

Search a strategy's parameters for the setting that maximizes an objective. `tune` runs each trial as a full backtest through the same `Backtester`, so the strategy you backtest is the strategy you tune, and the trials run across worker processes, so the search scales over cores. This is the path for a strategy you write.

```python
from emsl import tune
from emsl.backtest import Strategy

class SmaCross(Strategy):
    def __init__(self, fast, slow):           # the tunables, as constructor arguments
        self.fast, self.slow = fast, slow

    def init(self, engine):
        self.close = engine.closes
        self.warmup = self.slow               # read after init, so it can be computed

    def next(self, state, engine):
        i = state["tick_index"]
        fast = self.close[i - self.fast:i].mean()
        slow = self.close[i - self.slow:i].mean()
        if state["position"] == 0 and fast > slow:
            engine.market_buy(1.0)            # long when the fast average leads
        elif state["position"] > 0 and fast < slow:
            engine.close()                    # flat when it falls back

result = tune(
    SmaCross,                                 # your strategy, or any callable that builds one
    {"fast": (5, 40), "slow": (40, 200)},     # the space to search
    ohlcv,
    objective="sharpe", n_trials=200, oos=0.3,     # search 70%, keep 30% back
    n_jobs=-1,                                     # -1 uses every core
)
print(result.best_params)
print(result.best_value)                      # in-sample: the best of 200 tries
print(result.oos_stats["sharpe"])             # on bars no trial ever saw
```

**Read the second number, not the first.** `best_value` is the maximum of a noisy score over 200 attempts on the bars the search was allowed to see, so it is biased upward by the searching itself, by more the harder you search. `oos=0.3` fits every trial on the first 70% of the series and scores the winner on the last 30%, which no trial ever touched. The gap between the two is what the search invented. Leave `oos` out and it says so ([ADR 0049](.Documentation/Decisions.md)).

`objective` takes any stats key, or a function of your own over the finished run, so you can score on something the stats set does not name. It receives the whole `BacktestResult`, which puts the equity curve and every trade in reach, and a lambda or a closure survives the trip to the worker processes:

```python
def risk_adjusted(result):
    stats = result.stats
    if stats["num_trades"] < 20:                       # ignore configs that barely trade
        return -1e9
    return stats["sharpe"] - 0.1 * stats["max_drawdown_pct"]

result = tune(SmaCross, {"fast": (5, 40), "slow": (40, 200)}, ohlcv,
              objective=risk_adjusted, n_trials=200, oos=0.3, n_jobs=-1)
```

Higher wins by default; pass `direction="minimize"` when your metric is a cost. A trial whose objective returns `NaN` is failed and the search moves on. `n_jobs=-1` spreads the trials over every core, and the cost is reproducibility: a parallel search asks for trials before earlier ones have reported, so the sampler follows a different path each run. Pin `n_jobs=1` when a result has to reproduce from its seed ([ADR 0036](.Documentation/Decisions.md)).

`tune` needs optuna and cloudpickle (`pip install 'emsl[tune]'`); the [Python API](.Documentation/Python_API.md#tuning) covers the search space, objective, and result in full.

### Looking at the result

The four paths above produce numbers. `emsl.chart` is how you look at them: candles, the fills on the bars they happened on, an equity curve, a drawdown panel, a trade log, and any array of your own beside them.

```python
import emsl
from emsl.plot import Line

strategy = SmaCross(fast=20, slow=50)

result = emsl.backtest.Backtester(candles=frame, market="perp").run(strategy)

emsl.chart(
    frame=frame,
    marks=[
        Line(values=strategy.fast, name="SMA 20"),
        Line(values=strategy.slow, name="SMA 50", style="dashed"),
    ],
    run=result,
).show()
```

<div align="center">
  <img src=".Documentation/imgs/charts/chart-backtest.png" alt="A backtest drawn: candles, two averages, fills on the bars they filled on, momentum, equity and drawdown" width="100%" />
  <p style="margin: 0;"><i>One call, one year of hourly candles. Nothing above configures a panel, an axis, or a colour</i></p>
</div>

<br>

**Everything above the fold came free**, and everything else is one more mark in the same list. You add a `Line` for a moving average or an equity curve of your own, a `Band` for a channel or a conditional shading, a `Level` for a threshold, `Markers` for a glyph on every bar a condition holds, a `Background` for a regime, a `Histogram` for anything per-bar. Any of them goes on the candles or on a panel of its own, just by naming one, and panels are stretched, pinned to a range, put on a log scale, or hidden. Every mark takes one colour or **one colour per bar**, so a line can carry a second variable in its own shading, and the candles themselves can be tinted the same way. A series can even be drawn past the last candle, which is what an Ichimoku cloud needs.

None of it is required. A frame alone is a chart, an array finds its own home, and a run brings its own panels.

**`show()` draws it in the notebook cell**, inline, where you are already working. No second window, no server, no separate viewer.

**The chart stays in the notebook.** It is stored in the `.ipynb` itself, so opening that file next year shows the same charts, scrollable and zoomable, with no kernel running and nothing re-executed. Send somebody the notebook and they see what you saw.

**`save()` writes one HTML file** you open in a browser by double-clicking it, for the people who want the picture without the notebook.

```python
emsl.chart(frame=frame, run=result).save(path="reports/run.html")
```

It computes nothing and knows no indicator names: pass the strategy's own arrays and the line on screen cannot drift from the line that made the decision.

The [Plotting](.Documentation/Plotting.md) guide shows every one of those marks with the picture it draws, the length contract that keeps a series on the bar it belongs to, and what cannot be charted.

<br>
<br>

## Speed

Throughput depends on which path you drive, and they differ by orders of magnitude, so it is worth knowing which one you are on. Measured on a 28-thread x86_64 box against the built wheel over a 2,000-bar series:

| Path | Throughput | What bounds it |
| :--- | ---: | :--- |
| Single-env raw step | ~358K steps/s | the Python-to-Rust boundary, crossed on every `step` |
| Single-env with a Python strategy | ~77K steps/s | your per-bar Python, not the engine |
| Batched RL step | ~110-200K env-steps/s | building the observation, GIL released |

The single-env rows say something worth internalizing: the engine steps at about 358K per second with nothing attached, and a strategy that recomputes indicators with numpy on every bar drops that to about 77K, so a single backtest is bound by your own per-bar code, not by emsl. Vectorize the indicators once and the rate climbs back toward the engine. The batched RL step releases the GIL and moves every env together, bound by building the observation rather than by the fill. A parameter search scales the other way, across processes rather than per bar, so its rate is trials per second and it grows with the cores you give it.

These move with hardware and workload, so treat them as shape, not a promise. Run `benchmarks/throughput.py` for your own machine, and the Criterion bench in `crates/bar-engine/benches` for the pure-Rust step and the in-core sweep.

<br>
<br>

## Install

Every [release](https://github.com/atOCEANO/embeddable-market-simulation-library/releases) carries prebuilt wheels, so installing one needs no compiler. **emsl is not on PyPI**, so a bare `pip install emsl` finds nothing; point pip at a release instead.

Each build covers one platform, and one wheel per platform serves **Python 3.9 and up**, because the extension is stable-ABI:

| Platform | Wheel | Built |
| :--- | :--- | :--- |
| Linux x86_64 | manylinux | every release |
| Linux aarch64 | manylinux | best effort, cross-compiled; a release may not have one |
| macOS, Intel and Apple silicon | one universal2 file covering both | every release |
| Windows x64 | win_amd64 | every release |

Anything not in that table, Windows on ARM for instance, builds from source in the same command; you just need the Rust toolchain for it.

**The version below is the one to change.** It appears twice, here and in the tagged source install below, and a test asserts both match the shipping wheel, because a hardcoded tag in a README goes stale the day after a release; take the current one from the badge at the top of this page or from the releases page.

```bash
pip install --find-links https://github.com/atOCEANO/embeddable-market-simulation-library/releases/expanded_assets/v0.5.0 emsl
```

The optional extras use the same `--find-links` URL and stack:

| Extra | Pulls | For |
| :--- | :--- | :--- |
| `data` | pandas, pyarrow | DataFrame and parquet input |
| `tune` | optuna, cloudpickle | the `emsl.tune` parameter search |
| `sb3` | stable-baselines3 | the Stable-Baselines3 adapter |

```bash
pip install --find-links <the url above> "emsl[tune,sb3]"
```

To pick a wheel by hand, copy its link from the assets list on the [releases page](https://github.com/atOCEANO/embeddable-market-simulation-library/releases) and pass that URL to `pip install` directly.

Building from source needs the Rust toolchain ([rustup](https://rustup.rs)), and pip drives the build for you:

```bash
pip install "git+https://github.com/atOCEANO/embeddable-market-simulation-library.git"          # main
pip install "git+https://github.com/atOCEANO/embeddable-market-simulation-library.git@v0.5.0"   # or a tag
```

From a local checkout, either as a plain install or as a development build:

```bash
git clone https://github.com/atOCEANO/embeddable-market-simulation-library.git
cd embeddable-market-simulation-library

pip install .                    # a normal install from the checkout
                                 # or, to work on the library itself:
pip install maturin
maturin develop --release        # compiles the abi3 extension into the active venv
```

Or build the wheel once and install that copy wherever you need it:

```bash
maturin build --release --out dist
pip install dist/*.whl
```

`numpy` and `gymnasium` install with the package either way.

<br>
<br>

## Building and Validating

The pure-Rust crates validate locally with cargo; the Python surface validates on a Docker gate that builds the one abi3 wheel and runs the test suite across Python 3.9, 3.11, and 3.12. The riskiest code is adversarially verified, a process that has caught real bugs before they shipped; the full workflow is in the [Validation Guide](.Documentation/Validation_Guide.md).

```bash
cargo test -p emsl-core -p bar-engine     # pure-Rust core
cargo clippy --workspace --all-targets -- -D warnings
docker build --target test311 .           # wheel + pytest on Python 3.11
```
