<h1>OCEΛNO <small><code>embeddable-market-simulation-library</code></small></h1>


<div style="padding-top: 0px;">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+" /></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.88-orange.svg?logo=rust&logoColor=white" alt="Rust 1.88" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" /></a>
</div>

<sub>
  <a href="../README.md">Introduction</a> &nbsp;•&nbsp;
  <a href="Python_API.md">Python API</a> &nbsp;•&nbsp;
  <a href="RL_Guide.md">RL Guide</a> &nbsp;•&nbsp;
  <a href="Architecture.md">Architecture</a> &nbsp;•&nbsp;
  <b>Decisions</b> &nbsp;•&nbsp;
  <a href="Contributor_Guide.md">Contributor Guide</a> &nbsp;•&nbsp;
  <a href="Validation_Guide.md">Validation Guide</a>
</sub>

<br>
<br>
<br>
<br>

## Decisions

A market simulator's product is trust, and trust lives in the choices that had no single right answer. How realized PnL books when a position flips through zero, whether a spot account can short, when funding is charged relative to liquidation inside one bar: each could reasonably go more than one way, and once chosen it has to hold consistent across the whole engine. This page records those choices and the reasoning behind them, so a reader can check why a behavior is the way it is, and a contributor does not quietly undo a deliberate choice while fixing what looks odd.

Every decision keeps a number. The code and the other guides cite them by that number (`ADR 0017`), so the number is the stable handle even as the wording here is tidied.

<br>

## The Shape of the Library

The library is a stack of layers, and the one rule is that logic flows up, never sideways: each layer builds only on the one below it.

- **`emsl-core`** is pure primitives: the units, the order and fill types, the netted position, the account (equity, funding, liquidation, realized and unrealized PnL), the cost model, and the resting-order book. It knows nothing about candles, parallelism, or Python. This is the seam a future tick or L2 engine would reuse, so nothing bar-specific lives here.
- **`bar-engine`** is the bar-level realization: the candle series, the next-bar fill model, the single `step()` machine, the reporter and its stats, and the parallel batch and sweep. It builds only on `emsl-core`.
- **`emsl-py`** is the one crate that links Python. It parses arguments, copies candles once from numpy, hands state back as dicts, vends the zero-copy observation, and releases the GIL around the batched step. It holds no simulation logic.
- **`python/emsl`** is the thin wrappers: `backtest`, `rl`, `tune`. This is what "one engine, three surfaces" means in practice: a backtest, a Gymnasium RL env, and a parameter search are not three engines, they are three ways to drive the same one. A wrapper only configures the engine, drives its loop, and shapes the result. Keeping them thin is what keeps the core embeddable: it never assumes it is being used for a backtest rather than an RL rollout, so it drops into a system you already have instead of owning your loop.

<br>

## Orders and Fills

**Flip-through-zero PnL (0001).** A fill that crosses zero splits in two: it closes the old side, booking realized PnL on that part at the old average entry, then opens the remainder fresh at the fill price. A same-side add volume-weights the entry, and a non-finite or non-positive size fills nothing.

**Slippage (0004).** A market (taker) fill slips by `slippage_bps` off the next bar's open, adverse to the side.

**Volume cap (0005).** One order takes at most `max_fill_fraction` of a bar's volume, so a large limit fills over several bars rather than all at once.

**Market impact (0013).** An extra adverse slip proportional to the fraction of the bar's volume the fill takes, so a bigger order pays more. A per-fill cost function is deliberately refused; this coefficient is its fast-path stand-in.

**Time in force (0016).** `GTC` rests, `IOC` takes one bar then cancels the rest, `FOK` fills the whole size against one bar or nothing. It is meaningful on limits: a market order is `IOC` by nature and a stop rests until it triggers.

**Spot buys clamp to cash (0018).** A spot buy fills only what the quote balance affords, so a buy never drives the account negative.

**Spot cannot short (0015).** On spot a sell is clamped to the current long; shorting base needs a borrow this tier does not model, so a short and a flip through zero are perp behaviors.

**Bar-fidelity limits (0006).** At bar granularity the fill model cannot see the path inside a bar, so it can be slightly more generous than a live book, and two contradictory resting orders can both fill on one wide bar. These approximations are named rather than hidden.

<br>

## Accounting and Risk

**Funding (0002, 0017).** A perp charges funding on the held notional: a long pays a positive rate, a short receives. The cadence is measured in bars (`funding_rate`, `funding_interval`), anchored to the absolute bar index, and charged before the liquidation check, so a funding debit can bust the account on the same bar.

**Liquidation (0003).** A perp is force-closed at the bar's adverse extreme when its margin is exhausted, the low for a long and the high for a short.

**Leverage cap (0012).** `leverage` caps a perp's notional at that multiple of equity; the shipped default is a finite 10x, and `0.0` opts out into uncapped notional. Spot ignores it.

**A non-positive equity is terminal (0019).** Equity at or below zero ends the account for both markets; the RL env reads it as a termination, not a truncation.

**Statistics (0007).** The stats set (return, CAGR, Sharpe, Sortino, Calmar, drawdown, volatility, and the trade metrics) with fixed conventions: sample deviation, square-root annualization, and a degenerate or busted run returns finite numbers rather than `NaN`, so a bad run cannot poison a sweep's argmax.

**Trade recording (0009).** One trade row per closed portion of a position, carrying the closing-side fee only.

<br>

## The Parallel and RL Tier

**Zero-copy observation (0008).** A single env's observation is a read-only numpy view straight onto the shared candle buffer, no copy, made sound by the candle's `repr(C)` layout.

**Compiled sweep, and why it is not in the Python API (0011).** `bar-engine` carries an in-core sweep that runs built-in Rust strategies over a parameter grid with the GIL released and no Python in the loop. It is deliberately not exposed to Python: it can only run strategies compiled into the engine, so as a public surface it would be one nobody could extend without writing Rust, which is a promise not worth making. Parameter search from Python goes through tuning instead, and the sweep survives in the crate as the benchmark of the GIL-free path.

**RL autoreset (0010).** The vector env uses same-step autoreset: a finished env's `step` returns the next episode's first observation, with the true final observation and equity in `infos`. The observation is a `(window, F)` window over a feature matrix you supply. On gymnasium versions that tag autoreset style, it declares `SAME_STEP` in its metadata so a vector wrapper does not assume the next-step default.

**Stable-Baselines3 adapter (0022).** SB3 will not take a gymnasium vector env; it wants its own `VecEnv`. `emsl.sb3.EmslVecEnv` presents the batched env as one, a straight remap because emsl's same-step autoreset already matches SB3's contract, so PPO and the rest train on the GIL-free batch with no re-vectorization.

**Per-env cost randomization (0014).** Each cost knob accepts a `(low, high)` pair sampled once per env, so a batch trains across a spread of cost regimes.

**The batched tier is market-order-only (0020).** The parallel RL and batch path takes only market orders; a per-order Python decoder cannot ride the GIL-free step, the same reasoning that refuses a per-fill cost function.

<br>

## Tuning

**Tuning over the Strategy spine (0021).** `emsl.tune` searches a strategy's parameters by running the same backtest across worker processes. It never serializes a live engine: it rebuilds one per worker from the candle array, carries the strategy and objective with cloudpickle, and lets optuna drive the search.
