<h1>OCEΛNO <small><code>embeddable-market-simulation-library</code></small></h1>


<div style="padding-top: 0px;">
  <a href="https://github.com/atOCEANO/embeddable-market-simulation-library/releases"><img src="https://img.shields.io/github/v/release/atOCEANO/embeddable-market-simulation-library?label=release&color=2ea043" alt="Latest release" /></a>
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

- **`emsl-core`** is pure primitives: the units, the OHLCV bar, the order and fill types, the netted position, the account (equity, funding, liquidation, realized and unrealized PnL), the cost model, and the resting-order book. It knows nothing about a candle *series*, parallelism, or Python: it holds one bar, never the time series, the windows over it, or any bar-level fill logic, all of which live in `bar-engine`. This is the seam a future tick or L2 engine would reuse.
- **`bar-engine`** is the bar-level realization: the candle series, the next-bar fill model, the single `step()` machine, the reporter and its stats, and the parallel batch and sweep. It builds only on `emsl-core`.
- **`emsl-py`** is the one crate that links Python. It parses arguments, copies candles once from numpy, hands state back as dicts, vends the zero-copy observation, and releases the GIL around the batched step. It holds no simulation logic.
- **`python/emsl`** is the thin wrappers: `backtest`, `rl`, `tune`. This is what "one engine, three surfaces" means in practice: a backtest, a Gymnasium RL env, and a parameter search are not three engines, they are three ways to drive the same one. A wrapper only configures the engine, drives its loop, and shapes the result. Keeping them thin is what keeps the core embeddable: it never assumes it is being used for a backtest rather than an RL rollout, so it drops into a system you already have instead of owning your loop.

<br>

## Orders and Fills

**Flip-through-zero PnL (0001).** A fill that crosses zero splits in two: it closes the old side, booking realized PnL on that part at the old average entry, then opens the remainder fresh at the fill price. A same-side add volume-weights the entry, and a non-finite or non-positive size fills nothing.

**Slippage (0004).** A market (taker) fill slips by `slippage_bps` off the next bar's open, adverse to the side.

**Volume cap (0005).** One order takes at most `max_fill_fraction` of a bar's volume, so a large limit fills over several bars rather than all at once.

**Market impact (0013).** An extra adverse slip proportional to the fraction of the bar's volume the fill takes, so a bigger order pays more. A per-fill cost function is deliberately refused; this coefficient is its fast-path stand-in.

**Time in force (0016).** `GTC` rests, `IOC` takes one bar then cancels the rest, `FOK` fills the whole size against one bar or nothing. A market order is `IOC` unless `FOK` is asked for, since it never rests and so cannot tell `GTC` from `IOC`; a stop rests until it triggers, then fills as a market order. The one-bar rule is why an unfilled market order is cancelled rather than carried: on a bar that trades no volume it fills nothing and is gone, exactly as an `IOC` limit would be, while a `GTC` limit waits for the next bar.

**A stop that cannot fill has not triggered (0035).** The sentence above, read alone, says a stop becomes a market order the moment its trigger is crossed, and a market order that fills nothing is cancelled. That reading would delete a stop-loss on any bar whose volume is zero: the trigger is crossed, no liquidity exists, and the protection is gone with nothing to show for it. Three independent readers of this page reached exactly that conclusion, so the wording earned the correction. The engine binds triggering to filling instead: a stop is consumed only when it produces a fill, so on a bar that trades nothing it stays armed and protects the position on the next bar that can actually trade. The cost is that a stop can fire later than a live venue would have filled it, which is the honest direction for the error to run on illiquid data, and the alternative loses the protection silently. This is where the bar tier's fidelity limit (ADR 0006) shows most plainly.

**Moving a resting order (0032).** `replace(id, ...)` cancels a resting order and rests a replacement carrying its side, kind and flags. It returns `None` and places NOTHING when the id is no longer resting, and that is the whole point: re-placing a trailing stop with `stop()` each bar rests a new order every time, so the one that fills leaves its siblings live and on a perp they open a position on the other side. A trail written with `replace` cannot leave two orders alive, and once its stop has filled the next replacement quietly does nothing rather than arming a fresh one. There is still no OCO group; `replace` covers the case that kept costing people real answers.

**Order status is not modelled (0033).** An order's lifecycle state was a field on the order and an enum with three variants, but nothing ever wrote it after construction, so `Filled` and `Canceled` were unconstructible and the field was the constant `Resting`. Only resting orders are ever exposed, since a filled or cancelled one is removed from the book, so the state carried no information. The field and the enum are gone; the `status` key stays in the order dict, documented as the constant it always was.

**Protective stops and stable order ids (0028).** `stop()` takes `reduce_only`, because the shortcut's whole purpose is a stop-loss and without the flag a stop that outlives the one that closed the position opens a fresh position on the other side. Nothing links resting orders: there is no OCO and no replace, so re-placing a trailing stop each bar rests a new order every time and, once the book is full, every further placement is rejected and the trail keeps only its oldest triggers. Cancel before re-placing. Order ids are unique for the engine's life rather than per episode; restarting the counter on reset meant a handle carried across one addressed a live order belonging to the next episode.

**Spot buys clamp to cash (0018).** A spot buy fills only what the quote balance affords, so a buy never drives the account negative.

**Spot cannot short (0015).** On spot a sell is clamped to the current long; shorting base needs a borrow this tier does not model, so a short and a flip through zero are perp behaviors.

**Bar-fidelity limits (0006).** At bar granularity the fill model cannot see the path inside a bar, so it can be slightly more generous than a live book, and two contradictory resting orders can both fill on one wide bar. These approximations are named rather than hidden. Resting orders resolve in book-slot order, and a cancelled slot is reused by the next placement, so when two orders fill on the same bar and a cash or margin clamp binds on the first one applied, which of them gets the room depends on slot order rather than on placement time.

**The cap and the slip are bounded (0024).** The volume cap is a `min` against `max_fill_fraction * volume`, and `f64::min` keeps the other operand when one side is NaN, so both sides carry the guard: a non-finite remaining cannot launder a NaN size into a full-cap fill, and a non-finite fraction cannot delete the cap and let one order take more than the bar traded. Slippage and market impact are adverse by definition, so both are floored at zero, and their total is held just under 1: a slip of 1.0 would price a sell at zero and anything beyond it below zero, which is not a trade.

**All-or-nothing is decided after the clamps (0025).** `FOK` means the whole size fills or none of it, and the size that can actually fill is not the one the fill model offers: the perp margin cap, the spot cash and short clamps, and `reduce_only` all shrink it afterwards. So the decision is made against a clamped copy of the fill, and a `FOK` that could only fill in part books nothing. A resting `FOK` limit is judged against the account as it stands when its turn comes, not against a snapshot taken before the bar's earlier fills.

<br>

## The Input Boundary

**The boundary validates its arguments (0027).** The candle array was checked for non-finite values from the start, on the reasoning that one would poison every mark downstream. Every other argument crossing into the engine now gets the same treatment, because they poison the same things: a NaN `quote` makes every state field NaN, a NaN `max_fill_fraction` deletes the volume cap, a fee at or below -1 turns the spot cash clamp's `1 + rate` divisor non-positive and mints equity, a negative `slippage_bps` fills better than the market, an unbounded `max_open_orders` allocates its slots up front and aborts the process rather than raising, and a zero observation `window` reaches Rayon's chunker and surfaces as a `PanicException`, which is not an `Exception` and escapes an ordinary `except`. A non-finite limit price or stop trigger is refused a book slot for the same reason a non-finite size is (ADR 0001): every comparison against it is false, so it can never fill and never expire, and it holds a slot until the book is full and later orders are silently rejected. Errors name the argument and the expectation.

<br>

## Accounting and Risk

**Funding (0002, 0017).** A perp charges funding on the held notional: a long pays a positive rate, a short receives. The cadence is measured in bars (`funding_rate`, `funding_interval`), anchored to the absolute bar index, and charged before the liquidation check, so a funding debit can bust the account on the same bar.

**Liquidation (0003).** A perp is force-closed at the bar's adverse extreme when its margin is exhausted, the low for a long and the high for a short. The close is priced at that extreme after the whole bar has printed, so a gap through it books more loss than the margin behind it and leaves bad debt: a single event can cost more than the account, and a per-trade loss past -100% is arithmetic rather than an outcome.

**Cash moves only with the position (0023).** The netted position ignores a fill that is non-finite or sized at or below its dust epsilon, and the account gates its cash move and its fee on the same test. They used to disagree: the account moved quote for a fill the position had already refused, so a sub-dust trade shifted cash with the position frozen, and a fee rate below -1 drove the spot cash clamp to a negative size that credited the account without trading. Quote cannot change unless the position changes with it. The sizing helpers follow the same rule and return zero rather than an infinity when the mark is not finite and positive.

**Leverage cap (0012).** `leverage` caps a perp's notional at that multiple of equity; the shipped default is a finite 10x, and `0.0` opts out into uncapped notional. Spot ignores it. A position already over the cap because equity fell is not force-reduced, only blocked from growing.

**An insolvent perp may only shrink (0026).** The margin cap used to switch itself off once equity reached zero, on the grounds that liquidation would handle it. Liquidation cannot handle it: it force-closes a position, and an account that is flat has none, so a fill opened from negative equity was admitted uncapped and then closed at the bar's adverse extreme, deepening the bad debt instead of being refused. With no equity there is no notional to back, so the allowance is zero and the only fills that pass are those that reduce.

**A non-positive equity is terminal (0019).** Equity at or below zero ends the account for both markets; the RL env reads it as a termination, not a truncation.

**Statistics (0007).** The stats set (return, CAGR, Sharpe, Sortino, Calmar, drawdown, volatility, and the trade metrics) with fixed conventions: sample deviation for the volatility and the Sharpe denominator, the population divisor on Sortino's downside deviation as that ratio is conventionally defined, square-root annualization, and a degenerate or busted run returns finite numbers rather than `NaN`, so a bad run cannot poison a sweep's argmax. `avg_trade_pct` is measured against starting equity, so a trade taken in a drawn-down account counts for its nominal size rather than the smaller one it really had.

**Statistics stay finite, monotone, and net (0029).** Three ways the set could still mislead, closed. A `periods_per_year` of zero made the per-period risk-free rate `0.0 / 0.0` and handed back a `NaN` Sharpe, the one thing ADR 0007 exists to prevent, so a non-positive or non-finite annualization now falls back to none at all rather than propagating. Drawdown is capped at 100%: bad debt takes equity below zero, and an uncapped drawdown gave the deeper bust the larger divisor and therefore the higher Calmar, ranking it above a shallower one. And the trade metrics are computed net of the fee each row already records, because on gross PnL a strategy whose edge is smaller than its costs reported a perfect win rate and an infinite profit factor beside a negative total return.

**A trade carries its whole round trip (0030).** A trade row used to carry the closing fill's fee only, on the reasoning that the entry fee had already been charged to cash. That left roughly half the cost attributed to nothing, so the metrics built on it stayed optimistic even after ADR 0029 netted them: on one run the exit fees came to 31.60 and the unattributed entry fees to 31.63. The engine now tracks the fee paid to open the position it holds, and a close consumes the share belonging to the size it closes, so `fees` is the entry and exit sides together and `net_pnl` is `pnl - fees`. The identity that buys: on a run that ends flat with no liquidation, the net PnL of the logged trades equals the change in equity, which is asserted by a test.

**A run that never filled is visible (0031).** A series with no volume fills nothing, correctly and silently (ADR 0016), and the result was indistinguishable from a strategy that never placed an order: zero exposure, zero return, zero trades, and a full equity curve. `num_fills` counts the fills applied, so zero beside orders you know you placed says the feed is dead rather than the strategy quiet.

**A search needs a floor on activity (0034).** `tune` ranks on a point estimate, and a point estimate is best exactly where the sample is thinnest, so an unconstrained search walks to the cell with a handful of trades and the widest interval. `min_trades` fails a trial that closed fewer than that, the same treatment a `NaN` objective gets. It defaults to zero, so nothing changes unless asked for.

**Trade recording (0009).** One trade row per closed portion of a position, with `pnl` gross of fees, `fees` the whole round trip on that size (ADR 0030), and `net_pnl` the difference. A position still open when the data ends is never closed, so it appears in `total_return_pct` and `exposure_pct`, which mark it, and in none of the trade metrics, which count only completed round trips. Buy and hold therefore reports a real return beside zero trades; read `exposure_pct > 0` with `num_trades == 0` as "still open", not as "never traded".

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

**A parallel search is not reproducible from the seed (0036).** With `n_jobs=1` a seeded search is exact: the same seed replays the same trials in the same order and returns the same winner. With `n_jobs>1` it is not, and cannot be. The driver asks for `n_jobs` trials before any result has come back, then asks for a replacement as each future completes, so the order results reach the sampler depends on which worker finishes first. The sampler sees a different history and suggests different points. Nothing is wrong; this is what asynchronous search is. But a seed that does not reproduce is exactly the kind of thing that costs an afternoon, so: pin `n_jobs=1` when a result has to be reproducible, and treat a parallel search as a search rather than a computation.
