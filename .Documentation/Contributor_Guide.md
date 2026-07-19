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
  <a href="Decisions.md">Decisions</a> &nbsp;•&nbsp;
  <b>Contributor Guide</b> &nbsp;•&nbsp;
  <a href="Validation_Guide.md">Validation Guide</a>
</sub>

<br>
<br>
<br>
<br>

## Contributor Guide

The library is a cargo workspace of three crates plus a thin Python package. The one rule that shapes every change is that **logic flows up, never sideways**: each layer builds only on the one below it, and Python-specific concerns live only in the crate that links Python. Get the layer right and the change is small; put it in the wrong layer and it fights the design.

<br>

### Where a change belongs

| If it is... | It goes in... |
| :--- | :--- |
| A fidelity-neutral primitive (a unit, an order, position or account math, a cost model) | `crates/emsl-core` |
| A bar-level behavior (fills, the step machine, the reporter, batched stepping) | `crates/bar-engine` |
| A Python binding (a pyclass method, an argument type, a numpy bridge) | `crates/emsl-py` |
| A convenience over the bindings (a new runner, a wrapper) | `python/emsl` |

A good test of placement: could a future tick or L2 engine reuse it? If yes, it is a core primitive and belongs in `emsl-core`, which holds no candle series and no bar-level logic. If it only makes sense per candle, it belongs in `bar-engine`. The Python crate and package hold no simulation logic at all; they parse, convert, and orchestrate.

<br>

### The build loop

Pure-Rust work needs no Python, so the loop is short. `rust-toolchain.toml` pins the compiler to 1.88.0; add the two components once, since the toolchain file deliberately does not (declaring them there makes rustup try to install them into an already-present toolchain, which conflicts on some CI runners):

```bash
rustup component add rustfmt clippy     # once per checkout

cargo test -p emsl-core -p bar-engine
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

Python-facing work (anything in `emsl-py` or `python/emsl`) is validated on the Docker gate that builds the one abi3 wheel and runs the suite across Python versions; the full split is in the [Validation Guide](Validation_Guide.md). The `emsl-py` crate is an `extension-module` cdylib, so it cannot be unit-tested with `cargo test` (there is no interpreter to link); it is checked with `cargo build` and `clippy`, then exercised through the wheel.

<br>

### Code standards

- **Formatting is rustfmt.** Run `cargo fmt --all`; the CI checks it. No hand-formatting.
- **Clippy is a gate, not advice.** `clippy --all-targets -- -D warnings` must pass. Do not silence a lint without a comment saying why (the one crate-level allow, for a pyo3-macro false positive, is documented at its site).
- **Public items carry doc comments.** A `///` on every public type, method, and field, saying what it is, not restating its name.
- **Unsafe carries a `SAFETY` comment.** Every `unsafe` block states the invariant that makes it sound. The zero-copy observation is the worked example.
- **No em dashes in prose or comments.** Commas, semicolons, periods, or parentheses instead.

<br>

### Recording a decision

Every non-obvious decision, a PnL-booking rule, a cost convention, an autoreset semantics, is recorded in [Decisions](Decisions.md), numbered in order and settled before the code that depends on it. The entry states what was decided and why, so the next contributor reads the reasoning rather than reverse-engineering it. When you add behavior that had a real choice behind it, record its decision in the same change.

<br>

### Submitting a change

Run the build loop above green before you submit, and the Docker gate too for anything Python-facing. Commit subjects read `scope: summary`, where scope names the layer or layers touched (`core:`, `bar:`, `py:`, `docs:`, `test:`, comma-joined when a change spans them), detail goes in the body, and a governing ADR is cited in parentheses: for example `bar: charge funding on a bar interval (ADR 0017)`. A non-obvious decision lands its ADR in the same change, never a follow-up.

<br>

### Adding a wrapper

A new way to drive the engine (say a different backtest style, or a distributed runner) is a Python module under `python/emsl/`, built on the compiled `Engine` or `Batch`. It holds no simulation logic: it configures the engine, drives the loop, and shapes the results. Re-export it from `python/emsl/__init__.py` if it belongs on the top level, and add a test file under `tests/` so the Docker gate covers it. Keep documentation in this `.Documentation/` set and the top-level README; there are no per-folder READMEs.

<br>

### Adding a fidelity tier

The larger extension the workspace is built for is a second engine at a different fidelity, a tick or L2 sibling. It would be a new crate beside `bar-engine`, reusing `emsl-core` for the position, account, costs, and book, and replacing only the resolution layer (how a fill is decided). Nothing in `emsl-core` should assume bars, so that reuse stays clean; if a change to core starts to assume a candle, it belongs in `bar-engine` instead.
