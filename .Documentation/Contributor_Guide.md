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
  <a href="Plotting.md">Plotting</a> &nbsp;•&nbsp;
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
| A statistic read off a finished run | `python/emsl/metrics.py` |
| An indicator, which is a function of arrays and never sees the engine | `python/emsl/ta.py` |
| Anything a chart draws (a mark, a guard, a panel rule) | `python/emsl/_chart.py` and `plot.py` |
| How a chart is painted (a primitive, the legend, the controls) | `python/emsl/_static` |

A good test of placement: could a future tick or L2 engine reuse it? If yes, it is a core primitive and belongs in `emsl-core`, which holds no candle series and no bar-level logic. If it only makes sense per candle, it belongs in `bar-engine`. The Python crate and package hold no simulation logic at all; they parse, convert, and orchestrate.

The chart layer splits on the same principle one layer further out: Python decides what is drawn and where, and the JavaScript decides only how it is painted, so no colour, threshold, number format or arithmetic lives on that side ([ADR 0043](Decisions.md)). A test greps the shipped assets for a colour, and the only hit it allows is full transparency. Every asset is a plain script that declares and executes nothing at load, which is what lets the same files run as separate script tags in the development harness and as one bundle in the wheel; there is no npm and no build step anywhere in the project.

<br>

### The build loop

Pure-Rust work needs no Python, so the loop is short. `rust-toolchain.toml` pins the compiler to 1.88.0; add the two components once, since the toolchain file deliberately does not (declaring them there makes rustup try to install them into an already-present toolchain, which conflicts on some CI runners):

```bash
rustup component add rustfmt clippy     # once per checkout

cargo test -p emsl-core -p bar-engine
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

Python-facing work (anything in `emsl-py` or `python/emsl`) is validated on the Docker gate that builds the one abi3 wheel and runs the suite across Python versions; the full split is in the [Validation Guide](Validation_Guide.md). Two stages sit outside that gate because each pulls a much heavier image, and a change to what they cover has to run them by hand: `--target test-browser` executes the chart's JavaScript in a real browser, and `--target test-sb3` installs torch and exercises the Stable-Baselines3 adapter. Anything under `_static/` needs the first one, since nothing else in the project runs that code at all. The `emsl-py` crate is an `extension-module` cdylib, so it cannot be unit-tested with `cargo test` (there is no interpreter to link); it is checked with `cargo build` and `clippy`, then exercised through the wheel.

<br>

### The dev directory

`dev/` holds the tools that check the library rather than any part of it ([ADR 0084](Decisions.md)). Nothing there is packaged into the wheel and nothing under `python/emsl` imports it. The line between it and `tests/` is that a test asserts and a tool produces: the gate runs a test and fails a build, while a tool here is run by hand and writes a file you then read and commit. There are four, and each is explained in this documentation set rather than beside itself, which is the same rule every other folder follows.

`dev/differential/` is the second simulator and the two harnesses that drive it, `differential.py` over `Engine` and `batch_differential.py` over `Batch`, plus `trace.py` for shrinking a failure. It is the one part of `dev/` the gate runs, as its own stage, and the [Validation Guide](Validation_Guide.md) is where it is explained.

`dev/charts/` produces every chart image the documentation shows. `build.py` runs the documented examples for real against a frozen parquet and saves each one as a chart, and `shoot.py` points a headless browser at what it saved and screenshots it at the size it was drawn ([ADR 0085](Decisions.md)). Nothing is cropped or retouched, so what a reader sees is what the library drew, and the snippet printed beside a picture is the snippet that produced it. Both run in one opt-in stage, which needs the published [sample dataset](https://github.com/atOCEANO/sample-market-data) mounted because a picture built from live candles cannot be regenerated, only redrawn differently:

```bash
docker build --target charts -t emsl-charts .
docker run --rm -v "<sample-market-data>/data:/data:ro" \
  -v "${PWD}/.Documentation:/out" emsl-charts
```

Regenerating should change nothing unless the drawing changed, and that is the check worth making: thirteen of the fourteen images survived the move to this stage byte for byte. If a rerun rewrites images you did not touch, something moved that you did not mean to move, and the fonts are the first place to look.

`dev/golden.py` rewrites `tests/chart_schema.json`, the key-path golden for the chart spec. The renderer is JavaScript and the gate has no browser, so the contract between the two halves is the one thing a Python test cannot reach directly: a key renamed in `_chart.py` passes every test in the suite and draws a blank chart. Recording every key path the spec carries turns that rename into a one-line diff instead. Run it whenever the spec gains, loses or renames a key, and bump `schema` in `_chart.py` in the same commit. It imports the test module rather than rebuilding the fixture itself, so the golden cannot be recorded against a different chart from the one the test checks.

`dev/diagrams/` holds the mermaid sources for the images in `.Documentation/imgs/`. They are in the repository because the originals were not kept and had to be reconstructed from the rendered PNGs, which is a thing to do once. Render one file per `docker run`, and never a loop inside `sh -c`:

```bash
docker run --rm --shm-size=1g -v "${PWD}/dev/diagrams:/data" minlag/mermaid-cli \
  -i /data/205314.mmd -o /data/205314.png \
  -c /data/config.json -p /data/puppeteer.json -b transparent -s 3
```

Then copy the PNG into `.Documentation/imgs/`. Two things the recipe depends on. The image's bundled headless-shell is broken with an ENOENT, so `puppeteer.json` points `executablePath` at `/usr/bin/chromium`. And the background has to be `transparent` rather than white, or the images invert badly against GitHub's dark mode. The palette in `config.json` is the project's own: node fill `#16232e`, a teal `#2ee6a6` border for the engine stages, a blue `#4d9feb` one for the entry point, `#e6edf3` text, `#8b949e` arrows, trebuchet sans. The images are numbered rather than named and the descriptive name survives only in their alt text: 205310 is the README hero, 205312 the crate layering, 205314 the step lifecycle, 205316 no-lookahead, 205318 the RL loop. Only 205314 has its source here so far; the rest are still PNG alone, and reconstructing one is the price of changing it.

<br>

### Code standards

- **Formatting is rustfmt.** Run `cargo fmt --all`; the CI checks it. No hand-formatting.
- **Clippy is a gate, not advice.** `clippy --all-targets -- -D warnings` must pass. Do not silence a lint without a comment saying why (the one crate-level allow, for a pyo3-macro false positive, is documented at its site).
- **Public items carry doc comments.** A `///` on every public type, method, and field, saying what it is, not restating its name.
- **Unsafe carries a `SAFETY` comment.** Every `unsafe` block states the invariant that makes it sound. The zero-copy observation is the worked example.
- **No em dashes in prose or comments.** Commas, semicolons, periods, or parentheses instead.

There is no linter on the Python side and there is not going to be one, so those conventions are held by review and are written down here rather than left to be inferred:

- **No type annotations.** The pure-Python surface is deliberately untyped; only the compiled extension is described, by `_emsl.pyi`. `from __future__ import annotations` still opens every implementation module.
- **88 characters is the target and 100 the ceiling.** Multi-line calls hang by four and never align to an `=`.
- **Docstrings are prose, not numpydoc.** No `Parameters` or `Returns` headings anywhere. Parameters are named inside the sentences in double backticks, and what a function raises is the last sentence of the paragraph. Every module, public class and public function has one; a private function gets a `#` comment instead.
- **A comment says why, never what.** It goes on the first line of the body, before the code, opens lowercase, ends without a period, and cites its decision as `(ADR NNNN)`. The bar for adding one is that the code looks wrong until you have read it.
- **An error message names the offending argument as the caller typed it**, gives the value it got, and puts the fix after a semicolon. Lowercase, no closing period. `ValueError` is a wrong value, `TypeError` a wrong type, `KeyError` an unknown name in a known set, `ImportError` a missing extra; every re-raise chains with `from`.
- **Refusing beats lying.** A function handed something it cannot answer about raises and says what would have been wrong, rather than answering about the nearest thing it can. Most of the defects this library has shipped were a plausible number where an exception belonged.

<br>

### Recording a decision

Every non-obvious decision, a PnL-booking rule, a cost convention, an autoreset semantics, is recorded in [Decisions](Decisions.md), numbered in order and settled before the code that depends on it. The entry states what was decided and why, so the next contributor reads the reasoning rather than reverse-engineering it. When you add behavior that had a real choice behind it, record its decision in the same change.

<br>

### Submitting a change

Run the build loop above green before you submit, and the Docker gate too for anything Python-facing. Commit subjects read `scope: summary`, where scope names the layer or layers touched (`core:`, `bar:`, `py:`, `docs:`, `test:`, comma-joined when a change spans them), detail goes in the body, and a governing ADR is cited in parentheses: for example `bar: charge funding on a bar interval (ADR 0017)`. A non-obvious decision lands its ADR in the same change, never a follow-up.

<br>

### Cutting a release

Releases are built by CI, never by hand. Pushing a `v*` tag runs the release workflow, which builds the `abi3` wheels for Linux (x86_64 and aarch64), macOS (one universal2 file covering Intel and Apple silicon), and Windows, plus an sdist, and attaches them all to a GitHub Release:

```bash
git tag -a vX.Y.Z -m "emsl X.Y.Z: one line on what changed"
git push origin vX.Y.Z
```

Two things before you tag. The version in `Cargo.toml` is what names the wheels, so bump it first and let the tag match it, or the release says one number while its artifacts say another. And run the gate on the exact commit you intend to tag: a tag builds, it does not test, so nothing else will catch a regression at that point.

To rehearse without releasing, run the workflow by hand from the Actions tab. A manual dispatch builds every wheel and stops there; only a tag creates the Release. That distinction is worth using, because the macOS and Windows legs exist nowhere but the runners, so a tag is the first time they are exercised.

Those wheels are the install path for anyone without a Rust toolchain, so a release is what makes a version usable at all. emsl is not published to PyPI.

<br>

### Adding a wrapper

A new way to drive the engine (say a different backtest style, or a distributed runner) is a Python module under `python/emsl/`, built on the compiled `Engine` or `Batch`. It holds no simulation logic: it configures the engine, drives the loop, and shapes the results. Re-export it from `python/emsl/__init__.py` if it belongs on the top level, and add a test file under `tests/` so the Docker gate covers it. Keep documentation in this `.Documentation/` set and the top-level README; there are no per-folder READMEs.

<br>

### Adding a fidelity tier

The larger extension the workspace is built for is a second engine at a different fidelity, a tick or L2 sibling. It would be a new crate beside `bar-engine`, reusing `emsl-core` for the position, account, costs, and book, and replacing only the resolution layer (how a fill is decided). Nothing in `emsl-core` should assume bars, so that reuse stays clean; if a change to core starts to assume a candle, it belongs in `bar-engine` instead.
