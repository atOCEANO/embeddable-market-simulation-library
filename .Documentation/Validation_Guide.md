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
  <a href="Contributor_Guide.md">Contributor Guide</a> &nbsp;•&nbsp;
  <b>Validation Guide</b>
</sub>

<br>
<br>
<br>
<br>

## Validation Guide

Validation is split by where the risk lives. Every line of simulation is pure Rust, so it is tested locally with `cargo`, with no interpreter in the loop. The Python side simulates nothing, but it is no longer small: the indicators, the post-hoc statistics and the chart are roughly as much code as the engine, and they carry their own risk, which is a wrong number rather than a wrong fill. The Python boundary, that code, and the promise that one wheel works across Python versions are all checked in Docker.

<br>

### Pure Rust, locally

`emsl-core` and `bar-engine` hold the simulation, so they carry the bulk of the tests and every core-correctness invariant. They validate with cargo:

```bash
cargo test -p emsl-core -p bar-engine
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

This needs no interpreter, so it is the inner loop for anything below the Python boundary. The same three commands are also a stage of the gate, `docker build --target rust-checks .`, which runs them against the pinned toolchain with no Rust on the host. That stage exists because the wheel gate alone will not catch a formatting or lint failure: it builds the wheel and runs pytest, so a Rust change can pass it and still fail CI. Rust changes go through both.

<br>

### The Python wheel, in Docker

The `emsl-py` crate is an `extension-module` cdylib: it cannot be unit-tested with `cargo test`, because the test binary has no interpreter to link. So the binding is checked two ways. The Rust side compiles and lints with `cargo build -p emsl-py` and `clippy`. The behavior is checked through the built wheel on a Docker gate that builds one `abi3` wheel and installs it on Python 3.9, 3.11, and 3.12, running the `tests/` suite on each:

```bash
docker build --target test311 .    # 3.11; also test39, test312
```

Because the wheel is `abi3-py39`, the same artifact is what ships, so the gate tests the real deliverable, not a per-version rebuild. This is the "works on Python 3.9 and up" check, run before a Python-facing change is done, not on every edit.

<br>

### The differential gate

The sharpest check in the project, and the only one that can find a defect nobody thought to look for:

```bash
docker build --target test-differential .
```

`dev/differential/reference.py` is a second, deliberately slow implementation of the same simulator, written from [Decisions.md](Decisions.md) and never from the Rust. That constraint is the whole point of it. A test written against the implementation asks whether the code does what the code does; this asks whether two independent readings of the same prose produce the same number, over three thousand randomised paths through leverage, liquidation, funding, partial fills and the order book. It runs six seeds of five hundred cases at a relative tolerance of 1e-6, and any disagreement fails the build.

It has already earned its place. It found the slot leak of [ADR 0079](Decisions.md), which had survived every test ever written against the engine because none of them thought to ask whether an order filled down to float dust still holds its slot. It surfaced as five separate disagreements that all reduced to one line.

Read a disagreement as a question rather than a verdict: the reference is a second reading and can be the wrong one. On the way to that first agreement it was wrong three times and the engine right three times, and each time the decisions page was precise enough to convict the reference by reading it. `dev/differential/trace.py` shrinks a failing case to its smallest form and prints both simulators bar by bar, which is what makes the question answerable.

<br>

### The two stages outside the gate

Both are opt-in, because each pulls an image far heavier than the correctness gate, and each covers something the gate structurally cannot:

```bash
docker build --target test-browser .    # the chart's javascript, in chromium
docker build --target test-sb3 .        # the Stable-Baselines3 adapter, with torch
```

`test-browser` is the other half of the chart layer. Everything in `tests/test_chart.py` reads `Chart.spec()`, because the gate has no browser and a spec assertion is the sharper test of what Python decided ([ADR 0043](Decisions.md)). What that cannot reach is whether the shipped JavaScript parses, runs and draws, and roughly 1,250 lines of it had never been executed by anything in the project. The first run of it found that the vendored renderer joins a line straight across whitespace, so ADR 0038's central claim was false in the artifact while true in the document ([ADR 0073](Decisions.md)). `tests/test_render.py` skips wherever playwright is absent, so the correctness gate stays green without one.

<br>

### Cross-cutting invariants

The properties that would be easy to break silently are their own tests:

- **No same-bar lookahead.** An order decided on bar `t` fills on bar `t+1`; a market fill lands at the next open, never the decision bar's price.
- **Batched equals looped.** `Batch.step_all` over N envs is bit-identical, by exact state equality, to stepping N single engines in a loop. The envs are independent with no cross-env reduction, so parallelism introduces no drift.
- **View lifetime and read-only.** A zero-copy observation stays valid after the engine is dropped (its base keeps the buffer alive), and writing through it is rejected, so a view onto the shared candle buffer can never corrupt it.
- **Trade reconciliation.** The logged trades' realized PnL sums to the account's cumulative realized PnL, and on a run that ends flat the net PnL of the log equals the change in equity ([ADR 0030](Decisions.md)). A forced close is an ordinary row carrying `liquidated: true`, because it is booked through the one path every other fill takes; it used to reach none of that and appear in no trade at all ([ADR 0052](Decisions.md)).
- **Bad debt is unreachable, not caught.** A perp cannot end a bar owing money, on any path, because the liquidation is a price on the bar rather than a check at the end of it ([ADRs 0052, 0067](Decisions.md)). The guarantee is worth more than the check because it is what the exit price is derived from; when it was only a check, an ordinary `close()` on a bar that gapped past the liquidation stepped around it and booked the whole gap.
- **A mutation has to fail the suite.** A pass that breaks one line at a time and reruns the tests found 13 of 27 mutations surviving the Python tier, including one that held out the *start* of a series where [ADR 0049](Decisions.md) promises the end. The same pass over the Rust crates, with `cargo-mutants`, is the other half; a statistic or a fill rule whose defect no test can see is one nobody should quote, so a new one arrives with a test that fails when its arithmetic is wrong rather than one that only checks its shape. Run the Python pass against `python/emsl` after adding a statistic, and pass `-p no:cacheprovider`, or a stale `.pytest_cache` will hide a mutation.

<br>

### Adversarial verification of the riskiest code

Some code is dense enough that tests alone are not enough confidence: the netted position's PnL booking, the step state machine, the unsafe zero-copy view, and the trade accounting. Each was checked before it was committed by an adversarial pass, two independent reviewers working from the decision record, not the implementation:

- An **oracle** derives the expected result (the trade log, the fill, the PnL) from the ADR conventions by hand, without reading the code.
- A **code auditor** traces the implementation on the same scenario and hunts for the specific failure modes that code has (a use-after-free, an off-by-one in a close, a stale entry price).

When the two agree, and agree with the hand-written tests, the code ships. When they disagree, the gap is a bug or a missing decision. This caught real defects during the build (a float-dust close, an unenforced reduce-only, an aliasing hazard) before they reached a commit.

<br>

### Decisions gate the code

Every non-obvious behavior is settled as a numbered [decision](Decisions.md) before the code that depends on it, so a reviewer checks the implementation against a written intent rather than a guess. The decisions are the specification the tests and the adversarial passes verify against.

<br>

### Nothing ships unverified

A change below the Python boundary is not done until cargo test, fmt, and clippy pass; a Python-facing change is not done until the Docker gate is green across the version matrix. The riskiest changes add an adversarial pass on top. Continuous integration runs the same two loops (see [`.github/workflows/`](../.github/workflows/)).
