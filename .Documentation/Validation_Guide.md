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
  <a href="Decisions.md">Decisions</a> &nbsp;•&nbsp;
  <a href="Contributor_Guide.md">Contributor Guide</a> &nbsp;•&nbsp;
  <b>Validation Guide</b>
</sub>

<br>
<br>
<br>
<br>

## Validation Guide

Validation is split by where the risk lives. Almost all of the real logic is pure Rust, so it is tested locally with `cargo`, with no interpreter in the loop. The Python boundary, and the promise that one wheel works across Python versions, is checked in Docker.

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

### Cross-cutting invariants

The properties that would be easy to break silently are their own tests:

- **No same-bar lookahead.** An order decided on bar `t` fills on bar `t+1`; a market fill lands at the next open, never the decision bar's price.
- **Batched equals looped.** `Batch.step_all` over N envs is bit-identical, by exact state equality, to stepping N single engines in a loop. The envs are independent with no cross-env reduction, so parallelism introduces no drift.
- **View lifetime and read-only.** A zero-copy observation stays valid after the engine is dropped (its base keeps the buffer alive), and writing through it is rejected, so a view onto the shared candle buffer can never corrupt it.
- **Trade reconciliation.** The logged trades' realized PnL sums to the account's cumulative realized PnL. A forced liquidation books realized PnL without logging a trade (ADR 0009), so the identity is exact for a run without one; the test pins the no-liquidation case.

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
