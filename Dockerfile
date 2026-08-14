# Cross-version gate: build the one abi3 wheel, then prove it imports and passes
# the smoke tests on Python 3.9, 3.11, and 3.12. The wheel is built once in
# `builder`; each test stage installs that same wheel on a different interpreter.
#
# Run one stage at a time, any failure fails the build:
#   docker build --target test39  -t emsl-gate39  .
#   docker build --target test311 -t emsl-gate311 .
#   docker build --target test312 -t emsl-gate312 .
#
# The rust-checks stage is the other half of the same gate: it runs the pure-Rust
# format, lint, and test commands CI runs, so a Rust change is verified the same
# way a Python one is, with no toolchain on the host:
#   docker build --target rust-checks .
#
# The bench stage is opt-in and prints throughput; it is not a correctness gate,
# so its timing never fails a test build:
#   docker build --target bench   .

FROM python:3.11-slim AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain 1.88.0 --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --no-cache-dir maturin
WORKDIR /src
COPY . .
RUN maturin build --release --locked --out /wheels

# The pure-Rust half of CI. rust-toolchain.toml pins only the channel, so the two
# components are added here; declaring them in that file makes rustup install them
# into an already-provisioned toolchain, which conflicts on some CI runners. The
# base carries Python because clippy over emsl-py needs an interpreter for pyo3's
# build config.
FROM python:3.11-slim AS rust-checks
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain 1.88.0 --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup component add rustfmt clippy
WORKDIR /src
COPY . .
RUN cargo fmt --all --check \
    && cargo clippy --locked --workspace --all-targets -- -D warnings \
    && cargo test --locked -p emsl-core -p bar-engine

FROM python:3.9-slim AS test39
COPY --from=builder /wheels /wheels
COPY tests /tests
COPY .Documentation /docs
COPY README.md /README.md
# the tools, so test_docs.py can check that a page explains each one (ADR 0084).
# /devtools rather than /dev, which is the device filesystem on any linux box
COPY dev /devtools
RUN pip install --no-cache-dir /wheels/*.whl numpy gymnasium pandas pyarrow optuna cloudpickle pytest \
    && python -c "import emsl; print('py39 wheel import OK')" \
    && pytest -q /tests

FROM python:3.11-slim AS test311
COPY --from=builder /wheels /wheels
COPY tests /tests
COPY .Documentation /docs
COPY README.md /README.md
COPY dev /devtools
RUN pip install --no-cache-dir /wheels/*.whl numpy gymnasium pandas pyarrow optuna cloudpickle pytest \
    && python -c "import emsl; print('py311 wheel import OK')" \
    && pytest -q /tests

FROM python:3.12-slim AS test312
COPY --from=builder /wheels /wheels
COPY tests /tests
COPY .Documentation /docs
COPY README.md /README.md
COPY dev /devtools
RUN pip install --no-cache-dir /wheels/*.whl numpy gymnasium pandas pyarrow optuna cloudpickle pytest \
    && python -c "import emsl; print('py312 wheel import OK')" \
    && pytest -q /tests

FROM python:3.11-slim AS bench
COPY --from=builder /wheels /wheels
COPY benchmarks /benchmarks
RUN pip install --no-cache-dir /wheels/*.whl numpy \
    && python /benchmarks/throughput.py

# Per-surface throughput (Engine, Backtester, tune, VectorEnv, Batch).
# Needs the full deps, so it is its own stage; run it and read stdout:
#   docker build --target bench-surfaces -t emsl-bench-surfaces .
#   docker run --rm emsl-bench-surfaces
FROM python:3.11-slim AS bench-surfaces
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl numpy gymnasium optuna cloudpickle
COPY benchmarks /benchmarks
CMD ["python", "/benchmarks/surfaces.py"]

# The differential gate: the engine against a second implementation of the same
# decisions, over randomised paths. `dev/differential/reference.py` is written from
# Decisions.md and never from the Rust, which is the whole point of it, so a rule
# the engine gets wrong is a rule the reference gets right and the two part company
# on a number. It found the slot leak of ADR 0079, which had survived every test
# written against the implementation because none of them thought to ask.
#
# Six seeds of five hundred cases is about ninety seconds and has to be exact: the
# tolerance is relative 1e-6 and any disagreement fails the build.
#
# `batch_differential.py` is the same idea one tier up, over `Batch`: every
# reinforcement learning number comes out of `reset_at_each`, `advance` and
# `snapshot`, and what stood behind them was a Rust test comparing the batch
# against the engine it is built from, which shares every mistake the engine
# makes. Each env is compared against a reference started at that env's own
# offset, because an episode beginning part way into the series still funds on
# the bars the SERIES funds on (ADRs 0002, 0017, 0018) and nothing that starts
# every env at bar zero can see that rule at all.
#   docker build --target test-differential .
FROM python:3.11-slim AS test-differential
COPY --from=builder /wheels /wheels
COPY dev/differential /differential
RUN pip install --no-cache-dir /wheels/*.whl numpy \
    && cd /differential \
    && for seed in 1 7 42 1337 20260813 99991; do \
         python differential.py 500 $seed || exit 1; \
       done \
    && for seed in 7 42 20260814; do \
         python batch_differential.py 200 $seed || exit 1; \
       done

# The chart layer's other half. Every assertion in the correctness gate reads the
# JSON spec, because the gate has no browser and a spec assertion is the sharper
# test of what Python decided (ADR 0043). What it cannot reach is whether the
# shipped JavaScript parses, runs and draws. This stage supplies the browser, so
# it is opt-in and kept out of the correctness gate for the same reason test-sb3
# is, that the image is heavy:
#   docker build --target test-browser .
# The image ships the browsers under /ms-playwright but not the python package,
# and the two are versioned together, so the pin here has to match the tag above.
FROM mcr.microsoft.com/playwright/python:v1.47.0-jammy AS test-browser
COPY --from=builder /wheels /wheels
COPY tests /tests
RUN pip install --no-cache-dir /wheels/*.whl numpy pandas pytest playwright==1.47.0 \
    && pytest -q -p no:cacheprovider /tests/test_render.py

# Stable-Baselines3 integration: install torch and sb3 and run the adapter tests.
# torch is heavy, so this is opt-in and kept out of the correctness gate:
#   docker build --target test-sb3 .
FROM python:3.11-slim AS test-sb3
COPY --from=builder /wheels /wheels
COPY tests /tests
COPY .Documentation /docs
COPY README.md /README.md
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir /wheels/*.whl numpy gymnasium stable-baselines3 pytest \
    && pytest -q /tests/test_sb3.py
