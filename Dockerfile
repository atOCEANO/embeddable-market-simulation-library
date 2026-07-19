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
RUN pip install --no-cache-dir /wheels/*.whl numpy gymnasium pandas pyarrow optuna cloudpickle pytest \
    && python -c "import emsl; print('py39 wheel import OK')" \
    && pytest -q /tests

FROM python:3.11-slim AS test311
COPY --from=builder /wheels /wheels
COPY tests /tests
RUN pip install --no-cache-dir /wheels/*.whl numpy gymnasium pandas pyarrow optuna cloudpickle pytest \
    && python -c "import emsl; print('py311 wheel import OK')" \
    && pytest -q /tests

FROM python:3.12-slim AS test312
COPY --from=builder /wheels /wheels
COPY tests /tests
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

# Stable-Baselines3 integration: install torch and sb3 and run the adapter tests.
# torch is heavy, so this is opt-in and kept out of the correctness gate:
#   docker build --target test-sb3 .
FROM python:3.11-slim AS test-sb3
COPY --from=builder /wheels /wheels
COPY tests /tests
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir /wheels/*.whl numpy gymnasium stable-baselines3 pytest \
    && pytest -q /tests/test_sb3.py
