"""Wheel-level throughput benchmark for emsl.

Covers the paths a user actually runs, not just the raw step:

- single-env ``Engine.step``, which crosses the Python-to-Rust boundary each call;
- batched ``Batch.advance``, which releases the GIL and steps every env in parallel;
- a ``Backtester`` driving a Python ``Strategy`` callback each bar (flexible, slow);
- a full RL ``VectorEnv`` step, including the observation gather and the reward.

It prints steps/sec and env-steps/sec, the only figures a speed claim in the docs
may cite. Run it inside the Docker bench stage: ``docker build --target bench .``
"""

import time

import numpy as np

from emsl import Batch, Engine


def synthetic(bars):
    i = np.arange(bars, dtype=np.float64)
    base = 100.0 + 20.0 * np.sin(i / 25.0)
    volume = np.full(bars, 100_000.0)
    return np.stack([base, base + 2.0, base - 2.0, base, volume], axis=1)


def time_single(data, passes):
    engine = Engine(data, market="perp", quote=1_000_000.0, slippage_bps=2.0)
    bars = data.shape[0]
    start = time.perf_counter()
    for _ in range(passes):
        engine.reset()
        while not engine.done():
            engine.market_buy(0.001)
            engine.step()
    elapsed = time.perf_counter() - start
    return (passes * bars) / elapsed


def time_batch(data, num_envs, passes):
    batch = Batch(
        data, num_envs=num_envs, market="perp", quote=1_000_000.0, slippage_bps=2.0
    )
    bars = data.shape[0]
    sizes = np.full(num_envs, 0.001, dtype=np.float64)
    start = time.perf_counter()
    for _ in range(passes):
        batch.reset_all()
        while not batch.done():
            batch.advance(sizes)
    elapsed = time.perf_counter() - start
    return (passes * bars * num_envs) / elapsed


def time_backtest(data, passes):
    from emsl.backtest import Backtester, Strategy

    class SmaCross(Strategy):
        def init(self, engine):
            self.closes = engine.data[:, 3]
            self.fast = 10
            self.slow = 30

        def next(self, state, engine):
            i = state["tick_index"]
            if i < self.slow:
                return
            fast = self.closes[i - self.fast + 1 : i + 1].mean()
            slow = self.closes[i - self.slow + 1 : i + 1].mean()
            pos = state["position"]
            if pos == 0.0 and fast > slow:
                engine.market_buy(1.0)
            elif pos != 0.0 and fast < slow:
                engine.close()

    bars = data.shape[0]
    start = time.perf_counter()
    for _ in range(passes):
        Backtester(data, market="spot", quote=100_000.0, slippage_bps=2.0).run(SmaCross())
    elapsed = time.perf_counter() - start
    return (passes * bars) / elapsed


def time_rl(data, num_envs, steps):
    from emsl.rl import VectorEnv

    env = VectorEnv(
        data,
        features=data,
        num_envs=num_envs,
        window=32,
        market="perp",
        quote=1_000_000.0,
        slippage_bps=2.0,
    )
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    for _ in range(steps):
        env.step(rng.integers(0, 3, size=num_envs))
    elapsed = time.perf_counter() - start
    return (steps * num_envs) / elapsed


def main():
    data = synthetic(2_000)
    print("emsl throughput (synthetic 2000-bar series, honest costs on)")

    print("-- raw stepping --")
    print(f"single-env Engine.step     : {time_single(data, 5):>16,.0f} steps/sec")
    for num_envs in (1_024, 4_096, 8_192):
        rate = time_batch(data, num_envs, 3)
        print(f"batch advance n={num_envs:<5}       : {rate:>16,.0f} env-steps/sec")

    print("-- wrappers --")
    print(f"Backtester (python SMA)    : {time_backtest(data, 20):>16,.0f} bar-steps/sec")
    for num_envs in (1_024, 4_096):
        rate = time_rl(data, num_envs, 300)
        print(f"VectorEnv step n={num_envs:<5}      : {rate:>16,.0f} env-steps/sec")


if __name__ == "__main__":
    main()
