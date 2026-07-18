"""Throughput of each way to run emsl, with one simple strategy per surface.

Built and run in the Docker bench image so the numbers come from the same abi3
wheel the gate builds, not a dev checkout:

    docker build --target bench-surfaces -t emsl-bench-surfaces .
    docker run --rm emsl-bench-surfaces

Every number is wall-clock throughput on one machine and one series; treat them as
relative between surfaces, not as absolute hardware figures. The single-env timings
take the best of a few repeats; the batched ones run once over a fixed amount of
work.
"""

import os
import platform
import time

import numpy as np

from emsl import Batch, Engine, tune
from emsl.backtest import Backtester, Strategy
from emsl.batch import BatchRunner
from emsl.rl import VectorEnv

# one shared series and one shared strategy, so every surface runs the same logic
BARS = 10_000
FAST = 10
SLOW = 50


def make_series(n):
    # a rising, oscillating close so the crossover actually trades; open equals close
    # so a next-bar fill lands on the same value, high/low bracket it
    t = np.arange(n, dtype=np.float64)
    close = 100.0 + 20.0 * np.sin(t / 15.0) + 0.05 * t
    high = close + 1.0
    low = close - 1.0
    volume = np.full(n, 1000.0)
    return np.column_stack([close, high, low, close, volume])


class SmaCross(Strategy):
    """Fast/slow moving-average crossover: long when fast crosses above slow, flat
    when it crosses back. The two windows are the tunables.
    """

    def __init__(self, fast=FAST, slow=SLOW):
        self.fast = int(fast)
        self.slow = int(slow)

    def init(self, engine):
        self.close = engine.data[:, 3]

    def next(self, state, engine):
        i = state["tick_index"]
        if i < self.slow:
            return
        fast = self.close[i - self.fast:i].mean()
        slow = self.close[i - self.slow:i].mean()
        if state["position"] == 0 and fast > slow:
            engine.market_buy(1.0)
        elif state["position"] > 0 and fast < slow:
            engine.close()


def best_rate(work, amount, repeats=3):
    # run `work` a few times, return the highest amount/second seen
    best = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        work()
        dt = time.perf_counter() - t0
        best = max(best, amount / dt)
    return best


def bench_engine_raw(data):
    # the engine stepped with no strategy at all, so the number is the single-env
    # step ceiling: the Rust step plus the state dict built and handed back each bar,
    # with none of a strategy's own per-bar work
    steps = data.shape[0] - 1

    def run():
        eng = Engine(data, market="spot", quote=100_000)
        eng.reset()
        while not eng.done():
            eng.step()

    return best_rate(run, steps)


def bench_engine(data):
    # the raw engine, driven by a hand-written Python loop, one bar at a time
    close = data[:, 3]
    steps = data.shape[0] - 1

    def run():
        eng = Engine(data, market="spot", quote=100_000)
        state = eng.reset()
        while not eng.done():
            i = state["tick_index"]
            if i >= SLOW:
                fast = close[i - FAST:i].mean()
                slow = close[i - SLOW:i].mean()
                if state["position"] == 0 and fast > slow:
                    eng.market_buy(1.0)
                elif state["position"] > 0 and fast < slow:
                    eng.close()
            state = eng.step()

    return best_rate(run, steps)


def bench_backtester(data):
    # the same loop, driven for you by the reporting backtester
    bt = Backtester(data, market="spot", quote=100_000)
    steps = data.shape[0] - 1
    return best_rate(lambda: bt.run(SmaCross()), steps)


def bench_tune(data, n_trials, n_jobs):
    # the same strategy searched, each trial a full backtest
    space = {"fast": (5, 40), "slow": (40, 200)}

    def run():
        tune(SmaCross, space, data, objective="sharpe", n_trials=n_trials, n_jobs=n_jobs, seed=0)

    # one timed run: a search is expensive and its own warmup
    t0 = time.perf_counter()
    run()
    dt = time.perf_counter() - t0
    return n_trials / dt


def bench_batchrunner(data, n_configs):
    # the compiled sweep: the same crossover, but as the built-in Rust strategy, over
    # a grid with the GIL released
    runner = BatchRunner(data, market="spot", quote=100_000)
    params = np.random.default_rng(0).integers(5, 100, size=(n_configs, 2)).astype(float)
    steps = data.shape[0] - 1

    def run():
        runner.run_all("sma_cross", params)

    configs_per_sec = best_rate(run, n_configs, repeats=2)
    return configs_per_sec, configs_per_sec * steps


def bench_vectorenv(data, num_envs, steps):
    # the RL env, stepped by a cheap vectorized momentum policy over the batched
    # observation (window last close vs window first close)
    env = VectorEnv(data, num_envs=num_envs, window=32, market="perp", quote=10_000, seed=0)
    obs, _ = env.reset(seed=0)

    def run():
        nonlocal obs
        for _ in range(steps):
            actions = np.where(obs[:, -1, 3] > obs[:, 0, 3], 1, 2)
            obs, _r, _t, _tr, _i = env.step(actions)

    return best_rate(run, num_envs * steps, repeats=2)


def bench_batch(data, num_envs, steps):
    # the low-level batch under the RL env, stepped by a simple driver: hold a small
    # long when flat, reading state back each step
    batch = Batch(data, num_envs=num_envs, market="perp", quote=10_000)
    batch.reset_all()
    sizes = np.zeros(num_envs)

    def run():
        for _ in range(steps):
            if batch.done():
                batch.reset_all()
            position = batch.snapshot()["position"]
            sizes[:] = np.where(position == 0.0, 0.1, 0.0)
            batch.advance(sizes)

    return best_rate(run, num_envs * steps, repeats=2)


def main():
    cores = os.cpu_count()
    print(f"# emsl surface benchmarks")
    print(f"# python {platform.python_version()}, {platform.machine()}, cpu_count={cores}")
    print(f"# series: {BARS} bars, strategy: sma_cross(fast={FAST}, slow={SLOW})")
    print()

    data = make_series(BARS)

    engine_raw_bps = bench_engine_raw(data)
    engine_bps = bench_engine(data)
    backtester_bps = bench_backtester(data)

    tune_trials = 48
    tune_seq = bench_tune(data, n_trials=tune_trials, n_jobs=1)
    tune_par = bench_tune(data, n_trials=tune_trials, n_jobs=-1)

    br_configs = 3_000
    batchrunner_cps, batchrunner_bps = bench_batchrunner(data, n_configs=br_configs)

    rl_envs, rl_steps = 1_024, 2_000
    vectorenv_eps = bench_vectorenv(data, num_envs=rl_envs, steps=rl_steps)
    batch_eps = bench_batch(data, num_envs=rl_envs, steps=rl_steps)

    print("surface,metric,value,detail")
    print(f"engine_raw_step,bars_per_sec,{engine_raw_bps:.0f},single env no strategy")
    print(f"engine_own_loop,bars_per_sec,{engine_bps:.0f},single env python loop")
    print(f"backtester,bars_per_sec,{backtester_bps:.0f},single env reporting wrapper")
    print(f"tune,trials_per_sec,{tune_seq:.2f},n_jobs=1 {tune_trials} trials")
    print(f"tune,trials_per_sec,{tune_par:.2f},n_jobs=-1 {tune_trials} trials {cores} cores")
    print(f"batchrunner,configs_per_sec,{batchrunner_cps:.0f},{br_configs} compiled configs")
    print(f"batchrunner,bar_steps_per_sec,{batchrunner_bps:.0f},GIL released")
    print(f"vectorenv,env_steps_per_sec,{vectorenv_eps:.0f},{rl_envs} envs x {rl_steps} steps")
    print(f"batch,env_steps_per_sec,{batch_eps:.0f},{rl_envs} envs x {rl_steps} steps")


if __name__ == "__main__":
    main()
