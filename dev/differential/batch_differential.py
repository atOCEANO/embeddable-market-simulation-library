"""Run the BATCHED path against one independent reference per env.

`Batch` is the tier nothing independent has ever checked, and every reinforcement
learning number comes out of it: `VectorEnv` drives `reset_at_each`, `advance`
and `snapshot` and nothing else. What stood behind those was a Rust test
comparing the batch against the engine it is built from, which is worth having
and shares every mistake the engine makes. Here each env is compared against a
`Reference` started at that env's own offset, written from the decisions and
knowing nothing about either implementation.

The offsets are the point. An episode that starts part way into the series must
still fund on the bars the SERIES funds on (ADRs 0002, 0017, 0018), still hold
its own book, and still reach the end of the data when the data ends rather than
when its own step count runs out. All three are invisible when every env starts
at bar zero, which is what the batched tests do.

  python batch_differential.py [cases] [seed]
"""

import math
import random
import sys

import numpy as np

import emsl
from differential import close_enough, config, walk
from reference import Reference


def offsets_for(rng, bars, num_envs):
    # at least one bar to step into, and never all the same: two envs on the same
    # offset run the same episode, and then a batch that broadcast env 0 over the
    # rest of them would agree with the reference
    top = len(bars) - 2
    drawn = [rng.randint(0, top) for _ in range(num_envs)]
    if len(set(drawn)) == 1 and top > 0:
        drawn[0] = (drawn[0] + 1) % (top + 1)
    return drawn


def plan(rng, num_envs, steps):
    """A signed size per env per step: positive buys, negative sells, zero holds,
    which is exactly what `advance` reads."""
    return [
        [0.0 if rng.random() < 0.45 else rng.uniform(-40.0, 40.0)
         for _ in range(num_envs)]
        for _ in range(steps)
    ]


def drive_batch(bars, cfg, offsets, actions):
    array = np.array(bars, dtype=np.float64)
    batch = emsl.Batch(
        array, num_envs=len(offsets), market=cfg["market"], quote=cfg["quote"],
        fee_taker=cfg["fee_taker"], fee_maker=cfg["fee_maker"],
        slippage_bps=cfg["slippage_bps"], max_fill_fraction=cfg["max_fill_fraction"],
        leverage=cfg["leverage"], impact=cfg["impact"],
        funding_rate=cfg["funding_rate"], funding_interval=cfg["funding_interval"],
        report=False,
    )
    batch.reset_at_each(np.array(offsets, dtype=np.int64))
    seen = []
    for row in actions:
        batch.advance(np.array(row, dtype=np.float64))
        snap = batch.snapshot()
        seen.append({
            "position": snap["position"].copy(),
            "equity": snap["equity"].copy(),
            "mark_price": snap["mark_price"].copy(),
            "funding_paid": snap["funding_paid"].copy(),
            "bust": batch.busts().copy(),
            "done": batch.dones().copy(),
        })
    return seen, batch


def drive_references(bars, cfg, offsets, actions):
    refs = [Reference(bars, start=off, **cfg) for off in offsets]
    seen = []
    for row in actions:
        for ref, size in zip(refs, row):
            # `advance` skips a zero outright rather than placing an empty order
            if size > 0.0:
                ref.market_order("buy", size)
            elif size < 0.0:
                ref.market_order("sell", -size)
            ref.step()
        seen.append({
            "position": np.array([r.position for r in refs]),
            "equity": np.array([r.equity(r.bars[r.tick][3]) for r in refs]),
            "mark_price": np.array([r.bars[r.tick][3] for r in refs]),
            "funding_paid": np.array([r.funding_paid for r in refs]),
            "bust": np.array([r.bust for r in refs]),
            "done": np.array([r.tick + 1 >= len(bars) for r in refs]),
        })
    return seen, refs


def observation_disagreement(batch, bars, offsets, window):
    """The window each env observes, against the bars it should be looking at.

    Front-padded with zeros before the episode has `window` bars behind it, which
    is the one thing a warm-up rule can get wrong in a direction nothing notices:
    padding at the END would hand the agent the newest bar in the oldest row.
    """
    seen = batch.observe(window)
    for env, tick in enumerate(offsets):
        want = np.zeros((window, 5), dtype=np.float64)
        first = max(0, tick - window + 1)
        rows = np.array(bars[first:tick + 1], dtype=np.float64)
        want[window - rows.shape[0]:] = rows
        if not np.allclose(seen[env], want, rtol=0.0, atol=1e-9):
            return f"env {env} at bar {tick}: observation is not the window ending there"
    return None


def compare(mine, theirs, cfg, num_envs):
    for step, (a, b) in enumerate(zip(mine, theirs)):
        for env in range(num_envs):
            for key, scale in (("position", 1.0), ("equity", cfg["quote"]),
                               ("mark_price", 1.0), ("funding_paid", cfg["quote"])):
                if not close_enough(float(a[key][env]), float(b[key][env]), scale):
                    return (f"step {step + 1} env {env}: {key} "
                            f"{float(a[key][env]):.6f} against {float(b[key][env]):.6f}")
            for key in ("bust", "done"):
                if bool(a[key][env]) != bool(b[key][env]):
                    return (f"step {step + 1} env {env}: {key} "
                            f"{bool(a[key][env])} against {bool(b[key][env])}")
    return None


def main():
    cases = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260814
    rng = random.Random(seed)

    # what the cases actually reached. "0 disagreed" over a hundred flat accounts
    # that never funded and never died is a pass that means nothing, and there is
    # no way to tell one from the other after the fact
    reached = dict(traded=0, funded=0, busted=0, finished=0)
    agree, disagree = 0, []
    for case in range(cases):
        bars = walk(rng, rng.randint(12, 40))
        cfg = config(rng)
        num_envs = rng.randint(2, 5)
        offsets = offsets_for(rng, bars, num_envs)
        window = rng.randint(1, 6)
        # past the end for the earliest env, so a finished episode is asked to
        # keep going and has to sit still rather than wrap or run on
        steps = len(bars) - min(offsets)
        actions = plan(rng, num_envs, steps)
        try:
            mine, batch = drive_batch(bars, cfg, offsets, actions)
            theirs, refs = drive_references(bars, cfg, offsets, actions)
            fresh = emsl.Batch(
                np.array(bars, dtype=np.float64), num_envs=num_envs,
                market=cfg["market"], quote=cfg["quote"], report=False,
            )
            fresh.reset_at_each(np.array(offsets, dtype=np.int64))
            where = observation_disagreement(fresh, bars, offsets, window)
            if where is None:
                where = compare(mine, theirs, cfg, num_envs)
            last = mine[-1]
            reached["traded"] += any(abs(p) > 0.0 for p in last["position"]) or any(
                abs(p) > 0.0 for step in mine for p in step["position"]
            )
            reached["funded"] += any(f != 0.0 for f in last["funding_paid"])
            reached["busted"] += bool(last["bust"].any())
            reached["finished"] += bool(last["done"].any())
        except Exception as exc:
            where = f"raised: {type(exc).__name__}: {exc}"

        if where is None:
            agree += 1
        else:
            disagree.append((case, cfg, offsets, bars, where))

    print(f"\n{agree} of {cases} agreed, {len(disagree)} disagreed")
    print(f"cases that traded {reached['traded']}, funded {reached['funded']}, "
          f"busted {reached['busted']}, ran out of bars {reached['finished']}\n")
    for name, count in reached.items():
        if count == 0:
            print(f"NOTHING {name}: this run proves less than it looks like it does")
            return 1
    for case, cfg, offsets, bars, why in disagree[:8]:
        print(f"--- case {case} ({cfg['market']}) {why}")
        print(f"    cfg     {cfg}")
        print(f"    offsets {offsets}")
        print(f"    bars    {[tuple(round(x, 4) for x in b) for b in bars[:6]]}\n")
    if len(disagree) > 8:
        print(f"... and {len(disagree) - 8} more")
    return 1 if disagree else 0


if __name__ == "__main__":
    sys.exit(main())
