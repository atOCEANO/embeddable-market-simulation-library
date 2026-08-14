"""Run the engine and the reference over the same random cases and diff them.

A disagreement is not automatically an engine bug: the reference is a second
reading of the same decisions and can be the wrong one. What it buys is that
somebody has to decide which, and that question is never asked by a test written
against the implementation.

  python differential.py [cases] [seed]
"""

import math
import os
import random
import sys

import numpy as np

import emsl
from reference import Reference

TOL = 1e-6


def walk(rng, n):
    """Consistent OHLC from a random walk, violent enough to liquidate."""
    bars, price = [], 100.0
    for _ in range(n):
        ret = rng.uniform(-0.25, 0.25)
        close = max(price * (1.0 + ret), 0.01)
        high = max(price, close) * (1.0 + rng.uniform(0.0, 0.2))
        low = max(min(price, close) * (1.0 - rng.uniform(0.0, 0.2)), 0.001)
        volume = rng.choice([0.0, rng.uniform(0.001, 5.0), rng.uniform(50.0, 5_000.0)])
        bars.append((price, high, low, close, volume))
        price = close
    return bars


def config(rng):
    market = rng.choice(["spot", "perp"])
    return dict(
        market=market,
        quote=rng.choice([1_000.0, 10_000.0]),
        fee_taker=rng.choice([0.0, 0.0006, 0.002]),
        fee_maker=rng.choice([0.0, 0.0002]),
        slippage_bps=rng.choice([0.0, 5.0, 50.0]),
        max_fill_fraction=rng.choice([0.25, 1.0]),
        leverage=rng.choice([0.0, 3.0, 10.0]) if market == "perp" else 10.0,
        impact=rng.choice([0.0, 0.01, 0.5]),
        funding_rate=rng.choice([0.0, 0.0001, 0.01]) if market == "perp" else 0.0,
        funding_interval=rng.choice([0, 3]) if market == "perp" else 0,
    )


# every order surface the reference models, so the gate exercises all of them.
# Narrow it with KINDS=limit when a disagreement needs isolating, which is how
# ADR 0079 was found: whole-surface runs said "limits", and limits alone said
# which limit
KINDS = os.environ.get(
    "KINDS",
    "market,limit,stop,fok,post,cancel,replace,reduce,close,ioc_limit,fok_limit,"
    "cancel_all,reduce_limit,stop_entry",
).split(",")


def plan(rng, bars):
    """One action per bar, drawn from whichever order kinds are enabled."""
    out = []
    for _ in range(len(bars) - 1):
        if rng.random() < 0.45:
            out.append(None)
            continue
        kind = rng.choice(KINDS)
        side = rng.choice(["buy", "sell"])
        if kind == "market":
            out.append(("market", side, rng.uniform(0.1, 40.0)))
        elif kind == "fok":
            out.append(("fok", side, rng.uniform(0.1, 40.0)))
        elif kind == "post":
            out.append(("post", side, rng.uniform(0.1, 20.0), rng.uniform(50.0, 200.0)))
        elif kind == "cancel":
            out.append(("cancel", side, 0.0))
        elif kind == "cancel_all":
            out.append(("cancel_all", side, 0.0))
        elif kind == "close":
            out.append(("close", side, 0.0))
        elif kind == "reduce":
            # deliberately either side: a reduce_only order pointed the wrong way
            # is clamped to nothing rather than opening the other side (ADR 0028)
            out.append(("reduce", side, rng.uniform(0.1, 40.0)))
        elif kind in ("reduce_limit", "stop_entry"):
            out.append((kind, side, rng.uniform(0.1, 20.0), rng.uniform(50.0, 200.0)))
        else:
            # `replace` reads the same shape, and its `side` is deliberately
            # ignored by both simulators: a replacement keeps the side of the
            # order it moves (ADR 0032), which is the claim this exercises
            out.append((kind, side, rng.uniform(0.1, 20.0), rng.uniform(50.0, 200.0)))
    return out


def drive_engine(bars, cfg, actions):
    array = np.array([[b[0], b[1], b[2], b[3], b[4]] for b in bars], dtype=np.float64)
    engine = emsl.Engine(
        array, market=cfg["market"], quote=cfg["quote"],
        fee_taker=cfg["fee_taker"], fee_maker=cfg["fee_maker"],
        slippage_bps=cfg["slippage_bps"], max_fill_fraction=cfg["max_fill_fraction"],
        leverage=cfg["leverage"], impact=cfg["impact"],
        funding_rate=cfg["funding_rate"], funding_interval=cfg["funding_interval"],
        report=True,
    )
    engine.reset()
    seen, live = [], []
    for action in actions:
        place(engine, action, live)
        state = engine.step()
        seen.append((state["position"], state["equity"], state["funding_paid"]))
        if engine.done():
            break
    return seen, engine


def place(engine, action, live):
    """`live` collects the ids both sides believe are resting, so cancel and
    replace address the same order in each."""
    if action is None:
        return
    kind = action[0]
    if kind == "market":
        got = (engine.market_buy if action[1] == "buy" else engine.market_sell)(action[2])
    elif kind == "limit":
        got = (engine.limit_buy if action[1] == "buy" else engine.limit_sell)(action[2], action[3])
    elif kind == "fok":
        got = engine.order(action[1], action[2], "market", None, None, False, False, "fok")
    elif kind == "post":
        got = engine.order(action[1], action[2], "limit", action[3], None, False, True, "gtc")
    elif kind == "cancel":
        got = None
        if live:
            engine.cancel(live[0])
    elif kind == "replace":
        got = None
        if live:
            # a refused replacement puts the original back under its own id
            # (ADR 0069), so the slot is only re-pointed when one was placed
            moved = engine.replace(live[0], action[2], action[3])
            if moved is not None:
                live[0] = moved
    elif kind == "close":
        got = engine.close()
    elif kind == "reduce":
        got = engine.order(action[1], action[2], "market", None, None, True, False, "ioc")
    elif kind == "cancel_all":
        got = None
        engine.cancel_all()
        live.clear()
    elif kind in ("ioc_limit", "fok_limit"):
        tif = "ioc" if kind == "ioc_limit" else "fok"
        got = engine.order(action[1], action[2], "limit", action[3], None, False, False, tif)
    elif kind == "reduce_limit":
        got = engine.order(action[1], action[2], "limit", action[3], None, True, False, "gtc")
    elif kind == "stop_entry":
        # a breakout stop, which OPENS rather than protects, so reduce_only is off
        got = engine.stop(action[1], action[2], action[3], False)
    else:
        got = engine.stop(action[1], action[2], action[3], True)
    if kind in ("limit", "post", "stop", "ioc_limit", "fok_limit",
                "reduce_limit", "stop_entry") and got is not None:
        live.append(got)
    if kind == "cancel" and live:
        live.pop(0)


def place_reference(ref, action, live, reached=None):
    """The reference's half of `place`, kept beside it so the two dispatches stay
    in step. It used to be inlined in the driver and copied again in trace.py,
    which is how trace.py came to handle three of the seven order kinds."""
    if action is None:
        return
    kind = action[0]
    got = None
    if reached is not None:
        # what the action had to act on when it was placed. An action that never
        # meets a live position agrees with anything, and after the fact there is
        # no way to tell that from a real agreement
        if kind in ("close", "reduce") and ref.position != 0.0:
            reached[kind] += 1
        elif kind in ("cancel", "replace", "cancel_all") and live:
            reached[kind] += 1
    if kind == "market":
        got = ref.market_order(action[1], action[2])
    elif kind == "limit":
        got = ref.limit_order(action[1], action[2], action[3])
    elif kind == "fok":
        got = ref.market_order(action[1], action[2], fok=True)
    elif kind == "post":
        got = ref.limit_order(action[1], action[2], action[3], post_only=True)
    elif kind == "cancel":
        if live:
            ref.cancel(live.pop(0))
    elif kind == "replace":
        if live:
            moved = ref.replace(live[0], action[2], action[3])
            if moved is not None:
                live[0] = moved
    elif kind == "close":
        # what Engine::close does: a reduce_only market order for the whole
        # position, sized at PLACEMENT time, and nothing at all when flat
        if ref.position != 0.0:
            got = ref.market_order(
                "sell" if ref.position > 0.0 else "buy", abs(ref.position),
                reduce_only=True,
            )
    elif kind == "reduce":
        got = ref.market_order(action[1], action[2], reduce_only=True)
    elif kind == "cancel_all":
        ref.cancel_all()
        live.clear()
    elif kind in ("ioc_limit", "fok_limit"):
        got = ref.limit_order(action[1], action[2], action[3],
                              tif="ioc" if kind == "ioc_limit" else "fok")
    elif kind == "reduce_limit":
        got = ref.limit_order(action[1], action[2], action[3], reduce_only=True)
    elif kind == "stop_entry":
        got = ref.stop_order(action[1], action[2], action[3], False)
    elif kind == "stop":
        got = ref.stop_order(action[1], action[2], action[3], True)
    if kind in ("limit", "post", "stop", "ioc_limit", "fok_limit",
                "reduce_limit", "stop_entry") and got is not None:
        live.append(got)
    if reached is not None and got is not None and kind in (
        "ioc_limit", "fok_limit", "reduce_limit", "stop_entry"
    ):
        reached[kind] += 1


def drive_reference(bars, cfg, actions, reached=None):
    ref = Reference(bars, **cfg)
    seen, live = [], []
    for action in actions:
        place_reference(ref, action, live, reached)
        ref.step()
        seen.append((ref.position, ref.equity(ref.bars[ref.tick][3]), ref.funding_paid))
        if ref.tick + 1 >= len(bars):
            break
    return seen, ref


def close_enough(a, b, scale=1.0):
    if not math.isfinite(a) or not math.isfinite(b):
        return math.isfinite(a) == math.isfinite(b)
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b), scale)


def main():
    cases = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260813
    rng = random.Random(seed)

    reached = dict(close=0, reduce=0, cancel=0, replace=0, cancel_all=0,
                   ioc_limit=0, fok_limit=0, reduce_limit=0, stop_entry=0)
    agree, disagree = 0, []
    for case in range(cases):
        bars = walk(rng, rng.randint(8, 40))
        cfg = config(rng)
        actions = plan(rng, bars)
        try:
            mine, engine = drive_engine(bars, cfg, actions)
            theirs, ref = drive_reference(bars, cfg, actions, reached)
        except Exception as exc:
            disagree.append((case, seed, cfg, bars, actions, f"raised: {type(exc).__name__}: {exc}"))
            continue

        where = None
        # funding_paid is read as well as position and equity, because a
        # liquidation zeroes the equity on both sides and hides any disagreement
        # about what the account paid on its way there. That blind spot is exactly
        # where ADR 0082 was hiding: 3,000 cases over six seeds never saw it
        for i, ((pa, ea, fa), (pb, eb, fb)) in enumerate(zip(mine, theirs)):
            if not close_enough(pa, pb) or not close_enough(ea, eb, cfg["quote"]):
                where = (f"bar {i + 1}: position {pa:.6f} against {pb:.6f}, "
                         f"equity {ea:.6f} against {eb:.6f}")
                break
            if not close_enough(fa, fb, cfg["quote"]):
                where = f"bar {i + 1}: funding_paid {fa:.6f} against {fb:.6f}"
                break
        if where is None:
            agree += 1
        else:
            disagree.append((case, seed, cfg, bars, actions, where))

    print(f"\n{agree} of {cases} agreed, {len(disagree)} disagreed")
    print("actions placed with something to act on: "
          + ", ".join(f"{k} {v}" for k, v in sorted(reached.items())) + "\n")
    for kind, count in sorted(reached.items()):
        if kind in KINDS and count == 0:
            print(f"NO {kind} EVER MET A LIVE POSITION OR ORDER: it agrees for free")
            return 1
    for case, seed, cfg, bars, actions, why in disagree[:8]:
        print(f"--- case {case} ({cfg['market']}) {why}")
        print(f"    cfg  {cfg}")
        print(f"    bars {[tuple(round(x, 4) for x in b) for b in bars[:6]]}")
        acts = [a for a in actions[:6]]
        print(f"    acts {acts}\n")
    if len(disagree) > 8:
        print(f"... and {len(disagree) - 8} more")
    return 1 if disagree else 0


if __name__ == "__main__":
    sys.exit(main())
