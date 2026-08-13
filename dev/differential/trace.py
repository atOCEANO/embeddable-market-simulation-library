"""Shrink each disagreement to the smallest case that still shows it, and trace it.

The differential run says WHERE the two simulators part company. This says as
little as possible around that point, so the question can be argued from the
decisions rather than from a forty bar random walk.

  python trace.py <seed> [cases]
"""

import json
import random
import sys

import differential as D
from reference import Reference


def diverges(bars, cfg, actions):
    """The first bar the two disagree on, or None."""
    try:
        mine, _ = D.drive_engine(bars, cfg, actions)
        theirs, _ = D.drive_reference(bars, cfg, actions)
    except Exception:
        return None
    for i, ((pa, ea), (pb, eb)) in enumerate(zip(mine, theirs)):
        if not D.close_enough(pa, pb) or not D.close_enough(ea, eb, cfg["quote"]):
            return i
    return None


def shrink(bars, cfg, actions):
    """Drop bars off the tail and actions from anywhere, while it still diverges."""
    at = diverges(bars, cfg, actions)
    if at is None:
        return bars, actions, None

    # tail first: nothing after the divergence can be causing it
    keep = at + 2
    while keep < len(bars):
        if diverges(bars[:keep], cfg, actions[:keep - 1]) is not None:
            bars, actions = bars[:keep], actions[:keep - 1]
            break
        keep += 1

    # then blank one action at a time, keeping any blanking that preserves it
    changed = True
    while changed:
        changed = False
        for i in range(len(actions)):
            if actions[i] is None:
                continue
            trial = list(actions)
            trial[i] = None
            if diverges(bars, cfg, trial) is not None:
                actions = trial
                changed = True
    return bars, actions, diverges(bars, cfg, actions)


def trace(bars, cfg, actions):
    """Bar by bar, both sides, so the exact step that parts them is visible."""
    array = D.np.array([list(b) for b in bars], dtype=D.np.float64)
    engine = D.emsl.Engine(
        array, market=cfg["market"], quote=cfg["quote"], fee_taker=cfg["fee_taker"],
        fee_maker=cfg["fee_maker"], slippage_bps=cfg["slippage_bps"],
        max_fill_fraction=cfg["max_fill_fraction"], leverage=cfg["leverage"],
        impact=cfg["impact"], funding_rate=cfg["funding_rate"],
        funding_interval=cfg["funding_interval"], report=True,
    )
    engine.reset()
    ref = Reference(bars, **cfg)

    rows = []
    for i, action in enumerate(actions):
        D.place(engine, action)
        if action is not None:
            if action[0] == "market":
                ref.market_order(action[1], action[2])
            elif action[0] == "limit":
                ref.limit_order(action[1], action[2], action[3])
            else:
                ref.stop_order(action[1], action[2], action[3], True)
        state = engine.step()
        ref.step()
        rows.append(dict(
            bar=ref.tick,
            ohlcv=[round(x, 4) for x in bars[ref.tick]],
            placed=action,
            engine=dict(position=round(state["position"], 6),
                        equity=round(state["equity"], 6),
                        entry=round(state["avg_entry"], 6),
                        quote=round(state["quote"], 6)),
            reference=dict(position=round(ref.position, 6),
                           equity=round(ref.equity(bars[ref.tick][3]), 6),
                           entry=round(ref.entry, 6),
                           quote=round(ref.quote, 6)),
        ))
        if engine.done():
            break
    return rows, engine, ref


def main():
    seed = int(sys.argv[1])
    cases = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    rng = random.Random(seed)

    for case in range(cases):
        bars = D.walk(rng, rng.randint(8, 40))
        cfg = D.config(rng)
        actions = D.plan(rng, bars)
        if diverges(bars, cfg, actions) is None:
            continue

        bars, actions, at = shrink(bars, cfg, actions)
        rows, engine, ref = trace(bars, cfg, actions)
        out = dict(
            seed=seed, case=case, config=cfg, diverges_at_row=at,
            bars=[list(b) for b in bars], actions=actions, rows=rows,
            engine_trades=[{k: (round(v, 6) if isinstance(v, float) else v)
                            for k, v in t.items()} for t in (engine.trades() or [])],
            reference_trades=[{k: (round(v, 6) if isinstance(v, float) else v)
                               for k, v in t.items()} for t in ref.trades],
        )
        path = f"/diff/case-seed{seed}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=1)
        print(f"seed {seed} case {case}: shrank to {len(bars)} bars, "
              f"{sum(1 for a in actions if a)} orders, diverges at row {at} -> {path}")
        return 0

    print(f"seed {seed}: no disagreement in {cases} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
