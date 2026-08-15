"""Cross-version smoke tests for the emsl Engine over the single-env step.

These run against the installed abi3 wheel on each Python in the Docker gate, so
they double as the "works on most Python versions" check and the behavioral check
for the Python boundary: numpy candles in, a state dict out, and the cross-bar
no-lookahead rule holding through the binding.
"""

import re

import numpy as np
import pytest

import emsl


def test_version_is_single_sourced_from_the_build():
    # resolved from the wheel metadata (the Cargo.toml workspace version), not a
    # hardcoded string, so it can never drift from the built artifact
    assert emsl.__version__ != "0.0.0"
    assert re.match(r"^\d+\.\d+\.\d+", emsl.__version__)


def series():
    # open, high, low, close, volume
    return np.array(
        [
            [100.0, 160.0, 90.0, 150.0, 1000.0],
            [200.0, 260.0, 190.0, 250.0, 1000.0],
            [300.0, 360.0, 290.0, 350.0, 1000.0],
        ],
        dtype=np.float64,
    )


def engine(**overrides):
    params = dict(market="spot", quote=10_000.0, fee_taker=0.0, fee_maker=0.0)
    params.update(overrides)
    return emsl.Engine(series(), **params)


def test_reset_starts_flat_at_the_first_bar():
    e = engine()
    s = e.reset()
    assert s["tick_index"] == 0
    assert s["position"] == 0.0
    assert s["quote"] == 10_000.0
    assert s["bar_open"] == 100.0
    assert s["open_orders"] == []


def test_market_order_fills_on_the_next_open_not_the_decision_bar():
    e = engine()
    e.reset()
    oid = e.market_buy(1.0)
    assert isinstance(oid, int)
    s = e.step()
    assert s["tick_index"] == 1
    assert s["position"] == 1.0
    assert s["avg_entry"] == 200.0  # next bar open, not 150 or 250


def test_close_realizes_spot_pnl():
    e = engine()
    e.reset()
    e.market_buy(2.0)
    s1 = e.step()  # long 2 @ 200
    assert s1["position"] == 2.0
    e.close()
    s2 = e.step()  # closes at bar 2 open 300
    assert s2["position"] == 0.0
    assert s2["equity"] == 10_200.0  # bought 2 @ 200, sold 2 @ 300, +200


def test_done_only_at_the_last_bar():
    e = engine()
    e.reset()
    assert not e.done()
    e.step()
    assert not e.done()
    e.step()
    assert e.done()


def test_limit_rests_in_open_orders_then_cancels():
    e = engine()
    e.reset()
    oid = e.limit_buy(1.0, 50.0)  # 50 is never reached (lows are 90/190/290)
    s1 = e.step()
    assert s1["position"] == 0.0
    assert len(s1["open_orders"]) == 1
    order = s1["open_orders"][0]
    assert order["id"] == oid
    assert order["kind"] == "limit"
    assert order["side"] == "buy"
    assert order["price"] == 50.0
    assert e.cancel(oid)
    s2 = e.step()
    assert s2["open_orders"] == []


def test_stats_none_without_reporting_and_dict_with():
    off = engine()
    off.reset()
    off.step()
    assert off.stats() is None

    on = engine(report=True)
    on.reset()
    on.step()
    st = on.stats(periods_per_year=365.0)
    assert isinstance(st, dict)
    assert "sharpe" in st
    assert "max_drawdown_pct" in st


def test_observation_is_a_zero_copy_readonly_view():
    e = engine()
    e.reset()  # tick 0
    obs = e.observation(1)
    assert obs.shape == (1, 5)
    assert obs.dtype == np.float64
    assert obs[0].tolist() == [100.0, 160.0, 90.0, 150.0, 1000.0]  # bar 0
    # a borrowed view, not an owning copy, and read-only
    assert obs.flags.owndata is False
    assert obs.flags.writeable is False
    with pytest.raises(ValueError):
        obs[0, 0] = 1.0


def test_observation_grows_with_tick_and_clamps_at_warmup():
    e = engine()
    e.reset()  # tick 0
    assert e.observation(5).shape == (1, 5)  # only one bar exists yet, clamped
    e.step()  # tick 1
    obs = e.observation(2)
    assert obs.shape == (2, 5)
    assert obs[0].tolist() == [100.0, 160.0, 90.0, 150.0, 1000.0]  # bar 0
    assert obs[1].tolist() == [200.0, 260.0, 190.0, 250.0, 1000.0]  # bar 1


def test_observation_outlives_the_dropped_engine():
    import gc

    e = engine()
    e.reset()
    e.step()  # tick 1
    obs = e.observation(2)
    expected = [
        [100.0, 160.0, 90.0, 150.0, 1000.0],
        [200.0, 260.0, 190.0, 250.0, 1000.0],
    ]
    del e  # the view must keep the engine (and its Arc buffer) alive
    gc.collect()
    assert obs.tolist() == expected


def test_batch_matches_single_engines_and_tracks_done():
    n = 4
    b = emsl.Batch(series(), num_envs=n, market="spot", fee_taker=0.0, fee_maker=0.0)
    assert len(b) == n
    assert b.num_envs == n
    b.reset_all()
    assert not b.done()
    # each env buys a different size, applied as a batched action array
    actions = np.array([0.1 * (i + 1) for i in range(n)], dtype=np.float64)
    states = b.step_all(actions)
    assert len(states) == n
    for i, s in enumerate(states):
        # fills at bar 1 open (200), so position is exactly the action size
        assert s["tick_index"] == 1
        assert abs(s["position"] - 0.1 * (i + 1)) < 1e-12
        assert s["avg_entry"] == 200.0
    b.step_all()  # step with no new orders
    assert b.done()


def test_batch_step_without_actions_holds_position():
    b = emsl.Batch(series(), num_envs=2, fee_taker=0.0, fee_maker=0.0)
    b.reset_all()
    b.step_all(np.array([1.0, 2.0], dtype=np.float64))
    states = b.step_all()  # no actions: positions carry, no new fills
    assert states[0]["position"] == 1.0
    assert states[1]["position"] == 2.0


def test_batch_rejects_mismatched_action_length():
    b = emsl.Batch(series(), num_envs=3)
    b.reset_all()
    with pytest.raises(ValueError):
        b.step_all(np.array([1.0, 2.0], dtype=np.float64))  # length 2 != 3 envs


def test_batch_set_features_rejects_a_short_matrix():
    # features must carry one row per candle; a shorter matrix would index out of
    # bounds in the gather, so set_features rejects it rather than panicking across
    # the FFI
    b = emsl.Batch(series(), num_envs=2)  # series() has 3 bars
    b.reset_all()
    with pytest.raises(ValueError):
        b.set_features(np.zeros((2, 4), dtype=np.float64))  # 2 rows != 3 bars


class BuyAndHold:
    """Buys once on the first bar, then holds, through the engine's own API."""

    def __init__(self):
        self.entered = False

    def next(self, state, engine):
        if not self.entered:
            engine.market_buy(1.0)
            self.entered = True


class RecordsInit:
    def __init__(self):
        self.inited = False

    def init(self, engine):
        self.inited = True

    def next(self, state, engine):
        pass


def test_run_drives_a_python_strategy_that_places_orders():
    e = engine()
    final = e.run(BuyAndHold())
    assert final["tick_index"] == 2  # drove to the last bar
    assert final["position"] == 1.0  # bought 1 at bar 1 open, held
    # spot: quote 10000 - 200 spent, base 1 marked at the last close 350
    assert final["equity"] == 10_150.0


def test_run_calls_init_when_present():
    strat = RecordsInit()
    engine().run(strat)
    assert strat.inited is True


class BuyThenClose:
    """Buys on the first bar, then closes once it holds a position."""

    def next(self, state, engine):
        if state["tick_index"] == 0:
            engine.market_buy(1.0)
        elif state["position"] != 0.0:
            engine.close()


def test_run_with_reporting_yields_stats():
    e = engine(report=True)
    e.run(BuyAndHold())
    st = e.stats()
    assert isinstance(st, dict)
    assert "total_return_pct" in st
    assert st["total_return_pct"] > 0.0  # bought and held into a rising series


def test_closed_trades_flow_into_stats():
    e = engine(report=True)
    e.run(BuyThenClose())
    st = e.stats()
    assert st["num_trades"] == 1  # bought 1 @ 200, closed @ 300
    assert st["win_rate"] == 1.0
    assert st["profit_factor"] == float("inf")  # a win and no losses


def test_market_impact_worsens_a_large_fill():
    # buy 100 into a 1000-volume bar with impact 0.5: slip = 0.5 * 0.1 = 5%
    e = emsl.Engine(
        series(), market="spot", quote=1_000_000.0, fee_taker=0.0, fee_maker=0.0, impact=0.5
    )
    e.reset()
    e.market_buy(100.0)
    s = e.step()
    assert abs(s["avg_entry"] - 210.0) < 1e-9  # bar 1 open 200 * 1.05


def test_perp_funding_charges_a_held_position():
    # funding every bar at 0.001; a held long pays it, seen as a quote debit
    e = emsl.Engine(
        series(),
        market="perp",
        quote=10_000.0,
        fee_taker=0.0,
        fee_maker=0.0,
        funding_rate=0.001,
        funding_interval=1,
    )
    e.reset()
    e.market_buy(1.0)
    s = e.step()  # long 1 @ 200; funding at bar 1 close 250: pay 1 * 250 * 0.001
    assert abs(s["quote"] - (10_000.0 - 0.25)) < 1e-9


def test_perp_leverage_caps_the_position():
    e = emsl.Engine(
        series(), market="perp", quote=100.0, fee_taker=0.0, fee_maker=0.0, leverage=2.0
    )
    e.reset()
    e.market_buy(10.0)  # wants 10, but 2x on 100 equity at price 200 caps it at 1
    s = e.step()
    assert s["position"] == 1.0


def test_perp_leverage_defaults_to_a_finite_cap():
    # no leverage kwarg: the shipped 10x default bounds an oversized perp buy
    e = emsl.Engine(series(), market="perp", quote=100.0, fee_taker=0.0, fee_maker=0.0)
    e.reset()
    e.market_buy(10.0)  # 10x on 100 equity at price 200 caps the notional at 5 units
    s = e.step()
    assert s["position"] == 5.0


def test_per_env_cost_randomization_varies_costs_across_envs():
    # a per-env override array gives env1 a 1% taker fee and env0 none; both buy
    # 1 at bar 1 open 200, so only their quote after the fee differs
    b = emsl.Batch(
        series(),
        num_envs=2,
        market="spot",
        fee_taker=0.0,
        fee_taker_per_env=np.array([0.0, 0.01], dtype=np.float64),
    )
    b.reset_all()
    states = b.step_all(np.array([1.0, 1.0], dtype=np.float64))
    assert states[0]["quote"] == 9_800.0  # 10000 - 200
    assert states[1]["quote"] == 9_798.0  # 10000 - 200 - 0.01 * 200


def test_per_env_cost_override_length_must_match_num_envs():
    with pytest.raises(ValueError):
        emsl.Batch(
            series(),
            num_envs=3,
            fee_taker_per_env=np.array([0.0, 0.01], dtype=np.float64),  # length 2 != 3
        )


def test_order_reduce_only_market_only_shrinks():
    e = engine()
    e.reset()
    e.market_buy(1.0)
    e.step()  # long 1 @ 200
    e.order("buy", 1.0, type="market", reduce_only=True)  # cannot grow the long
    s = e.step()
    assert s["position"] == 1.0


def test_order_reduce_only_stop_loss_closes_a_long():
    # rise in, then a fall so a protective sell stop triggers and closes the long
    data = np.array(
        [
            [100.0, 110.0, 90.0, 100.0, 1000.0],
            [100.0, 110.0, 90.0, 100.0, 1000.0],
            [80.0, 85.0, 75.0, 80.0, 1000.0],
        ],
        dtype=np.float64,
    )
    e = emsl.Engine(data, market="spot", quote=10_000.0, fee_taker=0.0, fee_maker=0.0)
    e.reset()
    e.market_buy(1.0)
    e.step()  # long 1 @ bar 1 open 100
    e.order("sell", 1.0, type="stop", trigger=85.0, reduce_only=True)
    s = e.step()  # bar 2 low 75 crosses 85; the reduce-only stop closes the long
    assert s["position"] == 0.0


def test_order_ioc_limit_does_not_rest():
    e = engine()
    e.reset()
    e.order("buy", 1.0, type="limit", price=1.0, tif="IOC")  # 1.0 never reached
    s = e.step()
    assert s["open_orders"] == []  # IOC expired instead of resting


def test_order_validates_required_fields_and_tif():
    e = engine()
    e.reset()
    with pytest.raises(ValueError):
        e.order("buy", 1.0, type="limit")  # a limit needs a price
    with pytest.raises(ValueError):
        e.order("sell", 1.0, type="stop")  # a stop needs a trigger
    with pytest.raises(ValueError):
        e.order("buy", 1.0, tif="AON")  # unknown time-in-force


def test_cancel_all_drops_resting_orders():
    e = engine()
    e.reset()
    e.limit_buy(1.0, 50.0)
    e.limit_buy(1.0, 40.0)
    assert e.cancel_all() == 2
    s = e.step()
    assert s["open_orders"] == []


def test_qty_helpers_size_from_weight_and_quote():
    e = engine()
    e.reset()  # bar 0 close 150, equity 10000, flat
    assert abs(e.qty_from_weight(0.5) - 5_000.0 / 150.0) < 1e-9
    assert abs(e.qty_from_quote(300.0) - 2.0) < 1e-9


def test_stats_include_exposure_and_net_profit():
    e = engine(report=True)
    e.run(BuyAndHold())
    st = e.stats()
    assert "net_profit_pct" in st and "exposure_pct" in st
    assert abs(st["net_profit_pct"] - st["total_return_pct"]) < 1e-9
    assert 0.0 <= st["exposure_pct"] <= 100.0
    assert st["exposure_pct"] > 0.0  # BuyAndHold holds through the run


def test_open_order_dict_has_remaining_and_status():
    e = engine()
    e.reset()
    e.limit_buy(2.0, 50.0)  # rests, never reached
    s = e.step()
    order = s["open_orders"][0]
    assert order["remaining"] == 2.0  # nothing filled yet
    assert order["status"] == "resting"


def test_engine_shape_reports_the_candle_count():
    e = engine()
    assert e.shape == (3, 5)  # series() has 3 bars


def test_data_is_a_full_series_readonly_view():
    e = engine()
    d = e.data
    assert d.shape == (3, 5)  # the whole series, not a window
    assert d.flags.writeable is False
    assert d[:, 3].tolist() == [150.0, 250.0, 350.0]  # column 3 is close
    with pytest.raises(ValueError):
        d[0, 0] = 1.0


def test_bad_market_raises():
    with pytest.raises(ValueError):
        emsl.Engine(series(), market="options")


def test_candles_must_have_five_columns():
    with pytest.raises(ValueError):
        emsl.Engine(np.zeros((3, 4), dtype=np.float64))


def test_non_finite_candles_are_rejected():
    bad = series()
    bad[1, 2] = np.nan  # a NaN low on bar 1
    with pytest.raises(ValueError):
        emsl.Engine(bad)


def test_non_finite_config_scalars_are_rejected():
    # the candle array was always checked for non-finite values; every other knob
    # crossing the boundary poisons the same things and is now checked too (ADR 0027)
    for kwargs in (
        {"quote": float("nan")},
        {"fee_taker": float("nan")},
        {"fee_maker": float("inf")},
        {"slippage_bps": float("nan")},
        {"max_fill_fraction": float("nan")},
        {"leverage": float("nan")},
        {"impact": float("nan")},
        {"funding_rate": float("nan")},
    ):
        with pytest.raises(ValueError):
            emsl.Engine(series(), **kwargs)
        with pytest.raises(ValueError):
            emsl.Batch(series(), num_envs=2, **kwargs)


def test_out_of_range_per_env_costs_are_rejected_like_the_scalars():
    # the per-env arrays were checked for contiguity and length and nothing else,
    # so they were the one way past every guard below (ADR 0086). A fee of -2.0
    # reached an env: on a perp, four buys of five units took equity from 10,000
    # to 14,000, minted out of a rebate larger than the notional
    for name, bad in (
        ("fee_taker_per_env", -2.0),
        ("fee_maker_per_env", -2.0),
        ("slippage_bps_per_env", -5.0),
        ("impact_per_env", -5.0),
    ):
        with pytest.raises(ValueError):
            emsl.Batch(series(), num_envs=2,
                       **{name: np.array([0.0, bad], dtype=np.float64)})
        # and the good value beside it still builds, so the guard is a range check
        # rather than a refusal to accept the argument at all
        emsl.Batch(series(), num_envs=2,
                   **{name: np.array([0.0, 0.5], dtype=np.float64)})


def test_out_of_range_config_scalars_are_rejected():
    for kwargs in (
        {"quote": -1.0},
        {"slippage_bps": -1.0},  # slippage is adverse, never favourable
        {"impact": -0.5},
        {"leverage": -1.0},
        {"max_fill_fraction": 0.0},
        {"max_fill_fraction": -0.5},
        {"fee_maker": -1.0},  # a rebate cannot exceed the notional
        {"fee_taker": -2.0},
        {"max_open_orders": 0},
        {"max_open_orders": 10_000},
    ):
        with pytest.raises(ValueError):
            emsl.Engine(series(), **kwargs)


def test_a_zero_window_raises_instead_of_panicking():
    # the batched gathers chunk by the window and rayon asserts a nonzero chunk, so
    # this used to surface as a PanicException, which `except Exception` misses
    b = emsl.Batch(series(), num_envs=2)
    b.reset_all()
    with pytest.raises(ValueError):
        b.observe(0)
    b.set_features(np.zeros((3, 2), dtype=np.float64))
    with pytest.raises(ValueError):
        b.observe_features(0)


def test_non_finite_prices_and_triggers_are_rejected():
    e = engine()
    e.reset()
    with pytest.raises(ValueError):
        e.limit_buy(1.0, float("nan"))
    with pytest.raises(ValueError):
        e.limit_sell(1.0, float("inf"))
    with pytest.raises(ValueError):
        e.stop("sell", 1.0, float("nan"))
    with pytest.raises(ValueError):
        e.order("buy", 1.0, type="limit", price=float("nan"))


def test_stats_reject_a_non_positive_annualization():
    e = engine(report=True)
    e.reset()
    e.step()
    for ppy in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            e.stats(periods_per_year=ppy)


def test_stop_takes_reduce_only_so_it_cannot_open_the_other_side():
    # a fall that crosses the trigger while flat: a protective stop fills nothing,
    # a bare one opens a short on a perp (ADR 0028)
    data = np.array(
        [
            [100.0, 101.0, 99.0, 100.0, 1000.0],
            [100.0, 101.0, 80.0, 85.0, 1000.0],
            [85.0, 86.0, 84.0, 85.0, 1000.0],
        ],
        dtype=np.float64,
    )
    protective = emsl.Engine(data, market="perp", quote=10_000.0, fee_taker=0.0)
    protective.reset()
    protective.stop("sell", 1.0, 95.0, reduce_only=True)
    assert protective.step()["position"] == 0.0

    bare = emsl.Engine(data, market="perp", quote=10_000.0, fee_taker=0.0)
    bare.reset()
    bare.stop("sell", 1.0, 95.0)
    assert bare.step()["position"] == -1.0


def test_order_ids_do_not_restart_on_reset():
    e = engine()
    e.reset()
    first = e.limit_buy(1.0, 50.0)
    e.reset()
    second = e.limit_buy(1.0, 50.0)
    assert first != second
    assert e.cancel(first) is False  # the stale handle names nothing
    assert e.cancel(second) is True


def test_a_fok_market_blocked_by_the_cash_clamp_fills_nothing():
    # the bar has the volume, but the cash affords 50 of the 1000; FOK is all or
    # nothing against every clamp, not just liquidity (ADR 0025)
    e = engine()
    e.reset()
    e.order("buy", 1000.0, type="market", tif="FOK")
    s = e.step()
    assert s["position"] == 0.0
    assert s["quote"] == 10_000.0


def test_a_trade_carries_both_sides_of_its_fee():
    # charging only the closing fill left about half the round trip attributed to
    # nothing, so the metrics built on it stayed optimistic (ADR 0030)
    e = emsl.Engine(series(), market="spot", quote=10_000.0, fee_taker=0.01,
                    fee_maker=0.01, report=True)
    e.reset()
    e.market_buy(1.0)
    e.step()  # buy 1 at 200, entry fee 2.0
    e.close()
    e.step()  # sell 1 at 300, exit fee 3.0
    t = e.trades()[0]
    assert abs(t["pnl"] - 100.0) < 1e-9
    assert abs(t["fees"] - 5.0) < 1e-9
    assert abs(t["net_pnl"] - 95.0) < 1e-9


def test_stats_reproduce_from_the_trade_log():
    data = np.array(
        [[100.0 + (i % 7) - 3.0] * 4 + [10_000.0] for i in range(60)], dtype=np.float64
    )

    class PingPong:
        def __init__(self):
            self.hold = 0

        def next(self, state, engine):
            if state["position"] == 0.0:
                engine.market_buy(1.0)
                self.hold = 0
            else:
                self.hold += 1
                if self.hold >= 2:
                    engine.close()

    e = emsl.Engine(data, market="spot", quote=10_000.0, fee_taker=0.001,
                    fee_maker=0.001, report=True)
    e.run(PingPong())
    st, trades = e.stats(), e.trades()
    assert len(trades) > 3
    wins = sum(1 for t in trades if t["net_pnl"] > 0.0)
    assert abs(st["win_rate"] - wins / len(trades)) < 1e-12
    for t in trades:
        assert abs(t["net_pnl"] - (t["pnl"] - t["fees"])) < 1e-12


def test_num_fills_separates_a_dead_feed_from_a_quiet_strategy():
    dead = np.array([[100.0, 100.0, 100.0, 100.0, 0.0]] * 6, dtype=np.float64)
    e = emsl.Engine(dead, market="spot", report=True)
    e.reset()
    while not e.done():
        e.market_buy(1.0)
        e.step()
    assert e.num_fills() == 0
    assert e.stats()["num_fills"] == 0

    live = engine(report=True)
    live.reset()
    live.market_buy(1.0)
    live.step()
    assert live.num_fills() == 1


def test_replace_moves_one_order_and_will_not_arm_a_second():
    # the trailing-stop trap: stop() per bar rests a new order every time, and the
    # one that fills leaves its siblings live (ADR 0032)
    e = emsl.Engine(series(), market="perp", quote=10_000.0, fee_taker=0.0)
    e.reset()
    first = e.stop("sell", 1.0, 50.0, reduce_only=True)
    second = e.replace(first, trigger=60.0)
    s = e.step()
    assert len(s["open_orders"]) == 1
    assert s["open_orders"][0]["id"] == second
    assert s["open_orders"][0]["trigger"] == 60.0
    assert s["open_orders"][0]["reduce_only"] is True
    assert e.cancel(second)
    assert e.replace(second, trigger=70.0) is None  # gone, so nothing is placed
    assert e.step()["open_orders"] == []


def test_is_bust_reports_a_dead_account():
    e = emsl.Engine(series(), market="spot", quote=10_000.0)
    e.reset()
    assert e.is_bust() is False
    dead = emsl.Engine(series(), market="spot", quote=0.0)
    dead.reset()
    dead.step()
    assert dead.is_bust() is True


def test_trade_metrics_are_net_of_fees():
    # a creeping series whose per-trade gross edge is smaller than the round trip
    # cost: gross it wins every trade, net it loses money (ADR 0029)
    data = np.array(
        [[100.0 + i * 0.01] * 4 + [1000.0] for i in range(40)], dtype=np.float64
    )

    class PingPong:
        def next(self, state, engine):
            if state["position"] == 0.0:
                engine.market_buy(1.0)
            else:
                engine.close()

    e = emsl.Engine(data, market="spot", quote=10_000.0, fee_taker=0.001,
                    fee_maker=0.001, report=True)
    e.run(PingPong())
    st = e.stats()
    assert st["total_return_pct"] < 0.0
    assert st["win_rate"] == 0.0
    assert st["profit_factor"] == 0.0
    assert st["avg_trade_pct"] < 0.0


def test_spot_buy_cannot_spend_more_quote_than_held():
    # the Python view of the spot-buy cash clamp: a huge buy fills only what the
    # cash affords, and the balance never goes negative
    e = emsl.Engine(series(), market="spot", quote=10_000.0, fee_taker=0.0, fee_maker=0.0)
    e.reset()
    e.market_buy(1_000.0)  # wants 1000 base at bar 1 open 200, affords 50
    s = e.step()
    assert s["position"] == 50.0
    assert s["quote"] == 0.0


def costed_bars():
    # bar 1 is the one a decision on bar 0 fills against: its open is the fill
    # price, its volume is what an impact fraction is taken of, and its high sits
    # far enough above both that the taker clamp of ADR 0074 never binds and the
    # slip can be read straight off the price
    return np.array([
        [100.0, 200.0, 90.0, 100.0, 1000.0],
        [100.0, 200.0, 90.0, 100.0, 10.0],
        [100.0, 200.0, 90.0, 100.0, 1000.0],
    ], dtype=np.float64)


def test_a_per_env_slippage_lands_on_slippage_and_not_on_impact():
    # asserted by VALUE. The only thing standing behind these two arrays was
    # np.std(rewards) > 0.0, a property, and crossing the two assignments still
    # spreads the rewards: a slippage draw landing on impact moves the price by
    # size/volume times the number instead of by basis points, which is wrong by
    # orders of magnitude and reads identical to a property (ADR 0108)
    b = emsl.Batch(
        costed_bars(), num_envs=2, market="spot", fee_taker=0.0, fee_maker=0.0,
        slippage_bps_per_env=np.array([0.0, 50.0], dtype=np.float64),
        impact_per_env=np.array([0.0, 0.0], dtype=np.float64),
    )
    b.reset_all()
    states = b.step_all(np.array([1.0, 1.0], dtype=np.float64))
    assert states[0]["quote"] == 9_900.0            # one unit at the open of 100
    assert states[1]["quote"] == 9_899.5            # 50 bps of 100 is 0.5


def test_a_per_env_impact_lands_on_impact_and_not_on_slippage():
    # the mirror. One unit of a ten-volume bar is a tenth of it, so an impact of
    # 5.0 is half the price; the same 5.0 read as slippage_bps would be 5 bps
    b = emsl.Batch(
        costed_bars(), num_envs=2, market="spot", fee_taker=0.0, fee_maker=0.0,
        slippage_bps_per_env=np.array([0.0, 0.0], dtype=np.float64),
        impact_per_env=np.array([0.0, 5.0], dtype=np.float64),
    )
    b.reset_all()
    states = b.step_all(np.array([1.0, 1.0], dtype=np.float64))
    assert states[0]["quote"] == 9_900.0
    assert states[1]["quote"] == 9_850.0            # 5.0 * (1 / 10) is half of 100
