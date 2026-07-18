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


def test_spot_buy_cannot_spend_more_quote_than_held():
    # the Python view of the spot-buy cash clamp: a huge buy fills only what the
    # cash affords, and the balance never goes negative
    e = emsl.Engine(series(), market="spot", quote=10_000.0, fee_taker=0.0, fee_maker=0.0)
    e.reset()
    e.market_buy(1_000.0)  # wants 1000 base at bar 1 open 200, affords 50
    s = e.step()
    assert s["position"] == 50.0
    assert s["quote"] == 0.0
