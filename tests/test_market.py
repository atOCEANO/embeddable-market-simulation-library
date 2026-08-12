"""Tests for emsl.Market: it should carry the venue once and hand out every
surface configured identically, and refuse a knob passed twice.
"""

import numpy as np
import pytest

import emsl
from emsl import Market
from emsl.backtest import Strategy


def series(n=60):
    close = 100.0 + np.arange(n, dtype=np.float64) * 0.5
    return np.column_stack(
        [close, close + 1.0, close - 1.0, close, np.full(n, 1000.0)]
    )


class Alternate(Strategy):
    def next(self, state, engine):
        i = state["tick_index"]
        if i % 10 == 0:
            engine.market_buy(1.0)
        elif i % 10 == 5:
            engine.close()


def test_a_market_is_exported_and_defaults_to_the_engines_own():
    assert emsl.Market is Market
    assert "Market" in emsl.__all__
    settings = Market().as_dict()
    assert settings["market"] == "spot"
    assert settings["quote"] == 10_000.0
    assert settings["fee_taker"] == 0.0006
    assert settings["funding_interval"] == 0


def test_every_surface_gets_the_same_venue():
    # the whole point: one object, three surfaces, no chance for them to differ
    venue = Market(kind="perp", fee_taker=0.0004, slippage_bps=2.0, leverage=5.0)
    data = series()

    engine = venue.engine(data)
    assert engine.shape == (len(data), 5)

    result = venue.backtest(data, periods_per_year=365.0).run(Alternate())
    assert result.config["market"] == "perp"
    assert result.config["fee_taker"] == 0.0004
    assert result.config["slippage_bps"] == 2.0
    assert result.config["leverage"] == 5.0


def test_the_rl_env_gets_the_same_venue_too():
    pytest.importorskip("gymnasium")
    venue = Market(kind="perp", leverage=5.0, fee_taker=0.0004)
    env = venue.env(series(), num_envs=4, window=8)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 8, 5)
    env.close()


def test_a_market_backtest_matches_the_same_knobs_typed_out():
    from emsl.backtest import Backtester

    venue = Market(kind="perp", fee_taker=0.001, slippage_bps=3.0)
    data = series()
    by_object = venue.backtest(data, periods_per_year=365.0).run(Alternate())
    by_hand = Backtester(data, market="perp", fee_taker=0.001, slippage_bps=3.0,
                         periods_per_year=365.0).run(Alternate())
    assert by_object.stats == by_hand.stats
    assert by_object.config == by_hand.config


def test_a_knob_passed_beside_the_market_is_two_answers_to_one_question():
    venue = Market(kind="perp")
    with pytest.raises(TypeError) as excinfo:
        venue.tune(Alternate, {"x": (1, 2)}, series(), fee_taker=0.001)
    assert "belongs to the market" in str(excinfo.value)
    assert "replace" in str(excinfo.value)


def test_replace_changes_one_knob_and_leaves_the_rest():
    venue = Market(kind="perp", fee_taker=0.0004, leverage=5.0)
    dearer = venue.replace(fee_taker=0.002)
    assert dearer.fee_taker == 0.002
    assert dearer.kind == "perp"
    assert dearer.leverage == 5.0
    assert venue.fee_taker == 0.0004  # the original is untouched
    assert dearer != venue


def test_replacing_a_knob_that_does_not_exist_says_what_there_is():
    with pytest.raises(TypeError) as excinfo:
        Market().replace(fee_takre=0.001)
    assert "no ['fee_takre']" in str(excinfo.value)
    assert "fee_taker" in str(excinfo.value)


def test_an_unknown_kind_is_refused_at_construction():
    with pytest.raises(ValueError) as excinfo:
        Market(kind="futures")
    assert "'spot' or 'perp'" in str(excinfo.value)


def test_two_markets_with_the_same_knobs_are_the_same_market():
    assert Market(kind="perp", fee_taker=0.001) == Market(kind="perp", fee_taker=0.001)
    assert Market(kind="perp") != Market(kind="spot")
    assert len({Market(), Market()}) == 1


def test_a_market_prints_only_what_was_chosen():
    assert repr(Market()) == "Market(kind='spot')"
    shown = repr(Market(kind="perp", fee_taker=0.0004))
    assert "kind='perp'" in shown and "fee_taker=0.0004" in shown
    assert "max_open_orders" not in shown  # left at the default, so left unsaid


def test_the_keyword_form_still_works_untouched():
    from emsl.backtest import Backtester

    result = Backtester(series(), market="perp", fee_taker=0.001,
                        periods_per_year=365.0).run(Alternate())
    assert result.config["fee_taker"] == 0.001
