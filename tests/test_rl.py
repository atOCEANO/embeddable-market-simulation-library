"""Tests for emsl.rl.VectorEnv: it should present the Gymnasium vector API over
the Rust batch, with batched observations, delta-equity rewards, and same-step
autoreset carrying the final observation in infos.
"""

import numpy as np
import pytest

from emsl.rl import VectorEnv


def make_data(t=50):
    close = 100.0 + np.arange(t, dtype=np.float64)  # a gently rising series
    return np.stack(
        [close, close + 1.0, close - 1.0, close, np.full(t, 1000.0)], axis=1
    )


def flat_data(t=40, price=100.0):
    # a constant series: every start offset fills at the same price, so a per-env
    # cost is the only thing that can move the reward apart across envs
    col = np.full(t, price, dtype=np.float64)
    return np.stack([col, col, col, col, np.full(t, 1000.0)], axis=1)


def marked_data(t):
    # a series no two bars of which share a price: bar i opens at 100 + 10 * i and
    # closes 5 above that, so a price read anywhere names the bar it came from
    rise = np.arange(t, dtype=np.float64) * 10.0
    open_ = 100.0 + rise
    close = open_ + 5.0
    return np.stack(
        [open_, close + 1.0, open_ - 1.0, close, np.full(t, 1000.0)], axis=1
    )


def halving_data(t):
    # every bar opens where the last one closed and halves inside itself, so a
    # leveraged long bankrupts on the very step it opened whatever offset it
    # started from, which is what makes a termination independent of the draw
    open_ = 100.0 * np.power(0.5, np.arange(t, dtype=np.float64))
    close = open_ * 0.5
    # the volume is far past anything the position asks for, because the fill cap
    # is a size and the prices here span three orders of magnitude: a cap that bit
    # would leave the late offsets holding a position too small to bankrupt
    return np.stack([open_, open_, close, close, np.full(t, 1e9)], axis=1)


def marked_features(t, cols=3):
    # feature[i][c] = i * 10 + c, unique across the whole (T, cols) matrix, so one
    # value names the bar and the column it was gathered from and a window shifted
    # by a row cannot pass for the right one
    rows = np.arange(t, dtype=np.float64)[:, None] * 10.0
    return rows + np.arange(cols, dtype=np.float64)


def bar_index(mark_price):
    # invert marked_data's close, 105 + 10 * i, to name the bar a mark price came
    # from. The mark is read off the account state, not off the observation gather,
    # so it says which bar an env stands on independently of what it was shown
    return np.rint((np.asarray(mark_price) - 105.0) / 10.0).astype(np.int64)


def test_reset_returns_batched_observation_and_spaces():
    env = VectorEnv(make_data(), num_envs=4, window=8, market="spot", seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (4, 8, 5)
    assert obs.dtype == np.float32
    assert info == {}
    assert env.single_observation_space.shape == (8, 5)
    assert env.single_action_space.n == 3
    assert env.num_envs == 4


def test_step_returns_the_gym_vector_tuple():
    env = VectorEnv(make_data(), num_envs=4, window=8, seed=0)
    env.reset(seed=0)
    actions = np.array([1, 2, 0, 1])  # buy, sell, hold, buy
    obs, rewards, terminations, truncations, infos = env.step(actions)
    assert obs.shape == (4, 8, 5)
    assert obs.dtype == np.float32
    assert rewards.shape == (4,)
    assert rewards.dtype == np.float32
    assert terminations.shape == (4,) and terminations.dtype == np.bool_
    assert truncations.shape == (4,) and truncations.dtype == np.bool_
    assert not terminations.any()  # spot never liquidates


def test_envs_autoreset_at_the_end_with_final_obs_in_infos():
    env = VectorEnv(make_data(12), num_envs=2, window=4, seed=1)
    env.reset(seed=1)
    saw_final = False
    for _ in range(60):
        obs, rewards, term, trunc, infos = env.step(np.zeros(2, dtype=np.int64))
        assert obs.shape == (2, 4, 5)  # observation continues across resets
        if "final_observation" in infos:
            saw_final = True
            mask = infos["_final_observation"]
            for i in range(2):
                if mask[i]:
                    assert infos["final_observation"][i].shape == (4, 5)
                    assert np.isfinite(infos["final_equity"][i])
                else:
                    assert infos["final_observation"][i] is None
    assert saw_final  # a 12-bar series must truncate within 60 steps


def test_features_observation_uses_the_feature_columns():
    data = make_data(30)
    features = np.arange(30 * 3, dtype=np.float64).reshape(30, 3)  # (T, 3)
    env = VectorEnv(data, features=features, num_envs=4, window=6, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 6, 3)  # F = 3, not the 5 candle columns
    assert env.single_observation_space.shape == (6, 3)


def test_custom_reward_fn_is_called_with_batched_state():
    seen = {}

    def reward_fn(state, prev):
        seen["ok"] = True
        assert state.equity.shape == (3,)
        assert prev.equity.shape == (3,)
        return np.full(state.equity.shape, 7.0, dtype=np.float32)

    env = VectorEnv(make_data(30), num_envs=3, window=5, reward_fn=reward_fn, seed=0)
    env.reset(seed=0)
    _, rewards, _, _, _ = env.step(np.zeros(3, dtype=np.int64))
    assert seen.get("ok")
    assert np.allclose(rewards, 7.0)


def test_features_must_match_candle_length():
    with pytest.raises(ValueError):
        VectorEnv(make_data(30), features=np.zeros((10, 2)), num_envs=2, window=4)


def test_window_without_room_to_step_raises():
    # a window equal to or larger than the series leaves no bar to step into, so the
    # env would train on zero-padded observations; reject it instead of degrading
    with pytest.raises(ValueError):
        VectorEnv(make_data(8), num_envs=2, window=8)  # window == num_bars
    with pytest.raises(ValueError):
        VectorEnv(make_data(4), num_envs=2, window=8)  # window > num_bars


def test_episode_len_one_truncates_every_step():
    env = VectorEnv(make_data(200), num_envs=3, window=4, episode_len=1, seed=0)
    env.reset(seed=0)
    for _ in range(5):
        _, _, _, trunc, infos = env.step(np.zeros(3, dtype=np.int64))
        assert trunc.all()  # episode_len=1 truncates and resets every step
        assert "final_observation" in infos


def test_custom_action_fn_receives_actions_and_state():
    seen = {}

    def action_fn(actions, state):
        seen["called"] = True
        assert np.asarray(actions).shape == (3,)
        assert state.equity.shape == (3,)
        return np.zeros(3, dtype=np.float64)

    env = VectorEnv(make_data(50), num_envs=3, window=4, action_fn=action_fn, seed=0)
    env.reset(seed=0)
    env.step(np.array([1, 2, 0]))
    assert seen.get("called")


def test_per_env_cost_range_varies_reward_across_envs():
    # a (low, high) fee range draws a different taker fee per env; on a flat series
    # the only thing that can spread the delta-equity reward is that per-env fee
    env = VectorEnv(flat_data(), num_envs=8, window=4, fee_taker=(0.001, 0.02), seed=0)
    env.reset(seed=0)
    _, rewards, _, _, _ = env.step(np.ones(8, dtype=np.int64))  # every env buys 1.0
    assert rewards.std() > 0.0


def test_scalar_cost_gives_identical_reward_on_a_flat_series():
    # the control: one shared fee, so every env takes the identical hit
    env = VectorEnv(flat_data(), num_envs=8, window=4, fee_taker=0.01, seed=0)
    env.reset(seed=0)
    _, rewards, _, _, _ = env.step(np.ones(8, dtype=np.int64))
    assert rewards.std() == 0.0


def test_per_env_cost_range_is_reproducible_from_the_seed():
    def rollout():
        env = VectorEnv(flat_data(), num_envs=8, window=4, fee_taker=(0.001, 0.02), seed=5)
        env.reset(seed=5)
        _, rewards, _, _, _ = env.step(np.ones(8, dtype=np.int64))
        return rewards

    assert np.array_equal(rollout(), rollout())  # same seed draws the same costs


def drawn_fees(rewards, price=100.0, size=1.0):
    # on a flat series with no slippage the whole equity delta of a buy IS the fee,
    # so the reward reads back the per-env taker fee the env was given
    return -np.asarray(rewards, dtype=np.float64) / (price * size)


def test_a_per_env_cost_is_drawn_inside_its_range_and_fixed_for_the_envs_life():
    # ADR 0014 says the pair is sampled ONCE PER ENV, and the only thing asserted
    # about it was that the rewards have a spread. A draw taken per step, or per
    # episode, or from a range twice as wide as the one asked for, all produce a
    # spread: none of them is what "a batch trains across a spread of cost regimes"
    # means, because an env whose costs move underneath it is not a regime
    low, high = 0.001, 0.02
    env = VectorEnv(flat_data(), num_envs=8, window=4, fee_taker=(low, high), seed=0)
    env.reset(seed=0)
    _, first, _, _, _ = env.step(np.ones(8, dtype=np.int64))
    fees = drawn_fees(first)
    assert ((low <= fees) & (fees <= high)).all(), fees
    assert fees.std() > 0.0            # eight regimes, not one
    assert len(set(np.round(fees, 12))) == 8

    # the same env, one bar later: a cost redrawn per step reads differently here
    _, second, _, _, _ = env.step(np.ones(8, dtype=np.int64))
    assert np.allclose(drawn_fees(second), fees, rtol=0.0, atol=1e-6)

    # and a fresh episode keeps them: reset re-seeds the OFFSET stream only, so an
    # env's costs are its own for its whole life and not for one episode of it
    env.reset(seed=99)
    _, third, _, _, _ = env.step(np.ones(8, dtype=np.int64))
    assert np.allclose(drawn_fees(third), fees, rtol=0.0, atol=1e-6)


def test_the_cost_fallback_is_the_expensive_end_of_the_range():
    # _cost_arg hands back a scalar beside the per-env array, and the scalar is only
    # what would stand if the array ever failed to apply. Handing back the low end
    # makes that failure the cheapest possible market, which is the one direction a
    # cost model must never fail in, and nothing anywhere reads this value (ADR 0014)
    env = VectorEnv(flat_data(), num_envs=8, window=4, seed=0)
    scalar, drawn = env._cost_arg((0.001, 0.02))
    assert scalar == 0.02
    assert drawn.shape == (8,)
    assert ((0.001 <= drawn) & (drawn <= 0.02)).all()
    # a plain number is passed through with no array beside it
    assert env._cost_arg(0.007) == (0.007, None)


def ranged_data(t=40, price=100.0, width=10.0):
    # flat in open and close but with a real high and low, because a taker price is
    # clipped to its own bar (ADR 0074) and flat_data prints ONE price: slippage and
    # impact are both clipped away to nothing there, so a test of either on that
    # series reads zero however the knob is wired
    col = np.full(t, price, dtype=np.float64)
    return np.stack(
        [col, col + width, col - width, col, np.full(t, 1000.0)], axis=1
    )


def test_each_cost_knob_takes_its_own_range():
    # "each cost knob accepts a (low, high) pair" is four claims and only fee_taker
    # was ever read. A range wired to one knob and ignored on the others leaves this
    # file green, and a cost silently pinned to its default is the worst kind of
    # wrong number: it is plausible (ADR 0014)
    def spread(**kw):
        env = VectorEnv(ranged_data(), num_envs=8, window=4, seed=0, **kw)
        env.reset(seed=0)
        _, rewards, _, _, _ = env.step(np.ones(8, dtype=np.int64))
        return float(np.std(rewards))

    assert spread() == 0.0                                   # one shared venue
    assert spread(fee_taker=(0.001, 0.02)) > 0.0
    assert spread(slippage_bps=(1.0, 50.0), fee_taker=0.0) > 0.0
    assert spread(impact=(0.1, 5.0), fee_taker=0.0) > 0.0
    # fee_maker is the fourth, and the batched tier takes market orders only
    # (ADR 0020), so no maker fill can happen here and no reward can read it at
    # all. The draw itself is what there is to check
    env = VectorEnv(ranged_data(), num_envs=8, window=4, seed=0)
    scalar, drawn = env._cost_arg((0.0001, 0.001))
    assert scalar == 0.001
    assert drawn.shape == (8,) and drawn.std() > 0.0


def test_a_discrete_action_of_two_sells_and_one_buys():
    # ADR 0020's decoder: 1 buys, 2 sells, anything else holds. The sign is asserted
    # NOWHERE, in Rust or Python, and flipping it leaves every test in this file
    # green: the shapes, the dtypes and the spread of the rewards are all identical
    # under a decoder that shorts on 1 and buys on 2. An agent would then learn a
    # policy that trades the opposite way round from the one documented
    seen = {}

    def capture(cur, prev):
        seen["position"] = cur.position.copy()
        return np.zeros(cur.equity.shape, dtype=np.float64)

    env = VectorEnv(
        flat_data(), num_envs=4, window=4, market="perp", trade_size=3.0,
        fee_taker=0.0, fee_maker=0.0, reward_fn=capture, seed=0,
    )
    env.reset(seed=0)
    env.step(np.array([1, 2, 0, 1], dtype=np.int64))  # buy, sell, hold, buy
    assert np.array_equal(seen["position"], [3.0, -3.0, 0.0, 3.0])


def test_start_offsets_are_independent_of_cost_ranges():
    # the cost and offset RNG streams are independent, so a fixed constructor seed
    # gives the same start offsets (hence the same first observation) whether or not
    # a (low, high) cost range is passed
    def first_obs(**kw):
        env = VectorEnv(make_data(50), num_envs=4, window=4, seed=7, **kw)
        obs, _ = env.reset()  # the constructor seed, not a reset seed
        return obs

    assert np.array_equal(first_obs(), first_obs(fee_taker=(0.001, 0.02)))


def test_perp_liquidation_sets_a_termination():
    # a steep crash on a leveraged long wipes the account; terminations, the only
    # true-terminal path, must fire (nothing else in the suite observes it true)
    t = 40
    close = np.linspace(100.0, 1.0, t)
    data = np.stack([close, close, close, close, np.full(t, 1.0e9)], axis=1)
    env = VectorEnv(
        data,
        num_envs=8,
        window=3,
        market="perp",
        quote=100.0,
        leverage=5.0,
        trade_size=100.0,  # capped to 5x by the margin cap, then the crash busts it
        fee_taker=0.0,
        fee_maker=0.0,
        seed=0,
    )
    env.reset(seed=0)
    saw_term = False
    for _ in range(t + 5):
        _, _, term, _, _ = env.step(np.ones(8, dtype=np.int64))  # everyone long
        if bool(term.any()):
            saw_term = True
            break
    assert saw_term


def test_custom_action_space_and_continuous_actions():
    from gymnasium import spaces

    env = VectorEnv(
        make_data(50),
        num_envs=2,
        window=4,
        action_space=spaces.Box(-1.0, 1.0, shape=(1,)),
        action_fn=lambda a, s: np.asarray(a, dtype=np.float64).reshape(-1),
        seed=0,
    )
    assert isinstance(env.single_action_space, spaces.Box)
    env.reset(seed=0)
    obs, rewards, term, trunc, infos = env.step(np.zeros((2, 1), dtype=np.float64))
    assert rewards.shape == (2,)


def test_metadata_declares_same_step_autoreset_when_supported():
    # gymnasium's newer vector API has AutoresetMode; when present, the env declares
    # SAME_STEP so a vector wrapper does not assume the next-step default and misread
    # the same-step reset (ADR 0010). Older gymnasium has no enum, so the tag is absent
    try:
        from gymnasium.vector import AutoresetMode
    except ImportError:
        pytest.skip("gymnasium without AutoresetMode")
    env = VectorEnv(make_data(), num_envs=2, window=4, seed=0)
    assert env.metadata.get("autoreset_mode") == AutoresetMode.SAME_STEP


def test_the_feature_window_holds_exactly_the_bars_up_to_the_current_one():
    # The observation is the whole of what an agent sees, and every other test here
    # reads only its shape (ADR 0010). A window reaching one row further would hand
    # the agent bar tick + 1, the lookahead this library exists to prevent, and it
    # would ship green. Each feature value names the bar it came from, and the tick
    # is read back off the account's mark price, a channel the gather has no part
    # in, so a window off by a row cannot agree with both. A violation looks like
    # features[tick + 1] standing in the window, as its last row.
    t = 60
    window = 8
    features = marked_features(t)
    seen = {}

    def reward_fn(state, prev):
        seen["at_reset"] = np.array(prev.mark_price)
        seen["after_step"] = np.array(state.mark_price)
        return np.zeros(state.equity.shape, dtype=np.float32)

    env = VectorEnv(
        marked_data(t),
        features=features,
        num_envs=4,
        window=window,
        reward_fn=reward_fn,
        seed=11,
    )
    obs, _ = env.reset(seed=11)
    stepped, _, term, trunc, _ = env.step(np.zeros(4, dtype=np.int64))

    at_reset = bar_index(seen["at_reset"])  # prev is the snapshot reset() took
    for i in range(4):
        tick = int(at_reset[i])
        assert np.array_equal(obs[i], features[tick - window + 1 : tick + 1])
        assert not (obs[i] == features[tick + 1][0]).any()  # bar tick + 1 is absent

    after_step = bar_index(seen["after_step"])
    running = ~(term | trunc)
    assert running.any()  # a finished env shows the next episode, not this one
    for i in range(4):
        if not running[i]:
            continue
        tick = int(after_step[i])
        assert tick == int(at_reset[i]) + 1  # one action, one bar
        assert np.array_equal(stepped[i], features[tick - window + 1 : tick + 1])
        assert not (stepped[i] == features[tick + 1][0]).any()


def test_the_candle_window_holds_exactly_the_bars_up_to_the_current_one():
    # The default observation, the raw candles, carries the same risk and the same
    # blind spot: only its shape is ever read (ADR 0010). The bars are priced so no
    # two are alike and the tick comes back off the account's mark price, so a
    # window holding bar tick + 1 (lookahead) or ending a bar early is caught by
    # value. A violation looks like the last row being data[tick + 1].
    t = 40
    window = 6
    data = marked_data(t)
    seen = {}

    def reward_fn(state, prev):
        seen["at_reset"] = np.array(prev.mark_price)
        return np.zeros(state.equity.shape, dtype=np.float32)

    env = VectorEnv(data, num_envs=3, window=window, reward_fn=reward_fn, seed=4)
    obs, _ = env.reset(seed=4)
    env.step(np.zeros(3, dtype=np.int64))  # stepped only to read the reset snapshot

    at_reset = bar_index(seen["at_reset"])
    for i in range(3):
        tick = int(at_reset[i])
        assert np.array_equal(obs[i], data[tick - window + 1 : tick + 1])
        assert not (obs[i] == data[tick + 1][0]).any()  # bar tick + 1 is absent


def feature_bar(row):
    # marked_features holds bar * 10 + column, so a row's first column is ten times
    # the bar it was gathered from and one observation names its own start offset
    return int(round(float(row[0]) / 10.0))


# a series long enough that _random_offsets has a range to draw from. The three
# tests below used to run on window + 1 bars, the shortest the constructor accepts,
# where lo = window - 1 and hi = num_bars - 1 leave exactly ONE legal offset: every
# env then ran the same episode from the same bar, so an env handed another env's
# window, or a window cached at the first reset and re-served forever, was the same
# array as the right answer and no assertion could tell them apart. Both are live
# mutants that the shortest series cannot see, so the length is load-bearing
BARS = 40
EPISODE_SEED = 0


def test_a_finished_envs_step_returns_the_next_episodes_first_observation():
    # Same-step autoreset: the observation a finished env hands back belongs to the
    # NEXT episode, not to the one that just ended (ADR 0010). Returning the
    # terminal window instead leaves every shape, dtype and flag identical, so
    # nothing else in this file would notice, and the agent would learn from a
    # window its action never led to. episode_len=1 ends both episodes on the first
    # step, from two different offsets, and each window names the bars it came from
    window = 4
    features = marked_features(BARS)
    env = VectorEnv(
        marked_data(BARS), features=features, num_envs=2, window=window,
        episode_len=1, seed=0,
    )
    first, _ = env.reset(seed=EPISODE_SEED)
    started = [feature_bar(first[i][-1]) for i in range(2)]
    assert started[0] != started[1]  # two episodes, not one episode twice
    obs, _, term, trunc, _ = env.step(np.zeros(2, dtype=np.int64))

    assert trunc.all() and not term.any()
    for i in range(2):
        ended = started[i] + 1
        fresh = feature_bar(obs[i][-1])
        assert fresh != ended  # not the window the episode ended on
        assert not np.array_equal(obs[i], features[ended - window + 1 : ended + 1])
        assert not np.array_equal(obs[i], first[i])  # nor the one cached at reset
        # and it is a whole window off one bar rather than rows from several
        assert np.array_equal(obs[i], features[fresh - window + 1 : fresh + 1])
        assert window - 1 <= fresh < BARS - 1  # a legal offset for a fresh episode


def test_final_observation_holds_the_window_the_episode_actually_ended_on():
    # infos["final_observation"] is the only place the terminal window survives the
    # same-step reset, and today only its shape is read (ADR 0010), so handing back
    # the fresh post-reset window instead would pass. The two envs end on different
    # bars, so handing every finished env the first one's window is wrong for the
    # second and reads as two identical arrays here
    window = 4
    features = marked_features(BARS)
    env = VectorEnv(
        marked_data(BARS), features=features, num_envs=2, window=window,
        episode_len=1, seed=0,
    )
    first, _ = env.reset(seed=EPISODE_SEED)
    started = [feature_bar(first[i][-1]) for i in range(2)]
    assert started[0] != started[1]
    obs, _, _, trunc, infos = env.step(np.zeros(2, dtype=np.int64))

    assert trunc.all()
    for i in range(2):
        assert infos["_final_observation"][i]
        final = infos["final_observation"][i]
        ended = started[i] + 1
        assert np.array_equal(final, features[ended - window + 1 : ended + 1])
        assert np.array_equal(final[-1], features[ended])  # ends on the bar stepped to
        assert not np.array_equal(final, obs[i])  # not the fresh one
    finals = infos["final_observation"]
    assert not np.array_equal(finals[0], finals[1])  # each env's own, not env 0's


def test_final_equity_is_what_the_episode_ended_with_not_a_fresh_balance():
    # infos["final_equity"] is the finished episode's equity, read before the reset
    # (ADR 0010); a post-reset read hands back the starting 10000.0, which is finite
    # and passes the only assertion made about it today. The reward here is a
    # constant, not the equity delta, so reconstructing the final equity as the
    # previous equity plus the reward reads 10000.0 as well: under the DEFAULT
    # reward those two are the same number by definition, which is what let the
    # reconstruction pass. Every env buys 1.0 on the step that ends the episode, so
    # the number is hand computable per env: the market order fills at the next
    # bar's open, 100 + 10 * (offset + 1), pays a percent of that in taker fee, and
    # that bar closes five above it
    window = 4
    features = marked_features(BARS)
    env = VectorEnv(
        marked_data(BARS),
        features=features,
        num_envs=2,
        window=window,
        market="spot",
        quote=10_000.0,
        fee_taker=0.01,
        fee_maker=0.0,
        slippage_bps=0.0,
        impact=0.0,
        trade_size=1.0,
        reward_fn=lambda cur, prev: np.full(cur.equity.shape, 0.25),
        episode_len=1,
        seed=0,
    )
    first, _ = env.reset(seed=EPISODE_SEED)
    started = [feature_bar(first[i][-1]) for i in range(2)]
    assert started[0] != started[1]
    _, rewards, term, trunc, infos = env.step(np.ones(2, dtype=np.int64))

    assert trunc.all() and not term.any()
    assert (rewards == 0.25).all()  # the reward is not the equity delta here
    for i in range(2):
        fill = 100.0 + 10.0 * (started[i] + 1)
        ended_with = 10_000.0 - fill * 0.01 - fill + (fill + 5.0)
        assert infos["_final_equity"][i]
        assert abs(float(infos["final_equity"][i]) - ended_with) < 1e-9
        # and the reconstruction really is a different number here, which under the
        # default reward it could not be
        assert abs(ended_with - (10_000.0 + 0.25)) > 1e-6
    assert infos["final_equity"][0] != infos["final_equity"][1]


def test_a_bankrupt_env_autoresets_and_reports_what_the_episode_ended_with():
    # a termination is a finished episode too. Reading only the truncations for the
    # autoreset leaves a liquidated env running on a dead account for the rest of
    # the series, reporting neither its final observation nor its final equity, and
    # every other test in this file stays green: the only one that sees a
    # termination reads bool(term.any()) and never the observation or the infos
    # (ADR 0010, 0019)
    window = 4

    def all_in(actions, prev):
        # five times equity at whatever price the offset landed on, so the position
        # carries the same leverage wherever the episode started
        return 5.0 * prev.equity / prev.mark_price

    env = VectorEnv(
        halving_data(16),
        num_envs=2,
        window=window,
        market="perp",
        quote=10_000.0,
        leverage=10.0,
        fee_taker=0.0,
        fee_maker=0.0,
        slippage_bps=0.0,
        impact=0.0,
        action_fn=all_in,
        seed=0,
    )
    env.reset(seed=EPISODE_SEED)
    obs, rewards, term, trunc, infos = env.step(np.ones(2, dtype=np.int64))

    assert term.all() and not trunc.any()  # bankrupt, and nowhere near the last bar
    for i in range(2):
        assert infos["_final_observation"][i]
        assert infos["final_observation"][i] is not None
        assert infos["_final_equity"][i]
        assert float(infos["final_equity"][i]) == 0.0  # not the fresh 10000 balance
        # and the env was handed a fresh episode rather than left on the corpse
        assert not np.array_equal(obs[i], infos["final_observation"][i])
    # the default reward is the equity delta BY VALUE, which nothing else in this
    # file reads: elsewhere only its shape, its dtype and its spread are asserted,
    # so a reward scaled or signed wrongly would pass every one of them
    assert np.allclose(rewards, -10_000.0)
