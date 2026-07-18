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
