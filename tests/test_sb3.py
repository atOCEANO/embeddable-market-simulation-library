"""Tests for emsl.sb3.EmslVecEnv: it should present the VectorEnv as an SB3 VecEnv,
map the step tuple and same-step autoreset onto SB3's convention, and let an SB3
algorithm train through it. stable-baselines3 backs it, so the module skips without
it; the PPO smoke test also needs torch.
"""

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3.common.vec_env import VecEnv

from emsl.rl import VectorEnv
from emsl.sb3 import EmslVecEnv


def series(n=200):
    t = np.arange(n, dtype=np.float64)
    close = 100.0 + np.sin(t / 7.0) + 0.02 * t
    return np.stack([close, close + 1.0, close - 1.0, close, np.full(n, 1000.0)], axis=1)


def test_adapter_is_a_vecenv_and_steps():
    venv = EmslVecEnv(VectorEnv(series(), num_envs=4, window=8, seed=0))
    assert isinstance(venv, VecEnv)
    assert venv.num_envs == 4
    obs = venv.reset()
    assert obs.shape == (4, 8, 5)
    venv.step_async(np.array([1, 2, 0, 1]))
    obs, rewards, dones, infos = venv.step_wait()
    assert obs.shape == (4, 8, 5)
    assert rewards.shape == (4,)
    assert dones.shape == (4,) and dones.dtype == np.bool_
    assert isinstance(infos, list) and len(infos) == 4


def test_terminal_observation_lands_in_info_on_done():
    # episode_len=1 truncates every env each step, so every info carries the SB3
    # terminal observation, emsl's true final for the episode that just ended
    venv = EmslVecEnv(VectorEnv(series(), num_envs=3, window=4, episode_len=1, seed=0))
    venv.reset()
    venv.step_async(np.zeros(3, dtype=np.int64))
    _, _, dones, infos = venv.step_wait()
    assert dones.all()
    for info in infos:
        assert "terminal_observation" in info
        assert info["terminal_observation"].shape == (4, 5)
        assert info["TimeLimit.truncated"] is True  # a time limit, not a true terminal


def test_seed_forwards_from_sb3_to_the_env():
    # SB3 seeds an env through seed(); forwarding it to the next reset makes the
    # batch's episode offsets reproducible from an SB3 seed, not only the constructor
    def first_obs(s):
        venv = EmslVecEnv(VectorEnv(series(300), num_envs=4, window=8, seed=0))
        venv.seed(s)
        return venv.reset()

    assert np.array_equal(first_obs(11), first_obs(11))
    assert not np.array_equal(first_obs(11), first_obs(22))


def test_ppo_learns_through_the_adapter():
    pytest.importorskip("torch")
    from stable_baselines3 import PPO

    venv = EmslVecEnv(VectorEnv(series(400), num_envs=8, window=8, market="perp", seed=0))
    model = PPO("MlpPolicy", venv, n_steps=16, batch_size=32, n_epochs=1, verbose=0)
    model.learn(total_timesteps=8 * 16)  # a smoke run: it trains end to end
    obs = venv.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert np.asarray(action).shape == (8,)
