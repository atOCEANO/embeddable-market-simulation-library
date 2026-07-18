"""Stable-Baselines3 adapter for ``emsl.rl.VectorEnv``.

SB3 does not consume a Gymnasium vector env directly; it wants a single ``gym.Env``
it vectorizes itself, or its own ``VecEnv``. ``EmslVecEnv`` presents the batched,
GIL-free ``VectorEnv`` as an SB3 ``VecEnv``, so PPO, A2C, and the rest of SB3 train
on it without giving up the batch. The mapping is a straight remap rather than a
re-vectorization: emsl's same-step autoreset (a finished env's step returns the next
episode's first observation, the true final in ``infos``) is exactly the SB3
``VecEnv`` contract, so a done env's final observation becomes SB3's
``terminal_observation`` (ADR 0010).

```python
from stable_baselines3 import PPO
from emsl.rl import VectorEnv
from emsl.sb3 import EmslVecEnv

venv = EmslVecEnv(VectorEnv(ohlcv, num_envs=8, window=32, market="perp", seed=0))
model = PPO("MlpPolicy", venv)
model.learn(total_timesteps=100_000)
```

Needs stable-baselines3: ``pip install 'emsl[sb3]'``.
"""

from __future__ import annotations

import numpy as np

try:
    from stable_baselines3.common.vec_env import VecEnv
except ImportError as exc:
    raise ImportError(
        "stable-baselines3 is required for emsl.sb3; pip install 'emsl[sb3]'"
    ) from exc

__all__ = ["EmslVecEnv"]


class EmslVecEnv(VecEnv):
    """Present an ``emsl.rl.VectorEnv`` as a Stable-Baselines3 ``VecEnv``.

    Construct it from a built ``VectorEnv`` and hand it to any SB3 algorithm. The
    underlying env keeps ownership of the batch, the observation, the reward, and
    the action decoding; this only translates the step and reset conventions.
    """

    def __init__(self, env):
        self._env = env
        super().__init__(env.num_envs, env.single_observation_space, env.single_action_space)
        self._actions = None
        self._reset_seed = None

    def reset(self):
        if self._reset_seed is not None:
            obs, _info = self._env.reset(seed=self._reset_seed)
            self._reset_seed = None
        else:
            obs, _info = self._env.reset()
        return obs

    def seed(self, seed=None):
        # SB3 seeds the env through this; hold the seed for the next reset so the
        # batch's episode offsets are reproducible from an SB3 seed, not only from
        # the VectorEnv constructor seed
        self._reset_seed = seed
        return [seed] * self.num_envs

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, rewards, terminations, truncations, infos = self._env.step(self._actions)
        dones = np.logical_or(np.asarray(terminations), np.asarray(truncations))
        final_obs = infos.get("final_observation")
        final_mask = infos.get("_final_observation")
        info_list = []
        for i in range(self.num_envs):
            info = {}
            if dones[i]:
                # SB3 distinguishes a time-limit truncation from a true terminal so
                # it can bootstrap the value past a truncation; a done env's final
                # observation is emsl's, carried in infos, and becomes SB3's terminal
                info["TimeLimit.truncated"] = bool(truncations[i] and not terminations[i])
                if final_mask is not None and final_mask[i]:
                    info["terminal_observation"] = final_obs[i]
            info_list.append(info)
        return obs, np.asarray(rewards, dtype=np.float32), dones, info_list

    def close(self):
        self._env.close()

    def _count(self, indices):
        # SB3 addresses sub-envs by index; this adapter fronts one batched env, so a
        # per-env query returns the same value once per requested index
        if indices is None:
            return self.num_envs
        if isinstance(indices, int):
            return 1
        return len(list(indices))

    def get_attr(self, attr_name, indices=None):
        return [getattr(self._env, attr_name, None)] * self._count(indices)

    def set_attr(self, attr_name, value, indices=None):
        setattr(self._env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        result = getattr(self._env, method_name)(*method_args, **method_kwargs)
        return [result] * self._count(indices)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self._count(indices)
