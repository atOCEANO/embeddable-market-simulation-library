"""A Gymnasium-compatible vectorized market environment.

Wraps the Rust ``Batch``: ``num_envs`` independent envs over one shared candle
series, each starting at a random offset, stepped in parallel with the GIL
released. You control three things:

- **Observation**: pass ``features`` ``(T, F)`` and the agent sees a ``(window, F)``
  window of your indicators; omit it and it sees the raw ``(window, 5)`` candles.
- **Reward**: pass ``reward_fn(state, prev)`` where the two arguments carry the
  batched account fields as ``(num_envs,)`` arrays; omit it for the change in
  equity. It runs once per step over arrays, so it stays on the fast path.
- **Action**: ``Discrete(3)`` by default (0 hold, 1 buy, 2 sell a fixed
  ``trade_size``); pass ``action_fn`` and ``action_space`` to decode your own.

Episodes terminate on liquidation and truncate at the last bar, or at
``episode_len`` steps if set; finished envs auto-reset on the same step, with the
pre-reset observation and equity in ``infos`` (see ADR 0010).

Each cost knob (``fee_taker``, ``fee_maker``, ``slippage_bps``, ``impact``) takes
a scalar shared across envs, or a ``(low, high)`` pair sampled uniformly per env
so the batch trains across a spread of cost regimes (ADR 0014). The draw uses the
constructor ``seed`` and is fixed for each env's life. A per-fill ``cost_fn`` is
deliberately not offered; the ``impact`` coefficient is its fast-path substitute
(ADR 0013).

``data`` and ``features`` accept a numpy array, a pandas DataFrame, or a parquet
path.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from ._data import to_float2d, to_ohlcv
from ._emsl import Batch

# gymnasium's newer vector API tags a vector env's autoreset style so wrappers
# treat it correctly; emsl resets on the same step (ADR 0010), which older
# gymnasium has no enum to name, so declare it when present and skip it otherwise
try:
    from gymnasium.vector import AutoresetMode as _AutoresetMode

    _SAME_STEP = _AutoresetMode.SAME_STEP
except ImportError:
    _SAME_STEP = None


class VectorEnv(gym.vector.VectorEnv):
    """Vectorized bar-level trading env over the Rust batch."""

    def __init__(
        self,
        data,
        features=None,
        num_envs=8,
        window=32,
        market="spot",
        quote=10_000.0,
        fee_taker=0.0006,
        fee_maker=0.0002,
        slippage_bps=0.0,
        max_fill_fraction=1.0,
        max_open_orders=8,
        leverage=10.0,
        impact=0.0,
        funding_rate=0.0,
        funding_interval=0,
        trade_size=1.0,
        reward_fn=None,
        action_fn=None,
        action_space=None,
        episode_len=None,
        seed=None,
    ):
        data = to_ohlcv(data)
        self._num_bars = int(data.shape[0])
        self.window = int(window)
        if self._num_bars < self.window + 1:
            raise ValueError(
                f"window {self.window} needs at least {self.window + 1} bars "
                f"(a full window plus one to step into), but the series has "
                f"{self._num_bars}"
            )
        self.trade_size = float(trade_size)
        self.num_envs = int(num_envs)
        self._reward_fn = reward_fn
        self._action_fn = action_fn
        self._episode_len = None if episode_len is None else int(episode_len)

        n_features = 5
        if features is not None:
            features = to_float2d(features)
            if features.ndim != 2 or features.shape[0] != self._num_bars:
                raise ValueError("features must be (T, F) with T matching the candles")
            n_features = int(features.shape[1])
        self._has_features = features is not None

        # independent streams: the per-env cost draw and the start offsets come from
        # separate generators, so a fixed seed gives the same offsets whether or not
        # a (low, high) cost knob is passed. Costs are drawn once here and fixed for
        # the env's life; reset(seed=...) re-seeds only the offset stream (ADR 0014)
        cost_seed, offset_seed = np.random.SeedSequence(seed).spawn(2)
        self._cost_rng = np.random.default_rng(cost_seed)
        self._rng = np.random.default_rng(offset_seed)

        fee_taker, fee_taker_per_env = self._cost_arg(fee_taker)
        fee_maker, fee_maker_per_env = self._cost_arg(fee_maker)
        slippage_bps, slippage_bps_per_env = self._cost_arg(slippage_bps)
        impact, impact_per_env = self._cost_arg(impact)

        self._batch = Batch(
            data,
            num_envs=self.num_envs,
            market=market,
            quote=quote,
            fee_taker=fee_taker,
            fee_maker=fee_maker,
            slippage_bps=slippage_bps,
            max_fill_fraction=max_fill_fraction,
            max_open_orders=max_open_orders,
            leverage=leverage,
            impact=impact,
            funding_rate=funding_rate,
            funding_interval=funding_interval,
            report=False,
            fee_taker_per_env=fee_taker_per_env,
            fee_maker_per_env=fee_maker_per_env,
            slippage_bps_per_env=slippage_bps_per_env,
            impact_per_env=impact_per_env,
        )

        if features is not None:
            self._batch.set_features(features)     # cache once; observe_features gathers from it

        self.single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window, n_features), dtype=np.float32
        )
        self.single_action_space = (
            action_space if action_space is not None else spaces.Discrete(3)
        )
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        # set directly rather than via the version-sensitive base __init__
        self.metadata = {"autoreset_mode": _SAME_STEP} if _SAME_STEP is not None else {}
        self.render_mode = None
        self.closed = False

        self._prev = None
        self._steps = np.zeros(self.num_envs, dtype=np.int64)

    def _cost_arg(self, value):
        # a scalar shares one cost across the batch; a (low, high) pair samples a
        # per-env override array uniformly, so each env carries its own regime
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError("a cost range must be a (low, high) pair")
            low, high = float(value[0]), float(value[1])
            drawn = self._cost_rng.uniform(low, high, size=self.num_envs).astype(np.float64)
            return low, drawn
        return float(value), None

    def _random_offsets(self, n):
        # start at or after the warmup so the window is full, and leave at least one
        # bar to step into; the constructor guarantees num_bars >= window + 1, so
        # lo < hi always holds and the range always has a value to draw
        lo = self.window - 1
        hi = self._num_bars - 1
        return self._rng.integers(lo, hi, size=n, dtype=np.int64)

    def _obs(self):
        if self._has_features:
            arr = self._batch.observe_features(self.window)
        else:
            arr = self._batch.observe(self.window)
        return arr.astype(np.float32)

    def _snapshot(self):
        return SimpleNamespace(**self._batch.snapshot())

    def _reward(self, cur, prev):
        if self._reward_fn is not None:
            return np.asarray(self._reward_fn(cur, prev), dtype=np.float32)
        return (cur.equity - prev.equity).astype(np.float32)

    def _sizes(self, actions):
        # a custom decoder maps the raw actions and current state to per-env signed
        # sizes; the default reads Discrete(3): 1 buy, 2 sell, else hold
        if self._action_fn is not None:
            return np.asarray(self._action_fn(actions, self._prev), dtype=np.float64)
        actions = np.asarray(actions)
        return np.where(
            actions == 1, self.trade_size, np.where(actions == 2, -self.trade_size, 0.0)
        ).astype(np.float64)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        offsets = self._random_offsets(self.num_envs)
        self._batch.reset_at_each(offsets)
        self._steps[:] = 0
        self._prev = self._snapshot()
        return self._obs(), {}

    def step(self, actions):
        self._batch.advance(self._sizes(actions))
        self._steps += 1

        cur = self._snapshot()
        rewards = self._reward(cur, self._prev)
        terminations = self._batch.busts()
        truncations = self._batch.dones()
        if self._episode_len is not None:
            truncations = truncations | (self._steps >= self._episode_len)
        done = terminations | truncations

        obs = self._obs()
        infos = {}
        prev_for_next = cur
        if done.any():
            final_obs = obs
            final_equity = cur.equity

            new_offsets = self._random_offsets(self.num_envs)
            self._batch.reset_masked(done, new_offsets)
            self._steps[done] = 0
            obs = self._obs()
            prev_for_next = self._snapshot()  # the reset envs start fresh

            final_observation = np.empty(self.num_envs, dtype=object)
            for i in range(self.num_envs):
                final_observation[i] = final_obs[i] if done[i] else None
            infos["final_observation"] = final_observation
            infos["_final_observation"] = done.copy()
            fe = np.full(self.num_envs, np.nan, dtype=np.float64)
            fe[done] = final_equity[done]
            infos["final_equity"] = fe
            infos["_final_equity"] = done.copy()

        self._prev = prev_for_next
        return obs, rewards, terminations, truncations, infos

    def close(self, **kwargs):
        pass
