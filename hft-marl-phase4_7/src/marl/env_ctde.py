"""CTDE multi-agent env with more realistic fills and transient impact.

- Two maker agents: 'maker_bid', 'maker_ask' posting unit lots with tick offsets {-2..+2}
- Fill probabilities depend on spread, imbalance, offset, and a latent liquidity factor
- Transient price impact: mid_t+1 = mid_t + phi * signed_flow + eta * noise - kappa * (mid_t - anchor) (mean-revert)
- Reward: dPnL - lambda*|inventory| - kappaA*|offset|; PnL from executed fills and mark-to-market

Inputs:
- dataset windows (scaled features) for state context
- scaler.json for possible inverse-transforms (not required here)

This is a research-grade simulator; for production, plug into ABIDES/JAX-LOB.
"""
import gymnasium as gym
import numpy as np
import json
from pathlib import Path

class CTDEHFTEnv(gym.Env):
    metadata = {"render.modes": []}
    def __init__(self, dataset_path: str, scaler_path: str, tick_size: float = 0.01,
                 decision_ms: int = 100, reward_weights = None, episode_len: int = 1000,
                 seed: int = 123, impact=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.agents = ['maker_bid','maker_ask']
        self.action_space = gym.spaces.Discrete(5)  # {-2,-1,0,1,2}
        # obs = last frame features (8) + role one-hot(2) + latent liquidity (1) => 11 dims
        self.obs_dim = 8 + 2 + 1
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.dataset = np.load(dataset_path)
        self.X = self.dataset['X']  # [N, T, F], scaled features
        self.N, self.T, self.F = self.X.shape
        self.tick = tick_size
        self.decision_ms = decision_ms
        self.episode_len = min(episode_len, self.N-1) if self.N>1 else 0
        self.weights = reward_weights or {"lambda_inv":0.5, "kappa_imp":0.2, "eta_cvar":0.0}
        self.impact = impact or {"phi": 0.02, "eta": 0.01, "kappa_mr": 0.02}
        self.anchor = 100.0  # mean-reversion anchor

        self.reset_state()

    def reset_state(self):
        self.t = 0
        self.idx0 = int(self.rng.integers(low=0, high=max(1, self.N - self.episode_len)))
        self.inv = {a: 0.0 for a in self.agents}
        self.cash = {a: 0.0 for a in self.agents}
        self.last_mid = self.anchor
        self.liq = 1.0  # latent liquidity factor

    def _obs_from_frame(self, frame_last, role_id):
        role_vec = np.zeros(2, dtype=np.float32); role_vec[role_id] = 1.0
        liq = np.array([self.liq], dtype=np.float32)
        return np.concatenate([frame_last.astype(np.float32), role_vec, liq], axis=0)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.reset_state()
        frame = self.X[self.idx0 + self.t, -1, :]
        obs = {
            'maker_bid': self._obs_from_frame(frame, 0),
            'maker_ask': self._obs_from_frame(frame, 1),
        }
        return obs, {}

    def _fill_probability(self, spread, imbalance, offset_ticks, side):
        # Sigmoid over linear score of features
        base = 0.25 * self.liq
        score = base + 0.8*(max(0.0, -abs(offset_ticks))) + 0.3 * (imbalance * (-side))
        p = 1.0 / (1.0 + np.exp(-score))
        # Wider spreads reduce taker arrivals → lower p
        p *= float(np.clip(1.0 - 0.5*spread, 0.1, 1.0))
        return float(np.clip(p, 0.01, 0.99))

    def step(self, actions):
        offset_map = {-2:-2,-1:-1,0:0,1:1,2:2}
        off_bid = offset_map[int(actions.get('maker_bid', 2))]
        off_ask = offset_map[int(actions.get('maker_ask', 2))]

        frame = self.X[self.idx0 + self.t, -1, :]
        spread, imbalance = float(frame[0]), float(frame[1])

        # Fill simulation (unit lot each side)
        fill_bid = self.rng.random() < self._fill_probability(spread, imbalance, off_bid, side=-1)
        fill_ask = self.rng.random() < self._fill_probability(spread, imbalance, off_ask, side=+1)

        # Trade prices around mid ± spread/2 ± offset*tick
        px_bid = self.last_mid - (spread/2) + off_bid*self.tick
        px_ask = self.last_mid + (spread/2) + off_ask*self.tick

        # Signed flow for impact (buy +1, sell -1)
        flow = 0.0
        if fill_bid:  # we bought 1
            self.inv['maker_bid'] += 1.0
            self.cash['maker_bid'] -= px_bid
            flow += +1.0
        if fill_ask:  # we sold 1
            self.inv['maker_ask'] -= 1.0
            self.cash['maker_ask'] += px_ask
            flow += -1.0

        # Transient impact & mean-reversion of mid
        phi = self.impact.get("phi",0.02); eta = self.impact.get("eta",0.01); kappa_mr = self.impact.get("kappa_mr",0.02)
        noise = self.rng.normal(0.0, eta)
        self.last_mid = self.last_mid + phi*flow + noise - kappa_mr*(self.last_mid - self.anchor)

        # Update latent liquidity (mean-reverting)
        self.liq = 0.95*self.liq + 0.05*1.0 + self.rng.normal(0.0, 0.01)
        self.liq = float(np.clip(self.liq, 0.5, 1.5))

        # Mark-to-market PnL
        pnl_bid = self.cash['maker_bid'] + self.inv['maker_bid']*self.last_mid
        pnl_ask = self.cash['maker_ask'] + self.inv['maker_ask']*self.last_mid
        lam = self.weights.get("lambda_inv", 0.5)
        kap = self.weights.get("kappa_imp", 0.2)
        rew_bid = pnl_bid - lam*abs(self.inv['maker_bid']) - kap*abs(off_bid)
        rew_ask = pnl_ask - lam*abs(self.inv['maker_ask']) - kap*abs(off_ask)

        self.t += 1
        terminated = self.t >= self.episode_len
        frame_next = self.X[self.idx0 + min(self.t, self.episode_len-1), -1, :]
        obs = {
            'maker_bid': self._obs_from_frame(frame_next, 0),
            'maker_ask': self._obs_from_frame(frame_next, 1),
        }
        rewards = {'maker_bid': float(rew_bid), 'maker_ask': float(rew_ask)}
        term = {'maker_bid': terminated, 'maker_ask': terminated}
        trunc = {'maker_bid': False, 'maker_ask': False}
        infos = {'maker_bid': {}, 'maker_ask': {}}
        return obs, rewards, term, trunc, infos
