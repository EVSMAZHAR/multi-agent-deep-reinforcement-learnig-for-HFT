"""CTDE multi-agent env driven by dataset tensors.
Two maker agents (bid/ask). Actions = price offset in ticks {-2..+2}. Fixed lot size = 1 (configurable later).
Reward: dPnL - lambda*|inventory| - kappa*|offset|; PnL from simulated fills using imbalance & spread.
"""
import gymnasium as gym
import numpy as np
import json
from pathlib import Path

class CTDEHFTEnv(gym.Env):
    metadata = {"render.modes": []}
    def __init__(self, dataset_path: str, scaler_path: str, tick_size: float = 0.01,
                 decision_ms: int = 100, reward_weights = None, episode_len: int = 1000, seed: int = 123):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.agents = ['maker_bid','maker_ask']
        self.action_space = gym.spaces.Discrete(5)  # {-2,-1,0,1,2}
        # obs = last frame of window features + agent role one-hot(2)
        self.obs_dim = 8 + 2
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.dataset = np.load(dataset_path)
        self.X = self.dataset['X']  # [N, T, F], features are scaled
        self.N, self.T, self.F = self.X.shape
        with open(scaler_path,'r') as f:
            self.scaler = json.load(f)
        self.tick = tick_size
        self.decision_ms = decision_ms
        self.episode_len = min(episode_len, self.N-1) if self.N>1 else 0
        self.weights = reward_weights or {"lambda_inv":0.5, "kappa_imp":0.2, "eta_cvar":0.0}
        self.reset_state()

    def reset_state(self):
        self.t = 0
        # sample a random starting index for the episode
        self.idx0 = int(self.rng.integers(low=0, high=max(1, self.N - self.episode_len)))
        self.inv = {a: 0.0 for a in self.agents}
        self.cash = {a: 0.0 for a in self.agents}
        self.last_mid = 100.0

    def _obs_from_frame(self, frame_last, role_id):
        # frame_last: [F] for last time in window
        role_vec = np.zeros(2, dtype=np.float32); role_vec[role_id] = 1.0
        return np.concatenate([frame_last.astype(np.float32), role_vec], axis=0)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.reset_state()
        frame = self.X[self.idx0 + self.t, -1, :]  # last step features
        obs = {
            'maker_bid': self._obs_from_frame(frame, 0),
            'maker_ask': self._obs_from_frame(frame, 1),
        }
        return obs, {}

    def _fill_probability(self, spread, imbalance, offset_ticks, side):
        # Heuristic: better (more aggressive) offsets and favourable imbalance increase fill odds
        # side: +1 for ask (selling), -1 for bid (buying)
        base = 0.15 + 0.10 * (spread > 0.5).astype(float)
        imb_term = 0.2 * (imbalance * (-side))  # if bid side and imbalance negative -> more sellers -> higher fill
        off_term = 0.1 * (-abs(offset_ticks))   # larger offset reduces fill odds
        p = base + imb_term + off_term
        return float(np.clip(p, 0.01, 0.9))

    def step(self, actions):
        # decode inputs
        offset_map = {-2:-2,-1:-1,0:0,1:1,2:2}
        off_bid = offset_map[int(actions.get('maker_bid', 2))]  # default neutral
        off_ask = offset_map[int(actions.get('maker_ask', 2))]

        # current features
        frame = self.X[self.idx0 + self.t, -1, :]
        spread, imbalance = float(frame[0]), float(frame[1])
        mid = self.last_mid + float(self.rng.normal(0, 0.01))  # small noise to mid for dynamics

        # Fill simulation (unit lot)
        # Bid maker tries to buy; ask maker tries to sell
        fill_bid = self.rng.random() < self._fill_probability(spread, imbalance, off_bid, side=-1)
        fill_ask = self.rng.random() < self._fill_probability(spread, imbalance, off_ask, side=+1)

        # Trade prices roughly around microprice ± offset*tick
        px_bid = mid - (spread/2) + off_bid*self.tick
        px_ask = mid + (spread/2) + off_ask*self.tick

        # Update inventory/cash and compute pnl change
        pnl_delta = {}
        if fill_bid:
            self.inv['maker_bid'] += 1.0
            self.cash['maker_bid'] -= px_bid
        if fill_ask:
            self.inv['maker_ask'] -= 1.0
            self.cash['maker_ask'] += px_ask

        # Mark-to-market PnL at current mid
        m2m_bid = self.cash['maker_bid'] + self.inv['maker_bid'] * mid
        m2m_ask = self.cash['maker_ask'] + self.inv['maker_ask'] * mid
        pnl_delta['maker_bid'] = m2m_bid
        pnl_delta['maker_ask'] = m2m_ask

        # Risk penalties
        lam = self.weights.get("lambda_inv", 0.5)
        kap = self.weights.get("kappa_imp", 0.2)
        rew_bid = (pnl_delta['maker_bid']) - lam*abs(self.inv['maker_bid']) - kap*abs(off_bid)
        rew_ask = (pnl_delta['maker_ask']) - lam*abs(self.inv['maker_ask']) - kap*abs(off_ask)

        self.last_mid = mid
        self.t += 1
        terminated = self.t >= self.episode_len

        # Next observation
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
