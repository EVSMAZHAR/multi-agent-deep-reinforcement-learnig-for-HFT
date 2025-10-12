import gymnasium as gym
import numpy as np
class CTDEHFTEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.agents = ['maker_A','maker_B']
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)
        self.t = 0
    def reset(self, seed=None, options=None):
        self.t = 0
        obs = {a: self.observation_space.sample() for a in self.agents}
        return obs, {}
    def step(self, actions):
        rewards = {a: float(np.random.randn()*0.01) for a in self.agents}
        self.t += 1
        done = self.t >= 1000
        obs = {a: self.observation_space.sample() for a in self.agents}
        term = {a: done for a in self.agents}
        trunc = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, rewards, term, trunc, infos
