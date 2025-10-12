import torch, torch.nn as nn
class MAPPOPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim,64), nn.Tanh(), nn.Linear(64,64), nn.Tanh())
        self.pi = nn.Linear(64, act_dim)
        self.v = nn.Linear(64, 1)
    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h)
