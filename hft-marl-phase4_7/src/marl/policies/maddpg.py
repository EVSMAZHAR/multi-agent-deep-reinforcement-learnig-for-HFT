import torch, torch.nn as nn
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim,64), nn.ReLU(), nn.Linear(64,act_dim), nn.Tanh())
    def forward(self, x): return self.net(x)
class Critic(nn.Module):
    def __init__(self, joint_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(joint_dim+act_dim,128), nn.ReLU(), nn.Linear(128,1))
    def forward(self, s, a): return self.net(torch.cat([s,a],dim=-1))
