"""MAPPO with shared actor (per-agent) and central critic (joint state).
This is a compact educational implementation sufficient for experiments.
"""
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F

@dataclass
class PPOCfg:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.15
    entropy_coef: float = 0.02
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    epochs: int = 4

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim,128), nn.Tanh(), nn.Linear(128,128), nn.Tanh())
        self.pi = nn.Linear(128, act_dim)
    def forward(self, x):
        logits = self.pi(self.net(x))
        return logits

class CentralCritic(nn.Module):
    def __init__(self, joint_obs_dim):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(joint_obs_dim,128), nn.Tanh(), nn.Linear(128,64), nn.Tanh(), nn.Linear(64,1))
    def forward(self, joint_obs):
        return self.v(joint_obs)

class MAPPO:
    def __init__(self, obs_dim, act_dim, num_agents, ppo_cfg:PPOCfg):
        self.actor = Actor(obs_dim, act_dim)
        self.critic = CentralCritic(joint_obs_dim=obs_dim*num_agents)
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=ppo_cfg.lr_actor)
        self.v_opt = torch.optim.Adam(self.critic.parameters(), lr=ppo_cfg.lr_critic)
        self.cfg = ppo_cfg
        self.num_agents = num_agents
        self.act_dim = act_dim

    def act(self, obs_dict):
        # obs_dict: {agent: obs_np}
        obs = torch.from_numpy(
            torch.tensor([obs_dict[k] for k in sorted(obs_dict.keys())], dtype=torch.float32).numpy()
        )  # [A, obs_dim]
        logits = self.actor(obs)  # [A, act_dim]
        probs = torch.distributions.Categorical(logits=logits)
        a = probs.sample()
        logp = probs.log_prob(a)
        return a.numpy(), logp.detach().numpy()

    def evaluate_actions(self, obs_tensor, actions_tensor):
        # obs_tensor: [B*A, obs_dim] stacked; actions_tensor: [B*A]
        logits = self.actor(obs_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions_tensor)
        entropy = dist.entropy().mean()
        return logp, entropy

    def value(self, joint_obs_tensor):
        return self.critic(joint_obs_tensor).squeeze(-1)

