"""MADDPG with replay buffer, target networks, and TD3-style tricks (policy delay + noise).
Educational but functional.
"""
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from copy import deepcopy

@dataclass
class DDPGCfg:
    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    noise_sigma: float = 0.2
    noise_sigma_final: float = 0.05
    batch_size: int = 256
    policy_delay: int = 2

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,act_dim))
    def forward(self, x): return self.net(x)

class Critic(nn.Module):
    def __init__(self, joint_obs_dim, joint_act_dim):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(joint_obs_dim+joint_act_dim,256), nn.ReLU(), nn.Linear(256,256), nn.ReLU(), nn.Linear(256,1))
    def forward(self, s, a): return self.q(torch.cat([s,a], dim=-1))

class Replay:
    def __init__(self, obs_dim, act_dim, agents, capacity=1_000_000):
        self.capacity = capacity
        self.ptr = 0; self.size = 0
        self.agents = agents
        self.obs = {a: np.zeros((capacity, obs_dim), dtype=np.float32) for a in agents}
        self.next_obs = {a: np.zeros((capacity, obs_dim), dtype=np.float32) for a in agents}
        self.act = {a: np.zeros((capacity, 1), dtype=np.int64) for a in agents}  # discrete action index
        self.rew = {a: np.zeros((capacity, 1), dtype=np.float32) for a in agents}
        self.done = np.zeros((capacity, 1), dtype=np.float32)
    def add(self, obs, act, rew, next_obs, done):
        i = self.ptr
        for a in self.agents:
            self.obs[a][i] = obs[a]
            self.act[a][i,0] = act[a]
            self.rew[a][i,0] = rew[a]
            self.next_obs[a][i] = next_obs[a]
        self.done[i,0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size+1, self.capacity)
    def sample(self, batch, device):
        idx = np.random.randint(0, self.size, size=batch)
        batch_obs = torch.cat([torch.from_numpy(self.obs[a][idx]) for a in self.agents], dim=-1).to(device)
        batch_next = torch.cat([torch.from_numpy(self.next_obs[a][idx]) for a in self.agents], dim=-1).to(device)
        batch_act = torch.cat([torch.from_numpy(self.act[a][idx].astype(np.float32)) for a in self.agents], dim=-1).to(device)
        batch_rew = torch.cat([torch.from_numpy(self.rew[a][idx]) for a in self.agents], dim=-1).to(device)  # [B, A]
        batch_done = torch.from_numpy(self.done[idx]).to(device)  # [B,1]
        indiv_obs = {a: torch.from_numpy(self.obs[a][idx]).to(device) for a in self.agents}
        return indiv_obs, batch_obs, batch_act, batch_rew, batch_next, batch_done

class MADDPG:
    def __init__(self, obs_dim, act_dim, agents, cfg: DDPGCfg, device='cpu'):
        self.device = device
        self.agents = agents
        self.num_agents = len(agents)
        self.act_dim = act_dim
        # discrete -> one-hot for critic
        self.to_onehot = lambda a_idx: torch.nn.functional.one_hot(a_idx.long().squeeze(-1), num_classes=act_dim).float()
        # Actors per agent
        self.actor = {a: Actor(obs_dim, act_dim).to(device) for a in agents}
        self.target_actor = {a: deepcopy(self.actor[a]).to(device) for a in agents}
        # Central critic
        joint_obs_dim = obs_dim*self.num_agents
        joint_act_dim = act_dim*self.num_agents
        self.critic = Critic(joint_obs_dim, joint_act_dim).to(device)
        self.target_critic = deepcopy(self.critic).to(device)

        self.pi_opt = {a: torch.optim.Adam(self.actor[a].parameters(), lr=cfg.lr_actor) for a in agents}
        self.q_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

        self.cfg = cfg
        self.total_it = 0

    def select_actions(self, obs_dict, explore=True):
        acts = {}
        for i, a in enumerate(self.agents):
            o = torch.from_numpy(obs_dict[a]).float().unsqueeze(0).to(self.device)
            logits = self.actor[a](o)  # [1, act_dim]
            if explore:
                # Gumbel-softmax sampling to keep discrete
                g = -torch.log(-torch.log(torch.rand_like(logits)))
                a_idx = torch.argmax(logits + g, dim=-1).item()
            else:
                a_idx = torch.argmax(logits, dim=-1).item()
            acts[a] = int(a_idx) - 2  # map to {-2..+2}
        return acts

    def train_step(self, replay: Replay):
        self.total_it += 1
        indiv_obs, batch_obs, batch_act_idx, batch_rew, batch_next, batch_done = replay.sample(self.cfg.batch_size, self.device)

        # Current joint actions as one-hot
        onehots = []
        for a in self.agents:
            onehots.append(self.to_onehot(batch_act_idx[:, [self.agents.index(a)]]))
        A_joint = torch.cat(onehots, dim=-1)  # [B, A*act_dim]

        # Critic target: compute target actions (greedy + noise) for next state
        next_onehots = []
        for a in self.agents:
            logits = self.target_actor[a](indiv_obs[a])
            a_idx = torch.argmax(logits, dim=-1, keepdim=True)  # [B,1]
            next_onehots.append(self.to_onehot(a_idx))
        A_next = torch.cat(next_onehots, dim=-1)

        with torch.no_grad():
            q_next = self.target_critic(batch_next, A_next).squeeze(-1)
            # average rewards across agents as team reward
            r = batch_rew.mean(dim=-1)
            y = r + (1.0 - batch_done.squeeze(-1)) * self.cfg.gamma * q_next

        q = self.critic(batch_obs, A_joint).squeeze(-1)
        q_loss = torch.mean((q - y)**2)

        self.q_opt.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.q_opt.step()

        info = {"q_loss": q_loss.item()}

        # Delayed policy update
        if self.total_it % self.cfg.policy_delay == 0:
            # Actors try to maximise Q
            next_onehots_pi = []
            for a in self.agents:
                logits = self.actor[a](indiv_obs[a])
                a_idx = torch.argmax(logits, dim=-1, keepdim=True)
                next_onehots_pi.append(self.to_onehot(a_idx))
            A_pi = torch.cat(next_onehots_pi, dim=-1)
            q_pi = self.critic(batch_obs, A_pi).mean()
            pi_loss = -q_pi

            for a in self.agents:
                self.pi_opt[a].zero_grad()
            pi_loss.backward()
            for a in self.agents:
                torch.nn.utils.clip_grad_norm_(self.actor[a].parameters(), 1.0)
                self.pi_opt[a].step()

            # Polyak averaging
            for a in self.agents:
                for tp, p in zip(self.target_actor[a].parameters(), self.actor[a].parameters()):
                    tp.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

            info.update({"pi_loss": float(pi_loss.item())})
        return info
