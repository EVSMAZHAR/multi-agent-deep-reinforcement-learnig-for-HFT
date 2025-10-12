"""Working MAPPO training with GAE + PPO clipping over the CTDE env.
Saves model to models/mappo.pt
"""
import argparse, yaml, os, json
from pathlib import Path
import numpy as np, torch
from torch.nn.utils import clip_grad_norm_
from src.marl.env_ctde import CTDEHFTEnv
from src.marl.policies.mappo import MAPPO, PPOCfg

def to_torch(np_array): return torch.as_tensor(np_array, dtype=torch.float32)

def rollout(env, policy: MAPPO, steps, device='cpu'):
    agents = sorted(env.agents)
    obs_dict, _ = env.reset()
    obs_buf, act_buf, logp_buf, rew_buf, done_buf, joint_obs_buf = [], [], [], [], [], []
    for t in range(steps):
        # Build joint obs for critic
        obs_stack = np.stack([obs_dict[a] for a in agents], axis=0)  # [A, obs_dim]
        joint_obs = obs_stack.reshape(-1)  # [A*obs_dim]
        # Sample actions for each agent
        actions_np, logp_np = policy.act(obs_dict)
        actions = {a: int(actions_np[i]) for i,a in enumerate(agents)}

        next_obs, rewards, terms, truncs, infos = env.step(actions)

        obs_buf.append(obs_stack)                   # [A, obs_dim]
        joint_obs_buf.append(joint_obs)            # [A*obs_dim]
        act_buf.append(np.array(list(actions.values()), dtype=np.int64))    # [A]
        logp_buf.append(np.array(logp_np, dtype=np.float32))
        rew_buf.append(np.array([rewards[a] for a in agents], dtype=np.float32))
        done = any(list(terms.values()))
        done_buf.append(np.array([done]*len(agents), dtype=np.float32))

        obs_dict = next_obs
        if done:
            obs_dict, _ = env.reset()

    # Convert to arrays
    obs_arr = np.stack(obs_buf, axis=0)            # [T, A, obs_dim]
    joint_arr = np.stack(joint_obs_buf, axis=0)    # [T, A*obs_dim]
    act_arr = np.stack(act_buf, axis=0)            # [T, A]
    logp_arr = np.stack(logp_buf, axis=0)          # [T, A]
    rew_arr = np.stack(rew_buf, axis=0)            # [T, A]
    done_arr = np.stack(done_buf, axis=0)          # [T, A]

    return obs_arr, joint_arr, act_arr, logp_arr, rew_arr, done_arr

def compute_gae(values, rewards, dones, gamma, lam):
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(T-1)):
        nonterminal = 1.0 - dones[t+1]
        delta = rewards[t] + gamma * values[t+1] * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + values
    return adv, ret

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    # Paths
    feat_dir = Path(cfg["paths"]["features"])
    dataset_path = feat_dir/"dev_tensors.npz"
    scaler_path = feat_dir/"scaler.json"

    # Env
    env = CTDEHFTEnv(
        dataset_path=str(dataset_path),
        scaler_path=str(scaler_path),
        tick_size=0.01,
        decision_ms=cfg.get("decision_ms", 100),
        reward_weights=cfg.get("reward_weights", {"lambda_inv":0.5,"kappa_imp":0.2,"eta_cvar":0.0}),
        episode_len=cfg.get("episode_len", 1000),
        seed=cfg.get("seed", 123)
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    num_agents = len(env.agents)
    ppo = PPOCfg(
        gamma=cfg.get("gamma",0.99),
        gae_lambda=cfg.get("gae_lambda",0.95),
        clip_epsilon=cfg.get("clip_epsilon",0.15),
        entropy_coef=cfg.get("entropy_coef",0.02),
        lr_actor=cfg.get("lr_actor",3e-4),
        lr_critic=cfg.get("lr_critic",3e-4),
        epochs=cfg.get("epochs",4)
    )
    policy = MAPPO(obs_dim, act_dim, num_agents, ppo)

    steps_per_update = cfg.get("steps_per_update", 2048)
    updates = cfg.get("updates", 50)
    device = 'cpu'

    for update in range(updates):
        obs_arr, joint_arr, act_arr, logp_arr, rew_arr, done_arr = rollout(env, policy, steps_per_update, device=device)

        # Flatten across agents for actor; critic uses joint obs
        T, A, OD = obs_arr.shape
        obs_flat = to_torch(obs_arr.reshape(T*A, OD))
        joint_t = to_torch(joint_arr)                      # [T, A*OD]
        acts = torch.as_tensor(act_arr.reshape(T*A,), dtype=torch.int64)
        old_logp = to_torch(logp_arr.reshape(T*A,))
        rews = to_torch(rew_arr.reshape(T*A,))
        dones = to_torch(done_arr.reshape(T*A,))

        # Values for each time step from central critic
        vals = policy.value(joint_t)                       # [T]
        vals_ext = torch.cat([vals, vals[-1:]], dim=0)     # bootstrap
        rews_mean = rews.view(T, A).mean(dim=1)            # average team reward for critic target
        dones_any = dones.view(T, A).max(dim=1).values

        adv, ret = compute_gae(vals_ext, rews_mean, dones_any, ppo.gamma, ppo.gae_lambda)
        adv = adv[:-1]                                     # drop bootstrap step
        # Expand advantages/returns per agent
        adv_exp = adv.unsqueeze(1).expand(T, A).reshape(T*A)
        ret_exp = ret[:-1].unsqueeze(1).expand(T, A).reshape(T*A)

        # Normalise advantages
        adv_mean, adv_std = adv_exp.mean(), adv_exp.std().clamp_min(1e-6)
        adv_norm = (adv_exp - adv_mean) / adv_std

        # PPO update
        for _ in range(ppo.epochs):
            logp, entropy = policy.evaluate_actions(obs_flat, acts)
            ratio = (logp - old_logp).exp()
            surr1 = ratio * adv_norm
            surr2 = torch.clamp(ratio, 1.0-ppo.clip_epsilon, 1.0+ppo.clip_epsilon) * adv_norm
            pi_loss = -(torch.min(surr1, surr2).mean() + ppo.entropy_coef*entropy)

            v_pred = policy.value(joint_t).unsqueeze(1).expand(-1, A).reshape(T*A)
            v_loss = torch.mean((v_pred - ret_exp)**2)

            policy.pi_opt.zero_grad()
            pi_loss.backward()
            clip_grad_norm_(policy.actor.parameters(), 1.0)
            policy.pi_opt.step()

            policy.v_opt.zero_grad()
            v_loss.backward()
            clip_grad_norm_(policy.critic.parameters(), 1.0)
            policy.v_opt.step()

        if (update+1) % 5 == 0:
            print(f"[Update {update+1}/{updates}] pi_loss={pi_loss.item():.4f} v_loss={v_loss.item():.4f} adv_mean={adv_mean.item():.4f}")

    # Save model
    Path("models").mkdir(exist_ok=True)
    torch.save({
        "actor": policy.actor.state_dict(),
        "critic": policy.critic.state_dict(),
        "obs_dim": obs_dim,
        "act_dim": act_dim
    }, "models/mappo.pt")
    print("Saved model -> models/mappo.pt")

if __name__ == '__main__':
    main()
