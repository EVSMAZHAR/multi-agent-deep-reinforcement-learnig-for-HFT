"""MADDPG training with replay buffer, targets, and delayed policy updates.
Saves to models/maddpg.pt
"""
import argparse, yaml, os, json
from pathlib import Path
import numpy as np, torch
from src.marl.env_ctde import CTDEHFTEnv
from src.marl.policies.maddpg_full import MADDPG, DDPGCfg, Replay

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    feat_dir = Path(cfg["paths"]["features"])
    env = CTDEHFTEnv(
        dataset_path=str(feat_dir/'dev_tensors.npz'),
        scaler_path=str(feat_dir/'scaler.json'),
        tick_size=0.01, decision_ms=cfg.get("decision_ms",100),
        reward_weights=cfg.get("reward_weights", {"lambda_inv":0.5,"kappa_imp":0.2}),
        episode_len=cfg.get("episode_len", 1000), seed=cfg.get("seed", 123)
    )
    agents = env.agents
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    maddpg = MADDPG(obs_dim, act_dim, agents, DDPGCfg(
        gamma=cfg.get("gamma",0.99), tau=cfg.get("tau",0.005),
        lr_actor=cfg.get("lr_actor",1e-3), lr_critic=cfg.get("lr_critic",1e-3),
        noise_sigma=cfg.get("noise_sigma",0.2), noise_sigma_final=cfg.get("noise_sigma_final",0.05),
        batch_size=cfg.get("batch_size",256), policy_delay=cfg.get("policy_delay",2)
    ), device='cpu')

    replay = Replay(obs_dim, act_dim, agents, capacity=cfg.get("replay_size", 500_000))

    total_steps = cfg.get("total_steps", 100_000)
    warmup = cfg.get("warmup_steps", 10_000)
    start_obs, _ = env.reset()
    obs = start_obs

    for t in range(total_steps):
        # epsilon-like exploration via categorical Gumbel sampling already in policy
        acts = maddpg.select_actions(obs, explore=True)
        next_obs, rewards, terms, truncs, infos = env.step(acts)
        done = any(list(terms.values()))
        # map actions to indices 0..4 (offset +2)
        act_idx = {a: (acts[a] + 2) for a in agents}
        replay.add(obs, act_idx, rewards, next_obs, done)
        obs = next_obs
        if done:
            obs, _ = env.reset()

        if t > warmup and replay.size >= maddpg.cfg.batch_size:
            info = maddpg.train_step(replay)
            if (t+1) % 5000 == 0:
                print(f"[{t+1}/{total_steps}] q_loss={info.get('q_loss'):.4f} pi_loss={info.get('pi_loss', float('nan')):.4f}")

    Path("models").mkdir(exist_ok=True)
    torch.save({ "actors": {a: maddpg.actor[a].state_dict() for a in agents},
                 "critic": maddpg.critic.state_dict(),
                 "obs_dim": obs_dim, "act_dim": act_dim, "agents": agents},
               "models/maddpg.pt")
    print("Saved model -> models/maddpg.pt")

if __name__ == '__main__':
    main()
