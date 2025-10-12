"""Simple evaluation: roll an episode with greedy actions from current policy (argmax).
Reports mean reward per step per agent.
"""
import argparse, yaml, torch
from pathlib import Path
import numpy as np
from src.marl.env_ctde import CTDEHFTEnv
from src.marl.policies.mappo import Actor, CentralCritic

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
        episode_len=cfg.get("episode_len", 1000), seed=cfg.get("seed", 999)
    )
    ckpt = torch.load("models/mappo.pt", map_location="cpu")
    actor = Actor(env.observation_space.shape[0], env.action_space.n)
    actor.load_state_dict(ckpt["actor"]); actor.eval()

    obs, _ = env.reset()
    agents = sorted(env.agents)
    rewards_rec = {a: [] for a in agents}

    for _ in range(env.episode_len):
        obs_stack = torch.tensor([obs[a] for a in agents], dtype=torch.float32)
        logits = actor(obs_stack)
        act = torch.argmax(logits, dim=-1).numpy()
        actions = {a:int(act[i]) for i,a in enumerate(agents)}
        obs, rewards, terms, truncs, infos = env.step(actions)
        for a in agents:
            rewards_rec[a].append(float(rewards[a]))
        if any(list(terms.values())):
            break

    for a in agents:
        arr = np.array(rewards_rec[a], dtype=np.float32)
        print(f"{a}: steps={len(arr)}, mean_reward={arr.mean():.6f}, std={arr.std():.6f}")

if __name__ == '__main__':
    main()
