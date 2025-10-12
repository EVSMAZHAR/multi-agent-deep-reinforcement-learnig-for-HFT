"""Evaluation with CI across seeds: load trained actor and roll multiple episodes."""
import argparse, yaml, torch, numpy as np
from pathlib import Path
from src.marl.env_ctde import CTDEHFTEnv
from src.marl.policies.mappo import Actor
from src.reports.metrics import MetricsSuite
from src.reports.ci import bootstrap_ci

def run_once(env, actor):
    obs, _ = env.reset()
    agents = env.agents
    reward_rec = {a: [] for a in agents}
    for _ in range(env.episode_len):
        obs_stack = torch.tensor([obs[a] for a in agents], dtype=torch.float32)
        logits = actor(obs_stack)
        act = torch.argmax(logits, dim=-1).numpy()
        actions = {a:int(act[i]) for i,a in enumerate(agents)}
        obs, rewards, terms, truncs, infos = env.step(actions)
        for a in agents:
            reward_rec[a].append(float(rewards[a]))
        if any(list(terms.values())):
            break
    return reward_rec

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

    ms = MetricsSuite()
    seeds = cfg.get("seeds", [100,200,300])
    results = []
    for s in seeds:
        env.seed = s
        r = run_once(env, actor)
        arr = np.array(r['maker_bid'], dtype=np.float32)  # example agent
        results.append(ms.aggregate(arr))

    # Simple CI on mean reward across seeds (bootstrap)
    means = np.array([x['mean'] for x in results], dtype=np.float32)
    lo, hi = bootstrap_ci(means, fn=np.mean, alpha=0.05, iters=2000)
    print("MAPPO mean reward across seeds:", float(means.mean()))
    print("95% CI (bootstrap):", float(lo), float(hi))

if __name__ == '__main__':
    main()
