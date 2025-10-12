"""Run Avellaneda–Stoikov baseline in the env and compute CI metrics."""
import argparse, yaml, numpy as np
from pathlib import Path
from src.marl.env_ctde import CTDEHFTEnv
from src.baselines.avellaneda_stoikov import AvellanedaStoikovMM
from src.reports.metrics import MetricsSuite
from src.reports.ci import bootstrap_ci

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
        episode_len=cfg.get("episode_len", 1000), seed=cfg.get("seed", 321)
    )

    mm = AvellanedaStoikovMM(gamma=0.1, k=1.5, sigma=0.02, tick=0.01, T=60.0)
    obs, _ = env.reset()
    agents = env.agents
    pnl_rec = {a: [] for a in agents}
    reward_rec = {a: [] for a in agents}
    inv = 0.0
    for t in range(env.episode_len):
        # Simple mid proxy from env state
        # We don't observe true mid; use env.last_mid (internal); here we approximate with 100 as anchor
        mid = env.last_mid
        off_bid, off_ask = mm.quotes(mid, inv, t)
        actions = {'maker_bid': off_bid+2, 'maker_ask': off_ask+2}  # env expects 0..4; we'll convert back internally
        obs, rewards, terms, truncs, infos = env.step(actions)
        for a in agents:
            reward_rec[a].append(float(rewards[a]))
        inv = env.inv['maker_bid'] + env.inv['maker_ask']  # net inventory proxy
        if any(list(terms.values())):
            break

    # Metrics
    ms = MetricsSuite()
    per_agent = {a: ms.aggregate(np.array(reward_rec[a], dtype=np.float32)) for a in agents}
    print("Avellaneda–Stoikov baseline metrics:")
    for a in agents:
        print(a, per_agent[a])
    # Bootstrap CI for mean reward of maker_bid
    arr = np.array(reward_rec['maker_bid'], dtype=np.float32)
    lo, hi = bootstrap_ci(arr, fn=np.mean, alpha=0.05, iters=1000)
    print(f"maker_bid mean reward 95% CI: [{lo:.6f}, {hi:.6f}]")

if __name__ == '__main__':
    main()
