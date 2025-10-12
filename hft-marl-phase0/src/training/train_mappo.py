import argparse, yaml
from src.marl.env_ctde import CTDEHFTEnv
from src.marl.policies.mappo import MAPPOPolicy
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    env = CTDEHFTEnv(config={})
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy = MAPPOPolicy(obs_dim, act_dim)
    print("MAPPO skeleton initialised. Add rollout/GAE/clip training next.")
if __name__ == '__main__':
    main()
