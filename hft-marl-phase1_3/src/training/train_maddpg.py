import argparse, yaml
from src.marl.env_ctde import CTDEHFTEnv
from src.marl.policies.maddpg import Actor, Critic
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    env = CTDEHFTEnv(config={})
    print("MADDPG skeleton initialised. Add replay/targets/TD3-style updates next.")
if __name__ == '__main__':
    main()
