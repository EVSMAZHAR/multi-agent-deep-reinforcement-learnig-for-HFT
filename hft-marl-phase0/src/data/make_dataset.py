import argparse, yaml
from pathlib import Path
import pandas as pd, numpy as np
def split_by_date(df, splits):
    parts = {}
    for name, rng in splits.items():
        mask = (df['ts'] >= pd.Timestamp(rng['start'])) & (df['ts'] <= pd.Timestamp(rng['end']))
        parts[name] = df.loc[mask].reset_index(drop=True)
    return parts
def to_tensors(df):
    X = df[['best_bid','best_ask','spread','imbalance','microprice']].astype(np.float32).values
    return {"X": X}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    feat_dir = Path(cfg["paths"]["features"])
    df = pd.read_parquet(feat_dir/"features.parquet")
    parts = split_by_date(df, cfg["splits"])
    for name, part in parts.items():
        tensors = to_tensors(part)
        out = feat_dir / f"{name}_tensors.npz"
        np.savez_compressed(out, **tensors)
        print(f"Wrote tensors -> {out} (rows={len(part)})")
if __name__ == '__main__':
    main()
