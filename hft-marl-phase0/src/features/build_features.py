import argparse, yaml
from pathlib import Path
import pandas as pd, numpy as np
def add_basic_features(df):
    df = df.copy()
    df['spread'] = (df['best_ask'] - df['best_bid']).astype(np.float32)
    denom = (df['bid_qty_1'] + df['ask_qty_1']).replace(0, np.nan)
    df['imbalance'] = ((df['bid_qty_1'] - df['ask_qty_1'])/denom).fillna(0).astype(np.float32)
    df['microprice'] = ((df['best_ask']*df['bid_qty_1'] + df['best_bid']*df['ask_qty_1'])/denom)                       .fillna((df['best_ask']+df['best_bid'])/2).astype(np.float32)
    return df
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    interim = Path(cfg["paths"]["interim"])
    feat_dir = Path(cfg["paths"]["features"]); feat_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(interim/"snapshots.parquet")
    df = add_basic_features(df)
    df.to_parquet(feat_dir/"features.parquet", index=False)
    print(f"Features saved -> {feat_dir/'features.parquet'}")
if __name__ == '__main__':
    main()
