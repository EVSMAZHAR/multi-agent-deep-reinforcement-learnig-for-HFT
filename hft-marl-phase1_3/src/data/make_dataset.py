"""Create train/val/test shards with rolling windows and save robust scalers.
Inputs: data/features/features.parquet
Outputs: data/features/{split}_tensors.npz + data/features/scaler.json
"""
import argparse, yaml, json
from pathlib import Path
import pandas as pd, numpy as np

def robust_fit(x):
    med = np.nanmedian(x, axis=0)
    iqr = np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)
    iqr[iqr==0] = 1.0
    return med, iqr

def robust_transform(x, med, iqr):
    return (x - med) / iqr

def split_by_date(df, splits):
    parts = {}
    for name, rng in splits.items():
        mask = (df['ts'] >= pd.Timestamp(rng['start'])) & (df['ts'] <= pd.Timestamp(rng['end']))
        parts[name] = df.loc[mask].reset_index(drop=True)
    return parts

def build_windows(df, T, feature_cols):
    X, idx = [], []
    arr = df[feature_cols].to_numpy(dtype=np.float32)
    n = len(df)
    for t in range(T-1, n):
        X.append(arr[t-T+1:t+1, :])  # [T, F]
        idx.append(t)
    if not X:
        return np.zeros((0,T,len(feature_cols)), dtype=np.float32), np.array([], dtype=int)
    return np.stack(X), np.array(idx, dtype=int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    feat_dir = Path(cfg["paths"]["features"])
    T = int(yaml.safe_load(open("configs/features.yaml"))["history_T"])
    df = pd.read_parquet(feat_dir/"features.parquet")

    feature_cols = ['spread','imbalance','microprice','ofi','rv','trade_intensity','cancel_intensity','queue_proxy']
    parts = split_by_date(df, cfg["splits"])

    # Fit scaler on dev split only
    dev_arr = parts['dev'][feature_cols].to_numpy(dtype=np.float32)
    med, iqr = robust_fit(dev_arr)
    scaler = {'median': med.tolist(), 'iqr': iqr.tolist()}
    (feat_dir/"scaler.json").write_text(json.dumps(scaler), encoding="utf-8")

    for name, part in parts.items():
        # Transform features
        x = part[feature_cols].to_numpy(dtype=np.float32)
        x_scaled = robust_transform(x, med, iqr)
        part_scaled = part.copy()
        part_scaled[feature_cols] = x_scaled

        # Build windows per symbol and then concatenate
        X_list, ts_list = [], []
        for sym, g in part_scaled.groupby('symbol', sort=False):
            X_sym, idx = build_windows(g, T, feature_cols)
            X_list.append(X_sym)
            # also capture midprice at target step for reward proxy
            ts_list.append(g.iloc[idx]['ts'].to_numpy() if len(idx)>0 else np.array([], dtype='datetime64[ns]'))
        X = np.concatenate(X_list, axis=0) if X_list else np.zeros((0,T,len(feature_cols)),dtype=np.float32)
        timestamps = np.concatenate(ts_list, axis=0) if ts_list else np.array([], dtype='datetime64[ns]')

        # Targets for sanity (mid returns next step if available)
        y = np.zeros((X.shape[0],), dtype=np.float32)

        out = feat_dir / f"{name}_tensors.npz"
        np.savez_compressed(out, X=X, y=y, ts=timestamps.astype('datetime64[ns]').astype('int64'))
        print(f"Wrote {name}: X.shape={X.shape} -> {out}")

if __name__ == '__main__':
    main()
