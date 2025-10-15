"""
Dataset Preparation Module
==========================

Converts engineered features into training-ready tensors with time-series format
compatible with EnhancedCTDEHFTEnv.

Expected output format:
- X: [N, T, F] - N samples, T timesteps history, F features
- y: [N] - Target values (optional, for supervised tasks)
- ts: [N] - Timestamps for each sample
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def split_by_date(df: pd.DataFrame, splits: dict) -> dict:
    """Split data by date ranges with basic logging."""
    parts: dict[str, pd.DataFrame] = {}
    if 'ts' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['ts']):
        df = df.copy()
        df['ts'] = pd.to_datetime(df['ts'])
    for name, rng in splits.items():
        start = pd.Timestamp(rng['start'])
        end = pd.Timestamp(rng['end'])
        mask = (df['ts'] >= start) & (df['ts'] <= end)
        parts[name] = df.loc[mask].reset_index(drop=True)
        print(f"  {name:>5}: {len(parts[name]):>6} rows ({start.date()} to {end.date()})")
    return parts


def create_sequences(df: pd.DataFrame, history_T: int, feature_cols: list) -> np.ndarray:
    """Create sequences for temporal modeling with window length history_T.

    Produces N = len(df) - history_T sequences of shape [history_T, F].
    """
    features = df[feature_cols].values.astype(np.float32)
    n_samples = features.shape[0]
    if n_samples <= history_T:
        return np.zeros((0, history_T, len(feature_cols)), dtype=np.float32)

    sequences = []
    for i in range(history_T, n_samples):
        sequences.append(features[i - history_T:i, :])
    return np.asarray(sequences, dtype=np.float32)


def to_tensors(df: pd.DataFrame, history_T: int = 20) -> dict:
    """Convert DataFrame to tensor dict using a default feature set."""
    feature_cols = [
        'best_bid', 'best_ask', 'spread', 'imbalance', 'microprice',
        'bid_qty_1', 'ask_qty_1', 'mid_price'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = create_sequences(df, history_T, feature_cols)

    if 'returns' in df.columns and len(df) > history_T:
        y = df['returns'].iloc[history_T:].values.astype(np.float32)
        y = y[: len(X)]
    else:
        y = np.zeros(len(X), dtype=np.float32)

    if 'ts' in df.columns and len(df) > history_T:
        ts = df['ts'].iloc[history_T:].values[: len(X)]
    else:
        ts = np.arange(len(X), dtype=np.int64)

    return {"X": X, "y": y, "ts": ts}


def create_time_series_tensors(df: pd.DataFrame, feature_cols: list, history_T: int = 20) -> dict:
    """Create time-series tensors with sliding window, N = len(df) - history_T."""
    feature_data = df[feature_cols].astype(np.float32).values
    timestamps = df['ts'].values if 'ts' in df.columns else np.arange(len(df))

    N = len(df) - history_T
    F = len(feature_cols)
    if N <= 0:
        raise ValueError(f"Not enough data for history_T={history_T}. Need > {history_T} rows.")

    X = np.zeros((N, history_T, F), dtype=np.float32)
    ts = np.zeros(N, dtype='datetime64[ns]') if np.issubdtype(timestamps.dtype, np.datetime64) else np.zeros(N, dtype=np.int64)

    for i in range(N):
        X[i] = feature_data[i:i + history_T]
        ts[i] = timestamps[i + history_T - 1]

    if 'returns' in df.columns and len(df) > history_T:
        y = df['returns'].iloc[history_T:].values.astype(np.float32)
        y = y[:N]
    else:
        y = np.zeros(N, dtype=np.float32)

    return {"X": X, "y": y, "ts": ts}


def main():
    ap = argparse.ArgumentParser(description="Prepare training-ready datasets from features")
    ap.add_argument('--config', required=True, help='Path to data config file')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    feat_dir = Path(cfg["paths"]["features"])
    features_file = feat_dir / "features.parquet"

    if not features_file.exists():
        raise SystemExit(f"❌ Features file not found: {features_file}. Run feature engineering first.")

    print(f"Loading features from: {features_file}")
    df = pd.read_parquet(features_file)

    print(f"Total rows: {len(df)}")
    if 'ts' in df.columns:
        print(f"Date range: {df['ts'].min()} to {df['ts'].max()}")

    feature_cols = [
        'best_bid', 'best_ask', 'spread', 'imbalance',
        'microprice', 'mid_price', 'returns', 'volatility',
        'bid_value', 'ask_value', 'bid_qty_1', 'ask_qty_1'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

    history_T = cfg.get('history_T', 20)
    print(f"History window: {history_T} timesteps")

    print("\n📅 Splitting data by date ranges...")
    parts = split_by_date(df, cfg["splits"]) 

    print(f"\n🔄 Creating time-series tensors with history={history_T}...")
    for name, part in parts.items():
        if len(part) < history_T:
            print(f"  ⚠️  Skipping {name}: not enough data (need {history_T}, have {len(part)})")
            continue
        tensors = create_time_series_tensors(part, feature_cols, history_T=history_T)
        out_file = feat_dir / f"{name}_tensors.npz"
        np.savez_compressed(out_file, **tensors)
        print(f"  ✓ {name:>5} → {out_file.name}")
        print(f"      Shape: X={tensors['X'].shape}, y={tensors['y'].shape}")

    print(f"\n✅ Dataset preparation completed!")
    print(f"   Output directory: {feat_dir}")
    print(f"   Feature dimension: {len(feature_cols)}")
    print(f"   History length: {history_T}")


if __name__ == '__main__':
    main()
