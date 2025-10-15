"""
Dataset Preparation Module
===========================

Converts engineered features into training-ready tensors with time-series format
compatible with EnhancedCTDEHFTEnv.

Expected output format:
- X: [N, T, F] - N samples, T timesteps history, F features
- y: [N] - Target values (optional, for supervised tasks)
- ts: [N] - Timestamps for each sample
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np


def split_by_date(df, splits):
    """Split data by date ranges"""
    parts = {}
    
    for name, rng in splits.items():
        start = pd.Timestamp(rng['start'])
        end = pd.Timestamp(rng['end'])
        
        mask = (df['ts'] >= start) & (df['ts'] <= end)
        parts[name] = df.loc[mask].reset_index(drop=True)
        
        print(f"  {name:>5}: {len(parts[name]):>6} rows ({start.date()} to {end.date()})")
    
    return parts


def create_time_series_tensors(df, feature_cols, history_T=20):
    """
    Create time-series tensors with sliding window
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        history_T: Number of timesteps in history window
    
    Returns:
        Dictionary with X (features), y (targets), ts (timestamps)
    """
    # Extract feature values
    feature_data = df[feature_cols].astype(np.float32).values  # [total_rows, F]
    
    # Extract timestamps
    timestamps = df['ts'].values
    
    # Create sliding windows for time-series
    N = len(df) - history_T + 1  # Number of samples
    F = len(feature_cols)  # Number of features
    
    if N <= 0:
        raise ValueError(f"Not enough data for history_T={history_T}. Need at least {history_T} rows.")
    
    # Initialize arrays
    X = np.zeros((N, history_T, F), dtype=np.float32)
    ts = np.zeros(N, dtype='datetime64[ns]')
    
    # Fill with sliding windows
    for i in range(N):
        X[i] = feature_data[i:i+history_T]  # Window of T timesteps
        ts[i] = timestamps[i+history_T-1]  # Timestamp of the last point in window
    
    # Create target (mid-price return for next period, if available)
    if 'returns' in df.columns and len(df) > history_T:
        y = df['returns'].iloc[history_T:].values.astype(np.float32)
        y = y[:N]  # Ensure same length
    else:
        y = np.zeros(N, dtype=np.float32)
    
    return {
        "X": X,
        "y": y,
        "ts": ts
    }


def main():
    ap = argparse.ArgumentParser(description="Prepare training-ready datasets from features")
    ap.add_argument('--config', required=True, help='Path to data config file')
    args = ap.parse_args()
    
    # Load configuration
    cfg = yaml.safe_load(open(args.config))
    
    # Get paths
    feat_dir = Path(cfg["paths"]["features"])
    features_file = feat_dir / "features.parquet"
    
    if not features_file.exists():
        raise SystemExit(f"❌ Features file not found: {features_file}. Run feature engineering first.")
    
    print(f"Loading features from: {features_file}")
    df = pd.read_parquet(features_file)
    
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['ts'].min()} to {df['ts'].max()}")
    
    # Define feature columns to use for training
    # These are the scaled features compatible with the environment
    feature_cols = [
        'best_bid', 'best_ask', 'spread', 'imbalance', 
        'microprice', 'mid_price', 'returns', 'volatility',
        'bid_value', 'ask_value', 'bid_qty_1', 'ask_qty_1'
    ]
    
    # Filter to available columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
    
    # Get history length from config
    history_T = cfg.get('history_T', 20)
    print(f"History window: {history_T} timesteps")
    
    # Split data by date ranges
    print("\n📅 Splitting data by date ranges...")
    parts = split_by_date(df, cfg["splits"])
    
    # Convert each split to tensors
    print(f"\n🔄 Creating time-series tensors with history={history_T}...")
    
    for name, part in parts.items():
        if len(part) < history_T:
            print(f"  ⚠️  Skipping {name}: not enough data (need {history_T}, have {len(part)})")
            continue
        
        # Create tensors
        tensors = create_time_series_tensors(part, feature_cols, history_T=history_T)
        
        # Save tensors
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
