"""
Feature Engineering Module
==========================

Builds trading features from raw market snapshots including:
- Spread features
- Order book imbalance
- Microprice
- Additional technical indicators

These features are compatible with the EnhancedCTDEHFTEnv environment.
"""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic market microstructure features with robust fallbacks."""
    df = df.copy()

    # Fallback names
    if 'best_bid' not in df.columns and 'bid_price' in df.columns:
        df['best_bid'] = df['bid_price']
    if 'best_ask' not in df.columns and 'ask_price' in df.columns:
        df['best_ask'] = df['ask_price']

    # Spread
    if 'best_bid' in df.columns and 'best_ask' in df.columns:
        df['spread'] = (df['best_ask'] - df['best_bid']).astype(np.float32)
    else:
        df['spread'] = 0.0

    # Mid price
    if 'best_bid' in df.columns and 'best_ask' in df.columns:
        df['mid_price'] = ((df['best_bid'] + df['best_ask']) / 2.0).astype(np.float32)
    else:
        df['mid_price'] = 100.0

    # Quantities fallbacks
    if 'bid_qty_1' not in df.columns:
        if 'bid_volume' in df.columns:
            df['bid_qty_1'] = df['bid_volume']
        elif 'bid_size' in df.columns:
            df['bid_qty_1'] = df['bid_size']
        else:
            df['bid_qty_1'] = 100.0
    if 'ask_qty_1' not in df.columns:
        if 'ask_volume' in df.columns:
            df['ask_qty_1'] = df['ask_volume']
        elif 'ask_size' in df.columns:
            df['ask_qty_1'] = df['ask_size']
        else:
            df['ask_qty_1'] = 100.0

    # Imbalance and microprice
    denom = (df['bid_qty_1'] + df['ask_qty_1']).replace(0, np.nan)
    df['imbalance'] = ((df['bid_qty_1'] - df['ask_qty_1']) / denom).fillna(0).astype(np.float32)
    df['microprice'] = (
        (df['best_ask'] * df['bid_qty_1'] + df['best_bid'] * df['ask_qty_1']) / denom
    ).fillna((df['best_ask'] + df['best_bid']) / 2).astype(np.float32)

    # Price components
    df['bid_value'] = (df['best_bid'] * df['bid_qty_1']).astype(np.float32)
    df['ask_value'] = (df['best_ask'] * df['ask_qty_1']).astype(np.float32)

    return df


def add_technical_features(df: pd.DataFrame, windows: dict | None = None) -> pd.DataFrame:
    """Add technical indicators and derived features."""
    if windows is None:
        windows = {'fast': 10, 'slow': 30}

    df = df.copy()

    # Returns and rolling volatilities
    if 'mid_price' in df.columns:
        if 'symbol' in df.columns:
            df['returns'] = df.groupby('symbol')['mid_price'].pct_change().fillna(0).astype(np.float32)
        else:
            df['returns'] = df['mid_price'].pct_change().fillna(0).astype(np.float32)
        for name, window in windows.items():
            df[f'volatility_{name}'] = (
                df['returns'].rolling(window=window, min_periods=1).std().fillna(0).astype(np.float32)
            )

    # Rolling averages
    if 'spread' in df.columns:
        for name, window in windows.items():
            df[f'spread_ma_{name}'] = df['spread'].rolling(window=window, min_periods=1).mean().fillna(0).astype(np.float32)
    if 'imbalance' in df.columns:
        for name, window in windows.items():
            df[f'imbalance_ma_{name}'] = df['imbalance'].rolling(window=window, min_periods=1).mean().fillna(0).astype(np.float32)

    # Volume MA features
    if 'bid_qty_1' in df.columns and 'ask_qty_1' in df.columns:
        df['total_volume'] = (df['bid_qty_1'] + df['ask_qty_1']).astype(np.float32)
        for name, window in windows.items():
            df[f'volume_ma_{name}'] = df['total_volume'].rolling(window=window, min_periods=1).mean().fillna(0).astype(np.float32)

    return df


def compute_scaler_params(df: pd.DataFrame, feature_cols: list[str], method: str = 'robust') -> dict:
    """Compute scaling parameters for features."""
    scaler_params: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        if method == 'robust':
            median = float(df[col].median())
            q75, q25 = df[col].quantile([0.75, 0.25])
            iqr = float(q75 - q25) or 1.0
            scaler_params[col] = {'median': median, 'iqr': iqr}
        else:
            mean = float(df[col].mean())
            std = float(df[col].std()) or 1.0
            scaler_params[col] = {'mean': mean, 'std': std}
    return scaler_params


def apply_scaling(df: pd.DataFrame, scaler_params: dict, feature_cols: list[str], method: str = 'robust') -> pd.DataFrame:
    """Apply scaling to features and return DataFrame."""
    df = df.copy()
    for col in feature_cols:
        if col not in scaler_params:
            continue
        if method == 'robust':
            df[col] = (df[col] - scaler_params[col]['median']) / scaler_params[col]['iqr']
        else:
            df[col] = (df[col] - scaler_params[col]['mean']) / scaler_params[col]['std']
    return df


def main():
    ap = argparse.ArgumentParser(description="Build features from market snapshots")
    ap.add_argument('--config', required=True, help='Path to features config file')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    interim = Path(cfg["paths"]["interim"]) 
    feat_dir = Path(cfg["paths"]["features"]) 
    feat_dir.mkdir(parents=True, exist_ok=True)

    snapshots_file = interim / "snapshots.parquet"
    if not snapshots_file.exists():
        raise SystemExit(f"❌ Snapshots file not found: {snapshots_file}. Run data ingestion first.")

    print(f"Loading snapshots from: {snapshots_file}")
    df = pd.read_parquet(snapshots_file)
    print(f"Initial data: {len(df)} rows")

    print("\n🔨 Building features...")
    df = add_basic_features(df)
    print("  ✓ Basic features: spread, imbalance, microprice, mid_price")
    df = add_technical_features(df)
    print("  ✓ Technical features: returns, volatilities, rolling averages, volumes")

    feature_cols = [
        'best_bid', 'best_ask', 'bid_qty_1', 'ask_qty_1',
        'spread', 'imbalance', 'microprice', 'mid_price',
        'returns', 'volatility', 'bid_value', 'ask_value'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"\n📊 Feature columns ({len(feature_cols)}): {feature_cols}")

    scaling_method = cfg.get('scaler', {}).get('type', 'robust')
    print(f"\n🔧 Computing {scaling_method} scaler parameters...")
    scaler_params = compute_scaler_params(df, feature_cols, method=scaling_method)
    scaler_file = feat_dir / "scaler.json"
    with open(scaler_file, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print(f"  ✓ Scaler saved: {scaler_file}")

    df = apply_scaling(df, scaler_params, feature_cols, method=scaling_method)
    print("  ✓ Features scaled")

    output_file = feat_dir / "features.parquet"
    df.to_parquet(output_file, index=False)
    print(f"\n✅ Features saved → {output_file}")
    print(f"   Total features: {len(feature_cols)}")
    print(f"   Total rows: {len(df)}")
    print(f"   Feature columns: {feature_cols}")


if __name__ == '__main__':
    main()
