"""
Feature Engineering Module
===========================

Builds trading features from raw market snapshots including:
- Spread features
- Order book imbalance
- Microprice
- Additional technical indicators

These features are compatible with the EnhancedCTDEHFTEnv environment.
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import json


def add_basic_features(df):
    """Add basic market microstructure features"""
    df = df.copy()
    
    # Spread (in price units)
    df['spread'] = (df['best_ask'] - df['best_bid']).astype(np.float32)
    
    # Order book imbalance
    denom = (df['bid_qty_1'] + df['ask_qty_1']).replace(0, np.nan)
    df['imbalance'] = ((df['bid_qty_1'] - df['ask_qty_1']) / denom).fillna(0).astype(np.float32)
    
    # Microprice (volume-weighted mid-price)
    df['microprice'] = (
        (df['best_ask'] * df['bid_qty_1'] + df['best_bid'] * df['ask_qty_1']) / denom
    ).fillna((df['best_ask'] + df['best_bid']) / 2).astype(np.float32)
    
    return df


def add_technical_features(df):
    """Add technical indicators for enhanced learning"""
    df = df.copy()
    
    # Mid-price
    df['mid_price'] = ((df['best_bid'] + df['best_ask']) / 2).astype(np.float32)
    
    # Returns (1-period)
    df['returns'] = df.groupby('symbol')['mid_price'].pct_change().fillna(0).astype(np.float32)
    
    # Volatility (rolling standard deviation of returns)
    df['volatility'] = df.groupby('symbol')['returns'].rolling(window=20, min_periods=1).std().reset_index(0, drop=True).fillna(0).astype(np.float32)
    
    # Volume-weighted average price components
    df['bid_value'] = (df['best_bid'] * df['bid_qty_1']).astype(np.float32)
    df['ask_value'] = (df['best_ask'] * df['ask_qty_1']).astype(np.float32)
    
    return df


def add_time_features(df):
    """Add time-based features"""
    df = df.copy()
    
    # Time since start (in seconds, normalized)
    df['time_idx'] = df.groupby('symbol').cumcount().astype(np.float32)
    
    return df


def compute_scaler_params(df, feature_cols, method='robust'):
    """
    Compute scaling parameters for features
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        method: 'robust' (median/IQR) or 'standard' (mean/std)
    
    Returns:
        Dictionary with scaling parameters
    """
    scaler_params = {}
    
    for col in feature_cols:
        if method == 'robust':
            median = float(df[col].median())
            q75, q25 = df[col].quantile([0.75, 0.25])
            iqr = float(q75 - q25)
            iqr = iqr if iqr > 0 else 1.0
            scaler_params[col] = {'median': median, 'iqr': iqr}
        else:  # standard
            mean = float(df[col].mean())
            std = float(df[col].std())
            std = std if std > 0 else 1.0
            scaler_params[col] = {'mean': mean, 'std': std}
    
    return scaler_params


def apply_scaling(df, scaler_params, feature_cols, method='robust'):
    """Apply scaling to features"""
    df = df.copy()
    
    for col in feature_cols:
        if col in scaler_params:
            if method == 'robust':
                df[col] = (df[col] - scaler_params[col]['median']) / scaler_params[col]['iqr']
            else:  # standard
                df[col] = (df[col] - scaler_params[col]['mean']) / scaler_params[col]['std']
    
    return df


def main():
    ap = argparse.ArgumentParser(description="Build features from market snapshots")
    ap.add_argument('--config', required=True, help='Path to features config file')
    args = ap.parse_args()
    
    # Load configuration
    cfg = yaml.safe_load(open(args.config))
    
    # Get paths
    interim = Path(cfg["paths"]["interim"])
    feat_dir = Path(cfg["paths"]["features"])
    feat_dir.mkdir(parents=True, exist_ok=True)
    
    # Load consolidated snapshots
    snapshots_file = interim / "snapshots.parquet"
    if not snapshots_file.exists():
        raise SystemExit(f"❌ Snapshots file not found: {snapshots_file}. Run data ingestion first.")
    
    print(f"Loading snapshots from: {snapshots_file}")
    df = pd.read_parquet(snapshots_file)
    
    print(f"Initial data: {len(df)} rows")
    
    # Build features
    print("\n🔨 Building features...")
    df = add_basic_features(df)
    print("  ✓ Basic features: spread, imbalance, microprice")
    
    df = add_technical_features(df)
    print("  ✓ Technical features: returns, volatility, price components")
    
    df = add_time_features(df)
    print("  ✓ Time features: time_idx")
    
    # Define feature columns (excluding raw columns and identifiers)
    feature_cols = [
        'best_bid', 'best_ask', 'bid_qty_1', 'ask_qty_1',
        'spread', 'imbalance', 'microprice', 'mid_price',
        'returns', 'volatility', 'bid_value', 'ask_value'
    ]
    
    print(f"\n📊 Feature columns ({len(feature_cols)}): {feature_cols}")
    
    # Compute scaling parameters on full dataset (or train split if specified)
    scaling_method = cfg.get('scaler', {}).get('type', 'robust')
    print(f"\n🔧 Computing {scaling_method} scaler parameters...")
    
    scaler_params = compute_scaler_params(df, feature_cols, method=scaling_method)
    
    # Save scaler parameters
    scaler_file = feat_dir / "scaler.json"
    with open(scaler_file, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print(f"  ✓ Scaler saved: {scaler_file}")
    
    # Apply scaling
    df = apply_scaling(df, scaler_params, feature_cols, method=scaling_method)
    print("  ✓ Features scaled")
    
    # Save features
    output_file = feat_dir / "features.parquet"
    df.to_parquet(output_file, index=False)
    
    print(f"\n✅ Features saved → {output_file}")
    print(f"   Total features: {len(feature_cols)}")
    print(f"   Total rows: {len(df)}")
    print(f"   Feature columns: {feature_cols}")


if __name__ == '__main__':
    main()
