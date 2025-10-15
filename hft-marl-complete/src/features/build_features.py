"""
Feature Engineering Module
===========================

Builds trading features from raw market snapshots including:
- Spread features
- Order book imbalance
- Microprice
- Additional technical indicators

These features are compatible with the EnhancedCTDEHFTEnv environment.
Adapted from hft-marl-phase0 for compatibility with enhanced environment.
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic market microstructure features.
    
    Features computed:
    - spread: bid-ask spread
    - imbalance: orderbook imbalance
    - microprice: volume-weighted mid price
    - mid_price: simple mid price
    
    Args:
        df: DataFrame with orderbook data
        
    Returns:
        DataFrame with added features
    """
    df = df.copy()
    logger.info("Computing basic features...")
    
    # Ensure required columns exist
    required_cols = ['best_bid', 'best_ask']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")
        # Try alternative column names
        if 'bid_price' in df.columns:
            df['best_bid'] = df['bid_price']
        if 'ask_price' in df.columns:
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
        df['mid_price'] = 100.0  # Default
    
    # Quantities
    if 'bid_qty_1' not in df.columns:
        if 'bid_volume' in df.columns:
            df['bid_qty_1'] = df['bid_volume']
        elif 'bid_size' in df.columns:
            df['bid_qty_1'] = df['bid_size']
        else:
            df['bid_qty_1'] = 100.0  # Default
    
    if 'ask_qty_1' not in df.columns:
        if 'ask_volume' in df.columns:
            df['ask_qty_1'] = df['ask_volume']
        elif 'ask_size' in df.columns:
            df['ask_qty_1'] = df['ask_size']
        else:
            df['ask_qty_1'] = 100.0  # Default
    
    # Imbalance
    denom = (df['bid_qty_1'] + df['ask_qty_1']).replace(0, np.nan)
    df['imbalance'] = ((df['bid_qty_1'] - df['ask_qty_1']) / denom).fillna(0).astype(np.float32)
    
    # Microprice (volume-weighted mid)
>>>>>>> cursor/integrate-data-collection-and-feature-engineering-b52d
    df['microprice'] = (
        (df['best_ask'] * df['bid_qty_1'] + df['best_bid'] * df['ask_qty_1']) / denom
    ).fillna((df['best_ask'] + df['best_bid']) / 2).astype(np.float32)
    
<<<<<<< HEAD
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
=======
    logger.info(f"Added features: spread, imbalance, microprice, mid_price")
>>>>>>> cursor/integrate-data-collection-and-feature-engineering-b52d
    
    return df


<<<<<<< HEAD
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
=======
def add_technical_features(df: pd.DataFrame, windows: dict = None) -> pd.DataFrame:
    """
    Add technical indicators and derived features.
    
    Args:
        df: DataFrame with basic features
        windows: Dictionary of window sizes for technical indicators
        
    Returns:
        DataFrame with added technical features
    """
    if windows is None:
        windows = {'fast': 10, 'slow': 30}
    
    logger.info("Computing technical features...")
    
    # Returns
    if 'mid_price' in df.columns:
        df['returns'] = df['mid_price'].pct_change().fillna(0).astype(np.float32)
        
        # Rolling volatility
        for name, window in windows.items():
            col_name = f'volatility_{name}'
            df[col_name] = df['returns'].rolling(window=window, min_periods=1).std().fillna(0).astype(np.float32)
    
    # Rolling averages
    if 'spread' in df.columns:
        for name, window in windows.items():
            col_name = f'spread_ma_{name}'
            df[col_name] = df['spread'].rolling(window=window, min_periods=1).mean().fillna(0).astype(np.float32)
    
    if 'imbalance' in df.columns:
        for name, window in windows.items():
            col_name = f'imbalance_ma_{name}'
            df[col_name] = df['imbalance'].rolling(window=window, min_periods=1).mean().fillna(0).astype(np.float32)
    
    # Volume features
    if 'bid_qty_1' in df.columns and 'ask_qty_1' in df.columns:
        df['total_volume'] = (df['bid_qty_1'] + df['ask_qty_1']).astype(np.float32)
        
        for name, window in windows.items():
            col_name = f'volume_ma_{name}'
            df[col_name] = df['total_volume'].rolling(window=window, min_periods=1).mean().fillna(0).astype(np.float32)
    
    logger.info(f"Added technical features with windows: {windows}")
    
    return df


def compute_scaler(df: pd.DataFrame, feature_cols: list, method: str = 'robust') -> dict:
    """
    Compute scaling parameters for features.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature columns to scale
        method: Scaling method ('robust' or 'standard')
        
    Returns:
        Dictionary with scaling parameters
    """
    logger.info(f"Computing {method} scaler for {len(feature_cols)} features")
    
    scaler = {'method': method, 'features': feature_cols}
    
    if method == 'robust':
        # Robust scaling: median and IQR
        scaler['median'] = {}
        scaler['iqr'] = {}
        
        for col in feature_cols:
            if col in df.columns:
                median = df[col].median()
                q75 = df[col].quantile(0.75)
                q25 = df[col].quantile(0.25)
                iqr = q75 - q25
                
                scaler['median'][col] = float(median)
                scaler['iqr'][col] = float(iqr) if iqr > 0 else 1.0
    
    elif method == 'standard':
        # Standard scaling: mean and std
        scaler['mean'] = {}
        scaler['std'] = {}
        
        for col in feature_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                
                scaler['mean'][col] = float(mean)
                scaler['std'][col] = float(std) if std > 0 else 1.0
    
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    """
    Apply scaling to features.
    
    Args:
        df: DataFrame with features
        scaler: Scaler dictionary
        
    Returns:
        DataFrame with scaled features
    """
    df = df.copy()
    method = scaler['method']
    
    logger.info(f"Applying {method} scaling")
    
    if method == 'robust':
        for col in scaler['features']:
            if col in df.columns:
                median = scaler['median'][col]
                iqr = scaler['iqr'][col]
                df[col] = (df[col] - median) / iqr
    
    elif method == 'standard':
        for col in scaler['features']:
            if col in df.columns:
                mean = scaler['mean'][col]
                std = scaler['std'][col]
                df[col] = (df[col] - mean) / std
>>>>>>> cursor/integrate-data-collection-and-feature-engineering-b52d
    
    return df


def main():
<<<<<<< HEAD
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
=======
    """Main entry point for feature engineering"""
    ap = argparse.ArgumentParser(description="Build features from market data")
    ap.add_argument('--config', required=True, help='Path to features configuration file')
    args = ap.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load interim data
    interim_dir = Path(cfg["paths"]["interim"])
    snapshots_file = interim_dir / "snapshots.parquet"
    
    if not snapshots_file.exists():
        raise FileNotFoundError(f"Snapshots file not found: {snapshots_file}. Run data ingestion first.")
    
    logger.info(f"Loading data from {snapshots_file}")
    df = pd.read_parquet(snapshots_file)
    logger.info(f"Loaded {len(df)} rows")
    
    # Add basic features
    df = add_basic_features(df)
    
    # Add technical features if configured
    windows = cfg.get("windows", None)
    if windows:
        df = add_technical_features(df, windows)
    
    # Define feature columns for scaling
    base_features = ['best_bid', 'best_ask', 'spread', 'imbalance', 'microprice', 
                     'bid_qty_1', 'ask_qty_1', 'mid_price']
    
    # Add technical features to scaling if they exist
    tech_features = [col for col in df.columns if any(
        pattern in col for pattern in ['volatility_', 'ma_', 'volume_']
    )]
    
    feature_cols = [col for col in base_features + tech_features if col in df.columns]
    
    # Compute scaler on training data
    scaling_method = cfg.get("scaler", {}).get("type", "robust")
    scaler = compute_scaler(df, feature_cols, method=scaling_method)
    
    # Save scaler
    features_dir = Path(cfg["paths"]["features"])
    features_dir.mkdir(parents=True, exist_ok=True)
    
    scaler_file = features_dir / "scaler.json"
    with open(scaler_file, 'w') as f:
        json.dump(scaler, f, indent=2)
    logger.info(f"Saved scaler to {scaler_file}")
    
    # Apply scaling
    df = apply_scaler(df, scaler)
    
    # Save features
    features_file = features_dir / "features.parquet"
    df.to_parquet(features_file, index=False)
    logger.info(f"Features saved -> {features_file}")
    
    # Print feature summary
    logger.info("\nFeature Summary:")
    logger.info(f"  Total features: {len(feature_cols)}")
    logger.info(f"  Samples: {len(df)}")
    logger.info(f"  Columns: {list(df.columns)}")
    
    logger.info("Feature engineering completed successfully")
>>>>>>> cursor/integrate-data-collection-and-feature-engineering-b52d


if __name__ == '__main__':
    main()
