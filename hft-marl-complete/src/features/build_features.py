"""
Feature Engineering Module
===========================

Computes market features from raw orderbook snapshots.
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
    df['microprice'] = (
        (df['best_ask'] * df['bid_qty_1'] + df['best_bid'] * df['ask_qty_1']) / denom
    ).fillna((df['best_ask'] + df['best_bid']) / 2).astype(np.float32)
    
    logger.info(f"Added features: spread, imbalance, microprice, mid_price")
    
    return df


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
    
    return df


def main():
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


if __name__ == '__main__':
    main()
