"""
<<<<<<< HEAD
Dataset Preparation Module for HFT MARL
========================================

This module prepares the final datasets for training, validation, and testing.
It handles:
- Temporal splitting of data
- Feature scaling and normalization
- Sequence creation for temporal models
- Tensor conversion for PyTorch/JAX
=======
Dataset Preparation Module
===========================

Converts engineered features into training-ready tensors with time-series format
compatible with EnhancedCTDEHFTEnv.

Expected output format:
- X: [N, T, F] - N samples, T timesteps history, F features
- y: [N] - Target values (optional, for supervised tasks)
- ts: [N] - Timestamps for each sample
>>>>>>> origin/master
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
<<<<<<< HEAD
import json
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_by_date(df: pd.DataFrame, splits: Dict[str, Dict[str, str]]) -> Dict[str, pd.DataFrame]:
    """
    Split data by date ranges
    
    Args:
        df: Input DataFrame with 'ts' timestamp column
        splits: Dictionary of split definitions with start/end dates
        
    Returns:
        Dictionary of split name -> DataFrame
    """
    parts = {}
    
    for name, date_range in splits.items():
        start = pd.Timestamp(date_range['start'])
        end = pd.Timestamp(date_range['end'])
=======


def split_by_date(df, splits):
    """Split data by date ranges"""
    parts = {}
    
    for name, rng in splits.items():
        start = pd.Timestamp(rng['start'])
        end = pd.Timestamp(rng['end'])
>>>>>>> origin/master
        
        mask = (df['ts'] >= start) & (df['ts'] <= end)
        parts[name] = df.loc[mask].reset_index(drop=True)
        
<<<<<<< HEAD
        logger.info(f"Split '{name}': {len(parts[name])} rows ({start} to {end})")
=======
        print(f"  {name:>5}: {len(parts[name]):>6} rows ({start.date()} to {end.date()})")
>>>>>>> origin/master
    
    return parts


<<<<<<< HEAD
def create_scaler(scaling_method: str = 'robust'):
    """
    Create appropriate scaler based on method
    
    Args:
        scaling_method: Type of scaler ('robust', 'standard', 'minmax')
        
    Returns:
        Scaler object
    """
    if scaling_method == 'robust':
        return RobustScaler()
    elif scaling_method == 'standard':
        return StandardScaler()
    elif scaling_method == 'minmax':
        return MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {scaling_method}")


def fit_scaler(df: pd.DataFrame, feature_cols: List[str], config: Dict[str, Any]) -> RobustScaler:
    """
    Fit scaler on training data
    
    Args:
        df: Training DataFrame
        feature_cols: List of feature columns to scale
        config: Configuration dictionary
        
    Returns:
        Fitted scaler
    """
    scaling_method = config.get('scaling', {}).get('method', 'robust')
    scaler = create_scaler(scaling_method)
    
    # Select numeric features
    X = df[feature_cols].values
    
    # Fit scaler
    scaler.fit(X)
    
    logger.info(f"Fitted {scaling_method} scaler on {len(df)} samples")
    
    return scaler


def scale_features(df: pd.DataFrame, feature_cols: List[str], scaler) -> pd.DataFrame:
    """
    Scale features using fitted scaler
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature columns to scale
        scaler: Fitted scaler object
        
    Returns:
        DataFrame with scaled features
    """
    df = df.copy()
    
    # Scale features
    X_scaled = scaler.transform(df[feature_cols].values)
    
    # Update DataFrame
    for i, col in enumerate(feature_cols):
        df[col] = X_scaled[:, i]
    
    return df


def save_scaler(scaler, output_path: Path, feature_cols: List[str]):
    """
    Save scaler parameters to JSON
    
    Args:
        scaler: Fitted scaler object
        output_path: Path to save scaler
        feature_cols: List of feature column names
    """
    if isinstance(scaler, RobustScaler):
        scaler_dict = {
            'type': 'robust',
            'center': scaler.center_.tolist(),
            'scale': scaler.scale_.tolist(),
            'feature_names': feature_cols
        }
    elif isinstance(scaler, StandardScaler):
        scaler_dict = {
            'type': 'standard',
            'mean': scaler.mean_.tolist(),
            'std': scaler.scale_.tolist(),
            'feature_names': feature_cols
        }
    elif isinstance(scaler, MinMaxScaler):
        scaler_dict = {
            'type': 'minmax',
            'min': scaler.min_.tolist(),
            'scale': scaler.scale_.tolist(),
            'feature_names': feature_cols
        }
    else:
        raise ValueError(f"Unknown scaler type: {type(scaler)}")
    
    with open(output_path, 'w') as f:
        json.dump(scaler_dict, f, indent=2)
    
    logger.info(f"Saved scaler to: {output_path}")


def create_sequences(df: pd.DataFrame, feature_cols: List[str], 
                    history_T: int = 20) -> Dict[str, np.ndarray]:
    """
    Create sequences for temporal modeling
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature columns
        history_T: Length of historical window
        
    Returns:
        Dictionary with tensors
    """
    # Get feature values
    X_raw = df[feature_cols].values.astype(np.float32)
    
    # Create sequences with history
    N = len(X_raw)
    F = len(feature_cols)
    
    # Initialize output array [N, T, F]
    X = np.zeros((N, history_T, F), dtype=np.float32)
    
    for i in range(N):
        # Get historical window
        start_idx = max(0, i - history_T + 1)
        history = X_raw[start_idx:i+1]
        
        # Pad if needed
        if len(history) < history_T:
            padding = np.zeros((history_T - len(history), F), dtype=np.float32)
            history = np.vstack([padding, history])
        
        X[i] = history
    
    # Get timestamps
    timestamps = df['ts'].values
    
    return {
        'X': X,
        'ts': timestamps,
        'symbols': df['symbol'].values
    }


def create_target_variables(df: pd.DataFrame, target_windows: List[int] = [1, 5, 10]) -> Dict[str, np.ndarray]:
    """
    Create target variables for prediction
    
    Args:
        df: Input DataFrame with 'mid_price'
        target_windows: List of forward windows for targets
        
    Returns:
        Dictionary of target arrays
    """
    targets = {}
    
    for window in target_windows:
        # Forward returns as targets
        future_price = df['mid_price'].shift(-window)
        returns = (future_price - df['mid_price']) / df['mid_price']
        targets[f'target_{window}'] = returns.fillna(0).values.astype(np.float32)
    
    return targets


def to_tensors(df: pd.DataFrame, feature_cols: List[str], 
               history_T: int = 20) -> Dict[str, np.ndarray]:
    """
    Convert DataFrame to tensor format
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature columns
        history_T: Length of historical window
        
    Returns:
        Dictionary of tensors
    """
    # Create sequences
    tensors = create_sequences(df, feature_cols, history_T)
    
    # Add target variables
    targets = create_target_variables(df)
    tensors.update(targets)
    
    return tensors


def main():
    """Main entry point for dataset preparation"""
    parser = argparse.ArgumentParser(description="Prepare final datasets for training")
    parser.add_argument('--config', required=True, help="Path to data configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get paths
    features_dir = Path(config['paths']['features'])
    
    logger.info("=" * 60)
    logger.info("DATASET PREPARATION PIPELINE")
    logger.info("=" * 60)
    
    # Load features
    features_file = features_dir / "features.parquet"
    logger.info(f"Loading features from: {features_file}")
    df = pd.read_parquet(features_file)
    logger.info(f"Loaded {len(df)} rows")
    
    # Get feature columns (exclude metadata)
    metadata_cols = ['ts', 'symbol']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    logger.info(f"Number of features: {len(feature_cols)}")
    
    # Split data by date
    splits = config.get('splits', {})
    data_splits = split_by_date(df, splits)
    
    # Fit scaler on dev/train split
    fit_split = config.get('scaler', {}).get('fit_on', 'dev')
    logger.info(f"Fitting scaler on '{fit_split}' split...")
    scaler = fit_scaler(data_splits[fit_split], feature_cols, config)
    
    # Get history window
    history_T = config.get('features', {}).get('history_T', 20)
    
    # Process each split
    for split_name, split_df in data_splits.items():
        logger.info(f"\nProcessing '{split_name}' split...")
        
        # Scale features
        split_df_scaled = scale_features(split_df, feature_cols, scaler)
        
        # Convert to tensors
        tensors = to_tensors(split_df_scaled, feature_cols, history_T)
        
        # Save tensors
        output_file = features_dir / f"{split_name}_tensors.npz"
        np.savez_compressed(output_file, **tensors)
        
        logger.info(f"Saved {split_name} tensors to: {output_file}")
        logger.info(f"  Shape: {tensors['X'].shape} [N, T, F]")
        logger.info(f"  Features: {tensors['X'].shape[2]}")
        logger.info(f"  Samples: {tensors['X'].shape[0]}")
    
    # Save scaler
    scaler_path = features_dir / "scaler.json"
    save_scaler(scaler, scaler_path, feature_cols)
    
    logger.info("=" * 60)
    logger.info("DATASET PREPARATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Splits processed: {list(data_splits.keys())}")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"History window: {history_T}")
    logger.info(f"Scaling method: {config.get('scaling', {}).get('method', 'robust')}")
    logger.info(f"Output directory: {features_dir}")
    logger.info("=" * 60)
=======
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
>>>>>>> origin/master


if __name__ == '__main__':
    main()
