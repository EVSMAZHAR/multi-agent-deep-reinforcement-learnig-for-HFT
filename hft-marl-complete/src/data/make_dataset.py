"""
Dataset Creation Module
=======================

Creates train/val/test splits and prepares tensors for training.
Adapted from hft-marl-phase0 for compatibility with enhanced environment.
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_by_date(df: pd.DataFrame, splits: dict) -> dict:
    """
    Split data by date ranges.
    
    Args:
        df: DataFrame with timestamp column
        splits: Dictionary with split names and date ranges
        
    Returns:
        Dictionary of split DataFrames
    """
    parts = {}
    
    for name, rng in splits.items():
        logger.info(f"Creating {name} split: {rng['start']} to {rng['end']}")
        
        # Convert timestamps if needed
        if 'ts' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['ts']):
                df['ts'] = pd.to_datetime(df['ts'])
        
        # Create mask for date range
        start_date = pd.Timestamp(rng['start'])
        end_date = pd.Timestamp(rng['end'])
        mask = (df['ts'] >= start_date) & (df['ts'] <= end_date)
        
        parts[name] = df.loc[mask].reset_index(drop=True)
        logger.info(f"{name} split: {len(parts[name])} rows")
    
    return parts


def create_sequences(df: pd.DataFrame, history_T: int, feature_cols: list) -> np.ndarray:
    """
    Create sequences for temporal modeling.
    
    Args:
        df: DataFrame with features
        history_T: Number of historical timesteps
        feature_cols: List of feature column names
        
    Returns:
        Array of shape [N, T, F] where N=samples, T=timesteps, F=features
    """
    logger.info(f"Creating sequences with history_T={history_T}")
    
    # Extract feature values
    features = df[feature_cols].values.astype(np.float32)
    n_samples, n_features = features.shape
    
    # Create sequences
    sequences = []
    for i in range(history_T, n_samples):
        seq = features[i-history_T:i, :]
        sequences.append(seq)
    
    X = np.array(sequences, dtype=np.float32)
    logger.info(f"Created sequences with shape: {X.shape}")
    
    return X


def to_tensors(df: pd.DataFrame, history_T: int = 20) -> dict:
    """
    Convert DataFrame to tensor format for training.
    
    Args:
        df: DataFrame with engineered features
        history_T: Number of historical timesteps
        
    Returns:
        Dictionary with tensors
    """
    # Define feature columns (must match what build_features creates)
    feature_cols = [
        'best_bid', 'best_ask', 'spread', 'imbalance', 'microprice',
        'bid_qty_1', 'ask_qty_1', 'mid_price'
    ]
    
    # Filter to available columns
    available_cols = [col for col in feature_cols if col in df.columns]
    logger.info(f"Using features: {available_cols}")
    
    # Create sequences
    X = create_sequences(df, history_T, available_cols)
    
    # Create targets (next period returns if available)
    if 'returns' in df.columns:
        y = df['returns'].iloc[history_T:].values.astype(np.float32)
    else:
        y = np.zeros(len(X), dtype=np.float32)
    
    # Get timestamps
    if 'ts' in df.columns:
        timestamps = df['ts'].iloc[history_T:].values
    else:
        timestamps = np.arange(len(X), dtype=np.int64)
    
    return {
        "X": X,
        "y": y,
        "ts": timestamps
    }


def main():
    """Main entry point for dataset creation"""
    ap = argparse.ArgumentParser(description="Create training datasets")
    ap.add_argument('--config', required=True, help='Path to data configuration file')
    args = ap.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load features
    features_dir = Path(cfg["paths"]["features"])
    features_file = features_dir / "features.parquet"
    
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}. Run feature engineering first.")
    
    logger.info(f"Loading features from {features_file}")
    df = pd.read_parquet(features_file)
    
    # Split by date
    parts = split_by_date(df, cfg["splits"])
    
    # Get history length from config
    history_T = cfg.get("history_T", 20)
    
    # Convert each split to tensors
    for name, part in parts.items():
        if len(part) <= history_T:
            logger.warning(f"Skipping {name} split: insufficient data ({len(part)} <= {history_T})")
            continue
        
        tensors = to_tensors(part, history_T=history_T)
        
        # Save tensors
        output_file = features_dir / f"{name}_tensors.npz"
        np.savez_compressed(output_file, **tensors)
        logger.info(f"Wrote tensors -> {output_file} (samples={len(tensors['X'])})")
    
    logger.info("Dataset creation completed successfully")


if __name__ == '__main__':
    main()
