"""
Feature Engineering Module for HFT MARL
========================================

This module implements comprehensive feature engineering for high-frequency
trading data, including:
- Order book features (spread, imbalance, microprice)
- Technical indicators (returns, volatility, momentum)
- Microstructure features (order flow imbalance, realized volatility)
- Temporal features (time-of-day, rolling statistics)
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute spread-related features
    
    Args:
        df: Input DataFrame with best_bid and best_ask
        
    Returns:
        DataFrame with spread features
    """
    df = df.copy()
    
    # Raw spread
    df['spread'] = (df['best_ask'] - df['best_bid']).astype(np.float32)
    
    # Spread in basis points
    mid = (df['best_bid'] + df['best_ask']) / 2
    df['spread_bps'] = (df['spread'] / mid * 10000).astype(np.float32)
    
    # Relative spread (normalized)
    df['spread_rel'] = (df['spread'] / mid).astype(np.float32)
    
    return df


def compute_imbalance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute order book imbalance features
    
    Args:
        df: Input DataFrame with bid_qty_1 and ask_qty_1
        
    Returns:
        DataFrame with imbalance features
    """
    df = df.copy()
    
    # Volume imbalance at best levels
    total_qty = df['bid_qty_1'] + df['ask_qty_1']
    df['imbalance'] = ((df['bid_qty_1'] - df['ask_qty_1']) / total_qty).fillna(0).astype(np.float32)
    
    # Depth-weighted imbalance
    df['depth_imbalance'] = (df['bid_qty_1'] - df['ask_qty_1']).astype(np.float32)
    
    return df


def compute_microprice(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute microprice features
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with microprice
    """
    df = df.copy()
    
    # Volume-weighted microprice
    total_qty = df['bid_qty_1'] + df['ask_qty_1']
    df['microprice'] = (
        (df['best_ask'] * df['bid_qty_1'] + df['best_bid'] * df['ask_qty_1']) / total_qty
    ).fillna((df['best_ask'] + df['best_bid']) / 2).astype(np.float32)
    
    # Mid price
    df['mid_price'] = ((df['best_bid'] + df['best_ask']) / 2).astype(np.float32)
    
    return df


def compute_returns_features(df: pd.DataFrame, windows: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    Compute return features at various time windows
    
    Args:
        df: Input DataFrame with price data
        windows: List of window sizes for returns
        
    Returns:
        DataFrame with return features
    """
    df = df.copy()
    
    for window in windows:
        # Log returns
        df[f'log_return_{window}'] = (
            np.log(df['mid_price'] / df['mid_price'].shift(window))
        ).fillna(0).astype(np.float32)
        
        # Simple returns
        df[f'return_{window}'] = (
            (df['mid_price'] - df['mid_price'].shift(window)) / df['mid_price'].shift(window)
        ).fillna(0).astype(np.float32)
    
    return df


def compute_volatility_features(df: pd.DataFrame, windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
    """
    Compute volatility features
    
    Args:
        df: Input DataFrame with return data
        windows: List of window sizes for volatility
        
    Returns:
        DataFrame with volatility features
    """
    df = df.copy()
    
    # Need to have log_return_1 first
    if 'log_return_1' not in df.columns:
        df['log_return_1'] = np.log(df['mid_price'] / df['mid_price'].shift(1)).fillna(0)
    
    for window in windows:
        # Rolling standard deviation (realized volatility)
        df[f'realized_vol_{window}'] = (
            df['log_return_1'].rolling(window=window).std()
        ).fillna(0).astype(np.float32)
        
        # Parkinson volatility (high-low estimator)
        df[f'hl_vol_{window}'] = (
            np.sqrt((np.log(df['best_ask'] / df['best_bid']) ** 2) / (4 * np.log(2)))
        ).rolling(window=window).mean().fillna(0).astype(np.float32)
    
    return df


def compute_order_flow_imbalance(df: pd.DataFrame, window_ms: int = 1000) -> pd.DataFrame:
    """
    Compute order flow imbalance (OFI)
    
    Args:
        df: Input DataFrame
        window_ms: Window size in milliseconds
        
    Returns:
        DataFrame with OFI features
    """
    df = df.copy()
    
    # Estimate bid/ask arrivals from quantity changes
    df['bid_arrival'] = df['bid_qty_1'].diff().clip(lower=0).fillna(0)
    df['ask_arrival'] = df['ask_qty_1'].diff().clip(lower=0).fillna(0)
    
    # Estimate bid/ask cancellations
    df['bid_cancel'] = (-df['bid_qty_1'].diff()).clip(lower=0).fillna(0)
    df['ask_cancel'] = (-df['ask_qty_1'].diff()).clip(lower=0).fillna(0)
    
    # Order flow imbalance
    df['ofi'] = (df['bid_arrival'] - df['ask_arrival']).astype(np.float32)
    
    # Rolling OFI
    # Approximate window size in ticks (assuming regular sampling)
    window_ticks = max(1, window_ms // 100)  # Assuming 100ms tick
    df['ofi_rolling'] = df['ofi'].rolling(window=window_ticks).sum().fillna(0).astype(np.float32)
    
    # Trade intensity (total arrivals)
    df['trade_intensity'] = (
        (df['bid_arrival'] + df['ask_arrival']).rolling(window=window_ticks).sum()
    ).fillna(0).astype(np.float32)
    
    # Cancel intensity
    df['cancel_intensity'] = (
        (df['bid_cancel'] + df['ask_cancel']).rolling(window=window_ticks).sum()
    ).fillna(0).astype(np.float32)
    
    return df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with technical indicators
    """
    df = df.copy()
    
    # Moving averages
    for window in [10, 20, 50]:
        df[f'sma_{window}'] = (
            df['mid_price'].rolling(window=window).mean()
        ).fillna(df['mid_price']).astype(np.float32)
    
    # Exponential moving averages
    for span in [10, 20]:
        df[f'ema_{span}'] = (
            df['mid_price'].ewm(span=span, adjust=False).mean()
        ).fillna(df['mid_price']).astype(np.float32)
    
    # RSI-like momentum
    price_diff = df['mid_price'].diff()
    gain = price_diff.clip(lower=0).rolling(window=14).mean()
    loss = (-price_diff.clip(upper=0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = (100 - 100 / (1 + rs)).fillna(50).astype(np.float32)
    
    return df


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute temporal features
    
    Args:
        df: Input DataFrame with timestamp
        
    Returns:
        DataFrame with temporal features
    """
    df = df.copy()
    
    # Time of day features (normalized)
    df['hour'] = df['ts'].dt.hour
    df['minute'] = df['ts'].dt.minute
    df['second'] = df['ts'].dt.second
    
    # Cyclical encoding of time
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype(np.float32)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype(np.float32)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60).astype(np.float32)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60).astype(np.float32)
    
    # Time since market open (in minutes)
    market_open = df['ts'].dt.normalize() + pd.Timedelta(hours=9, minutes=30)
    df['time_since_open'] = (
        (df['ts'] - market_open).dt.total_seconds() / 60
    ).astype(np.float32)
    
    return df


def build_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Build comprehensive feature set
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        DataFrame with all engineered features
    """
    logger.info("Building comprehensive feature set...")
    
    # Sort by symbol and timestamp
    df = df.sort_values(['symbol', 'ts']).reset_index(drop=True)
    
    # Compute features by symbol
    dfs = []
    for symbol in df['symbol'].unique():
        logger.info(f"Processing features for {symbol}...")
        symbol_df = df[df['symbol'] == symbol].copy()
        
        # Core order book features
        symbol_df = compute_spread_features(symbol_df)
        symbol_df = compute_imbalance_features(symbol_df)
        symbol_df = compute_microprice(symbol_df)
        
        # Price and return features
        symbol_df = compute_returns_features(symbol_df)
        symbol_df = compute_volatility_features(symbol_df)
        
        # Microstructure features
        ofi_window = config.get('features', {}).get('windows', {}).get('ofi_ms', 1000)
        symbol_df = compute_order_flow_imbalance(symbol_df, window_ms=ofi_window)
        
        # Technical indicators
        symbol_df = compute_technical_indicators(symbol_df)
        
        # Temporal features
        symbol_df = compute_temporal_features(symbol_df)
        
        dfs.append(symbol_df)
    
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(['symbol', 'ts']).reset_index(drop=True)
    
    # Fill any remaining NaNs
    df = df.fillna(0)
    
    logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
    
    return df


def select_final_features(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select final feature set for model training
    
    Args:
        df: DataFrame with all features
        config: Configuration dictionary
        
    Returns:
        Tuple of (DataFrame with selected features, list of feature names)
    """
    # Define core features for training
    core_features = [
        # Order book features
        'spread', 'spread_bps', 'spread_rel',
        'imbalance', 'depth_imbalance',
        'microprice', 'mid_price',
        
        # Returns
        'log_return_1', 'log_return_5', 'log_return_10',
        
        # Volatility
        'realized_vol_10', 'realized_vol_20',
        
        # Order flow
        'ofi', 'ofi_rolling',
        'trade_intensity', 'cancel_intensity',
        
        # Technical indicators
        'sma_10', 'sma_20',
        'ema_10', 'ema_20',
        'rsi',
        
        # Temporal
        'hour_sin', 'hour_cos',
        'minute_sin', 'minute_cos',
        'time_since_open'
    ]
    
    # Filter to available features
    available_features = [f for f in core_features if f in df.columns]
    
    logger.info(f"Selected {len(available_features)} features for training")
    
    # Keep metadata columns
    metadata_cols = ['ts', 'symbol']
    final_cols = metadata_cols + available_features
    
    return df[final_cols], available_features


def main():
    """Main entry point for feature engineering"""
    parser = argparse.ArgumentParser(description="Engineer features from market data")
    parser.add_argument('--config', required=True, help="Path to data configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get paths
    interim_dir = Path(config['paths']['interim'])
    features_dir = Path(config['paths']['features'])
    features_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)
    
    # Load interim data
    input_file = interim_dir / "snapshots.parquet"
    logger.info(f"Loading data from: {input_file}")
    df = pd.read_parquet(input_file)
    logger.info(f"Loaded {len(df)} rows")
    
    # Build features
    df = build_features(df, config)
    
    # Select final features
    df_final, feature_names = select_final_features(df, config)
    
    # Save features
    output_file = features_dir / "features.parquet"
    df_final.to_parquet(output_file, index=False)
    logger.info(f"Saved features to: {output_file}")
    
    # Save feature names
    feature_names_file = features_dir / "feature_names.txt"
    with open(feature_names_file, 'w') as f:
        f.write('\n'.join(feature_names))
    logger.info(f"Saved feature names to: {feature_names_file}")
    
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total rows: {len(df_final)}")
    logger.info(f"Number of features: {len(feature_names)}")
    logger.info(f"Symbols: {df_final['symbol'].unique().tolist()}")
    logger.info(f"Date range: {df_final['ts'].min()} to {df_final['ts'].max()}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
