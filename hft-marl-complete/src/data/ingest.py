"""
Data Ingestion Module for HFT MARL
===================================

This module ingests raw market data from simulators, combines them,
and prepares interim data for feature engineering.
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_simulator_files(sim_dir: Path) -> List[pd.DataFrame]:
    """
    Load all simulator output files from directory
    
    Args:
        sim_dir: Directory containing simulator outputs
        
    Returns:
        List of DataFrames
    """
    parquet_files = list(sim_dir.glob("*_snapshots*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No simulator files found in {sim_dir}. Run simulators first.")
    
    logger.info(f"Found {len(parquet_files)} simulator output files")
    
    dataframes = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            logger.info(f"Loaded {len(df)} rows from {file.name}")
            dataframes.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {file}: {e}")
    
    return dataframes


def combine_and_clean_data(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine multiple DataFrames and clean the data
    
    Args:
        dataframes: List of DataFrames to combine
        
    Returns:
        Combined and cleaned DataFrame
    """
    # Concatenate all dataframes
    df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined {len(dataframes)} dataframes into {len(df)} rows")
    
    # Sort by symbol and timestamp
    df = df.sort_values(['symbol', 'ts']).reset_index(drop=True)
    
    # Remove exact duplicates
    original_len = len(df)
    df = df.drop_duplicates(['symbol', 'ts'], keep='first')
    if len(df) < original_len:
        logger.info(f"Removed {original_len - len(df)} duplicate rows")
    
    # Basic data validation
    required_columns = ['ts', 'symbol', 'best_bid', 'best_ask', 'bid_qty_1', 'ask_qty_1']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Remove invalid data
    df = df[
        (df['best_bid'] > 0) & 
        (df['best_ask'] > 0) & 
        (df['best_ask'] > df['best_bid']) &
        (df['bid_qty_1'] > 0) &
        (df['ask_qty_1'] > 0)
    ]
    logger.info(f"After validation: {len(df)} rows")
    
    return df


def add_basic_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic derived features that are needed for validation
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    # Compute mid price if not already present
    if 'mid_price' not in df.columns:
        df['mid_price'] = ((df['best_bid'] + df['best_ask']) / 2).astype(np.float32)
    
    # Compute spread
    df['spread'] = (df['best_ask'] - df['best_bid']).astype(np.float32)
    
    # Compute spread in basis points
    df['spread_bps'] = (df['spread'] / df['mid_price'] * 10000).astype(np.float32)
    
    return df


def main():
    """Main entry point for data ingestion"""
    parser = argparse.ArgumentParser(description="Ingest market data from simulators")
    parser.add_argument('--config', required=True, help="Path to data configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get paths from config
    sim_dir = Path(config['paths']['sim'])
    interim_dir = Path(config['paths']['interim'])
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("DATA INGESTION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Simulator directory: {sim_dir}")
    logger.info(f"Output directory: {interim_dir}")
    
    # Load simulator files
    dataframes = load_simulator_files(sim_dir)
    
    # Combine and clean
    df = combine_and_clean_data(dataframes)
    
    # Add basic features
    df = add_basic_derived_features(df)
    
    # Save to interim
    output_file = interim_dir / "snapshots.parquet"
    df.to_parquet(output_file, index=False)
    
    # Print summary statistics
    logger.info("=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Symbols: {df['symbol'].unique().tolist()}")
    logger.info(f"Date range: {df['ts'].min()} to {df['ts'].max()}")
    logger.info(f"Average spread (bps): {df['spread_bps'].mean():.2f}")
    logger.info(f"Output file: {output_file}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
