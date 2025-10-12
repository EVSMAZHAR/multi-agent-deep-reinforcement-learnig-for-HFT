"""
ABIDES Market Simulator for HFT MARL
=====================================

This module generates synthetic market data using an ABIDES-like approach
for testing and training multi-agent reinforcement learning algorithms.
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_abides_data(config: dict, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic market data using ABIDES-like simulation
    
    Args:
        config: Configuration dictionary
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with market snapshots
    """
    np.random.seed(seed)
    
    symbols = config.get('symbols', ['SYMA'])
    tick_ms = config.get('tick_ms', 100)
    num_snapshots = config.get('simulation', {}).get('num_snapshots', 10000)
    
    dataframes = []
    
    for symbol in symbols:
        # Generate time series
        timestamps = pd.date_range(
            "2025-01-01 09:30:00", 
            periods=num_snapshots, 
            freq=f'{tick_ms}ms'
        )
        
        # Generate realistic price movement with mean reversion
        returns = np.random.randn(num_snapshots) * 0.0001
        log_prices = 100 + np.cumsum(returns)
        mid_prices = np.exp(log_prices / 100) * 100
        
        # Add intraday volatility pattern (U-shaped)
        time_of_day = np.linspace(0, 2 * np.pi, num_snapshots)
        volatility_multiplier = 1.0 + 0.3 * (np.cos(time_of_day) ** 2)
        mid_prices = mid_prices * volatility_multiplier
        
        # Generate bid-ask spread
        spread_bps = 5 + 5 * np.abs(np.random.randn(num_snapshots))
        spread = mid_prices * (spread_bps / 10000)
        
        best_bid = mid_prices - spread / 2
        best_ask = mid_prices + spread / 2
        
        # Generate quantities with realistic patterns
        base_qty = 1000
        bid_qty_1 = base_qty + (np.random.rand(num_snapshots) * 500).astype(int)
        ask_qty_1 = base_qty + (np.random.rand(num_snapshots) * 500).astype(int)
        
        # Add volume clustering
        volume_clusters = np.random.binomial(1, 0.05, num_snapshots)
        bid_qty_1 = bid_qty_1 * (1 + volume_clusters * 5)
        ask_qty_1 = ask_qty_1 * (1 + volume_clusters * 5)
        
        # Create DataFrame
        df = pd.DataFrame({
            'ts': timestamps,
            'symbol': symbol,
            'best_bid': best_bid.astype(np.float32),
            'best_ask': best_ask.astype(np.float32),
            'bid_qty_1': bid_qty_1.astype(np.int64),
            'ask_qty_1': ask_qty_1.astype(np.int64),
            'mid_price': mid_prices.astype(np.float32),
        })
        
        dataframes.append(df)
    
    return pd.concat(dataframes, ignore_index=True)


def main():
    """Main entry point for ABIDES simulation"""
    parser = argparse.ArgumentParser(description="Run ABIDES-like market simulation")
    parser.add_argument('--config', required=True, help="Path to data configuration file")
    parser.add_argument('--out', required=True, help="Output directory for simulation data")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running ABIDES simulation with config: {args.config}")
    print(f"Output directory: {out_dir}")
    
    # Generate data
    df = generate_abides_data(config, seed=args.seed)
    
    # Save individual symbol files
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        output_file = out_dir / f"{symbol}_snapshots_abides.parquet"
        symbol_df.to_parquet(output_file, index=False)
        print(f"Generated {len(symbol_df)} snapshots for {symbol} -> {output_file}")
    
    print(f"ABIDES simulation completed. Total snapshots: {len(df)}")


if __name__ == '__main__':
    main()
