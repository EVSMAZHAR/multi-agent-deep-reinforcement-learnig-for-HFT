"""
JAX-LOB Market Simulator for HFT MARL
======================================

This module generates synthetic market data using a JAX-LOB-like approach
with limit order book dynamics for training multi-agent RL algorithms.
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_jaxlob_data(config: dict, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic market data using JAX-LOB-like simulation
    
    Args:
        config: Configuration dictionary
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with market snapshots
    """
    np.random.seed(seed + 1)  # Different seed from ABIDES
    
    symbols = config.get('symbols', ['SYMA'])
    tick_ms = config.get('tick_ms', 100)
    num_snapshots = config.get('simulation', {}).get('num_snapshots', 10000)
    
    dataframes = []
    
    for symbol in symbols:
        # Generate time series
        timestamps = pd.date_range(
            "2025-01-02 09:30:00",  # Different start date from ABIDES
            periods=num_snapshots, 
            freq=f'{tick_ms}ms'
        )
        
        # Generate price process with jumps
        t = np.linspace(0, 20, num_snapshots)
        
        # Multiple frequency components for realistic market dynamics
        price_component_1 = 2.0 * np.cos(t * 2)
        price_component_2 = 1.0 * np.sin(t * 5)
        price_component_3 = 0.5 * np.cos(t * 10)
        
        # Add random walk component
        random_walk = np.cumsum(np.random.randn(num_snapshots) * 0.02)
        
        mid_prices = 100.0 + price_component_1 + price_component_2 + price_component_3 + random_walk
        
        # Generate time-varying spread
        base_spread = 0.01
        spread_variation = 0.005 * np.abs(np.sin(t * 3))
        spread = base_spread + spread_variation
        
        best_bid = mid_prices - spread / 2
        best_ask = mid_prices + spread / 2
        
        # Generate quantities with order book dynamics
        # Simulate queue buildup and depletion
        base_qty = 950
        qty_variation = 200 * np.abs(np.cos(t * 4))
        
        bid_qty_1 = (base_qty + qty_variation + np.random.rand(num_snapshots) * 150).astype(int)
        ask_qty_1 = (base_qty + qty_variation + np.random.rand(num_snapshots) * 150).astype(int)
        
        # Add microstructure effects (quote stuffing events)
        quote_stuffing = np.random.binomial(1, 0.02, num_snapshots)
        bid_qty_1 = bid_qty_1 * (1 + quote_stuffing * 10)
        ask_qty_1 = ask_qty_1 * (1 + quote_stuffing * 10)
        
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
    """Main entry point for JAX-LOB simulation"""
    parser = argparse.ArgumentParser(description="Run JAX-LOB-like market simulation")
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
    
    print(f"Running JAX-LOB simulation with config: {args.config}")
    print(f"Output directory: {out_dir}")
    
    # Generate data
    df = generate_jaxlob_data(config, seed=args.seed)
    
    # Save individual symbol files
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        output_file = out_dir / f"{symbol}_snapshots_jaxlob.parquet"
        symbol_df.to_parquet(output_file, index=False)
        print(f"Generated {len(symbol_df)} snapshots for {symbol} -> {output_file}")
    
    print(f"JAX-LOB simulation completed. Total snapshots: {len(df)}")


if __name__ == '__main__':
    main()
