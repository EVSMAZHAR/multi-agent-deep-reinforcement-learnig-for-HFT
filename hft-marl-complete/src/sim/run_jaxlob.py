"""
JAX-LOB Market Simulator
========================

Generates synthetic JAX-LOB-like market snapshots for HFT MARL training.
This creates order book snapshots with different price dynamics than ABIDES.
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Run JAX-LOB-like market simulator")
    ap.add_argument('--config', required=True, help='Path to simulation config file')
    ap.add_argument('--out', required=True, help='Output directory for generated data')
    args = ap.parse_args()
    
    # Load configuration
    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic market data for each symbol
    for sym in cfg['symbols']:
        # Generate time series
        num_samples = cfg.get('num_samples', 5000)
        tick_ms = cfg.get('tick_ms', 100)
        
        # Create timestamp range (different dates to distinguish from ABIDES)
        ts = pd.date_range("2020-01-04", periods=num_samples, freq=f'{tick_ms}ms')
        
        # Generate realistic price movements with different dynamics
        np.random.seed(cfg.get('seed', 42) + 1)  # Different seed for variation
        
        # Base price with jump diffusion
        price_base = 100.0
        volatility = cfg.get('volatility', 0.02)
        
        # Generate price path with occasional jumps
        prices = [price_base]
        for i in range(1, num_samples):
            # Regular diffusion
            regular_change = volatility * np.random.randn() * 0.5
            
            # Jump component (rare large moves)
            jump_prob = 0.01
            jump = 0
            if np.random.rand() < jump_prob:
                jump = np.random.choice([-1, 1]) * volatility * np.random.exponential(2)
            
            prices.append(prices[-1] + regular_change + jump)
        
        prices = np.array(prices)
        
        # Generate tighter spread than ABIDES
        spread_bps = cfg.get('spread_target_bps', 5) * 0.8  # Tighter spread
        spread = prices * (spread_bps / 10000)
        
        # Add some noise to spread
        spread += np.random.randn(num_samples) * spread * 0.15
        spread = np.clip(spread, cfg.get('tick_size', 0.01), None)
        
        # Calculate best bid and ask
        best_bid = prices - spread / 2
        best_ask = prices + spread / 2
        
        # Round to tick size
        tick_size = cfg.get('tick_size', 0.01)
        best_bid = np.round(best_bid / tick_size) * tick_size
        best_ask = np.round(best_ask / tick_size) * tick_size
        
        # Generate quantities with different depth profile
        base_qty = cfg.get('lot_size', 100) * 12  # Deeper book
        
        bid_qty_1 = base_qty + (np.random.gamma(3, 40, num_samples)).astype(int)
        ask_qty_1 = base_qty + (np.random.gamma(3, 40, num_samples)).astype(int)
        
        # Create dataframe
        df = pd.DataFrame({
            "ts": ts,
            "symbol": sym,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_qty_1": bid_qty_1,
            "ask_qty_1": ask_qty_1
        })
        
        # Save to parquet
        output_file = out / f"{sym}_snapshots_jaxlob.parquet"
        df.to_parquet(output_file, index=False)
        print(f"✓ JAX-LOB-like data generated: {output_file} ({len(df)} rows)")
    
    print(f"\n✓ All JAX-LOB simulations completed. Output: {out}")


if __name__ == '__main__':
    main()
