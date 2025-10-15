"""
ABIDES Market Simulator
=======================

Generates synthetic ABIDES-like market snapshots for HFT MARL training.
This creates realistic order book snapshots with best bid/ask prices and quantities.
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Run ABIDES-like market simulator")
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
        
        # Create timestamp range
        ts = pd.date_range("2020-01-01", periods=num_samples, freq=f'{tick_ms}ms')
        
        # Generate realistic price movements
        np.random.seed(cfg.get('seed', 42))
        
        # Base price with mean reversion and volatility
        price_base = 100.0
        volatility = cfg.get('volatility', 0.02)
        mean_reversion_speed = cfg.get('mean_reversion_speed', 0.05)
        
        # Generate price path using Ornstein-Uhlenbeck process
        prices = [price_base]
        for i in range(1, num_samples):
            drift = -mean_reversion_speed * (prices[-1] - price_base)
            diffusion = volatility * np.random.randn()
            prices.append(prices[-1] + drift + diffusion)
        
        prices = np.array(prices)
        
        # Generate spread
        spread_bps = cfg.get('spread_target_bps', 5)
        spread = prices * (spread_bps / 10000)
        
        # Add some noise to spread
        spread += np.random.randn(num_samples) * spread * 0.1
        spread = np.clip(spread, cfg.get('tick_size', 0.01), None)
        
        # Calculate best bid and ask
        best_bid = prices - spread / 2
        best_ask = prices + spread / 2
        
        # Round to tick size
        tick_size = cfg.get('tick_size', 0.01)
        best_bid = np.round(best_bid / tick_size) * tick_size
        best_ask = np.round(best_ask / tick_size) * tick_size
        
        # Generate quantities with realistic depth
        base_qty = cfg.get('lot_size', 100) * 10  # 10 lots base
        
        bid_qty_1 = base_qty + (np.random.gamma(2, 50, num_samples)).astype(int)
        ask_qty_1 = base_qty + (np.random.gamma(2, 50, num_samples)).astype(int)
        
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
        output_file = out / f"{sym}_snapshots_abides.parquet"
        df.to_parquet(output_file, index=False)
        print(f"✓ ABIDES-like data generated: {output_file} ({len(df)} rows)")
    
    print(f"\n✓ All ABIDES simulations completed. Output: {out}")


if __name__ == '__main__':
    main()
