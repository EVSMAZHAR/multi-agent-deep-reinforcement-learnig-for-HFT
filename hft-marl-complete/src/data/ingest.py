"""
Data Ingestion Module
=====================

Ingests raw market snapshots from simulators and consolidates them into
a unified interim dataset for feature engineering.
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Ingest and consolidate market snapshots")
    ap.add_argument('--config', required=True, help='Path to data config file')
    args = ap.parse_args()
    
    # Load configuration
    cfg = yaml.safe_load(open(args.config))
    
    # Get paths
    sim_dir = Path(cfg["paths"]["sim"])
    out_dir = Path(cfg["paths"]["interim"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all snapshot files
    parts = []
    snapshot_files = list(sim_dir.glob("*_snapshots*.parquet"))
    
    if not snapshot_files:
        raise SystemExit(f"❌ No simulator files found in {sim_dir}. Run simulators first.")
    
    print(f"Found {len(snapshot_files)} snapshot file(s):")
    for p in snapshot_files:
        print(f"  - {p.name}")
        df = pd.read_parquet(p)
        parts.append(df)
    
    # Consolidate all data
    df = pd.concat(parts, ignore_index=True)
    
    # Sort by symbol and timestamp
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    
    # Remove duplicates (same symbol and timestamp)
    df = df.drop_duplicates(["symbol", "ts"]).reset_index(drop=True)
    
    # Save consolidated data
    output_file = out_dir / "snapshots.parquet"
    df.to_parquet(output_file, index=False)
    
    print(f"\n✓ Consolidated {len(df)} rows → {output_file}")
    print(f"  Time range: {df['ts'].min()} to {df['ts'].max()}")
    print(f"  Symbols: {df['symbol'].unique().tolist()}")


if __name__ == '__main__':
    main()
