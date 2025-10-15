"""
Data Ingestion Module
=====================

Ingests raw market snapshots from simulators and consolidates them into
a unified interim dataset for feature engineering.
Adapted from hft-marl-phase0 for compatibility with enhanced environment.
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_simulator_data(sim_dir: Path, output_dir: Path, config: dict) -> pd.DataFrame:
    """
    Ingest data from simulator outputs and consolidate into a single DataFrame.
    
    Args:
        sim_dir: Directory containing simulator output files
        output_dir: Directory to save consolidated data
        config: Configuration dictionary
        
    Returns:
        Consolidated DataFrame with market snapshots
    """
    logger.info(f"Ingesting data from {sim_dir}")
    
    # Find all simulator snapshot files
    snapshot_files = list(sim_dir.glob("*_snapshots*.parquet"))
    
    if not snapshot_files:
        # Also check for CSV files
        snapshot_files = list(sim_dir.glob("*_snapshots*.csv"))
    
    if not snapshot_files:
        raise SystemExit(f"No simulator files found in {sim_dir}. Run simulators first.")
    
    logger.info(f"Found {len(snapshot_files)} snapshot files")
    
    # Load and concatenate all parts
    parts = []
    for file_path in snapshot_files:
        logger.info(f"Loading {file_path}")
        if file_path.suffix == '.parquet':
            df_part = pd.read_parquet(file_path)
        else:
            df_part = pd.read_csv(file_path)
        parts.append(df_part)
    
    # Concatenate and sort
    df = pd.concat(parts, ignore_index=True)
    
    # Sort by symbol and timestamp
    if 'symbol' in df.columns and 'ts' in df.columns:
        df = df.sort_values(['symbol', 'ts'])
    elif 'ts' in df.columns:
        df = df.sort_values('ts')
    
    # Drop duplicates
    if 'symbol' in df.columns and 'ts' in df.columns:
        df = df.drop_duplicates(['symbol', 'ts'])
    elif 'ts' in df.columns:
        df = df.drop_duplicates('ts')
    
    df = df.reset_index(drop=True)
    
    logger.info(f"Consolidated {len(df)} rows")
    
    return df


def save_consolidated_data(df: pd.DataFrame, output_path: Path):
    """
    Save consolidated data to disk.
    
    Args:
        df: DataFrame to save
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet for efficiency
    df.to_parquet(output_path, index=False)
    logger.info(f"Wrote {len(df)} rows -> {output_path}")


def main():
    """Main entry point for data ingestion"""
    ap = argparse.ArgumentParser(description="Ingest simulator data")
    ap.add_argument('--config', required=True, help='Path to data configuration file')
    args = ap.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Get paths
    sim_dir = Path(cfg["paths"]["sim"])
    interim_dir = Path(cfg["paths"]["interim"])
    
    # Ingest data
    df = ingest_simulator_data(sim_dir, interim_dir, cfg)
    
    # Save consolidated data
    output_path = interim_dir / "snapshots.parquet"
    save_consolidated_data(df, output_path)
    
    logger.info("Data ingestion completed successfully")
>>>>>>> cursor/integrate-data-collection-and-feature-engineering-b52d


if __name__ == '__main__':
    main()
