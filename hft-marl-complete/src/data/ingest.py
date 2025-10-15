"""
Data Ingestion Module
=====================

Ingests raw market snapshots from simulators and consolidates them into
a unified interim dataset for feature engineering.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml


def _load_snapshot_frames(sim_dir: Path) -> list[pd.DataFrame]:
    """Load all snapshot files (parquet preferred, fallback to CSV)."""
    parquet_files = list(sim_dir.glob("*_snapshots*.parquet"))
    csv_files = list(sim_dir.glob("*_snapshots*.csv")) if not parquet_files else []

    files = parquet_files if parquet_files else csv_files
    if not files:
        raise SystemExit(f"No simulator files found in {sim_dir}. Run simulators first.")

    frames: list[pd.DataFrame] = []
    for file_path in files:
        if file_path.suffix == ".parquet":
            frames.append(pd.read_parquet(file_path))
        else:
            frames.append(pd.read_csv(file_path))
    return frames


def _consolidate_snapshots(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate, sort, and de-duplicate snapshot frames."""
    df = pd.concat(frames, ignore_index=True)

    if "ts" in df.columns and "symbol" in df.columns:
        df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
        df = df.drop_duplicates(["symbol", "ts"]).reset_index(drop=True)
    elif "ts" in df.columns:
        df = df.sort_values(["ts"]).reset_index(drop=True)
        df = df.drop_duplicates(["ts"]).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest and consolidate market snapshots")
    ap.add_argument("--config", required=True, help="Path to data config file")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    sim_dir = Path(cfg["paths"]["sim"])  # Directory with simulator outputs
    out_dir = Path(cfg["paths"]["interim"])  # Directory to write consolidated data
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = _load_snapshot_frames(sim_dir)
    print(f"Found {len(frames)} snapshot file(s) in {sim_dir}")

    df = _consolidate_snapshots(frames)

    output_file = out_dir / "snapshots.parquet"
    df.to_parquet(output_file, index=False)

    print(f"✓ Consolidated {len(df)} rows → {output_file}")
    if "ts" in df.columns:
        print(f"  Time range: {df['ts'].min()} to {df['ts'].max()}")
    if "symbol" in df.columns:
        try:
            symbols = df["symbol"].unique().tolist()
            print(f"  Symbols: {symbols}")
        except Exception:
            pass


if __name__ == "__main__":
    main()


# Backward-compatible API for tests and scripts that import these names
def ingest_simulator_data(sim_dir: Path, output_dir: Path, config: dict) -> pd.DataFrame:  # type: ignore[override]
    """Load simulator outputs from sim_dir and return consolidated DataFrame."""
    frames = _load_snapshot_frames(sim_dir)
    return _consolidate_snapshots(frames)


def save_consolidated_data(df: pd.DataFrame, output_path: Path) -> None:  # type: ignore[override]
    """Save consolidated DataFrame to parquet at output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
