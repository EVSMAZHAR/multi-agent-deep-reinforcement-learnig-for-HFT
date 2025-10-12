import argparse, yaml
from pathlib import Path
import pandas as pd
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    sim_dir = Path(cfg["paths"]["sim"])
    out_dir = Path(cfg["paths"]["interim"]); out_dir.mkdir(parents=True, exist_ok=True)
    parts = [pd.read_parquet(p) for p in sim_dir.glob("*_snapshots*.parquet")]
    if not parts:
        raise SystemExit(f"No simulator files found in {sim_dir}. Run simulators first.")
    df = pd.concat(parts).sort_values(["symbol","ts"]).drop_duplicates(["symbol","ts"]).reset_index(drop=True)
    df.to_parquet(out_dir/"snapshots.parquet", index=False)
    print(f"Wrote {len(df)} rows -> {out_dir/'snapshots.parquet'}")
if __name__ == '__main__':
    main()
