import argparse, yaml
from pathlib import Path
import pandas as pd, numpy as np
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for sym in cfg['symbols']:
        df = pd.DataFrame({
            "ts": pd.date_range("2025-01-01", periods=2000, freq=f'{cfg.get("tick_ms",100)}ms'),
            "symbol": sym,
            "best_bid": 100 + np.sin(np.linspace(0,10,2000))*0.05,
            "best_ask": 100.01 + np.sin(np.linspace(0,10,2000))*0.05,
            "bid_qty_1": 900 + (np.random.rand(2000)*200).astype(int),
            "ask_qty_1": 900 + (np.random.rand(2000)*200).astype(int)
        })
        df.to_parquet(out / f"{sym}_snapshots.parquet", index=False)
    print(f"Synthetic ABIDES-like data written to {out}")
if __name__ == '__main__':
    main()
