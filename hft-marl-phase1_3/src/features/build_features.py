"""Feature engineering for microstructure-aware state (enhanced).
Computes spread, imbalance, microprice, OFI proxy, realised volatility, trade/cancel intensities, and a queue position proxy.
Inputs: data/interim/snapshots.parquet (regular decision grid, per Phase 0 ingestion).
Outputs: data/features/features.parquet
"""
import argparse, yaml
from pathlib import Path
import pandas as pd, numpy as np

def realised_vol(mid, window):
    r = np.log(mid).diff().fillna(0.0)
    # Rolling std of returns, scaled to per-step; leave as is (window steps)
    rv = r.rolling(window=window, min_periods=1).std().fillna(0.0)
    return rv

def ofi_proxy(df):
    # Simplified OFI using top-of-book deltas (since we may not have full L3 events)
    d_bid_p = df['best_bid'].diff().fillna(0.0)
    d_ask_p = df['best_ask'].diff().fillna(0.0)
    d_bid_q = df['bid_qty_1'].diff().fillna(0.0)
    d_ask_q = df['ask_qty_1'].diff().fillna(0.0)
    # If bid price up or unchanged -> positive buy pressure from d_bid_q; if down, reduce by previous qty
    ofi = ( (d_bid_q * (d_bid_p>=0).astype(float)) - (d_ask_q * (d_ask_p<=0).astype(float)) )
    return ofi.astype(np.float32)

def intensities(df, window):
    # Approximate trade intensity as absolute mid move count; cancel intensity as reduction in top qty
    mid = (df['best_bid'] + df['best_ask'])/2.0
    price_move = (mid.diff().fillna(0.0) != 0).astype(int)
    trade_int = price_move.rolling(window=window, min_periods=1).sum().astype(np.float32)

    dq_bid = (-df['bid_qty_1'].diff().fillna(0.0)).clip(lower=0.0)
    dq_ask = (-df['ask_qty_1'].diff().fillna(0.0)).clip(lower=0.0)
    cancel_int = (dq_bid + dq_ask).rolling(window=window, min_periods=1).mean().astype(np.float32)
    return trade_int, cancel_int

def queue_proxy(df, window):
    # Proxy: time since last touch change & imbalance magnitude
    touch_change = ((df['best_bid'].diff().fillna(0)!=0) | (df['best_ask'].diff().fillna(0)!=0)).astype(int)
    c = 0; times = []
    for x in touch_change.values:
        c = 0 if x==1 else c+1
        times.append(c)
    qtime = pd.Series(times, index=df.index).astype(np.float32)
    imb = ((df['bid_qty_1'] - df['ask_qty_1'])/ (df['bid_qty_1'] + df['ask_qty_1']).replace(0,np.nan)).fillna(0.0).astype(np.float32)
    qp = (qtime.rolling(window=window, min_periods=1).mean() * (1.0 + imb.abs())).astype(np.float32)
    return qp

def build_features(df, cfg):
    df = df.sort_values(['symbol','ts']).reset_index(drop=True)
    grouped = []
    ofi_win = max(1, int(cfg['windows'].get('ofi_ms',1000)/cfg['decision_ms']))
    rv_win = max(2, int(cfg['windows'].get('realised_vol_ms',2000)/cfg['decision_ms']))
    int_win = max(1, int(cfg['windows'].get('intensity_ms',1000)/cfg['decision_ms']))
    q_win = max(1, int(cfg['windows'].get('queue_ms',2000)/cfg['decision_ms']))
    for sym, g in df.groupby('symbol', sort=False):
        g = g.reset_index(drop=True)
        g['mid'] = (g['best_bid'] + g['best_ask'])/2.0
        g['spread'] = (g['best_ask'] - g['best_bid']).astype(np.float32)
        denom = (g['bid_qty_1'] + g['ask_qty_1']).replace(0, np.nan)
        g['imbalance'] = ((g['bid_qty_1'] - g['ask_qty_1'])/denom).fillna(0).astype(np.float32)
        g['microprice'] = ((g['best_ask']*g['bid_qty_1'] + g['best_bid']*g['ask_qty_1'])/denom)                           .fillna(g['mid']).astype(np.float32)
        g['ofi'] = ofi_proxy(g)
        g['rv'] = realised_vol(g['mid'], rv_win).astype(np.float32)
        ti, ci = intensities(g, int_win)
        g['trade_intensity'] = ti; g['cancel_intensity'] = ci
        g['queue_proxy'] = queue_proxy(g, q_win)
        grouped.append(g)
    feat = pd.concat(grouped, ignore_index=True)
    # select output columns
    cols = ['ts','symbol','best_bid','best_ask','bid_qty_1','ask_qty_1',
            'mid','spread','imbalance','microprice','ofi','rv','trade_intensity','cancel_intensity','queue_proxy']
    return feat[cols]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    interim = Path(cfg['paths']['interim'])
    feat_dir = Path(cfg['paths']['features']); feat_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(interim/'snapshots.parquet')
    feat = build_features(df, cfg | {'windows': cfg.get('windows', {})})
    feat.to_parquet(feat_dir/'features.parquet', index=False)
    print(f"Features saved -> {feat_dir/'features.parquet'} with {len(feat)} rows")

if __name__ == '__main__':
    main()
