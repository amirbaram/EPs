#!/usr/bin/env python3
"""
Enhanced Re-Entry ML Dataset Builder (Trade 2 / Trade 3 Continuation Setups)
Extracts winning intermediate ribbon features: bandwidth, 3-day expansion, compression,
distance to moving averages, consolidation duration, and volume digestion.
Zero lookahead bias; optimized O(N) symbol grouping.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import datastore

REENTRY_FEATURES = [
    "feature_days_since_ep",
    "feature_drawdown_from_peak",
    "feature_dist_10ema",
    "feature_dist_20ema",
    "feature_dist_50sma",
    "feature_vol_contraction",
    "feature_is_yellow_flip",
    "feature_is_interm_flip",
    "feature_ribbon_bandwidth",
    "feature_ribbon_expansion_3d",
    "feature_ribbon_compression",
    "feature_cons_days",
    "feature_digestion_ratio",
]


def build_reentry_dataset():
    print("=" * 80)
    print("🚀 BUILDING ENHANCED RE-ENTRY ML DATASET (TRADE 2 / TRADE 3)")
    print("=" * 80)
    t0 = time.time()

    parquet_file = "data/simulations/ep_combined_study_scored.parquet"
    if not os.path.exists(parquet_file):
        print(f"❌ Error: {parquet_file} not found.")
        return

    df_ep = pd.read_parquet(parquet_file)
    events = df_ep[["symbol", "date"]].to_dict("records")
    print(f"📊 Total EP events: {len(events)}")

    # Group events by symbol for fast O(N) processing
    by_symbol = {}
    for ev in events:
        by_symbol.setdefault(ev["symbol"], []).append(ev["date"])

    print(f"🔄 Grouped into {len(by_symbol)} unique symbols.")

    dataset = []

    for sym_idx, (sym, ep_dates) in enumerate(by_symbol.items()):
        if (sym_idx + 1) % 250 == 0 or sym_idx == len(by_symbol) - 1:
            print(f"Processing symbol {sym_idx + 1}/{len(by_symbol)} ({sym})...")

        df = datastore.load_bars(sym)
        if df is None or len(df) < 60:
            continue

        c = df["close"]
        l = df["low"]
        h = df["high"]
        o = df["open"]
        v = df["volume"]

        # Intermediate ribbon: (8, 12, 16, 21)
        ema8 = c.ewm(span=8, adjust=False).mean()
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema16 = c.ewm(span=16, adjust=False).mean()
        ema21 = c.ewm(span=21, adjust=False).mean()

        # Benchmark MAs
        ema3 = c.ewm(span=3, adjust=False).mean()
        ema8_low = l.ewm(span=8, adjust=False).mean()
        ema10 = c.ewm(span=10, adjust=False).mean()
        ema20 = c.ewm(span=20, adjust=False).mean()
        sma50 = c.rolling(50).mean()
        vol20 = v.rolling(20).mean()

        # Ribbon dynamics
        ribbon_bullish = (ema8 >= ema12) & (ema12 >= ema16) & (ema16 >= ema21)
        is_interm_flip = (ribbon_bullish & (~ribbon_bullish.shift(1).fillna(False))).astype(int)

        ribbon_bw = (ema8 - ema21) / np.where(ema21 > 0, ema21, 1.0)
        ribbon_exp_3d = ribbon_bw - ribbon_bw.shift(3).fillna(0.0)

        # Ribbon compression: std of EMAs / mean of EMAs
        ema_matrix = pd.concat([ema8, ema12, ema16, ema21], axis=1)
        ribbon_comp = (ema_matrix.std(axis=1) / ema_matrix.mean(axis=1)).fillna(0.0)

        # Legacy 3/8 flip
        is_yellow_flip = ((ema3 > ema8_low) & (ema3.shift(1) <= ema8_low.shift(1))).astype(int)

        # Process each EP anchor for this symbol
        for ep_date in ep_dates:
            if ep_date not in df.index:
                continue

            ep_idx = df.index.get_loc(ep_date)
            if ep_idx + 25 >= len(df):
                continue

            # Scan days 5 to 120 post-EP
            scan_end = min(ep_idx + 120, len(df) - 20)
            for t_idx in range(ep_idx + 5, scan_end):
                c_t = float(c.iloc[t_idx])
                sma50_t = float(sma50.iloc[t_idx])

                # Structural integrity gate: must hold above 50 SMA (within 2% buffer)
                if not np.isnan(sma50_t) and c_t < sma50_t * 0.98:
                    continue

                # Find peak high since EP
                peak_high = float(h.iloc[ep_idx:t_idx].max())
                drawdown_from_peak = (c_t - peak_high) / peak_high if peak_high > 0 else 0.0

                # Must be in a pullback of at least 4% from peak
                if drawdown_from_peak > -0.04:
                    continue

                # Must not have totally collapsed (>55% drop from peak)
                if drawdown_from_peak < -0.55:
                    continue

                # Compute distances to moving averages
                ema10_t = float(ema10.iloc[t_idx])
                ema20_t = float(ema20.iloc[t_idx])
                vol20_t = float(vol20.iloc[t_idx])
                vol_t = float(v.iloc[t_idx])

                dist_10 = (c_t - ema10_t) / ema10_t if ema10_t > 0 else 0.0
                dist_20 = (c_t - ema20_t) / ema20_t if ema20_t > 0 else 0.0
                dist_50 = (c_t - sma50_t) / sma50_t if (not np.isnan(sma50_t) and sma50_t > 0) else 0.0
                vol_contraction = vol_t / vol20_t if vol20_t > 0 else 1.0

                # Consolidation duration (count bars since peak)
                peak_bar_rel = int(h.iloc[ep_idx:t_idx].argmax())
                peak_bar_idx = ep_idx + peak_bar_rel
                cons_days = t_idx - peak_bar_idx

                # Up volume vs down volume during the consolidation window
                sub_pullback = df.iloc[peak_bar_idx:t_idx + 1]
                up_v = sub_pullback[sub_pullback["close"] >= sub_pullback["open"]]["volume"].sum()
                dn_v = sub_pullback[sub_pullback["close"] < sub_pullback["open"]]["volume"].sum()
                digestion_ratio = float(up_v / (dn_v + 1.0))

                # Forward 20-day evaluation
                f_20d = df.iloc[t_idx + 1:t_idx + 21]
                max_fwd = float((f_20d["high"].max() - c_t) / c_t)
                min_fwd = float((f_20d["low"].min() - c_t) / c_t)

                # Target: Hits +15% before dropping -8% (or stop loss)
                hit_15 = False
                for f_idx in range(len(f_20d)):
                    if f_20d["low"].iloc[f_idx] < c_t * 0.92:
                        break
                    if f_20d["high"].iloc[f_idx] >= c_t * 1.15:
                        hit_15 = True
                        break

                dataset.append({
                    "symbol": sym,
                    "date": str(df.index[t_idx].date()),
                    "entry_date": ep_date,
                    "feature_days_since_ep": t_idx - ep_idx,
                    "feature_drawdown_from_peak": drawdown_from_peak,
                    "feature_dist_10ema": dist_10,
                    "feature_dist_20ema": dist_20,
                    "feature_dist_50sma": dist_50,
                    "feature_vol_contraction": vol_contraction,
                    "feature_is_yellow_flip": int(is_yellow_flip.iloc[t_idx]),
                    "feature_is_interm_flip": int(is_interm_flip.iloc[t_idx]),
                    "feature_ribbon_bandwidth": float(ribbon_bw.iloc[t_idx]),
                    "feature_ribbon_expansion_3d": float(ribbon_exp_3d.iloc[t_idx]),
                    "feature_ribbon_compression": float(ribbon_comp.iloc[t_idx]),
                    "feature_cons_days": cons_days,
                    "feature_digestion_ratio": digestion_ratio,
                    "target_15pct_win": int(hit_15),
                    "forward_max_20d": max_fwd,
                    "forward_min_20d": min_fwd,
                })

    out_df = pd.DataFrame(dataset)
    out_dir = Path("data/ml_datasets/reentry")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_reentry.parquet"
    out_df.to_parquet(out_path)

    t1 = time.time()
    print("=" * 80)
    print(f"✅ Enhanced Re-entry dataset built with {len(out_df)} candidate rows in {t1 - t0:.2f}s.")
    print(f"Positive +15% Hit Rate: {out_df['target_15pct_win'].mean() * 100:.2f}%")
    print(f"Features: {REENTRY_FEATURES}")
    print(f"Saved to: {out_path}")
    print("=" * 80)
    return out_df


if __name__ == "__main__":
    build_reentry_dataset()
