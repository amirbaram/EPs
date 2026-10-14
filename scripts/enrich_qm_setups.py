#!/usr/bin/env python3
"""scripts/enrich_qm_setups.py — Enriches qm_historical_setups.parquet with r2 and pinch_ratio."""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import datastore

DATA_PATH = Path("data/simulations/qm_historical_setups.parquet")


def process_symbol_batch(items):
    out = []
    for sym, setups in items:
        df_bars = datastore.load_bars(sym)
        if df_bars is None or len(df_bars) < 20:
            for s_idx, _, _, _, _ in setups:
                out.append((s_idx, 0.0, np.nan))
            continue

        std20 = df_bars["close"].rolling(20).std()
        adr20 = (df_bars["high"] - df_bars["low"]).rolling(20).mean()
        pinch = (4.0 * std20) / adr20
        dates_idx = {str(d.date()): i for i, d in enumerate(df_bars.index)}
        closes = df_bars["close"].to_numpy(float)

        for s_idx, p_st, p_en, c_st, c_en in setups:
            r2_val = 0.0
            if p_st in dates_idx and p_en in dates_idx:
                i1, i2 = dates_idx[p_st], dates_idx[p_en]
                if i2 > i1 and i2 - i1 >= 3:
                    pc = closes[i1 : i2 + 1]
                    x = np.arange(len(pc))
                    slope, intercept = np.polyfit(x, pc, 1)
                    if slope > 0:
                        y_pred = slope * x + intercept
                        ss_res = np.sum((pc - y_pred) ** 2)
                        ss_tot = np.sum((pc - np.mean(pc)) ** 2)
                        r2_val = round(float(1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0), 3)

            pinch_val = np.nan
            if c_st in dates_idx and c_en in dates_idx:
                i1, i2 = dates_idx[c_st], dates_idx[c_en]
                p_slice = pinch.iloc[i1 : i2 + 1].dropna()
                if len(p_slice) > 0:
                    pinch_val = round(float(p_slice.min()), 2)

            out.append((s_idx, r2_val, pinch_val))
    return out


def main():
    t0 = time.time()
    if not DATA_PATH.exists():
        print(f"File {DATA_PATH} not found.")
        return 1

    print(f"Loading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    n = len(df)
    print(f"Loaded {n} setups.")

    grouped = {}
    for idx, row in df.iterrows():
        grouped.setdefault(row["symbol"], []).append((
            idx,
            str(row["qm_pole_start"]),
            str(row["qm_pole_end"]),
            str(row["qm_cons_start"]),
            str(row["qm_cons_end"])
        ))

    items = list(grouped.items())
    num_batches = 32
    batch_size = (len(items) + num_batches - 1) // num_batches
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    print(f"Processing {len(items)} symbols across {len(batches)} batches...")
    all_results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_symbol_batch, b) for b in batches]
        for f in as_completed(futures):
            all_results.extend(f.result())

    print(f"Enrichment calculation finished in {time.time() - t0:.2f}s.")
    all_results.sort(key=lambda x: x[0])
    r2_arr = [x[1] for x in all_results]
    pinch_arr = [x[2] for x in all_results]

    df["r2"] = r2_arr
    df["pinch_ratio"] = pinch_arr

    print(f"Saving enriched dataset back to {DATA_PATH}...")
    df.to_parquet(DATA_PATH, index=False)
    print(f"Successfully saved {len(df)} setups with r2 and pinch_ratio in {time.time() - t0:.2f}s total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
