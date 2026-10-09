"""Simulate ML-Enhanced Continuation Execution and compare Before vs After:
1. Model 1: U&R Qualification Classifier (filters out false reclaims / consolidation traps)
2. Model 2: Dynamic Trailing Stop Quantile Regressor (protects profits on >= +2.5R runners)
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import datastore
import scanner_core
from ep_ml_engine import engine


def run_comparison(ribbon_spans=(8, 12, 16, 21), max_eps=None):
    study_path = Path("data/simulations/ep_combined_study_scored.parquet")
    backup_path = Path("data/simulations/ep_combined_study_scored_pre_dual_track.parquet")
    source_path = backup_path if backup_path.exists() else study_path
    df_ep = pd.read_parquet(source_path)
    if max_eps:
        df_ep = df_ep.head(max_eps)

    print(f"Loaded {len(df_ep)} EP events. Comparing Baseline vs ML-Enhanced Execution...")

    results = []
    t0 = time.time()

    for i, (_, row) in enumerate(df_ep.iterrows()):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(df_ep)} ({time.time() - t0:.1f}s)...")

        sym = row["symbol"]
        date_str = str(row["date"])
        raw = datastore.load_bars(sym)
        if raw is None or len(raw) < 30:
            continue

        d = scanner_core.calc_intermediate_ribbon(raw, spans=ribbon_spans)
        if "sma50" not in d.columns:
            d["sma50"] = d["close"].rolling(50).mean()
        if "vol20" not in d.columns:
            d["vol20"] = d["volume"].rolling(20).mean()

        ts = pd.Timestamp(date_str)
        if ts not in d.index:
            continue
        pos = d.index.get_loc(ts)
        if isinstance(pos, (slice, np.ndarray)):
            pos = pos[0] if isinstance(pos, np.ndarray) else pos.start
        if pos >= len(d) - 2:
            continue

        fwd = d.iloc[pos:min(len(d), pos + 250)].copy()
        if len(fwd) < 5:
            continue

        d1 = fwd.iloc[0]
        d1_close = float(d1["close"])
        d1_low = float(d1["low"])
        d1_high = float(d1["high"])
        risk1_pct = (d1_close - d1_low) / d1_close * 100.0 if d1_close > 0 else 11.1

        # Simulate Trade 1
        t1_stopped = False
        t1_exit_bar = None
        t1_exit_price = None
        t1_peak = d1_close
        seen_bull = False

        for b in range(1, len(fwd)):
            c = float(fwd["close"].iloc[b])
            l = float(fwd["low"].iloc[b])
            h = float(fwd["high"].iloc[b])
            b_open = float(fwd["open"].iloc[b])
            r_state = fwd["ribbon_state"].iloc[b]
            if h > t1_peak:
                t1_peak = h
            if pd.notna(r_state) and r_state in ["yellow", "gray"]:
                seen_bull = True

            if l <= d1_low:
                t1_stopped = True
                t1_exit_bar = b
                t1_exit_price = b_open if b_open < d1_low else d1_low
                break
            if seen_bull and pd.notna(r_state) and r_state == "blue":
                t1_exit_bar = b
                t1_exit_price = c
                break
            sma50 = fwd["sma50"].iloc[b]
            if seen_bull and pd.notna(sma50) and c < sma50:
                t1_exit_bar = b
                t1_exit_price = c
                break

        if t1_exit_bar is None:
            t1_exit_bar = len(fwd) - 1
            t1_exit_price = float(fwd["close"].iloc[-1])

        t1_ret = (t1_exit_price / d1_close - 1.0) * 100.0
        t1_r = t1_ret / risk1_pct if risk1_pct > 0 else 0.0

        # --- SIMULATE BASELINE T2 & T3 (WITHOUT ML) ---
        base_t2 = {"has_reentry": False, "r_mult": 0.0, "track": "none"}
        base_t3 = {"has_reentry": False, "r_mult": 0.0}

        # Track 1 U&R Detection
        ur_signal_bar = None
        ur_stop = None
        if t1_stopped and 1 <= t1_exit_bar <= 10:
            shakeout_low = float(fwd["low"].iloc[t1_exit_bar])
            controlled_ur = (shakeout_low >= d1_low * 0.85)
            if controlled_ur:
                ur_end_search = min(len(fwd), t1_exit_bar + 16)
                for b in range(t1_exit_bar + 1, ur_end_search):
                    l_b = float(fwd["low"].iloc[b])
                    c_b = float(fwd["close"].iloc[b])
                    if l_b < shakeout_low:
                        shakeout_low = l_b
                    if shakeout_low < d1_low * 0.85:
                        controlled_ur = False
                        break
                    if c_b >= d1_low:
                        ur_signal_bar = b
                        ur_stop = round(shakeout_low, 2)
                        break

            if controlled_ur and ur_signal_bar is not None and ur_signal_bar + 1 < len(fwd):
                entry_bar = ur_signal_bar + 1
                t2_entry = float(fwd["open"].iloc[entry_bar])
                t2_risk_pct = (t2_entry - ur_stop) / t2_entry * 100.0
                if 1.0 <= t2_risk_pct <= 35.0:
                    day1_low = float(fwd["low"].iloc[entry_bar])
                    day1_open = float(fwd["open"].iloc[entry_bar])
                    if day1_low <= ur_stop:
                        t2_exit_price = day1_open if day1_open < ur_stop else ur_stop
                        t2_exit_bar = entry_bar
                    else:
                        t2_exit_bar = None
                        for b in range(entry_bar + 1, len(fwd)):
                            c = float(fwd["close"].iloc[b])
                            l = float(fwd["low"].iloc[b])
                            b_open = float(fwd["open"].iloc[b])
                            s = fwd["ribbon_state"].iloc[b]
                            sma50 = fwd["sma50"].iloc[b]
                            if l <= ur_stop:
                                t2_exit_bar = b
                                t2_exit_price = b_open if b_open < ur_stop else ur_stop
                                break
                            if pd.notna(s) and s == "blue":
                                t2_exit_bar = b
                                t2_exit_price = c
                                break
                            if pd.notna(sma50) and c < sma50:
                                t2_exit_bar = b
                                t2_exit_price = c
                                break
                        if t2_exit_bar is None:
                            t2_exit_bar = len(fwd) - 1
                            t2_exit_price = float(fwd["close"].iloc[-1])

                    t2_ret = (t2_exit_price / t2_entry - 1.0) * 100.0
                    t2_r = t2_ret / t2_risk_pct if t2_risk_pct > 0 else 0.0
                    base_t2 = {"has_reentry": True, "r_mult": round(t2_r, 3), "track": "track_1_ur", "exit_bar": t2_exit_bar, "entry_bar": entry_bar}

        # Track 2 Ribbon Base Breakout Detection
        if not base_t2["has_reentry"]:
            seen_cons = False
            reentry_signal_bar = None
            cons_low = float(fwd["low"].iloc[t1_exit_bar])
            curr_peak = t1_peak
            search_end = min(len(fwd), t1_exit_bar + 66)
            for b in range(t1_exit_bar, search_end):
                l = float(fwd["low"].iloc[b])
                h = float(fwd["high"].iloc[b])
                c = float(fwd["close"].iloc[b])
                s = fwd["ribbon_state"].iloc[b]
                if h > curr_peak:
                    curr_peak = h
                    cons_low = l
                elif l < cons_low:
                    cons_low = l
                if pd.notna(s) and s in ["blue", "gray"]:
                    seen_cons = True
                pullback_pct = (curr_peak - cons_low) / curr_peak * 100.0 if curr_peak > 0 else 0.0
                if seen_cons and b > t1_exit_bar and pd.notna(s) and s == "yellow":
                    reentry_signal_bar = b
                    break
                if not seen_cons and b > t1_exit_bar + 2 and pullback_pct >= 5.0 and pd.notna(s) and s == "yellow":
                    prev_5d_high = float(fwd["high"].iloc[max(0, b - 5):b].max())
                    if c > prev_5d_high:
                        reentry_signal_bar = b
                        break

            if reentry_signal_bar is not None and reentry_signal_bar + 1 < len(fwd):
                c_sig = float(fwd["close"].iloc[reentry_signal_bar])
                sma50_sig = fwd["sma50"].iloc[reentry_signal_bar] if "sma50" in fwd.columns else None
                held_50 = pd.isna(sma50_sig) or c_sig >= sma50_sig
                elapsed_days = int(reentry_signal_bar - t1_exit_bar)
                allowed_cons = (elapsed_days <= 65 if held_50 else elapsed_days <= 45)
                drop_from_peak = (curr_peak - cons_low) / curr_peak * 100.0 if curr_peak > 0 else 0.0

                if allowed_cons and drop_from_peak <= 55.0:
                    swing5_low = float(fwd["low"].iloc[max(0, reentry_signal_bar - 4):reentry_signal_bar + 1].min())
                    t2_stop = round(swing5_low, 2)
                    entry_bar = reentry_signal_bar + 1
                    t2_entry = float(fwd["open"].iloc[entry_bar])
                    t2_risk_pct = (t2_entry - t2_stop) / t2_entry * 100.0
                    if 1.0 <= t2_risk_pct <= 35.0:
                        day1_low = float(fwd["low"].iloc[entry_bar])
                        day1_open = float(fwd["open"].iloc[entry_bar])
                        if day1_low <= t2_stop:
                            t2_exit_price = day1_open if day1_open < t2_stop else t2_stop
                            t2_exit_bar = entry_bar
                        else:
                            t2_exit_bar = None
                            for b in range(entry_bar + 1, len(fwd)):
                                c = float(fwd["close"].iloc[b])
                                l = float(fwd["low"].iloc[b])
                                b_open = float(fwd["open"].iloc[b])
                                s = fwd["ribbon_state"].iloc[b]
                                sma50 = fwd["sma50"].iloc[b]
                                if l <= t2_stop:
                                    t2_exit_bar = b
                                    t2_exit_price = b_open if b_open < t2_stop else t2_stop
                                    break
                                if pd.notna(s) and s == "blue":
                                    t2_exit_bar = b
                                    t2_exit_price = c
                                    break
                                if pd.notna(sma50) and c < sma50:
                                    t2_exit_bar = b
                                    t2_exit_price = c
                                    break
                            if t2_exit_bar is None:
                                t2_exit_bar = len(fwd) - 1
                                t2_exit_price = float(fwd["close"].iloc[-1])

                        t2_ret = (t2_exit_price / t2_entry - 1.0) * 100.0
                        t2_r = t2_ret / t2_risk_pct if t2_risk_pct > 0 else 0.0
                        base_t2 = {"has_reentry": True, "r_mult": round(t2_r, 3), "track": "track_2_ribbon", "exit_bar": t2_exit_bar, "entry_bar": entry_bar}

        # --- SIMULATE ML-ENHANCED T2 & T3 ---
        # 1. Model 1 (U&R Qualification Classifier): filters out false reclaims
        # 2. Model 2 (Dynamic Trailer): activates at >= 2.5R to protect profits on runners
        ml_t2 = {"has_reentry": False, "r_mult": 0.0, "track": "none", "prob": 0.0}

        # Track 1 U&R with Model 1 Qualification
        if ur_signal_bar is not None and ur_signal_bar + 1 < len(fwd) and t1_stopped and 1 <= t1_exit_bar <= 10:
            ur_feats = engine.compute_ur_reentry_features(sym, 0, t1_exit_bar, ur_signal_bar, shakeout_low, ribbon_spans=ribbon_spans, df=fwd)
            ur_pred = engine.predict_ur_reentry(ur_feats) if ur_feats else {"prob_win": 0.5, "is_qualified": True}

            entry_bar = ur_signal_bar + 1
            t2_entry = float(fwd["open"].iloc[entry_bar])
            t2_risk_pct = (t2_entry - ur_stop) / t2_entry * 100.0

            # Only enter if qualified by Model 1!
            if 1.0 <= t2_risk_pct <= 35.0 and ur_pred["is_qualified"]:
                risk_pts = t2_entry - ur_stop
                day1_low = float(fwd["low"].iloc[entry_bar])
                day1_open = float(fwd["open"].iloc[entry_bar])

                if day1_low <= ur_stop:
                    ml_exit_price = day1_open if day1_open < ur_stop else ur_stop
                    ml_exit_bar = entry_bar
                else:
                    ml_exit_bar = None
                    trade_peak = t2_entry
                    dynamic_trail_stop = ur_stop

                    for b in range(entry_bar + 1, len(fwd)):
                        c = float(fwd["close"].iloc[b])
                        l = float(fwd["low"].iloc[b])
                        h = float(fwd["high"].iloc[b])
                        b_open = float(fwd["open"].iloc[b])
                        s = fwd["ribbon_state"].iloc[b]
                        sma50 = fwd["sma50"].iloc[b]

                        if h > trade_peak:
                            trade_peak = h

                        # Check Dynamic Trailer activation at >= +2.5R
                        unrealized_r = (trade_peak - t2_entry) / risk_pts
                        if unrealized_r >= 2.5:
                            c_feats = engine.compute_continuation_trailer_features(fwd, b, t2_entry, trade_peak, risk_pts, ribbon_spans=ribbon_spans, entry_bar=entry_bar)
                            pred_buffer = engine.predict_continuation_trailer(c_feats)
                            trail_from_peak = trade_peak * (1.0 - pred_buffer / 100.0)

                            e21_col = f"ribbon_ema{ribbon_spans[3]}"
                            e21 = float(fwd[e21_col].iloc[b]) if e21_col in fwd.columns else c
                            shelf5 = float(fwd["low"].iloc[max(0, b - 4):b + 1].min())
                            support_lvl = min(e21 * 0.98, shelf5 * 0.98)
                            dyn_stop_cand = min(trail_from_peak, support_lvl)
                            dynamic_trail_stop = max(dynamic_trail_stop, dyn_stop_cand)

                        # Hard stop
                        if l <= ur_stop:
                            ml_exit_bar = b
                            ml_exit_price = b_open if b_open < ur_stop else ur_stop
                            break

                        # Dynamic trailer stop
                        if dynamic_trail_stop > ur_stop and l <= dynamic_trail_stop:
                            ml_exit_bar = b
                            ml_exit_price = b_open if b_open < dynamic_trail_stop else dynamic_trail_stop
                            break

                        if pd.notna(s) and s == "blue":
                            ml_exit_bar = b
                            ml_exit_price = c
                            break

                        if pd.notna(sma50) and c < sma50:
                            ml_exit_bar = b
                            ml_exit_price = c
                            break

                    if ml_exit_bar is None:
                        ml_exit_bar = len(fwd) - 1
                        ml_exit_price = float(fwd["close"].iloc[-1])

                ml_ret = (ml_exit_price / t2_entry - 1.0) * 100.0
                ml_r = ml_ret / t2_risk_pct if t2_risk_pct > 0 else 0.0
                ml_t2 = {"has_reentry": True, "r_mult": round(ml_r, 3), "track": "track_1_ur", "prob": ur_pred["prob_win"]}

        # If Track 1 did not take a trade (or was filtered out), evaluate Track 2 with Dynamic Trailer!
        if not ml_t2["has_reentry"] and base_t2.get("track") == "track_2_ribbon":
            # Track 2 with Model 2 Dynamic Trailer
            entry_bar = base_t2["entry_bar"]
            swing5_low = float(fwd["low"].iloc[max(0, entry_bar - 5):entry_bar].min())
            t2_stop = round(swing5_low, 2)
            t2_entry = float(fwd["open"].iloc[entry_bar])
            t2_risk_pct = (t2_entry - t2_stop) / t2_entry * 100.0

            if 1.0 <= t2_risk_pct <= 35.0:
                risk_pts = t2_entry - t2_stop
                day1_low = float(fwd["low"].iloc[entry_bar])
                day1_open = float(fwd["open"].iloc[entry_bar])

                if day1_low <= t2_stop:
                    ml_exit_price = day1_open if day1_open < t2_stop else t2_stop
                    ml_exit_bar = entry_bar
                else:
                    ml_exit_bar = None
                    trade_peak = t2_entry
                    dynamic_trail_stop = t2_stop

                    for b in range(entry_bar + 1, len(fwd)):
                        c = float(fwd["close"].iloc[b])
                        l = float(fwd["low"].iloc[b])
                        h = float(fwd["high"].iloc[b])
                        b_open = float(fwd["open"].iloc[b])
                        s = fwd["ribbon_state"].iloc[b]
                        sma50 = fwd["sma50"].iloc[b]

                        if h > trade_peak:
                            trade_peak = h

                        unrealized_r = (trade_peak - t2_entry) / risk_pts
                        if unrealized_r >= 2.5:
                            c_feats = engine.compute_continuation_trailer_features(fwd, b, t2_entry, trade_peak, risk_pts, ribbon_spans=ribbon_spans, entry_bar=entry_bar)
                            pred_buffer = engine.predict_continuation_trailer(c_feats)
                            trail_from_peak = trade_peak * (1.0 - pred_buffer / 100.0)

                            e21_col = f"ribbon_ema{ribbon_spans[3]}"
                            e21 = float(fwd[e21_col].iloc[b]) if e21_col in fwd.columns else c
                            shelf5 = float(fwd["low"].iloc[max(0, b - 4):b + 1].min())
                            support_lvl = min(e21 * 0.98, shelf5 * 0.98)
                            dyn_stop_cand = min(trail_from_peak, support_lvl)
                            dynamic_trail_stop = max(dynamic_trail_stop, dyn_stop_cand)

                        if l <= t2_stop:
                            ml_exit_bar = b
                            ml_exit_price = b_open if b_open < t2_stop else t2_stop
                            break

                        if dynamic_trail_stop > t2_stop and l <= dynamic_trail_stop:
                            ml_exit_bar = b
                            ml_exit_price = b_open if b_open < dynamic_trail_stop else dynamic_trail_stop
                            break

                        if pd.notna(s) and s == "blue":
                            ml_exit_bar = b
                            ml_exit_price = c
                            break

                        if pd.notna(sma50) and c < sma50:
                            ml_exit_bar = b
                            ml_exit_price = c
                            break

                    if ml_exit_bar is None:
                        ml_exit_bar = len(fwd) - 1
                        ml_exit_price = float(fwd["close"].iloc[-1])

                ml_ret = (ml_exit_price / t2_entry - 1.0) * 100.0
                ml_r = ml_ret / t2_risk_pct if t2_risk_pct > 0 else 0.0
                ml_t2 = {"has_reentry": True, "r_mult": round(ml_r, 3), "track": "track_2_ribbon", "prob": 0.5}

        results.append({
            "symbol": sym,
            "date": date_str,
            "t1_r": round(t1_r, 3),
            "breached_d1_low_5d": bool(row.get("breached_d1_low_5d", False)),
            "base_t2_has": base_t2["has_reentry"],
            "base_t2_track": base_t2["track"],
            "base_t2_r": base_t2["r_mult"],
            "ml_t2_has": ml_t2["has_reentry"],
            "ml_t2_track": ml_t2["track"],
            "ml_t2_r": ml_t2["r_mult"],
            "ml_t2_prob": ml_t2.get("prob", 0.0),
        })

    res_df = pd.DataFrame(results)
    return res_df


def calc_metrics(r_series: pd.Series):
    if len(r_series) == 0:
        return {"trades": 0, "win_rate": 0.0, "ev_r": 0.0, "total_r": 0.0, "pf": 0.0, "max_dd_r": 0.0}
    n = len(r_series)
    wins = r_series[r_series > 0]
    losses = r_series[r_series <= 0]
    wr = len(wins) / n * 100.0
    ev = r_series.mean()
    tot = r_series.sum()
    loss_sum = abs(losses.sum())
    pf = wins.sum() / loss_sum if loss_sum > 0 else 999.0
    cum = r_series.cumsum()
    peak = cum.cummax()
    max_dd = (cum - peak).min()
    return {
        "trades": n,
        "win_rate": round(wr, 1),
        "ev_r": round(ev, 3),
        "total_r": round(tot, 1),
        "pf": round(pf, 2),
        "max_dd_r": round(max_dd, 2),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-parquet", action="store_true", help="Update ep_combined_study_scored.parquet")
    parser.add_argument("--max-eps", type=int, default=None)
    args = parser.parse_args()

    res_df = run_comparison(max_eps=args.max_eps)

    if args.update_parquet:
        study_path = Path("data/simulations/ep_combined_study_scored.parquet")
        backup_path = Path("data/simulations/ep_combined_study_scored_pre_ml_continuation.parquet")
        if study_path.exists() and not backup_path.exists():
            import shutil
            shutil.copy2(study_path, backup_path)
            print(f"Backed up to {backup_path}")

        df_orig = pd.read_parquet(study_path)
        merge_cols = ["symbol", "date", "ml_t2_has", "ml_t2_track", "ml_t2_r", "ml_t2_prob"]
        merged = df_orig.merge(res_df[merge_cols], on=["symbol", "date"], how="left")
        merged["t2_has_reentry"] = merged["ml_t2_has"].fillna(False)
        merged["t2_real_r"] = merged["ml_t2_r"].fillna(0.0)
        merged["t2_track"] = merged["ml_t2_track"].fillna("none")
        merged["t2_prob_reentry"] = merged["ml_t2_prob"].fillna(0.0)
        merged = merged.drop(columns=["ml_t2_has", "ml_t2_track", "ml_t2_r", "ml_t2_prob"])

        tmp_p = study_path.with_suffix(".tmp.parquet")
        merged.to_parquet(tmp_p, index=False)
        tmp_p.replace(study_path)
        print(f"✅ Successfully updated {study_path} with ML-Enhanced continuation columns!")

    print("\n" + "=" * 80)
    print("📊 1. TRACK 1: UNDERCUT & RECLAIM (U&R) PERFORMANCE")
    print("=" * 80)
    base_ur = res_df[res_df["base_t2_track"] == "track_1_ur"]["base_t2_r"]
    ml_ur = res_df[res_df["ml_t2_track"] == "track_1_ur"]["ml_t2_r"]
    m_base_ur = calc_metrics(base_ur)
    m_ml_ur = calc_metrics(ml_ur)

    print(f"{'Metric':<20} | {'Baseline (Unfiltered)':<22} | {'ML-Qualified (Model 1)':<24} | {'Delta':<15}")
    print("-" * 85)
    for k in ["trades", "win_rate", "ev_r", "total_r", "pf", "max_dd_r"]:
        unit = "%" if k == "win_rate" else (" R" if "r" in k else "")
        v_old = m_base_ur[k]
        v_new = m_ml_ur[k]
        diff = v_new - v_old
        diff_str = f"{diff:+.2f}{unit}" if isinstance(diff, float) else f"{diff:+d}"
        print(f"{k:<20} | {v_old}{unit:<20} | {v_new}{unit:<22} | {diff_str:<15}")

    print("\n" + "=" * 80)
    print("📊 2. ALL CONTINUATION TRADES (T2 DUAL-TRACK: U&R + RIBBON BREAKOUT)")
    print("=" * 80)
    base_all = res_df[res_df["base_t2_has"]]["base_t2_r"]
    ml_all = res_df[res_df["ml_t2_has"]]["ml_t2_r"]
    m_base_all = calc_metrics(base_all)
    m_ml_all = calc_metrics(ml_all)

    print(f"{'Metric':<20} | {'Baseline (Standard Exits)':<22} | {'ML-Enhanced (Model 1+2)':<24} | {'Delta':<15}")
    print("-" * 85)
    for k in ["trades", "win_rate", "ev_r", "total_r", "pf", "max_dd_r"]:
        unit = "%" if k == "win_rate" else (" R" if "r" in k else "")
        v_old = m_base_all[k]
        v_new = m_ml_all[k]
        diff = v_new - v_old
        diff_str = f"{diff:+.2f}{unit}" if isinstance(diff, float) else f"{diff:+d}"
        print(f"{k:<20} | {v_old}{unit:<20} | {v_new}{unit:<22} | {diff_str:<15}")

    print("\n" + "=" * 80)
    print("📊 3. FULL PORTFOLIO LIFECYCLE (TRADE 1 + TRADE 2)")
    print("=" * 80)
    base_seq = res_df["t1_r"] + res_df["base_t2_r"]
    ml_seq = res_df["t1_r"] + res_df["ml_t2_r"]
    m_base_seq = calc_metrics(base_seq)
    m_ml_seq = calc_metrics(ml_seq)

    print(f"{'Metric':<20} | {'Baseline Lifecycle':<22} | {'ML-Enhanced Lifecycle':<24} | {'Delta':<15}")
    print("-" * 85)
    for k in ["trades", "win_rate", "ev_r", "total_r", "pf", "max_dd_r"]:
        unit = "%" if k == "win_rate" else (" R" if "r" in k else "")
        v_old = m_base_seq[k]
        v_new = m_ml_seq[k]
        diff = v_new - v_old
        diff_str = f"{diff:+.2f}{unit}" if isinstance(diff, float) else f"{diff:+d}"
        print(f"{k:<20} | {v_old}{unit:<20} | {v_new}{unit:<22} | {diff_str:<15}")
