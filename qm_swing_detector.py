"""
qm_swing_detector.py — Multi-Timeframe QM & Bull Flag Pole Detector.
Supports two complementary algorithms:
  1. "linear_regression" (Recommended):
     - Discovers launch troughs (L) and peaks (H)
     - Validates pole linearity via Linear Regression R^2 >= min_r2
     - Validates momentum efficiency via Kaufman ER >= min_er
     - Evaluates flag consolidation (10/20 EMA surf, 50 SMA floor, breakout stack)
  2. "higher_low" (Sequential):
     - Bar-by-bar higher low sequence with ATR buffer
     - Type A (HL break) and Type B (Inside Bars)
Leaves all production files untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

import datastore
import indicators
import trendlab
import setups
import patterns
import config



@dataclass
class SwingPole:
    symbol: str
    tf: str
    algo: str                       # "linear_regression" or "higher_low"
    pole_type: str                  # "Linear R^2 Pole", "Type A (HL Break)", or "Type B (Inside Bars)"
    state: str                      # "breakout" or "building"
    date: str                       # date of breakout or active setup
    pole_start_date: str
    pole_end_date: str
    pole_low: float
    pole_high: float
    move_pct: float
    bars_in_swing: int
    r2: float
    er: float
    cons_start_date: str
    cons_end_date: str
    cons_bars: int
    pullback_pct: float
    trigger: float
    vol_x: Optional[float]
    ride_frac: float
    shakeouts: int
    levels: int


def _resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "1D":
        return df
    return datastore.resample_daily(df[["open", "high", "low", "close", "volume"]], tf)


def _fast_breakout_stack(
    high: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    piv_indices: set[int],
    hi_pos: int,
    end_pos: int,
    vol_ok: np.ndarray,
    dbars: int
) -> dict:
    """Precomputed-pivots breakout stack evaluation (runs in microseconds)."""
    pole_high = float(high[hi_pos])
    frac = config.STAIRSTEP_ATR_FRACTION
    stack: list[tuple[float, str]] = [(pole_high, "pole")]
    dr = 0
    n_broken = 0
    for i in range(hi_pos + 1, end_pos + 1):
        if i in piv_indices:
            stack.append((float(high[i]), "pivot"))
            dr = 0
        else:
            a = atr[i]
            dr = dr + 1 if (np.isfinite(a) and high[i] < high[i - 1] + a * frac) else 0
            if dr >= dbars:
                lvl = float(high[i - 1])
                if stack[-1][1] == "down":
                    stack[-1] = (lvl, "down")
                else:
                    stack.append((lvl, "down"))
        if vol_ok[i]:
            while len(stack) > 1 and close[i] > stack[-1][0]:
                stack.pop()
                n_broken += 1
            if len(stack) == 1 and close[i] > stack[-1][0]:
                stack.pop()
                n_broken += 1
        if not stack and close[i] < pole_high:
            stack = [(pole_high, "pole")]
    levels = [lv for lv, _ in stack] or [pole_high]
    return {
        "trigger": float(levels[-1]),
        "broke_today": bool(vol_ok[end_pos] and close[end_pos] > levels[-1]),
        "n_levels": len(levels),
    }


def detect_linear_regression_poles(
    df_daily: pd.DataFrame,
    symbol: str,
    tf: str = "1D",
    min_move: float = 15.0,
    min_r2: float = 0.65,
    min_er: float = 0.40,
    min_cons: int = 3,
    max_cons: int = 45,
    max_pb: float = 25.0,
    max_pole_bars: int = 65,
    piv_daily: Optional[set[int]] = None
) -> List[Dict[str, Any]]:
    """
    Detects bull flag poles using Linear Regression (R^2) & Kaufman Efficiency Ratio (ER),
    then evaluates daily consolidation base and breakout stack.
    """
    df_tf = _resample(df_daily, tf)
    if df_tf is None or len(df_tf) < 30:
        return []

    tf_highs = df_tf["high"].to_numpy(float)
    tf_lows = df_tf["low"].to_numpy(float)
    tf_closes = df_tf["close"].to_numpy(float)
    tf_dates = df_tf.index
    n_tf = len(df_tf)

    daily_highs = df_daily["high"].to_numpy(float)
    daily_lows = df_daily["low"].to_numpy(float)
    daily_closes = df_daily["close"].to_numpy(float)
    daily_vols = df_daily["volume"].to_numpy(float)
    daily_e10 = df_daily["ema10"].to_numpy(float)
    daily_e20 = df_daily["ema20"].to_numpy(float)
    daily_s50 = df_daily["sma50"].to_numpy(float)
    daily_atr = df_daily["atr14"].to_numpy(float)
    daily_dates = df_daily.index

    if piv_daily is None:
        piv_daily = {i for i, _ in trendlab.swing_highs(df_daily, len(df_daily))}

    # Strictly causal candidate pole discovery (0 bars to the right of evaluation bar):
    cand_poles = []
    seen_poles = set()
    for eval_i in range(min_cons + 10, n_tf):
        h_start = max(0, eval_i - max_cons)
        h_end = eval_i - min_cons
        if h_end < h_start:
            continue

        hi_idx = h_start + int(np.argmax(tf_highs[h_start : h_end + 1]))
        hi_p = tf_highs[hi_idx]

        lo_start = max(0, hi_idx - max_pole_bars)
        if hi_idx - lo_start < 4:
            continue

        cand_low_indices = set()
        cand_low_indices.add(lo_start + int(np.argmin(tf_lows[lo_start : hi_idx])))
        for k in range(max(2, lo_start), hi_idx - 3):
            if tf_lows[k] <= tf_lows[k - 2 : k].min() and tf_lows[k] <= tf_lows[k + 1 : k + 3].min():
                cand_low_indices.add(k)

        best_cand = None
        for lo_idx in cand_low_indices:
            lo_p = tf_lows[lo_idx]
            if tf_highs[lo_idx : hi_idx + 1].max() > hi_p + 1e-4:
                continue
            if tf_lows[lo_idx : hi_idx + 1].min() < lo_p - 1e-4:
                continue

            move_pct = (hi_p / lo_p - 1.0) * 100.0
            if move_pct < min_move:
                continue

            p_c = tf_closes[lo_idx : hi_idx + 1]
            n_bars = len(p_c)
            x = np.arange(n_bars)
            slope, intercept = np.polyfit(x, p_c, 1)
            if slope <= 0:
                continue
            y_pred = slope * x + intercept
            ss_res = np.sum((p_c - y_pred) ** 2)
            ss_tot = np.sum((p_c - np.mean(p_c)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            er = abs(p_c[-1] - p_c[0]) / np.sum(np.abs(np.diff(p_c))) if len(p_c) > 1 else 1.0

            if r2 >= min_r2 and er >= min_er:
                score = r2 * er
                if best_cand is None or score > best_cand["score"]:
                    best_cand = {
                        "score": score,
                        "tf": tf,
                        "lo_idx": lo_idx,
                        "hi_idx": hi_idx,
                        "pole_start_date": str(tf_dates[lo_idx].date()),
                        "pole_end_date": str(tf_dates[hi_idx].date()),
                        "pole_low": round(float(lo_p), 2),
                        "pole_high": round(float(hi_p), 2),
                        "move_pct": round(float(move_pct), 1),
                        "bars_in_swing": n_bars,
                        "r2": round(float(r2), 3),
                        "er": round(float(er), 3),
                    }

        if best_cand is not None:
            pole_key = (best_cand["lo_idx"], best_cand["hi_idx"])
            if pole_key not in seen_poles:
                seen_poles.add(pole_key)
                cand_poles.append(best_cand)

    final_setups = []
    for cand in cand_poles:
        p_start_d = pd.to_datetime(cand["pole_start_date"])
        p_end_d = pd.to_datetime(cand["pole_end_date"])

        d_start_matches = np.where(daily_dates >= p_start_d)[0]
        d_end_matches = np.where(daily_dates <= p_end_d)[0]
        if len(d_start_matches) == 0 or len(d_end_matches) == 0:
            continue

        d_start_idx = int(d_start_matches[0])
        d_end_idx = int(d_end_matches[-1])
        if d_end_idx <= d_start_idx or d_end_idx >= len(daily_dates) - 1:
            continue

        cons_limit = min(d_end_idx + max_cons + 1, len(daily_dates))
        pole_high_p = cand["pole_high"]

        found_breakout = False
        latest_building = None

        # Precompute volume thresholds for the slice
        win = config.QMB_CONTRACT_WIN
        vol_ok_arr = np.zeros(len(daily_dates), bool)
        for k in range(d_end_idx + 1, cons_limit):
            vb = daily_vols[max(0, k - win):k].mean()
            vol_ok_arr[k] = vb > 0 and daily_vols[k] >= config.QMB_BREAKOUT_VOL * vb

        for eval_idx in range(d_end_idx + min_cons, cons_limit):
            cons_bars = eval_idx - d_end_idx

            # 1. Controlled pullback
            c_low = float(daily_lows[d_end_idx : eval_idx + 1].min())
            pullback = (1.0 - c_low / pole_high_p) * 100.0
            if pullback > max_pb:
                break

            # 2. 10/20 EMA surf & 50 SMA floor
            cons_sl = slice(d_end_idx + 1, eval_idx + 1)
            ride = setups._qm_ride(
                daily_closes[cons_sl], daily_lows[cons_sl],
                daily_e10[cons_sl], daily_e20[cons_sl],
                daily_s50[cons_sl]
            )
            if ride is None:
                continue

            # 3. Fast Breakout stack
            st = _fast_breakout_stack(
                daily_highs, daily_closes, daily_atr, piv_daily,
                d_end_idx, eval_idx, vol_ok_arr, config.BREAKOUT_STACK_DOWNSWING
            )
            trig = st["trigger"]
            state = "breakout" if st["broke_today"] else "building"
            vbase = daily_vols[max(0, eval_idx - win):eval_idx].mean() if eval_idx > win else daily_vols[:eval_idx].mean()
            vol_x = round(float(daily_vols[eval_idx] / vbase), 2) if vbase > 0 else None

            setup_dict = {
                "symbol": symbol,
                "tf": tf,
                "algo": "Linear R^2 + ER",
                "pole_type": f"Linear R²={cand['r2']}",
                "state": state,
                "date": str(daily_dates[eval_idx].date()),
                "pole_start_date": cand["pole_start_date"],
                "pole_end_date": cand["pole_end_date"],
                "pole_low": cand["pole_low"],
                "pole_high": cand["pole_high"],
                "move_pct": cand["move_pct"],
                "bars_in_swing": cand["bars_in_swing"],
                "r2": cand["r2"],
                "er": cand["er"],
                "cons_start_date": str(daily_dates[d_end_idx + 1].date()),
                "cons_end_date": str(daily_dates[eval_idx].date()),
                "cons_bars": cons_bars,
                "pullback_pct": round(pullback, 1),
                "trigger": round(trig, 2),
                "vol_x": vol_x,
                "ride_frac": ride["ride_frac"],
                "shakeouts": ride["shakeouts"],
                "levels": st["n_levels"],
            }

            if state == "breakout":
                final_setups.append(setup_dict)
                found_breakout = True
                break
            else:
                latest_building = setup_dict

        if not found_breakout and latest_building is not None:
            final_setups.append(latest_building)

    # Debounce setups reaching the same consolidation area:
    # If setups share the same date or adjacent dates (within 4 trading days) with overlapping bases,
    # keep the dominant, highest quality pole (highest r2 * move_pct).
    debounced = []
    final_setups.sort(key=lambda s: s["date"])
    for s in final_setups:
        if not debounced:
            debounced.append(s)
            continue
        last = debounced[-1]
        dt_s = pd.to_datetime(s["date"])
        dt_last = pd.to_datetime(last["date"])
        days_apart = abs((dt_s - dt_last).days)

        if days_apart <= 4 or s["date"] == last["date"]:
            if s["state"] == "breakout" and last["state"] != "breakout":
                debounced[-1] = s
            elif last["state"] == "breakout" and s["state"] != "breakout":
                pass
            else:
                q_new = (s.get("r2", 1.0) or 1.0) * s["move_pct"]
                q_last = (last.get("r2", 1.0) or 1.0) * last["move_pct"]
                if q_new > q_last:
                    debounced[-1] = s
        else:
            debounced.append(s)

    return debounced


def detect_higher_low_poles(
    df_daily: pd.DataFrame,
    symbol: str,
    tf: str = "1D",
    atr_frac: float = 0.3,
    min_move: float = 15.0,
    min_cons: int = 3,
    max_cons: int = 45,
    max_pb: float = 25.0,
    inside_mode: str = "pole_range",
    piv_daily: Optional[set[int]] = None
) -> List[Dict[str, Any]]:
    """Sequential higher-low sequence pole detection (Type A / Type B)."""
    df_tf = _resample(df_daily, tf)
    if df_tf is None or len(df_tf) < 20:
        return []

    atr_tf = trendlab.atr_series(df_tf, n=14)
    highs = df_tf["high"].to_numpy(float)
    lows = df_tf["low"].to_numpy(float)
    closes = df_tf["close"].to_numpy(float)
    tf_dates = df_tf.index

    daily_highs = df_daily["high"].to_numpy(float)
    daily_lows = df_daily["low"].to_numpy(float)
    daily_closes = df_daily["close"].to_numpy(float)
    daily_vols = df_daily["volume"].to_numpy(float)
    daily_e10 = df_daily["ema10"].to_numpy(float)
    daily_e20 = df_daily["ema20"].to_numpy(float)
    daily_s50 = df_daily["sma50"].to_numpy(float)
    daily_atr = df_daily["atr14"].to_numpy(float)
    daily_dates = df_daily.index

    if piv_daily is None:
        piv_daily = {i for i, _ in trendlab.swing_highs(df_daily, len(df_daily))}

    n_tf = len(df_tf)
    detected_candidates = []

    i = 1
    while i < n_tf:
        delta = atr_tf[i] * atr_frac
        if lows[i] >= lows[i - 1] - delta and highs[i] > highs[i - 1]:
            start_idx = i - 1
            cur_low = lows[start_idx]
            cur_high = max(highs[start_idx], highs[i])
            cur_high_idx = i if highs[i] >= highs[start_idx] else start_idx

            j = i + 1
            inside_count = 0
            swing_ended = False
            end_type = None
            end_high_idx = cur_high_idx

            while j < n_tf:
                d_j = atr_tf[j] * atr_frac
                if inside_mode == "strict":
                    is_inside = (highs[j] <= highs[j - 1] + 0.05 * atr_tf[j]) and (lows[j] >= lows[j - 1] - 0.05 * atr_tf[j])
                else:
                    is_inside = (highs[j] <= cur_high + 0.1 * atr_tf[j]) and (lows[j] >= cur_high * (1.0 - max_pb / 100.0))

                is_hl = lows[j] >= lows[j - 1] - d_j

                if highs[j] > cur_high:
                    cur_high = highs[j]
                    cur_high_idx = j
                    inside_count = 0
                elif is_inside:
                    inside_count += 1
                    if inside_count >= min_cons:
                        swing_ended = True
                        end_type = "Type B (Inside Bars)"
                        end_high_idx = cur_high_idx
                        break

                if not is_hl and not is_inside:
                    swing_ended = True
                    end_type = "Type A (HL Break)"
                    end_high_idx = cur_high_idx
                    break

                j += 1

            if not swing_ended and j >= n_tf:
                end_type = "Developing"
                end_high_idx = cur_high_idx

            p_low = lows[start_idx]
            p_high = cur_high
            move_pct = (p_high / p_low - 1.0) * 100.0

            if move_pct >= min_move and end_type is not None:
                p_c = closes[start_idx : end_high_idx + 1]
                x = np.arange(len(p_c))
                slope, intercept = np.polyfit(x, p_c, 1)
                y_pred = slope * x + intercept
                ss_res = np.sum((p_c - y_pred) ** 2)
                ss_tot = np.sum((p_c - np.mean(p_c)) ** 2)
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                er = abs(p_c[-1] - p_c[0]) / np.sum(np.abs(np.diff(p_c))) if len(p_c) > 1 else 1.0

                detected_candidates.append({
                    "tf": tf,
                    "type": end_type,
                    "pole_start_date": str(tf_dates[start_idx].date()),
                    "pole_end_date": str(tf_dates[end_high_idx].date()),
                    "pole_low": round(float(p_low), 2),
                    "pole_high": round(float(p_high), 2),
                    "move_pct": round(float(move_pct), 1),
                    "bars_in_swing": end_high_idx - start_idx + 1,
                    "r2": round(float(r2), 3),
                    "er": round(float(er), 3),
                })

            i = max(i + 1, cur_high_idx + 1)
        else:
            i += 1

    final_setups = []
    for cand in detected_candidates:
        p_start_d = pd.to_datetime(cand["pole_start_date"])
        p_end_d = pd.to_datetime(cand["pole_end_date"])

        d_start_matches = np.where(daily_dates >= p_start_d)[0]
        d_end_matches = np.where(daily_dates <= p_end_d)[0]
        if len(d_start_matches) == 0 or len(d_end_matches) == 0:
            continue

        d_start_idx = int(d_start_matches[0])
        d_end_idx = int(d_end_matches[-1])
        if d_end_idx <= d_start_idx or d_end_idx >= len(daily_dates) - 1:
            continue

        cons_limit = min(d_end_idx + max_cons + 1, len(daily_dates))
        pole_high_p = cand["pole_high"]

        found_breakout = False
        latest_building = None

        win = config.QMB_CONTRACT_WIN
        vol_ok_arr = np.zeros(len(daily_dates), bool)
        for k in range(d_end_idx + 1, cons_limit):
            vb = daily_vols[max(0, k - win):k].mean()
            vol_ok_arr[k] = vb > 0 and daily_vols[k] >= config.QMB_BREAKOUT_VOL * vb

        for eval_idx in range(d_end_idx + min_cons, cons_limit):
            cons_bars = eval_idx - d_end_idx

            c_low = float(daily_lows[d_end_idx : eval_idx + 1].min())
            pullback = (1.0 - c_low / pole_high_p) * 100.0
            if pullback > max_pb:
                break

            cons_sl = slice(d_end_idx + 1, eval_idx + 1)
            ride = setups._qm_ride(
                daily_closes[cons_sl], daily_lows[cons_sl],
                daily_e10[cons_sl], daily_e20[cons_sl],
                daily_s50[cons_sl]
            )
            if ride is None:
                continue

            st = _fast_breakout_stack(
                daily_highs, daily_closes, daily_atr, piv_daily,
                d_end_idx, eval_idx, vol_ok_arr, config.BREAKOUT_STACK_DOWNSWING
            )
            trig = st["trigger"]
            state = "breakout" if st["broke_today"] else "building"
            vbase = daily_vols[max(0, eval_idx - win):eval_idx].mean() if eval_idx > win else daily_vols[:eval_idx].mean()
            vol_x = round(float(daily_vols[eval_idx] / vbase), 2) if vbase > 0 else None

            setup_dict = {
                "symbol": symbol,
                "tf": tf,
                "algo": "Higher-Low Seq",
                "pole_type": cand["type"],
                "state": state,
                "date": str(daily_dates[eval_idx].date()),
                "pole_start_date": cand["pole_start_date"],
                "pole_end_date": cand["pole_end_date"],
                "pole_low": cand["pole_low"],
                "pole_high": cand["pole_high"],
                "move_pct": cand["move_pct"],
                "bars_in_swing": cand["bars_in_swing"],
                "r2": cand["r2"],
                "er": cand["er"],
                "cons_start_date": str(daily_dates[d_end_idx + 1].date()),
                "cons_end_date": str(daily_dates[eval_idx].date()),
                "cons_bars": cons_bars,
                "pullback_pct": round(pullback, 1),
                "trigger": round(trig, 2),
                "vol_x": vol_x,
                "ride_frac": ride["ride_frac"],
                "shakeouts": ride["shakeouts"],
                "levels": st["n_levels"],
            }

            if state == "breakout":
                final_setups.append(setup_dict)
                found_breakout = True
                break
            else:
                latest_building = setup_dict

        if not found_breakout and latest_building is not None:
            final_setups.append(latest_building)

    # Debounce setups reaching the same consolidation area
    debounced = []
    final_setups.sort(key=lambda s: s["date"])
    for s in final_setups:
        if not debounced:
            debounced.append(s)
            continue
        last = debounced[-1]
        dt_s = pd.to_datetime(s["date"])
        dt_last = pd.to_datetime(last["date"])
        days_apart = abs((dt_s - dt_last).days)

        if days_apart <= 4 or s["date"] == last["date"]:
            if s["state"] == "breakout" and last["state"] != "breakout":
                debounced[-1] = s
            elif last["state"] == "breakout" and s["state"] != "breakout":
                pass
            else:
                if s["move_pct"] > last["move_pct"]:
                    debounced[-1] = s
        else:
            debounced.append(s)

    return debounced


def detect_ema_catchup_bb_pinch_poles(
    df_daily: pd.DataFrame,
    symbol: str,
    tf: str = "1D",
    min_move: float = 15.0,
    min_ext_10: float = 3.0,
    min_ext_20: float = 5.0,
    max_pinch_ratio: float = 3.2,
    max_pb: float = 25.0,
    max_cons: int = 50,
    max_pole_bars: int = 50,
    piv_daily: Optional[set[int]] = None
) -> List[Dict[str, Any]]:
    """
    EMA Extension -> EMA Catchup & Surfing -> Bollinger Band Pinch vs 20-Day ADR Bull Flag Detection.
    Mechanical logic:
      1. Pole Impulse: Stock runs up >= min_move%, becoming extended from EMA10 (>= min_ext_10%)
         and EMA20 (>= min_ext_20%). Grinding new highs debounce to confirmed swing peaks.
      2. Natural Maturity (No Hardcoded Min Days): Price consolidates sideways/downwards while rising
         EMAs catch up (Low <= EMA10 * 1.005 or Low <= EMA20 * 1.01).
      3. EMA Surfing: Price surfs above rising EMA20 and 50 SMA floor with max pullback <= max_pb%.
      4. Breakout Priming: Bollinger Bands (20, 2) pinch to a bandwidth significantly less than ADR20
         (Pinch Ratio = BB_width / ADR20 <= max_pinch_ratio).
      5. Breakout vs Primed: Fast Breakout Stack evaluates whether resistance has been broken on volume
         ('breakout') or if the stock is tightly coiled and ready ('primed').
    """
    if "ema10" not in df_daily.columns:
        df_daily = indicators.add_indicators(df_daily)

    df_tf = _resample(df_daily, tf)
    if len(df_tf) < 25:
        return []

    tf_highs = df_tf["high"].to_numpy(float)
    tf_lows = df_tf["low"].to_numpy(float)
    tf_closes = df_tf["close"].to_numpy(float)
    tf_dates = df_tf.index
    n_tf = len(df_tf)

    # Timeframe EMAs for pole extension verification
    tf_e10 = df_tf["close"].ewm(span=10, adjust=False).mean().to_numpy(float)
    tf_e20 = df_tf["close"].ewm(span=20, adjust=False).mean().to_numpy(float)

    daily_highs = df_daily["high"].to_numpy(float)
    daily_lows = df_daily["low"].to_numpy(float)
    daily_closes = df_daily["close"].to_numpy(float)
    daily_vols = df_daily["volume"].to_numpy(float)
    daily_e10 = df_daily["ema10"].to_numpy(float)
    daily_e20 = df_daily["ema20"].to_numpy(float)
    daily_s50 = df_daily["sma50"].to_numpy(float)
    daily_atr = df_daily["atr14"].to_numpy(float)
    daily_dates = df_daily.index

    # Standard Bollinger Bands (20, 2) and 20-Day ADR on daily
    std20 = df_daily["close"].rolling(20).std().to_numpy(float)
    bb_w = 4.0 * std20
    adr20 = (df_daily["high"] - df_daily["low"]).rolling(20).mean().to_numpy(float)
    pinch_ratio = bb_w / adr20

    if piv_daily is None:
        piv_daily = {i for i, _ in trendlab.swing_highs(df_daily, len(df_daily))}

    # Discover candidate poles causally
    cand_poles = []
    for H in range(15, n_tf - 1):
        # Debounce grinding days: H must be a local peak (higher than preceding bars and next bar)
        if tf_highs[H] < tf_highs[max(0, H - 2) : H].max() or tf_highs[H] < tf_highs[H + 1]:
            continue

        ext10 = (tf_highs[H] - tf_e10[H]) / tf_e10[H] * 100.0
        ext20 = (tf_highs[H] - tf_e20[H]) / tf_e20[H] * 100.0
        if ext10 < min_ext_10 or ext20 < min_ext_20:
            continue

        lo_start = max(0, H - max_pole_bars)
        L = lo_start + int(np.argmin(tf_lows[lo_start:H]))
        if H - L < 3:
            continue

        lo_p = tf_lows[L]
        hi_p = tf_highs[H]
        if tf_highs[L : H + 1].max() > hi_p + 1e-4 or tf_lows[L : H + 1].min() < lo_p - 1e-4:
            continue

        move_pct = (hi_p / lo_p - 1.0) * 100.0
        if move_pct < min_move:
            continue

        cand_poles.append({
            "L": L,
            "H": H,
            "pole_start_date": str(tf_dates[L].date()),
            "pole_end_date": str(tf_dates[H].date()),
            "pole_low": round(float(lo_p), 2),
            "pole_high": round(float(hi_p), 2),
            "move_pct": round(float(move_pct), 1),
            "bars_in_swing": H - L + 1,
            "ext10": round(float(ext10), 1),
            "ext20": round(float(ext20), 1),
        })

    final_setups = []
    win = config.QMB_CONTRACT_WIN
    vol_ok_arr = np.zeros(len(daily_dates), bool)
    for k in range(win, len(daily_dates)):
        vb = daily_vols[k - win : k].mean()
        vol_ok_arr[k] = vb > 0 and daily_vols[k] >= config.QMB_BREAKOUT_VOL * vb

    for cand in cand_poles:
        p_start_d = pd.to_datetime(cand["pole_start_date"])
        p_end_d = pd.to_datetime(cand["pole_end_date"])
        if p_start_d not in daily_dates or p_end_d not in daily_dates:
            d_start_idx = daily_dates.get_indexer([p_start_d], method="nearest")[0]
            d_end_idx = daily_dates.get_indexer([p_end_d], method="nearest")[0]
        else:
            d_start_idx = daily_dates.get_loc(p_start_d)
            d_end_idx = daily_dates.get_loc(p_end_d)

        pole_high_p = cand["pole_high"]
        cons_limit = min(len(daily_dates), d_end_idx + 1 + max_cons)

        caught_up = False
        was_pinched = False
        min_pinch_seen = 999.0
        latest_primed = None
        found_breakout = False

        for eval_idx in range(d_end_idx + 1, cons_limit):
            cons_bars = eval_idx - d_end_idx

            # If price breaks above pole high before EMA catchup, runup was still underway
            if daily_highs[eval_idx] > pole_high_p and not caught_up:
                break

            # 1. EMA Catchup verification (moving average catches up to price)
            if not caught_up:
                if daily_lows[eval_idx] <= daily_e10[eval_idx] * 1.005 or daily_lows[eval_idx] <= daily_e20[eval_idx] * 1.01:
                    caught_up = True

            # 2. Controlled pullback limit
            c_low = float(daily_lows[d_end_idx : eval_idx + 1].min())
            pullback = (1.0 - c_low / pole_high_p) * 100.0
            if pullback > max_pb:
                break

            # 3. Surfing EMAs & 50 SMA floor
            if daily_closes[eval_idx] < daily_s50[eval_idx] or daily_lows[eval_idx] < daily_e20[eval_idx] * 0.97:
                break

            cons_sl = slice(d_end_idx + 1, eval_idx + 1)
            ride = setups._qm_ride(
                daily_closes[cons_sl], daily_lows[cons_sl],
                daily_e10[cons_sl], daily_e20[cons_sl],
                daily_s50[cons_sl]
            )
            if ride is None:
                continue

            # 4. Volatility Pinch vs 20-Day ADR:
            # Once the base pinches, it enters the "primed" regime. Breakouts naturally expand bands.
            pr = pinch_ratio[eval_idx]
            if np.isfinite(pr):
                if pr <= max_pinch_ratio:
                    was_pinched = True
                min_pinch_seen = min(min_pinch_seen, pr)

            if caught_up and was_pinched:
                st = _fast_breakout_stack(
                    daily_highs, daily_closes, daily_atr, piv_daily,
                    d_end_idx, eval_idx, vol_ok_arr, config.BREAKOUT_STACK_DOWNSWING
                )
                trig = st["trigger"]
                state = "breakout" if st["broke_today"] else "primed"
                vbase = daily_vols[max(0, eval_idx - win) : eval_idx].mean() if eval_idx > win else daily_vols[:eval_idx].mean()
                vol_x = round(float(daily_vols[eval_idx] / vbase), 2) if vbase > 0 else None

                best_pinch = min_pinch_seen if min_pinch_seen < 999 else (pr if np.isfinite(pr) else 0.0)

                setup_dict = {
                    "symbol": symbol,
                    "tf": tf,
                    "algo": "EMA Catchup & BB Pinch",
                    "pole_type": f"Ext {cand['ext10']}% / Pinch {round(float(best_pinch), 2)}x",
                    "state": state,
                    "date": str(daily_dates[eval_idx].date()),
                    "pole_start_date": cand["pole_start_date"],
                    "pole_end_date": cand["pole_end_date"],
                    "pole_low": cand["pole_low"],
                    "pole_high": cand["pole_high"],
                    "move_pct": cand["move_pct"],
                    "ext10": cand["ext10"],
                    "ext20": cand["ext20"],
                    "bars_in_swing": cand["bars_in_swing"],
                    "cons_start_date": str(daily_dates[d_end_idx + 1].date()),
                    "cons_end_date": str(daily_dates[eval_idx].date()),
                    "cons_bars": cons_bars,
                    "pullback_pct": round(pullback, 1),
                    "pinch_ratio": round(float(best_pinch), 2),
                    "trigger": round(trig, 2),
                    "vol_x": vol_x,
                    "ride_frac": ride["ride_frac"],
                    "shakeouts": ride["shakeouts"],
                    "levels": st["n_levels"],
                }

                if state == "breakout":
                    final_setups.append(setup_dict)
                    found_breakout = True
                    break
                else:
                    latest_primed = setup_dict

        if not found_breakout and latest_primed is not None:
            final_setups.append(latest_primed)

    # Debounce setups sharing the same evaluation date
    debounced = []
    final_setups.sort(key=lambda s: s["date"])
    for s in final_setups:
        if not debounced:
            debounced.append(s)
            continue
        last = debounced[-1]
        if s["date"] == last["date"]:
            if s["state"] == "breakout" and last["state"] != "breakout":
                debounced[-1] = s
            elif s["move_pct"] > last["move_pct"]:
                debounced[-1] = s
        else:
            debounced.append(s)

    return debounced


def detect_original_qm_poles(
    df_daily: pd.DataFrame,
    symbol: str,
    tf: str = "1D",
    min_move: float = 15.0,
    min_cons: int = 3,
    max_cons: int = 45,
    max_pb: float = 35.0,
    max_pinch_ratio: float = 3.5,
    max_pole_bars: int = 63,
    piv_daily: Optional[set[int]] = None
) -> List[Dict[str, Any]]:
    """
    Original Qullamaggie Breakout Setup Detector (from setups.py / mine_qm_setups.py)
    with optional Bollinger Band Squeeze confirmation.
    Mechanical logic:
      1. Pole Impulse: Stock runs up >= min_move% from launch low to absolute high within lookback.
      2. 10/20 EMA Surf & 50 SMA Floor: Price surfs above 20 EMA and stays above rising 50 SMA floor.
      3. Higher Lows: Lows across the consolidation base trend net higher.
      4. Bollinger Band Squeeze: BB width / ADR20 contracts to <= max_pinch_ratio prior to breakout.
      5. Breakout Stack: Dynamic stairstep levels identify nearest resistance.
      6. Breakout vs Building: State is 'breakout' on volume expansion above trigger, or 'building' while coiling.
    """
    if "ema10" not in df_daily.columns:
        df_daily = indicators.add_indicators(df_daily)
    df_tf = _resample(df_daily, tf)
    if df_tf is None or len(df_tf) < 30:
        return []

    tf_highs = df_tf["high"].to_numpy(float)
    tf_lows = df_tf["low"].to_numpy(float)
    tf_dates = df_tf.index
    n_tf = len(df_tf)

    daily_highs = df_daily["high"].to_numpy(float)
    daily_lows = df_daily["low"].to_numpy(float)
    daily_closes = df_daily["close"].to_numpy(float)
    daily_vols = df_daily["volume"].to_numpy(float)
    daily_e10 = df_daily["ema10"].to_numpy(float)
    daily_e20 = df_daily["ema20"].to_numpy(float)
    daily_s50 = df_daily["sma50"].to_numpy(float)
    daily_atr = df_daily["atr14"].to_numpy(float)
    daily_dates = df_daily.index

    # Standard Bollinger Bands (20, 2) and 20-Day ADR on daily
    std20 = df_daily["close"].rolling(20).std().to_numpy(float)
    bb_w = 4.0 * std20
    adr20 = (df_daily["high"] - df_daily["low"]).rolling(20).mean().to_numpy(float)
    pinch_ratio = bb_w / adr20

    if piv_daily is None:
        piv_daily = {i for i, _ in trendlab.swing_highs(df_daily, len(df_daily))}

    cand_poles = []
    seen = set()
    for H in range(min_cons + 5, n_tf):
        h_start = max(0, H - max_cons)
        h_end = H - min_cons
        if h_end < h_start:
            continue
        for hi_pos in range(h_end, h_start - 1, -1):
            mstart = max(0, hi_pos - max_pole_bars)
            mlow_pos = mstart + int(np.argmin(tf_lows[mstart : hi_pos + 1]))
            sub_hi = tf_highs[mlow_pos : h_end + 1]
            hi_abs = mlow_pos + int(np.argmax(sub_hi))
            if hi_pos != hi_abs:
                continue
            lo_p = float(tf_lows[mlow_pos])
            hi_p = float(tf_highs[hi_pos])
            if lo_p <= 0:
                continue
            move = (hi_p / lo_p - 1.0) * 100.0
            if move < min_move:
                continue
            key = (mlow_pos, hi_pos)
            if key not in seen:
                seen.add(key)
                cand_poles.append({
                    "lo_idx": mlow_pos,
                    "hi_idx": hi_pos,
                    "pole_start_date": str(tf_dates[mlow_pos].date()),
                    "pole_end_date": str(tf_dates[hi_pos].date()),
                    "pole_low": round(lo_p, 2),
                    "pole_high": round(hi_p, 2),
                    "move_pct": round(move, 1),
                    "bars_in_swing": hi_pos - mlow_pos + 1,
                })

    final_setups = []
    win = config.QMB_CONTRACT_WIN
    vol_ok_arr = np.zeros(len(daily_dates), bool)
    for k in range(1, len(daily_dates)):
        vb = daily_vols[max(0, k - win) : k].mean()
        vol_ok_arr[k] = vb > 0 and daily_vols[k] >= config.QMB_BREAKOUT_VOL * vb

    for cand in cand_poles:
        p_start_d = pd.to_datetime(cand["pole_start_date"])
        p_end_d = pd.to_datetime(cand["pole_end_date"])
        d_start_matches = np.where(daily_dates >= p_start_d)[0]
        d_end_matches = np.where(daily_dates <= p_end_d)[0]
        if len(d_start_matches) == 0 or len(d_end_matches) == 0:
            continue
        d_start_idx = int(d_start_matches[0])
        d_end_idx = int(d_end_matches[-1])
        if d_end_idx <= d_start_idx or d_end_idx >= len(daily_dates) - 1:
            continue

        cons_limit = min(d_end_idx + max_cons + 1, len(daily_dates))
        pole_high_p = cand["pole_high"]
        found_breakout = False
        latest_building = None
        was_pinched = False
        min_pinch_seen = 999.0

        for eval_idx in range(d_end_idx + min_cons, cons_limit):
            cons_bars = eval_idx - d_end_idx

            # Track Bollinger Band pinch
            pr = pinch_ratio[eval_idx]
            if np.isfinite(pr):
                if pr < min_pinch_seen:
                    min_pinch_seen = pr
                if pr <= max_pinch_ratio:
                    was_pinched = True

            # If squeeze is required, base must have pinched
            if max_pinch_ratio is not None and max_pinch_ratio > 0 and not was_pinched:
                continue

            c_low = float(daily_lows[d_end_idx : eval_idx + 1].min())
            pullback = (1.0 - c_low / pole_high_p) * 100.0
            if pullback > max_pb:
                continue

            cons_sl = slice(d_end_idx + 1, eval_idx + 1)
            ride = setups._qm_ride(
                daily_closes[cons_sl], daily_lows[cons_sl],
                daily_e10[cons_sl], daily_e20[cons_sl], daily_s50[cons_sl]
            )
            if ride is None:
                continue

            lo_start = d_end_idx + 1 + int(np.argmin(daily_lows[d_end_idx + 1 : eval_idx + 1]))
            if not setups._qm_higher_lows(df_daily, lo_start, eval_idx):
                continue

            st = _fast_breakout_stack(
                daily_highs, daily_closes, daily_atr, piv_daily,
                d_end_idx, eval_idx, vol_ok_arr, config.BREAKOUT_STACK_DOWNSWING
            )
            trig = st["trigger"]
            state = "breakout" if st["broke_today"] else "building"
            vbase = daily_vols[max(0, eval_idx - win) : eval_idx].mean() if eval_idx > win else daily_vols[:eval_idx].mean()
            vol_x = round(float(daily_vols[eval_idx] / vbase), 2) if vbase > 0 else None

            try:
                pat = patterns.classify_consolidation(df_daily, d_end_idx + 1, eval_idx)
                pat_name = pat.get("pattern", "Consolidation")
            except Exception:
                pat_name = "Consolidation"

            best_pinch = min_pinch_seen if min_pinch_seen < 999 else (pr if np.isfinite(pr) else 0.0)

            setup_dict = {
                "symbol": symbol,
                "tf": tf,
                "algo": "Original QM",
                "pole_type": pat_name,
                "pattern": pat_name,
                "state": state,
                "date": str(daily_dates[eval_idx].date()),
                "pole_start_date": cand["pole_start_date"],
                "pole_end_date": cand["pole_end_date"],
                "pole_low": cand["pole_low"],
                "pole_high": cand["pole_high"],
                "move_pct": cand["move_pct"],
                "bars_in_swing": cand["bars_in_swing"],
                "cons_start_date": str(daily_dates[d_end_idx + 1].date()),
                "cons_end_date": str(daily_dates[eval_idx].date()),
                "cons_bars": cons_bars,
                "pullback_pct": round(pullback, 1),
                "pinch_ratio": round(float(best_pinch), 2),
                "trigger": round(trig, 2),
                "vol_x": vol_x,
                "ride_frac": ride["ride_frac"],
                "shakeouts": ride["shakeouts"],
                "levels": st["n_levels"],
            }
            if state == "breakout":
                final_setups.append(setup_dict)
                found_breakout = True
                break
            else:
                latest_building = setup_dict

        if not found_breakout and latest_building is not None:
            final_setups.append(latest_building)

    # Debounce overlapping setups
    final_setups.sort(key=lambda s: s["date"])
    debounced = []
    for s in final_setups:
        if not debounced:
            debounced.append(s)
            continue
        last = debounced[-1]
        if s["date"] == last["date"] or (
            s["pole_end_date"] == last["pole_end_date"]
            and abs((pd.to_datetime(s["date"]) - pd.to_datetime(last["date"])).days) <= 7
        ):
            if s["state"] == "breakout" and last["state"] != "breakout":
                debounced[-1] = s
            elif s["move_pct"] > last["move_pct"]:
                debounced[-1] = s
        else:
            debounced.append(s)

    return debounced


def scan_symbol_multi_tf(
    symbol: str,
    algo: str = "original_qm",
    tfs: List[str] = ("1D", "2D", "3D", "1W"),
    min_move: float = 15.0,
    min_r2: float = 0.65,
    min_er: float = 0.40,
    atr_frac: float = 0.3,
    min_cons: int = 3,
    max_cons: int = 45,
    max_pb: float = 35.0,
    min_ext_10: float = 3.0,
    min_ext_20: float = 5.0,
    max_pinch_ratio: float = 3.2,
    inside_mode: str = "pole_range"
) -> List[Dict[str, Any]]:
    """Runs chosen pole detection algorithm across multiple requested timeframes."""
    df_daily = datastore.load_bars(symbol)
    if df_daily is None or len(df_daily) < 60:
        return []
    df_daily = indicators.add_indicators(df_daily)
    piv_daily = {i for i, _ in trendlab.swing_highs(df_daily, len(df_daily))}

    all_hits = []
    for tf in tfs:
        if algo == "original_qm":
            hits = detect_original_qm_poles(
                df_daily, symbol, tf=tf,
                min_move=min_move, min_cons=min_cons, max_cons=max_cons,
                max_pb=max_pb, max_pinch_ratio=max_pinch_ratio,
                piv_daily=piv_daily
            )
        elif algo == "linear_regression":
            hits = detect_linear_regression_poles(
                df_daily, symbol, tf=tf,
                min_move=min_move, min_r2=min_r2, min_er=min_er,
                min_cons=min_cons, max_cons=max_cons, max_pb=max_pb,
                piv_daily=piv_daily
            )
        elif algo == "ema_catchup_bb_pinch":
            hits = detect_ema_catchup_bb_pinch_poles(
                df_daily, symbol, tf=tf,
                min_move=min_move, min_ext_10=min_ext_10, min_ext_20=min_ext_20,
                max_pinch_ratio=max_pinch_ratio, max_pb=max_pb,
                max_cons=max_cons, piv_daily=piv_daily
            )
        else:
            hits = detect_higher_low_poles(
                df_daily, symbol, tf=tf, atr_frac=atr_frac,
                min_move=min_move, min_cons=min_cons,
                max_cons=max_cons, max_pb=max_pb,
                inside_mode=inside_mode,
                piv_daily=piv_daily
            )
        all_hits.extend(hits)

    # Sort hits by date descending
    all_hits.sort(key=lambda x: x["date"], reverse=True)
    return all_hits


