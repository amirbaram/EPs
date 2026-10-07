"""Build data/setup_registry/{setup}.parquet — every historical TRIGGER of a setup across the
full bar cache: (symbol, date, state). Precomputed ONCE so the app's setup-history panel never
recomputes (Amir 2026-07-03); re-run after detector changes or to extend history.

Engines:
  fast    — one pass per symbol reusing the detector's own event primitives
            (backburner: _bb_os_state; episodic_pivot: _is_ep_event per row;
             gapper / hvc: per-row event tests)
  sliding — full detector called on every as-of slice (structural setups); slow but exact.
            Run these overnight: nohup .venv/bin/python build_setup_registry.py --sliding &

    .venv/bin/python build_setup_registry.py            # fast setups only (minutes)
    .venv/bin/python build_setup_registry.py --sliding  # the structural rest (hours)
    .venv/bin/python build_setup_registry.py --setups qm_breakout,flat_base
"""

from __future__ import annotations

import argparse
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

import config
import datastore
import emarider
import setups
from indicators import add_indicators

OUT = config.DATA_DIR / "setup_registry"
TRIGGERS = {"breakout", "breakout_lowvol", "breakdown", "entry1", "entry2"}
# ONLY chart-verified setups get a registry/history build (Amir 2026-07-03). Add a setup here
# when its verdict pass completes (see the graduation checklist in DOCS.md); until then a
# one-off build needs an explicit --setups <name> --unvalidated.
VALIDATED = {"backburner", "backburner_rsi25", "backburner_rsi20", "episodic_pivot",
             "cup_handle", "double_top", "head_shoulders", "inverse_hs",
             # Amir's verdict pass 2026-07-03 (all-ok JSON): flat base, gapper subtypes, HVC,
             # delayed HVC, high tight flag, QM breakout. Stairstep passed at the 8-bar retune;
             # MIN_BARS then bumped to 10 (a strict subset of the verified detections).
             "flat_base", "gapper", "hvc", "delayed_hvc", "high_tight_flag", "qm_breakout",
             "stairstep",
             # EMA-rider arm-exit state machine (2026-07-07): NVDA chart-verified + Amir go. The registry
             # stores the FULL state timeline (every armed/saved/break transition, per ride side).
             "ema_rider_bull", "ema_rider_bear"}
FAST = ["backburner", "backburner_rsi25", "backburner_rsi20", "episodic_pivot", "gapper", "hvc",
        "ema_rider_bull", "ema_rider_bear"]
SLIDING = ["flat_base", "high_tight_flag", "qm_breakout", "delayed_hvc", "stairstep",
           "higher_low_ma", "undercut_rally", "cup_handle", "double_top",
           "head_shoulders", "inverse_hs"]
_DET = {name: fn for fn, name in setups.SETUP_OF.items()}


def _frames():
    import universe as uni
    active = uni.active_symbols()
    for sym in datastore.list_symbols():
        if active is not None and sym not in active:
            continue
        d = datastore.load_bars(sym)
        if d is not None and len(d) >= 120:
            yield sym, add_indicators(d)


def fast_rows(setup: str, sym: str, d: pd.DataFrame) -> list[dict]:
    out = []
    dates = [x.date().isoformat() for x in d.index]
    if setup.startswith("backburner"):
        lvl = float(setup.rsplit("rsi", 1)[1]) if "rsi" in setup else None   # backburner_rsi25 -> 25
        fires: list = []
        setups._bb_os_state(d, collect=fires, l1=lvl)
        out += [{"symbol": sym, "date": dates[t], "state": st} for t, st in fires]
    elif setup in ("rsi_extreme_revert", "rsi_extreme_fade"):
        import rsi_extremes as rx
        fires: list = []
        rx.fires(d, "long" if setup.endswith("revert") else "short", collect=fires)
        out += [{"symbol": sym, "date": dates[t], "state": "entry1"}
                for t in fires if t >= config.RSIX_MIN_HISTORY]   # skip young-series artifacts
    elif setup == "episodic_pivot":
        med60 = (d["close"] * d["volume"]).rolling(60, min_periods=10).median().shift(1)
        for i in range(30, len(d)):
            sub = setups._is_ep_event(d.iloc[i], med60.iloc[i] if pd.notna(med60.iloc[i]) else None)
            if sub:
                out.append({"symbol": sym, "date": dates[i], "state": f"breakout:{sub}"})
    elif setup == "gapper":
        m = (d["gap_pct"] >= config.GAP_MIN_PCT) & (d["rvol"] >= config.GAP_MIN_RVOL)
        out += [{"symbol": sym, "date": dates[i], "state": "breakout"}
                for i in list(m[m].index.map(d.index.get_loc))]
    elif setup == "hvc":
        for i in range(60, len(d)):
            if setups._is_hvc_bar(d.iloc[i]):
                out.append({"symbol": sym, "date": dates[i], "state": "breakout"})
    elif setup in ("ema_rider_bull", "ema_rider_bear"):
        # full state timeline: every armed/saved/break transition, attributed to the ride side.
        # armed/saved carry the ride's own direction; a break's dir is POST-flip, so it belongs to -dir.
        want = 1 if setup.endswith("bull") else -1
        for ev in emarider.state_timeline(d, config.ER_EMA_LEN, config.ER_ATR_LEN, config.ER_ATR_FRAC,
                                          config.ER_USE_STREAK_THRESH, config.ER_STREAK_ATR_FRAC,
                                          config.ER_USE_ARM_EXIT):
            side = ev["dir"] if ev["event"] != "break" else -ev["dir"]
            if side == want:
                out.append({"symbol": sym, "date": ev["date"], "state": ev["event"]})
    return out


def _trigger_candidates(setup: str, d: pd.DataFrame):
    """RETIRED (returns None = full bar-by-bar replay, Amir 2026-07-03). Price-only prefilters
    provably LOSE fires: the 2026-07-03 spot check vs full replay found misses under both the
    "20-bar-high or ±1.5%" proxy (MRVL 2025-01-06, ARKK 2025-08-13 — Amir-verdict-passed dates)
    AND the up-close superset (SSB 2024-08-26, BBWI 2020-09-18, BEP 2021-02-09 — a pattern can
    become VALID days after the price cross, so the first reported breakout can be a DOWN day).
    With the DETECT_WINDOW cap the full replay costs ~0.3-0.9s/symbol — affordable; correctness
    wins. The REGISTRY_BARS / DETECT_WINDOW caps below still apply."""
    return None


REGISTRY_BARS = 1500          # slide the last ~6 years (some parquets carry decades)
DETECT_WINDOW = 420           # bars fed per detector call — every template's structure + prior-
                              # trend lookback fits in ~260; 420 keeps calls O(1) on old giants


def sliding_rows(setup: str, sym: str, d: pd.DataFrame, step: int = 1) -> list[dict]:
    fn = _DET[setup]
    cand = _trigger_candidates(setup, d)
    out = []
    prev = None
    for k in range(max(90, len(d) - REGISTRY_BARS), len(d), step):
        if cand is not None and not bool(cand.iloc[k]):
            prev = None                              # off-candidate day: episode boundary resets
            continue
        try:
            m = fn(d.iloc[max(0, k - DETECT_WINDOW + 1):k + 1])
        except Exception:
            m = None
        st = m["state"] if m and m.get("state") in TRIGGERS else None
        if st and prev != st:                        # first day of each trigger episode
            out.append({"symbol": sym, "date": d.index[k].date().isoformat(), "state": st})
        prev = st
    return out


def build(setup: str, engine, force: bool = False) -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if not force and (OUT / f"{setup}.parquet").exists():
        print(f"{setup}: already built — skipping (use --force to rebuild)", flush=True)
        return
    rows = []
    n = 0
    for sym, d in _frames():
        rows += engine(setup, sym, d)
        n += 1
        if n % 200 == 0:
            print(f"  {setup}: {n} symbols, {len(rows)} fires, {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows, columns=["symbol", "date", "state"])
    df.to_parquet(OUT / f"{setup}.parquet", index=False)
    print(f"{setup}: {len(df)} fires across {df['symbol'].nunique() if len(df) else 0} symbols "
          f"-> {OUT}/{setup}.parquet ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sliding", action="store_true", help="build the slow structural setups")
    ap.add_argument("--setups", help="comma list override")
    ap.add_argument("--force", action="store_true", help="rebuild even if the parquet exists")
    ap.add_argument("--unvalidated", action="store_true",
                    help="allow building a setup that is NOT on the VALIDATED whitelist "
                         "(one-off use during a validation pass; requires --setups)")
    args = ap.parse_args()
    if args.setups:
        todo = args.setups.split(",")
    else:
        todo = SLIDING if args.sliding else FAST
    if not args.unvalidated:
        skipped = [s for s in todo if s not in VALIDATED]
        if skipped:
            print(f"skipping unvalidated (no registry until chart-verified): {', '.join(skipped)}")
        todo = [s for s in todo if s in VALIDATED]
    for s in todo:
        build(s, sliding_rows if s in SLIDING else fast_rows, force=args.force)


if __name__ == "__main__":
    main()
