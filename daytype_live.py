"""DAY-TYPE LIVE SCORER (H3, Amir approved 2026-07-04): evaluate the frozen production model
artifacts (daytype_model{_sym}.json from `daytype_prob.py --production`) OUTSIDE the research
batch — the module the app's banner strip calls.

Stages (graceful degradation by design: a missing feature contributes 0 log-lift, so the
probability falls toward the base rate instead of breaking):

  --prep       nightly / pre-open: compute TODAY's bucket-A features (everything knowable
               before the open) via the SAME dayfeatures blocks, using the stub-row trick —
               append today's date to the calendar; every a_* feature is a shift(1), so
               today's row pulls yesterday's settled values with no intraday bars needed.
               Self-verifies: recomputes the LAST SETTLED session and diffs it against the
               batch features parquet before writing data/features/daytype_live_pre.json.
  --score      score a checkpoint from whatever features are cached/available; prints the
               probabilities + skill-gated narrative. serve.py calls score_payload().

STAGED (anchored in BUGS.md): intraday b_/c_ features live via tiingo IEX today-bars for
sym+SPDRs+QQQ (extra_bars param into spy_intraday_block) and sample-universe %>VWAP with
rank-mapped edges. Until then intraday checkpoints score from a_* alone (labeled in payload).

    .venv/bin/python daytype_live.py --prep
    .venv/bin/python daytype_live.py --score --sym SPY
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config
import datastore
import daytype
import dayfeatures

FDIR = config.DATA_DIR / "features"
PRE = FDIR / "daytype_live_pre.json"
LOG = FDIR / "daytype_live_log.csv"
STORE = config.DATA_DIR / "tiingo" / "bars_5min"
SYMS = ["SPY", "QQQ", "IWM", "DIA"]


# ── artifact evaluation ──────────────────────────────────────────────────────────

def _bin_value(v, bdef: dict):
    """Raw value -> bin id per the artifact's edges (mirror of daytype_prob.bin_ids)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return -1
    if bdef["kind"] == "bool":
        return int(bool(v))
    for i, e in enumerate(bdef["edges"]):
        if v <= e:
            return i
    return 4


def _inter_state(feats: dict, bins: dict, bdef: dict) -> int:
    def side(col, tag):
        v = feats.get(col)
        bd = bins.get(col)
        if v is None or bd is None:
            return False
        if tag == "flag":
            return bool(v)
        b = _bin_value(v, bd)
        return b == 4 if tag == "top-q" else (b == 0 if b >= 0 else False)
    return int(side(bdef["c1"], bdef["t1"]) and side(bdef["c2"], bdef["t2"]))


def score_features(model: dict, feats: dict, cp: str) -> dict:
    """feats = {raw column: value}. Missing/NaN features contribute 0."""
    out = {}
    bins = model["bins"]
    for tname, m in model["checkpoints"].get(cp, {}).items():
        raw, used, missing = 0.0, 0, 0
        for c, bl in m["feats"].items():
            bd = bins.get(c)
            if bd is None:
                missing += 1
                continue
            if bd["kind"] == "inter":
                b = _inter_state(feats, bins, bd)
                if bd["c1"] not in feats and bd["c2"] not in feats:
                    missing += 1
                    continue
            else:
                if c not in feats:
                    missing += 1
                    continue
                b = _bin_value(feats[c], bd)
            lift = bl.get(str(b), 0.0)
            raw += lift
            used += 1
        base = m["base"]
        p = 1 / (1 + np.exp(-(np.log(base / (1 - base)) + raw / m["T"])))
        out[tname] = {"p": round(float(p), 4), "base": base,
                      "feats_used": used, "feats_missing": missing}
    return out


def score_features2(model2: dict, sfeats: dict, dfeats: dict, mins: int) -> dict:
    """PROB V2 runtime scorer: per-bar probabilities from daytype_model2 artifacts.
    sfeats = {s_ column: raw value} for THIS bar, dfeats = {day-level column: value}.
    Missing/unavailable features contribute 0 lift (degrade toward base)."""
    if mins < 575:
        tmap, bins_src = model2["pre"], None
    else:
        tmap = None
        for bname, b in model2["buckets"].items():
            if b["lo"] <= mins <= b["hi"]:
                tmap = b["targets"]
                break
        if tmap is None:                       # after 16:00: score with the last bucket
            tmap = model2["buckets"]["1430"]["targets"]
    bar = max(0, min((mins - 575) // 5, 77))
    dbins = model2["dbins"]
    c_avail = model2.get("c_avail", {})

    def avail(col: str) -> bool:
        for pref, t0 in sorted(c_avail.items(), key=lambda kv: -len(kv[0])):
            if col.startswith(pref):
                return mins >= t0
        return True                            # a_/b_: known pre-open

    out = {}
    for tname, m in tmap.items():
        raw, used, missing = 0.0, 0, 0
        for c, bl in m["feats"].items():
            b = -1
            if c.startswith("s_"):
                v = sfeats.get(c)
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    missing += 1
                    continue
                if c in model2.get("sbinary", []):
                    b = int(bool(v))
                else:
                    edges = model2["sbins"].get(c, {}).get(str(bar))
                    if edges is None:
                        missing += 1
                        continue
                    b = _bin_value(v, {"kind": "num", "edges": edges})
            else:
                bd = dbins.get(c)
                if bd is None or not avail(c):
                    missing += 1
                    continue
                if bd["kind"] == "inter":
                    if (bd["c1"] not in dfeats and bd["c2"] not in dfeats) \
                            or not (avail(bd["c1"]) and avail(bd["c2"])):
                        missing += 1
                        continue
                    b = _inter_state(dfeats, dbins, bd)
                else:
                    if c not in dfeats:
                        missing += 1
                        continue
                    b = _bin_value(dfeats[c], bd)
            raw += bl.get(str(b), 0.0)
            used += 1
        base = m["base"]
        p = 1 / (1 + np.exp(-(np.log(base / (1 - base)) + raw / m["T"])))
        out[tname] = {"p": round(float(p), 4), "base": base,
                      "feats_used": used, "feats_missing": missing}
    return out


def h1_tier(w1b, breadth_dhr) -> str | None:
    """THE shared H1 warning-tier definition (Amir's operating point, 2026-07-05) — the
    dev-page cursor box and the live strip must both call/mirror exactly this:
      AMBER 'attention'  = W1b has fired (two consecutive closes on the wrong side of VWAP
                           after a >=90-min one-sided run; ~30min lead, FP 31%);
      RED 'de-risk with-trend exposure' = W1b AND breadth decay <= -10 pts vs one hour ago
                           (17%->43% reversal odds, era-stable)."""
    if not w1b:
        return None
    if breadth_dhr is not None and np.isfinite(breadth_dhr) and breadth_dhr <= -10:
        return "red"
    return "amber"


def narrative(sc: dict, skill: dict, cp: str) -> tuple[str, str]:
    """Server-side twin of the review page's skill-gated stance()."""
    t, u, d = (sc.get(k, {}) for k in ("trend", "up", "down"))
    if not t or not u or not d:
        return "no model read", "#8aa0b8"
    ur, dr, tr = u["p"] / u["base"], d["p"] / d["base"], t["p"] / t["base"]
    sk = skill.get(cp, {})
    gate = lambda s: 2 if (s or 0) > 5 else (1 if (s or 0) >= 1.5 else 0)
    if ur >= 1.3 and dr <= 0.8:
        g = gate(sk.get("up"))
        if g == 0:
            return f"slight up tilt — direction evidence WEAK at this hour (skill {sk.get('up')}%)", "#8aa0b8"
        if g == 1 or not (ur >= 1.7 and dr <= 0.5):
            return f"up lean — favor longs, normal size (skill {sk.get('up')}%)", "#8fd6bb"
        return f"strong UP lean — longs with size (skill {sk.get('up')}%)", "#2fbf8f"
    if dr >= 1.3 and ur <= 0.8:
        g = gate(sk.get("down"))
        if g == 0:
            return f"slight down tilt — direction evidence WEAK at this hour (skill {sk.get('down')}%)", "#8aa0b8"
        if g == 1 or not (dr >= 1.7 and ur <= 0.5):
            return f"down lean — favor shorts, normal size (skill {sk.get('down')}%)", "#e8a3ae"
        return f"strong DOWN lean — shorts with size (skill {sk.get('down')}%)", "#e05a6d"
    if tr <= 0.6 and ur < 1.2 and dr < 1.2:
        return "quiet day likely — fade extremes, small size", "#5aa2e0"
    if tr >= 1.5 and gate(sk.get("trend")) > 0:
        return "expansion risk elevated, direction unclear — wait for the break", "#e0a43a"
    return "no edge yet — let the tape develop", "#8aa0b8"


# ── --prep: today's bucket-A features via the batch blocks + stub row ────────────

A_ONLY_PREFIXES = ("a_",)


def prep(today: str | None = None) -> dict:
    if today is None:                             # next TRADING day (weekend/holiday-safe-ish)
        t = pd.Timestamp(dt.date.today())
        today = str((t if t.dayofweek < 5 else t + pd.offsets.BDay(1)).date())
    try:
        datastore.update_macro()                  # freshen BTC/HSI/DAX etc. for the MED block
    except Exception as e:                        # (never fatal; stale rows just ffill)
        print(f"  macro refresh skipped: {e}")
    out = {}
    for sym in SYMS:
        df = daytype.classify_all(pd.read_parquet(STORE / f"{sym}.parquet"))
        if today in df.index:                      # market already traded today: nothing to stub
            cal_x = dayfeatures._sess(df.index)
            df_x = df
        else:
            stub = df.iloc[[-1]].copy()
            stub.index = [today]
            df_x = pd.concat([df, stub])           # a_* are shift(1): stub content never read
            cal_x = dayfeatures._sess(df_x.index)
        blocks = [dayfeatures.spy_daily_block(df_x)]
        ub = dayfeatures.universe_daily_block(cal_x).shift(1).add_prefix("a_")
        blocks.append(ub.set_axis(df_x.index))
        blocks.append(pd.DataFrame(
            {"a_dd_spy": dayfeatures.dist_days("SPY", cal_x).shift(1).values,
             "a_dd_qqq": dayfeatures.dist_days("QQQ", cal_x).shift(1).values},
            index=df_x.index))
        blocks.append(dayfeatures.vol_block(cal_x).set_axis(df_x.index))
        blocks.append(dayfeatures.external_block(cal_x, sym).set_axis(df_x.index))
        blocks.append(dayfeatures.med_block(cal_x).set_axis(df_x.index))
        blocks.append(dayfeatures.calendar_block(cal_x).set_axis(df_x.index))
        vix = datastore.load_bars("^VIX")["close"].reindex(cal_x, method="ffill")
        rv = (df_x["range"] / df_x["close"]).values * 100
        blocks.append(pd.DataFrame({"a_rv_vs_iv": (pd.Series(rv, index=df_x.index)
                                                   / (vix.values / np.sqrt(252))).shift(1)}))
        F = pd.concat(blocks, axis=1)
        acols = [c for c in F.columns if c.startswith(A_ONLY_PREFIXES) or c.startswith("cal_")]
        # ── self-check: the LAST SETTLED row must match the batch parquet ──
        fp = FDIR / ("daytype_features.parquet" if sym == "SPY"
                     else f"daytype_features_{sym}.parquet")
        n_ok = n_diff = 0
        if fp.exists():
            B = pd.read_parquet(fp)
            last = df.index[-1]
            if last in B.index and last in F.index:
                for c in acols:
                    if c not in B.columns:
                        continue
                    a, b = F.at[last, c], B.at[last, c]
                    try:
                        if (pd.isna(a) and pd.isna(b)):
                            n_ok += 1
                        elif isinstance(a, str) or isinstance(b, str):
                            n_ok += int(a == b); n_diff += int(a != b)
                        elif isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
                            n_ok += int(bool(a) == bool(b)); n_diff += int(bool(a) != bool(b))
                        elif pd.notna(a) and pd.notna(b) and np.isclose(
                                float(a), float(b), rtol=1e-6, atol=1e-9):
                            n_ok += 1
                        else:
                            n_diff += 1
                    except Exception:
                        n_diff += 1
        row = F.iloc[-1]
        feats = {}
        for c in acols:
            v = row[c]
            if isinstance(v, str) or (not isinstance(v, (bool, np.bool_)) and pd.isna(v)):
                continue                            # labels/strings aren't model features
            feats[c] = bool(v) if isinstance(v, (bool, np.bool_)) else round(float(v), 6)
        out[sym] = {"feats": feats, "check_ok": n_ok, "check_diff": n_diff}
        print(f"{sym}: {len(feats)} pre-open features for {today} "
              f"(self-check vs batch on {df.index[-1]}: {n_ok} match, {n_diff} differ)")
    payload = {"date": today, "generated": dt.datetime.now().isoformat(timespec='seconds'),
               "syms": out}
    PRE.write_text(json.dumps(payload))
    print(f"-> {PRE}")
    return payload


# ── scoring entry points (serve.py imports these) ────────────────────────────────

CP_ORDER = [("open", 0), ("0935", 575), ("0945", 585), ("1000", 600), ("1030", 630),
            ("1130", 690), ("1230", 750), ("1330", 810), ("1430", 870)]
CP_LABEL = {"open": "pre-open", "0935": "09:35", "0945": "09:45", "1000": "10:00",
            "1030": "10:30", "1130": "11:30", "1230": "12:30", "1330": "13:30",
            "1430": "14:30"}


def _cp_for(t: str | None) -> str:
    """Latest checkpoint at or before HH:MM replay time; None (settled view) -> last."""
    if not t:
        return "1430"
    try:
        mins = int(t[:2]) * 60 + int(t[3:5])
    except Exception:
        return "1430"
    cp = "open"
    for name, m in CP_ORDER:
        if m <= mins:
            cp = name
    return cp


_BARP_CACHE: dict = {}          # sfx -> (bar probs df, bar feats df) for the history path


def _bar_frames(sfx: str):
    if sfx not in _BARP_CACHE:
        pp = FDIR / f"daytype_bar_probs{sfx}.parquet"
        bf = FDIR / f"daytype_bar_features{sfx}.parquet"
        P = pd.read_parquet(pp).set_index(["date", "mins"]) if pp.exists() else None
        W = (pd.read_parquet(bf, columns=["date", "mins", "s_w1b", "s_breadth_dhr"])
             .set_index(["date", "mins"]) if bf.exists() else None)
        _BARP_CACHE[sfx] = (P, W)
    return _BARP_CACHE[sfx]


def score_bar_history(sym: str, date: str, t: str | None) -> dict | None:
    """PROB V2 time-travel path: the batch PER-BAR probabilities read at the replay moment
    (latest 5-min bar close <= t). None -> fall back to the checkpoint path."""
    sfx = "" if sym == "SPY" else f"_{sym}"
    P, W = _bar_frames(sfx)
    if P is None or date not in P.index.get_level_values(0):
        return None
    mins = 960 if not t else int(t[:2]) * 60 + int(t[3:5])
    if mins < 575:
        return None                            # pre-open: checkpoint 'open' path handles it
    day = P.loc[date]
    at = day.index[day.index <= mins]
    if not len(at):
        return None
    m_ = int(at.max())
    row = day.loc[m_]
    mp = FDIR / f"daytype_model2{sfx}.json"
    model2 = json.loads(mp.read_text()) if mp.exists() else {}
    bname, base_map = "1430", {}
    for bn, b in model2.get("buckets", {}).items():
        if b["lo"] <= m_ <= b["hi"]:
            bname = bn
            base_map = {tn: v["base"] for tn, v in b["targets"].items()}
            break
    sc = {}
    for tgt in ("trend", "up", "down"):
        v = row.get(f"p_{tgt}")
        if v is None or pd.isna(v):
            continue
        sc[tgt] = {"p": round(float(v), 4),
                   "base": base_map.get(tgt, round(float(P[f"p_{tgt}"].mean()), 4))}
    if not sc:
        return None
    sp = FDIR / f"daytype_prob2_skill{sfx}.json"
    skill = json.loads(sp.read_text()).get("buckets", {}) if sp.exists() else {}
    txt, col = narrative(sc, skill, bname)
    out = {"sym": sym, "date": date, "cp": bname,
           "cp_label": f"{m_ // 60:02d}:{m_ % 60:02d} bar", "granularity": "5m",
           "mode": "history", "probs": sc, "stance": txt, "color": col}
    if W is not None and (date, m_) in W.index:
        wr = W.loc[(date, m_)]
        tier = h1_tier(bool(wr.get("s_w1b")), wr.get("s_breadth_dhr"))
        if tier:
            out["tier"] = tier
    hp = FDIR / f"daytype_hold_probs{sfx}.parquet"
    cp_hold = "1430" if m_ >= 870 else ("1330" if m_ >= 810 else ("1230" if m_ >= 750 else None))
    if hp.exists() and cp_hold:
        H = pd.read_parquet(hp)
        if date in H.index:
            h = H.loc[date]
            v = h.get(f"p_hold_{cp_hold}")
            if v is not None and pd.notna(v):
                out["hold"] = {"dir": int(h["dir"]), "event": str(h["event"]),
                               "p": round(float(v), 3)}
    return out


def score_history(sym: str, date: str, t: str | None) -> dict:
    """Time-travel path: per-bar (PROB V2) when the bar-probs parquet covers the date,
    else the v1 BATCH checkpoint probabilities (latest checkpoint <= t)."""
    try:
        v2 = score_bar_history(sym, date, t)
        if v2 is not None:
            return v2
    except Exception:
        pass
    sfx = "" if sym == "SPY" else f"_{sym}"
    pp = FDIR / f"daytype_probs{sfx}.parquet"
    if not pp.exists():
        return {"error": "no daytype_probs parquet — run daytype_prob.py"}
    P = pd.read_parquet(pp)
    if date not in P.index:
        return {"error": f"no day-type read for {date}"}
    mp = FDIR / f"daytype_model{sfx}.json"
    bases = {}
    if mp.exists():
        bases = json.loads(mp.read_text()).get("checkpoints", {})
    sp = FDIR / f"daytype_prob_skill{sfx}.json"
    skill = json.loads(sp.read_text()) if sp.exists() else {}
    cp = _cp_for(t)
    row = P.loc[date]
    sc = {}
    for tgt in ("trend", "up", "down"):
        v = row.get(f"p_{tgt}_{cp}")
        if v is None or pd.isna(v):
            continue
        base = bases.get(cp, {}).get(tgt, {}).get("base")
        sc[tgt] = {"p": round(float(v), 4),
                   "base": base if base is not None else round(float(P[f"p_{tgt}_{cp}"].mean()), 4)}
    if not sc:
        return {"error": f"no probabilities at checkpoint {cp} for {date}"}
    txt, col = narrative(sc, skill, cp)
    out = {"sym": sym, "date": date, "cp": cp, "cp_label": CP_LABEL.get(cp, cp),
           "mode": "history", "probs": sc, "stance": txt, "color": col}
    hp = FDIR / f"daytype_hold_probs{sfx}.parquet"
    if hp.exists():
        H = pd.read_parquet(hp)
        if date in H.index and cp in ("1230", "1330", "1430"):
            h = H.loc[date]
            v = h.get(f"p_hold_{cp}")
            if v is not None and pd.notna(v):
                out["hold"] = {"dir": int(h["dir"]), "event": str(h["event"]),
                               "p": round(float(v), 3)}
    return out

def score_payload(sym: str = "SPY") -> dict:
    mp = FDIR / f"daytype_model{'' if sym == 'SPY' else '_' + sym}.json"
    sp = FDIR / f"daytype_prob_skill{'' if sym == 'SPY' else '_' + sym}.json"
    if not mp.exists():
        return {"error": "no production model — run daytype_prob.py --production"}
    model = json.loads(mp.read_text())
    skill = json.loads(sp.read_text()) if sp.exists() else {}
    feats, fdate, fnote = {}, None, "no pre-open cache — run daytype_live.py --prep"
    if PRE.exists():
        pre = json.loads(PRE.read_text())
        fdate = pre["date"]
        feats = pre["syms"].get(sym, {}).get("feats", {})
        fnote = f"pre-open features for {fdate} ({len(feats)} cols)"
    sc = score_features(model, feats, "open")
    txt, col = narrative(sc, skill, "open")
    row = {"asof": dt.datetime.now().isoformat(timespec='seconds'), "sym": sym,
           "date": fdate, "cp": "open",
           **{f"p_{t}": v["p"] for t, v in sc.items()}}
    try:                                            # emission log -> daytype_monitor.py
        pd.DataFrame([row]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    except Exception:
        pass
    return {"sym": sym, "date": fdate, "cp": "open", "model_fitted": model.get("fitted"),
            "probs": sc, "stance": txt, "color": col, "features": fnote,
            "staged": "intraday checkpoints go live after the b_/c_ live-feature stage "
                      "(BUGS anchor) — pre-open expectations only for now"}


# ── H3 INTRADAY LIVE STAGE: today's 5-min bars -> the PROB V2 per-bar scorer ─────
# Today's bars come from the Tiingo IEX intraday endpoint (same source as the 5-min store)
# for the scored sym + its counterpart + the 11 SPDRs; b_/c_ day features come from the SAME
# dayfeatures.spy_intraday_block code via extra_bars (one definition). Universe breadth
# (s_breadth*/c_pct_above_vwap*) is STAGED pending the rank-map study — those features are
# simply missing, so probabilities degrade toward base, never break.

_IEX_URL = "https://api.tiingo.com/iex/{sym}/prices"
_LIVE_BARS: dict = {"key": None, "bars": None}     # one fetch cycle per closed 5-min bar
_DAILY_CACHE: dict = {}
_MODEL2_CACHE: dict = {}


def _tiingo_key() -> str | None:
    try:
        return (config.DATA_DIR / "tiingo_key.txt").read_text().strip()
    except OSError:
        return None


def _iex_5m(sym: str, key: str, date: str) -> pd.DataFrame:
    import urllib.parse
    import urllib.request
    q = urllib.parse.urlencode({"startDate": date, "resampleFreq": "5min",
                                "afterHours": "true",
                                "columns": "open,high,low,close,volume", "token": key})
    with urllib.request.urlopen(_IEX_URL.format(sym=sym) + "?" + q, timeout=15) as r:
        js = json.loads(r.read())
    if not js:
        return pd.DataFrame()
    f = pd.DataFrame(js)
    f.index = (pd.DatetimeIndex(pd.to_datetime(f["date"], utc=True))
               .tz_convert("America/New_York").tz_localize(None))
    return f[["open", "high", "low", "close", "volume"]].sort_index()


def _daily(sym: str) -> pd.DataFrame:
    k = (sym, dt.date.today().isoformat())
    if _DAILY_CACHE.get("k") != k:
        _DAILY_CACHE.clear()
        _DAILY_CACHE["k"] = k
        _DAILY_CACHE["df"] = daytype.classify_all(pd.read_parquet(STORE / f"{sym}.parquet"))
    return _DAILY_CACHE["df"]


def _model2(sym: str) -> dict | None:
    sfx = "" if sym == "SPY" else f"_{sym}"
    mp = FDIR / f"daytype_model2{sfx}.json"
    if not mp.exists():
        return None
    k = (sym, mp.stat().st_mtime)
    if _MODEL2_CACHE.get("k" + sym) != k:
        _MODEL2_CACHE["k" + sym] = k
        _MODEL2_CACHE["m" + sym] = json.loads(mp.read_text())
    return _MODEL2_CACHE["m" + sym]


def _cycle_bars(date: str, cutoff_mins: int, sim: bool) -> dict:
    """5-min frames (closed bars only, close time <= cutoff) for every sym the feature
    pass needs. Live: one IEX fetch per sym per closed-bar cycle. Sim: read the store."""
    import dayfeatures
    import datastore
    need = sorted(set(SYMS) | set(dayfeatures.SPDR))
    ck = (date, cutoff_mins, sim)
    if _LIVE_BARS["key"] == ck:
        return _LIVE_BARS["bars"]
    key = _tiingo_key()
    out = {}
    for s in need:
        f = pd.DataFrame()
        try:
            if sim:
                f = pd.read_parquet(STORE / f"{s}.parquet").loc[date]
            elif key:
                f = _iex_5m(s, key, date)                        # freshest: direct IEX for today
        except (KeyError, OSError):
            f = pd.DataFrame()
        except Exception:
            f = pd.DataFrame()
        if not sim and not len(f):
            try:                                                # live fallback: the overlay-current 5m
                b = datastore.load_bars_5m(s, calibrate=False)  # store (RAW-IEX basis, poller-fed) —
                if b is not None and len(b):                    # keeps the day-type read live if the
                    day = b[b.index.normalize() == pd.Timestamp(date)]   # direct fetch is unavailable
                    if len(day):
                        f = day[["open", "high", "low", "close", "volume"]]
            except Exception:
                f = pd.DataFrame()
        if len(f):
            m = f.index.hour * 60 + f.index.minute + 5          # bar CLOSE minutes
            f = f[m <= cutoff_mins]
        out[s] = f
    _LIVE_BARS["key"] = ck
    _LIVE_BARS["bars"] = out
    return out


# Universe %>VWAP live: the 15m RAM store's measure maps onto the census
# c_pct_above_vwap_* scale essentially 1:1 — pooled fit census = 0.9919*live + 0.009,
# r 0.992, per-checkpoint bias < 0.5pt (data/validation/breadth_map_study.txt, 60
# sessions x 1,651 shared symbols). Bins are ~20pts wide, so +-1.9pt noise rarely flips one.
BREADTH_MAP = (0.9919, 0.009)
BREADTH_CPS = [585, 600, 630, 690, 750, 810, 870]      # 09:45..14:30 — all 15m bar closes
BCOL_BY_MIN = {585: "c0945_pct_above_vwap_0945", 600: "c1000_pct_above_vwap_1000",
               630: "c_pct_above_vwap_1030", 690: "c1130_pct_above_vwap_1130",
               750: "c1230_pct_above_vwap_1230", 810: "c1330_pct_above_vwap_1330",
               870: "c1430_pct_above_vwap_1430"}
_BR_CACHE: dict = {}


def _breadth_ckpts_live(frames15: dict, date: str, cutoff: int) -> dict:
    """{checkpoint close-minute: census-mapped %>VWAP} from the 15m RAM frames — recomputed
    from the session's bars each cycle (stateless: survives serve restarts mid-day)."""
    ck = (date, cutoff)
    if _BR_CACHE.get("k") == ck:
        return _BR_CACHE["v"]
    cps = [m for m in BREADTH_CPS if m <= cutoff]
    if not cps or not frames15:
        return {}
    try:                                                # bar-close minute is base-aware: +5 on the
        import datastore                                # 5m live base, +15 on the legacy 15m base
        _bm = datastore._infer_base_min(next(iter(frames15.values())).index)
    except Exception:
        _bm = 15
    acc = {m: [0, 0] for m in cps}
    for s, f in frames15.items():
        try:
            g = f.loc[date].between_time("09:30", "15:59")
        except (KeyError, TypeError):
            continue
        if not len(g):
            continue
        tp = ((g["high"] + g["low"] + g["close"]) / 3).to_numpy(float)
        v = np.maximum(g["volume"].to_numpy(float), 1.0)
        vw = np.cumsum(tp * v) / np.cumsum(v)
        cl = g["close"].to_numpy(float)
        mm = g.index.hour * 60 + g.index.minute + _bm   # bar-close minute (base-aware)
        for m in cps:
            ix = np.where(mm <= m)[0]
            if not len(ix):
                continue
            i = ix[-1]
            acc[m][0] += int(cl[i] > vw[i])
            acc[m][1] += 1
    a, b = BREADTH_MAP
    out = {m: a * (100 * c / n) + b for m, (c, n) in acc.items() if n >= 500}
    _BR_CACHE["k"] = ck
    _BR_CACHE["v"] = out
    return out


def _breadth_ckpts_sim(sym: str, date: str, cutoff: int) -> dict:
    """Sim path: the settled census checkpoint values (<= cutoff) from the batch parquet."""
    fp = FDIR / ("daytype_features.parquet" if sym == "SPY"
                 else f"daytype_features_{sym}.parquet")
    if not fp.exists():
        return {}
    row = pd.read_parquet(fp, columns=list(BCOL_BY_MIN.values()))
    if date not in row.index:
        return {}
    row = row.loc[date]
    return {m: float(row[c]) for m, c in BCOL_BY_MIN.items()
            if m <= cutoff and pd.notna(row[c])}


def score_live(sym: str = "SPY", sim: tuple[str, str] | None = None,
               frames15: dict | None = None, live_asof: tuple[str, str] | None = None) -> dict:
    """Score the CURRENT session at the latest closed 5-min bar with the frozen v2 model.
    sim=(date,'HH:MM') replays a settled date through this exact path (plumbing check —
    compare against the batch daytype_bar_probs). frames15 = serve's live RAM store,
    used for the universe %>VWAP breadth features (mapped to the census scale)."""
    import dayfeatures
    import daytype_bar_features as dbf
    model2 = _model2(sym)
    if model2 is None:
        return {"error": "no daytype_model2 artifact — run daytype_prob2.py --production"}
    if sim:
        date, hm = sim
        cutoff = int(hm[:2]) * 60 + int(hm[3:5])
    elif live_asof:                                             # offline check of the LIVE data path
        date, hm = live_asof                                    # (direct-IEX / overlay, NOT the store) vs a
        cutoff = int(hm[:2]) * 60 + int(hm[3:5])                # settled date — verify the plumbing w/o RTH
    else:
        from zoneinfo import ZoneInfo
        now = dt.datetime.now(ZoneInfo("America/New_York"))
        date = now.date().isoformat()
        cutoff = (now.hour * 60 + now.minute) // 5 * 5           # forming bar excluded
    bars = _cycle_bars(date, cutoff, bool(sim))                 # sim=False for live_asof -> live data path
    g = bars.get(sym, pd.DataFrame())
    g = g.between_time("09:30", "15:59") if len(g) else g
    if not len(g):
        if not sim:
            print(f"[daytype.score_live] {sym} {date}: no closed 5m bars <= cutoff {cutoff} "
                  f"(direct-fetch + overlay both empty) -> pre-open fallback")
        return score_payload(sym)                                # pre-open fallback
    daily = _daily(sym)
    dd = daily[daily.index < date]
    prev_c = dd["close"].shift(1)
    tr = pd.concat([dd["high"] - dd["low"], (dd["high"] - prev_c).abs(),
                    (dd["low"] - prev_c).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        if not sim:
            print(f"[daytype.score_live] {sym} {date}: bad ATR ({atr}) -> pre-open fallback")
        return score_payload(sym)
    sf = dbf.session_features(float(g["open"].iloc[0]), g["high"].to_numpy(float),
                              g["low"].to_numpy(float), g["close"].to_numpy(float),
                              g["volume"].to_numpy(float), atr)
    i = len(g) - 1
    mins = int(g.index[i].hour * 60 + g.index[i].minute + 5)
    sfeats = {c: float(v[i]) for c, v in sf.items() if np.isfinite(v[i])}
    # universe breadth (stage B): checkpoint %>VWAP carried forward, exactly like the
    # batch daytype_bar_features.breadth_track — live values census-mapped (BREADTH_MAP)
    br = (_breadth_ckpts_sim(sym, date, mins) if sim
          else _breadth_ckpts_live(frames15 or {}, date, mins))
    if br:
        cur_m = max(br)
        sfeats["s_breadth"] = br[cur_m]
        if 630 in br and cur_m >= 630:
            sfeats["s_breadth_d30"] = br[cur_m] - br[630]
        if cur_m - 60 in br:
            sfeats["s_breadth_dhr"] = br[cur_m] - br[cur_m - 60]
    # day-level features: a_ from the pre-open cache (live) / batch parquet (sim);
    # b_/c_ from the SAME batch block on a 40-session tail + today's bars
    dfeats: dict = {}
    if sim:
        fp = FDIR / ("daytype_features.parquet" if sym == "SPY"
                     else f"daytype_features_{sym}.parquet")
        if fp.exists():
            row = pd.read_parquet(fp).loc[date]
            dfeats = {c: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
                      for c, v in row.items()
                      if c.startswith("a_") and not c.endswith("_pct")
                      and not isinstance(v, str) and pd.notna(v)}
    elif PRE.exists():
        pre = json.loads(PRE.read_text())
        if pre.get("date") == date:
            dfeats = dict(pre["syms"].get(sym, {}).get("feats", {}))
    tail = daily[daily.index <= date].iloc[-41:]
    if date not in tail.index:
        stub = tail.iloc[[-1]].copy()
        stub.index = [date]
        tail = pd.concat([tail, stub])
    try:
        crow = dayfeatures.spy_intraday_block(tail, sym, extra_bars=bars).iloc[-1]
        for c, v in crow.items():
            if c in dfeats or isinstance(v, str) or (not isinstance(v, (bool, np.bool_))
                                                     and pd.isna(v)):
                continue
            dfeats[c] = bool(v) if isinstance(v, (bool, np.bool_)) else float(v)
    except Exception as e:
        print(f"live c-block failed ({e}) — scoring on a_/s_ only")
    for m, val in br.items():                            # census-scale checkpoint columns
        dfeats.setdefault(BCOL_BY_MIN[m], val)
    sc = score_features2(model2, sfeats, dfeats, mins)
    sp = FDIR / f"daytype_prob2_skill{'' if sym == 'SPY' else '_' + sym}.json"
    skill = json.loads(sp.read_text()).get("buckets", {}) if sp.exists() else {}
    bname = "1430"
    for bn, b in model2["buckets"].items():
        if b["lo"] <= mins <= b["hi"]:
            bname = bn
            break
    txt, col = narrative(sc, skill, bname)
    out = {"sym": sym, "date": date, "cp": bname,
           "cp_label": f"{mins // 60:02d}:{mins % 60:02d} bar", "granularity": "5m",
           "mode": "sim" if sim else "live-intraday", "probs": sc, "stance": txt,
           "color": col, "model_fitted": model2.get("fitted"),
           "breadth": ("census-mapped live (15m store)" if (br and not sim)
                       else ("batch (sim)" if br else "unavailable this cycle"))}
    tier = h1_tier(bool(sfeats.get("s_w1b")), sfeats.get("s_breadth_dhr"))
    if tier:
        out["tier"] = tier
    if not sim:
        try:
            row = {"asof": dt.datetime.now().isoformat(timespec="seconds"), "sym": sym,
                   "date": date, "cp": f"bar{mins}",
                   **{f"p_{t}": v["p"] for t, v in sc.items()}}
            pd.DataFrame([row]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
        except Exception:
            pass
    return out


def sim_check(sym: str, date: str, times: list[str]) -> None:
    """Replay a settled date through the LIVE path and diff against the batch per-bar probs."""
    sfx = "" if sym == "SPY" else f"_{sym}"
    P = pd.read_parquet(FDIR / f"daytype_bar_probs{sfx}.parquet").set_index(["date", "mins"])
    for hm in times:
        r = score_live(sym, sim=(date, hm))
        mins = int(hm[:2]) * 60 + int(hm[3:5])
        mins = (mins // 5) * 5
        want = P.loc[(date, mins)] if (date, mins) in P.index else None
        cmp = ("  batch: " + " ".join(f"{t} {100 * want[f'p_{t}']:.1f}%"
                                      for t in ("trend", "up", "down"))
               if want is not None else "  (no batch row)")
        got = " ".join(f"{t} {100 * v['p']:.1f}%(m{v['feats_missing']})"
                       for t, v in r.get("probs", {}).items())
        print(f"{date} {hm} [{r.get('cp_label')}] live-path: {got}\n{cmp}"
              f"   -> {r.get('stance')}{' · tier ' + r['tier'] if r.get('tier') else ''}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--live", action="store_true", help="score the current session per-bar")
    ap.add_argument("--sim", nargs=2, metavar=("DATE", "HH:MM"),
                    help="replay a settled date through the live path (plumbing check)")
    ap.add_argument("--sym", default="SPY")
    ap.add_argument("--date", default=None, help="--prep for a specific date (test)")
    args = ap.parse_args()
    if args.prep:
        prep(args.date)
    if args.score:
        print(json.dumps(score_payload(args.sym), indent=1))
    if args.live:
        print(json.dumps(score_live(args.sym), indent=1))
    if args.sim:
        sim_check(args.sym, args.sim[0], [args.sim[1]])


if __name__ == "__main__":
    main()
