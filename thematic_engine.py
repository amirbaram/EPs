"""Thematic Scoring & Emerging Sector/Theme Engine.

Analyzes multi-timeframe relative strength across 124 sectors and 163 themes
from data/perf_history/groups.parquet (1W, 1M, 3M, YTD).
Applies pattern recognition for:
- 🚀 Velocity Surge (W1 >= 75th, M3 < 60th)
- ⚡ Multi-TF Power Cluster (W1, M1 >= 70th, M3 >= 60th)
- 📈 Acceleration Cascade (W1 > M1 > M3 >= 65th)
- 👑 Top Board Persistence (>=3 horizons in Top Quartile)
- ⚠️ Severe Headwind Veto (M1 < 35th and M3 < 35th)

Integrates historical EP outcome feedback (Incubator Score) and persists
learned scores to data/thematic_scores.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import labels

STORE_PATH = Path("data/thematic_scores.json")
GROUPS_PARQUET = Path("data/perf_history/groups.parquet")
STUDY_PARQUET = Path("data/simulations/ep_combined_study.parquet")

_THEMATIC_ENGINE_CACHE: Optional[Dict[str, Any]] = None


class ThematicScoringEngine:
    def __init__(self):
        self.groups_df: Optional[pd.DataFrame] = None
        self.ep_df: Optional[pd.DataFrame] = None
        self.scores: Dict[str, Any] = {}
        self.latest_date_str: str = ""
        self.load_or_build()

    def load_or_build(self, force_rebuild: bool = False):
        if not force_rebuild and STORE_PATH.exists():
            try:
                with open(STORE_PATH, "r") as f:
                    self.scores = json.load(f)
                self.latest_date_str = self.scores.get("latest_date", "")
                if self.scores.get("themes") and self.scores.get("sectors"):
                    return
            except Exception as e:
                print(f"Failed to load cached thematic scores: {e}")

        self.rebuild_scores()

    def rebuild_scores(self):
        if not GROUPS_PARQUET.exists():
            print("WARNING: groups.parquet not found!")
            return

        print("ThematicScoringEngine: Loading performance history...")
        self.groups_df = pd.read_parquet(GROUPS_PARQUET)
        self.groups_df["date_str"] = self.groups_df["date"].dt.strftime("%Y-%m-%d")
        self.latest_date_str = self.groups_df["date_str"].max()

        # Load historical EP outcomes for incubator feedback
        incubator_feedback = {}
        if STUDY_PARQUET.exists():
            try:
                self.ep_df = pd.read_parquet(STUDY_PARQUET)
                # Compute historical performance per theme & sector
                for kind, col in [("theme", "theme"), ("sector", "sector")]:
                    grp = self.ep_df.groupby(col).agg(
                        n=("symbol", "count"),
                        win20=("ret_20d", lambda x: float((x > 0).mean() * 100)),
                        monsters=("outcome", lambda x: float((x == "Runaway Monster").mean() * 100)),
                        traps=("breached_d1_low_5d", lambda x: float(x.mean() * 100))
                    )
                    for name, r in grp.iterrows():
                        if r["n"] >= 5:
                            # Edge bonus: high monster rate and low trap rate
                            bonus = 0.0
                            if r["monsters"] >= 40.0: bonus += 10.0
                            elif r["monsters"] >= 30.0: bonus += 5.0
                            if r["traps"] <= 20.0: bonus += 5.0
                            elif r["traps"] >= 35.0: bonus -= 10.0
                            incubator_feedback[(kind, name)] = {
                                "bonus": bonus,
                                "n": int(r["n"]),
                                "win20": round(r["win20"], 1),
                                "monsters": round(r["monsters"], 1),
                                "traps": round(r["traps"], 1)
                            }
            except Exception as e:
                print(f"Could not load incubator feedback: {e}")

        # Compute latest percentiles for latest date
        latest_rows = self.groups_df[self.groups_df["date_str"] == self.latest_date_str]
        
        horizons = ["w1", "m1", "m3", "ytd"]
        records: Dict[str, Dict[str, Any]] = {"themes": {}, "sectors": {}}

        for kind, bucket_key in [("theme", "themes"), ("sector", "sectors")]:
            sub = latest_rows[latest_rows["kind"] == kind].copy()
            if len(sub) < 3:
                continue

            for h in horizons:
                if h in sub.columns:
                    valid = sub.dropna(subset=[h])
                    ranks = valid.set_index("name")[h].rank(pct=True) * 100.0
                    sub[f"{h}_pct"] = sub["name"].map(ranks)
                else:
                    sub[f"{h}_pct"] = 50.0

            for _, r in sub.iterrows():
                name = r["name"]
                w1_p = float(r.get("w1_pct", 50.0)) if pd.notna(r.get("w1_pct")) else 50.0
                m1_p = float(r.get("m1_pct", 50.0)) if pd.notna(r.get("m1_pct")) else 50.0
                m3_p = float(r.get("m3_pct", 50.0)) if pd.notna(r.get("m3_pct")) else 50.0
                ytd_p = float(r.get("ytd_pct", 50.0)) if pd.notna(r.get("ytd_pct")) else 50.0

                w1_ret = round(float(r["w1"]), 2) if pd.notna(r.get("w1")) else 0.0
                m1_ret = round(float(r["m1"]), 2) if pd.notna(r.get("m1")) else 0.0
                m3_ret = round(float(r["m3"]), 2) if pd.notna(r.get("m3")) else 0.0
                ytd_ret = round(float(r["ytd"]), 2) if pd.notna(r.get("ytd")) else 0.0
                member_count = int(r["n"]) if pd.notna(r.get("n")) else 0

                # Base multi-timeframe momentum score (0-100)
                base_score = 0.35 * w1_p + 0.30 * m1_p + 0.20 * m3_p + 0.15 * ytd_p

                # Add historical incubator feedback bonus
                fb = incubator_feedback.get((kind, name), {})
                bonus = fb.get("bonus", 0.0)
                final_score = float(np.clip(base_score + bonus, 0.0, 100.0))

                # Archetype Detection
                archetypes = []
                badge_type = "neutral"
                
                # 1. Severe Headwind Veto Check
                if m1_p < 35.0 and m3_p < 35.0:
                    archetypes.append("Severe Headwind (Hard Veto)")
                    badge_type = "veto"
                else:
                    if w1_p >= 75.0 and m3_p < 60.0:
                        archetypes.append("Velocity Surge")
                        badge_type = "surge"
                    if w1_p >= 70.0 and m1_p >= 70.0 and m3_p >= 60.0:
                        archetypes.append("Multi-TF Power Cluster")
                        if badge_type != "surge": badge_type = "power"
                    if w1_p > m1_p and m1_p > m3_p and w1_p >= 65.0:
                        archetypes.append("Acceleration Cascade")
                        if badge_type == "neutral": badge_type = "acceleration"
                    top_q_cnt = sum([w1_p >= 75.0, m1_p >= 75.0, m3_p >= 75.0, ytd_p >= 75.0])
                    if top_q_cnt >= 3:
                        archetypes.append("Top Board Persistence")
                        if badge_type == "neutral": badge_type = "persistence"

                primary_archetype = archetypes[0] if archetypes else ("Leading Trend" if final_score >= 60 else ("Neutral Consolidation" if final_score >= 40 else "Lagging Drag"))

                records[bucket_key][name] = {
                    "name": name,
                    "kind": kind,
                    "score": round(final_score, 1),
                    "base_score": round(base_score, 1),
                    "bonus": bonus,
                    "w1_pct": round(w1_p, 1),
                    "m1_pct": round(m1_p, 1),
                    "m3_pct": round(m3_p, 1),
                    "ytd_pct": round(ytd_p, 1),
                    "w1_ret": w1_ret,
                    "m1_ret": m1_ret,
                    "m3_ret": m3_ret,
                    "ytd_ret": ytd_ret,
                    "member_count": member_count,
                    "archetypes": archetypes,
                    "primary_archetype": primary_archetype,
                    "badge_type": badge_type,
                    "incubator_history": fb
                }

        self.scores = {
            "latest_date": self.latest_date_str,
            "themes": records["themes"],
            "sectors": records["sectors"]
        }

        # Atomically write to cache
        os.makedirs(STORE_PATH.parent, exist_ok=True)
        tmp_path = STORE_PATH.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.scores, f, indent=2)
        os.replace(tmp_path, STORE_PATH)
        print(f"ThematicScoringEngine: Saved {len(records['themes'])} themes and {len(records['sectors'])} sectors to {STORE_PATH}.")

    def get_group_info(self, kind: str, name: str) -> Dict[str, Any]:
        bucket = "themes" if kind == "theme" else "sectors"
        grp_data = self.scores.get(bucket, {})
        if name in grp_data:
            return grp_data[name]
        return {
            "name": name,
            "kind": kind,
            "score": 50.0,
            "w1_pct": 50.0, "m1_pct": 50.0, "m3_pct": 50.0, "ytd_pct": 50.0,
            "archetypes": [],
            "primary_archetype": "Neutral Baseline",
            "badge_type": "neutral",
            "incubator_history": {}
        }

    def evaluate_symbol(self, sym: str) -> Dict[str, Any]:
        sec = labels.sector(sym) or "Unknown"
        thms = labels.themes(sym) or []
        sec_info = self.get_group_info("sector", sec)

        # Evaluate ALL themes for this symbol and select the one with highest momentum conviction
        if not thms:
            best_thm_info = self.get_group_info("theme", "General")
        else:
            thm_infos = [self.get_group_info("theme", t) for t in thms]
            # Prioritize themes that are NOT vetoes; take highest score
            best_thm_info = max(thm_infos, key=lambda x: (x["badge_type"] != "veto", x["score"]))

        thm = best_thm_info["name"]
        thm_info = best_thm_info

        # Composite score chooses highest conviction of theme or sector
        best_score = max(thm_info["score"], sec_info["score"])

        # Hard Veto logic: applies ONLY if the stock lacks any leading theme (score < 65)
        # AND its primary theme or sector is stuck in a multi-month severe headwind (bottom 35th percentile)
        is_veto = (thm_info["badge_type"] == "veto" and sec_info["score"] < 60.0) or \
                  (sec_info["badge_type"] == "veto" and thm_info["score"] < 65.0)

        best_archetype = thm_info["primary_archetype"] if thm_info["score"] >= sec_info["score"] else sec_info["primary_archetype"]
        best_badge = thm_info["badge_type"] if thm_info["score"] >= sec_info["score"] else sec_info["badge_type"]
        if is_veto:
            best_badge = "veto"
            best_archetype = "Severe Headwind (Hard Veto)"

        return {
            "symbol": sym,
            "sector": sec,
            "theme": thm,
            "composite_score": round(best_score, 1),
            "is_veto": is_veto,
            "primary_archetype": best_archetype,
            "badge_type": best_badge,
            "theme_info": thm_info,
            "sector_info": sec_info,
        }

    def get_emerging_summary(self) -> Dict[str, Any]:
        themes = list(self.scores.get("themes", {}).values())
        sectors = list(self.scores.get("sectors", {}).values())

        # Top emerging themes: score >= 65 and not veto
        emerging_themes = [t for t in themes if t["score"] >= 65.0 and t["badge_type"] != "veto"]
        emerging_themes.sort(key=lambda x: x["score"], reverse=True)

        # Top emerging sectors
        emerging_sectors = [s for s in sectors if s["score"] >= 65.0 and s["badge_type"] != "veto"]
        emerging_sectors.sort(key=lambda x: x["score"], reverse=True)

        # Headwind Veto list
        headwinds = [t for t in themes if t["badge_type"] == "veto"]
        headwinds.extend([s for s in sectors if s["badge_type"] == "veto"])
        headwinds.sort(key=lambda x: x["score"])

        return {
            "latest_date": self.latest_date_str,
            "total_themes": len(themes),
            "total_sectors": len(sectors),
            "emerging_themes": emerging_themes[:20],
            "emerging_sectors": emerging_sectors[:15],
            "headwind_veto": headwinds[:20],
            "all_themes_ranked": sorted(themes, key=lambda x: x["score"], reverse=True),
            "all_sectors_ranked": sorted(sectors, key=lambda x: x["score"], reverse=True),
        }

    def get_group_stocks(self, name: str, kind: str = "theme") -> Dict[str, Any]:
        """Returns the constituent stocks for a given theme or sector, ranked by multi-horizon returns."""
        labels.load()
        if kind == "theme":
            syms = list(labels.tickers_in_theme(name))
        else:
            syms = [s for s, sc in (labels._SECTOR or {}).items() if sc.lower() == name.lower()]

        if not syms:
            return {"group": name, "kind": kind, "count": 0, "stocks": []}

        tickers_df = None
        for p in [Path("data/perf_history/tickers_2026.parquet"), Path("data/perf_history/tickers_2025.parquet")]:
            if p.exists():
                try:
                    df_t = pd.read_parquet(p)
                    latest_dt = df_t["date"].max()
                    tickers_df = df_t[df_t["date"] == latest_dt].set_index("symbol")
                    break
                except Exception:
                    pass

        ep_counts = {}
        if STUDY_PARQUET.exists():
            try:
                ep_df = pd.read_parquet(STUDY_PARQUET)
                ep_counts = ep_df["symbol"].value_counts().to_dict()
            except Exception:
                pass

        stocks = []
        for sym in syms:
            r = tickers_df.loc[sym] if tickers_df is not None and sym in tickers_df.index else None
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]

            d1 = round(float(r["d1"]), 2) if r is not None and pd.notna(r.get("d1")) else None
            w1 = round(float(r["w1"]), 2) if r is not None and pd.notna(r.get("w1")) else None
            m1 = round(float(r["m1"]), 2) if r is not None and pd.notna(r.get("m1")) else None
            m3 = round(float(r["m3"]), 2) if r is not None and pd.notna(r.get("m3")) else None
            ytd = round(float(r["ytd"]), 2) if r is not None and pd.notna(r.get("ytd")) else None

            px = None
            vol_m = None
            try:
                import datastore
                b = datastore.load_bars(sym)
                if b is not None and len(b):
                    last_b = b.iloc[-1]
                    px = round(float(last_b["close"]), 2)
                    vol_m = round(float(last_b["volume"] * last_b["close"]) / 1e6, 1)
            except Exception:
                pass

            n_eps = int(ep_counts.get(sym, 0))
            stocks.append({
                "symbol": sym,
                "price": px,
                "dvol_m": vol_m,
                "d1": d1,
                "w1": w1,
                "m1": m1,
                "m3": m3,
                "ytd": ytd,
                "ep_count": n_eps,
                "has_ep": n_eps > 0
            })

        stocks.sort(key=lambda x: (x["w1"] is not None, x["w1"] if x["w1"] is not None else -9999), reverse=True)

        return {
            "group": name,
            "kind": kind,
            "count": len(stocks),
            "info": self.get_group_info(kind, name),
            "stocks": stocks
        }


def get_thematic_engine(force_rebuild: bool = False) -> ThematicScoringEngine:
    global _THEMATIC_ENGINE_CACHE
    if _THEMATIC_ENGINE_CACHE is None or force_rebuild:
        _THEMATIC_ENGINE_CACHE = ThematicScoringEngine()
    return _THEMATIC_ENGINE_CACHE


if __name__ == "__main__":
    eng = ThematicScoringEngine()
    summary = eng.get_emerging_summary()
    print(f"\nEngine Date: {summary['latest_date']}")
    print(f"\nTop 10 Emerging Themes:")
    for t in summary["emerging_themes"][:10]:
        print(f"  {t['name']:35s} | Score: {t['score']:4.1f} | Archetype: {t['primary_archetype']:25s} | W1: {t['w1_ret']:+5.1f}% | M1: {t['m1_ret']:+5.1f}%")

    print(f"\nTop 5 Emerging Sectors:")
    for s in summary["emerging_sectors"][:5]:
        print(f"  {s['name']:35s} | Score: {s['score']:4.1f} | Archetype: {s['primary_archetype']:25s} | W1: {s['w1_ret']:+5.1f}% | M1: {s['m1_ret']:+5.1f}%")

    print(f"\nSample Severe Headwinds (Hard Veto):")
    for h in summary["headwind_veto"][:5]:
        print(f"  {h['name']:35s} | Score: {h['score']:4.1f} | M1 Pctile: {h['m1_pct']:4.1f}th | M3 Pctile: {h['m3_pct']:4.1f}th")
