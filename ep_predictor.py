"""
Prediction Engine for Episodic Pivots.
This module encapsulates the quantitative insights derived from the 10-year study.
It maps live/historical conditions to mathematical probabilities, target ranges, and actionable alerts.
"""

def classify_archetype(features):
    """Returns the primary Playbook Insight archetype based on conditions."""
    if features.get("violated_48h") or features.get("close_pos", 0) < 0.50:
        return "Toxic Traps"
        
    theme = features.get("theme", "")
    sector = features.get("sector", "")
    rvol = features.get("rvol", 0)
    close_pos = features.get("close_pos", 0)
    gap_pct = features.get("gap_pct", 0)
    
    top_clusters = ["Computer Hardware", "Semiconductors", "Biotechnology", "Semiconductor Equipment", "Bitcoin Miners", "Quantum Computing", "Uranium & Nuclear", "Clean Energy", "Software - Application", "Software - Infrastructure", "Aerospace & Defense"]
    hb_clusters = ["Bitcoin Miners", "Quantum Computing", "Semiconductor Equipment", "Clean Energy", "Data Centers"]
    
    if theme in hb_clusters and rvol >= 3.5:
        return "High Beta Momentum"
    if features.get("neglect_6m", 0) <= -15.0 and rvol >= 4.0 and close_pos >= 0.70:
        return "Turnarounds"
    if (sector in top_clusters or theme in top_clusters) and not features.get("violated_48h"):
        return "Compounders"
    
    boost_tailwind = features.get("sec_m1_pctile", 0) >= 60 or features.get("thm_m1_pctile", 0) >= 60
    
    if close_pos >= 0.65 and rvol >= 3.0 and gap_pct >= 6.0:
        return "Inst. Sweet Spot"
    if close_pos >= 0.65 and rvol >= 2.5 and (features.get("sec_m1_pctile", 0) >= 65 or features.get("thm_m1_pctile", 0) >= 65):
        return "Emerging Leaders"
    if close_pos >= 0.65 and rvol >= 2.5 and gap_pct >= 5.0 and boost_tailwind:
        return "Pinnacle Elite"
        
    return "Baseline"

def get_predictions(features):
    """
    Returns probabilities, recommended targets, and alerts based on the setup's conditions.
    """
    archetype = classify_archetype(features)
    
    # 1. Base Probabilities (Hardcoded from our ML matrix)
    probs = {
        "High Beta Momentum": {"p_50": 65.6, "p_100": 46.6, "p_consol": 10.7, "p_clean": 26.7, "mae": -9.7, "ev": 34.9},
        "Compounders":        {"p_50": 55.3, "p_100": 31.3, "p_consol": 2.9,  "p_clean": 31.7, "mae": -5.7, "ev": 18.8},
        "Turnarounds":        {"p_50": 53.1, "p_100": 31.9, "p_consol": 5.8,  "p_clean": 23.3, "mae": -8.8, "ev": 13.8},
        "Inst. Sweet Spot":   {"p_50": 50.2, "p_100": 26.3, "p_consol": 2.3,  "p_clean": 27.2, "mae": -4.8, "ev": 11.4},
        "Emerging Leaders":   {"p_50": 49.0, "p_100": 27.1, "p_consol": 2.5,  "p_clean": 27.0, "mae": -5.1, "ev": 11.4},
        "Pinnacle Elite":     {"p_50": 48.7, "p_100": 26.6, "p_consol": 2.1,  "p_clean": 25.8, "mae": -4.7, "ev": 5.9},
        "Toxic Traps":        {"p_50": 38.1, "p_100": 20.1, "p_consol": 13.3, "p_clean": 3.4,  "mae": -14.1,"ev": -0.3},
        "Baseline":           {"p_50": 40.0, "p_100": 20.0, "p_consol": 5.0,  "p_clean": 15.0, "mae": -6.0, "ev": 2.0}
    }
    
    base = probs.get(archetype, probs["Baseline"])
    
    # 2. Dynamic Upgrades based on Live Boosters
    p_100 = base["p_100"]
    alerts = []
    caution = None
    
    # 5D Support Held Booster upgrades the 100% target probability
    if not features.get("breached_d1_low_5d", False) and features.get("days_since_ep", 0) >= 5:
        if archetype != "Toxic Traps":
            p_100 += 1.5
            alerts.append("🔥 5D Support Held: Upgrading +100% Target Probability.")
            
    # Theme/Market Regime dynamics
    theme_state = features.get("theme_std_state", "gray")
    qqq_risk = features.get("qqq_risk_on", False)
    
    if qqq_risk and theme_state == "gray":
        alerts.append("🚀 Massive EV Expected (Theme Rotating + QQQ Risk On). Target +3x ADR.")
    elif theme_state == "yellow":
        alerts.append("⚠️ Theme Exhaustion (Yellow). Scale out heavily. Expectancy dropping.")
        
    # Archetype specific cautions
    if archetype == "Toxic Traps":
        caution = f"TRAP CONFIRMED: 13.3% chance of deep immediate collapse ({base['mae']}% MAE). CUT POSITION."
    elif archetype in ["Turnarounds", "High Beta Momentum"]:
        caution = f"Wide Volatility Expected: Average MAE is {base['mae']}%. Allow for deep pullback before run."
    elif archetype == "Compounders":
        alerts.append(f"✨ Clean Momentum Expected ({base['p_clean']}%). Hold through initial chop.")

    # Top-Down Sector Consolidation Drag (ARKG / KOD Insight)
    sec_pcts = features.get("sec_pctiles", {})
    if isinstance(sec_pcts, dict):
        w1_pct = sec_pcts.get("1w", 50)
        if w1_pct < 35:
            caution_msg = f"⚠️ Sector Consolidation Drag ({w1_pct}th Pctile): Sector is actively pulling back on the weekly timeframe. Expect sympathy chop and delayed momentum. Pause secondary adds until sector confirms a higher low."
            if caution is None:
                caution = caution_msg
            else:
                alerts.append(caution_msg)
        
    return {
        "archetype": archetype,
        "prob_hit_50": base["p_50"],
        "prob_hit_100": round(p_100, 1),
        "prob_consolidation": base["p_consol"],
        "prob_clean_run": base["p_clean"],
        "avg_mae": base["mae"],
        "alerts": alerts,
        "caution": caution
    }
