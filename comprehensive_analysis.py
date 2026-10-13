import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, brier_score_loss

def get_group_masks(q):
    TOP_SECTORS = [
        "Computer Hardware", "Biotechnology", "Healthcare", "Technology", "Semiconductors",
        "Industrials", "Capital Markets", "Financial Services", "Specialty Industrial Machinery"
    ]
    TOP_THEMES = [
        "Semiconductors", "Biotechnology", "Software - Infrastructure", "Computer Hardware",
        "Health Information Services", "Medical Devices", "Software - Application"
    ]
    masks = {}
    masks["Pinnacle Elite"] = (
        (~q["violated_48h"]) & (q["close_pos"] >= 0.65) & (q["rvol"] >= 2.5) & (q["gap_pct"] >= 5.0) &
        (q["sector"].isin(TOP_SECTORS) | q["theme"].isin(TOP_THEMES)) &
        ((q["sec_m1_pctile"] >= 50) | (q["thm_m1_pctile"] >= 50))
    )
    masks["Emerging Leaders"] = (
        (~q["violated_48h"]) & (q["close_pos"] >= 0.65) & (q["rvol"] >= 2.5) &
        ((q["sec_m1_pctile"] >= 65) | (q["thm_m1_pctile"] >= 65) | (q["sec_m3_pctile"] >= 65))
    )
    masks["Institutional Sweet Spot"] = (
        (~q["violated_48h"]) & (q["close_pos"] >= 0.65) & (q["rvol"] >= 3.0) & (q["gap_pct"] >= 6.0)
    )
    top_clusters = [
        "Computer Hardware", "Semiconductors", "Biotechnology", "Semiconductor Equipment",
        "Bitcoin Miners", "Quantum Computing", "Uranium & Nuclear", "Clean Energy",
        "Software - Application", "Software - Infrastructure", "Aerospace & Defense"
    ]
    masks["Multi Quarter Leaders"] = (
        (~q["violated_48h"]) & (q["sector"].isin(top_clusters) | q["theme"].isin(top_clusters))
    )
    masks["Neglected Turnaround"] = (
        (q["neglect_6m"] <= -15.0) & (q["rvol"] >= 4.0) & (q["close_pos"] >= 0.70)
    )
    hb = ["Bitcoin Miners", "Quantum Computing", "Semiconductor Equipment", "Clean Energy", "Data Centers"]
    masks["High Beta Momentum"] = (
        (q["theme"].isin(hb)) & (q["rvol"] >= 3.5)
    )
    is_headwind = (((q["sec_m1_pctile"] < 35) & (q["sec_m3_pctile"] < 35)) | (q["thm_m1_pctile"] < 35))
    masks["Idiosyncratic Alpha"] = (
        is_headwind & (~q["violated_48h"]) & (q["close_pos"] >= 0.80) & (q["rvol"] >= 6.0) & (q["dvol_m"] >= 75.0)
    )
    return masks

def main():
    print("Loading data...")
    df_ep = pd.read_parquet("data/simulations/ep_combined_study_scored.parquet")
    df_ep["entry_date"] = pd.to_datetime(df_ep["date"]).dt.strftime("%Y-%m-%d")
    
    masks = get_group_masks(df_ep)
    for group, m in masks.items():
        df_ep[group] = m

    df_groups = df_ep[["symbol", "entry_date"] + list(masks.keys())]
    
    df_v2 = pd.read_parquet("data/ml_datasets/amir_rolling/dataset_rolling.parquet")
    df_v2["date_idx"] = pd.to_datetime(df_v2["date"])
    
    # Merge V2 with EP groups based on symbol and entry_date
    df_v2 = df_v2.merge(df_groups, on=["symbol", "entry_date"], how="inner")
    df_v2_oos = df_v2[df_v2["date_idx"] >= pd.Timestamp("2023-01-01")].copy()
    
    v2_model = joblib.load("data/models/amir_rolling_xgboost_ordinal.pkl")
    features_v2 = [c for c in df_v2_oos.columns if c.startswith("feature_") and c not in list(masks.keys())]
    
    probs_v2 = v2_model.predict_proba(df_v2_oos[features_v2])
    
    df_v2_oos["prob_50"] = np.clip(probs_v2[:, 1:].sum(axis=1), 0, 1)
    if probs_v2.shape[1] > 2: df_v2_oos["prob_100"] = np.clip(probs_v2[:, 2:].sum(axis=1), 0, 1)
    if probs_v2.shape[1] > 3: df_v2_oos["prob_150"] = np.clip(probs_v2[:, 3:].sum(axis=1), 0, 1)
    if probs_v2.shape[1] > 4: df_v2_oos["prob_200"] = np.clip(probs_v2[:, 4:].sum(axis=1), 0, 1)
    
    df_v2_oos["y_50"] = (df_v2_oos["target_50"] == 1).astype(int)
    if "target_100" in df_v2_oos.columns: df_v2_oos["y_100"] = (df_v2_oos["target_100"] == 1).astype(int)
    if "target_150" in df_v2_oos.columns: df_v2_oos["y_150"] = (df_v2_oos["target_150"] == 1).astype(int)
    if "target_200" in df_v2_oos.columns: df_v2_oos["y_200"] = (df_v2_oos["target_200"] == 1).astype(int)

    targets = [t for t in [50, 100, 150, 200] if f"y_{t}" in df_v2_oos.columns]

    print("\n--- Day 1-20 Metrics by Group & Target ---")
    for group in masks.keys():
        g_df = df_v2_oos[df_v2_oos[group] == True]
        if len(g_df) == 0: continue
        print(f"\n[{group}]")
        
        for tgt in targets:
            days_res = []
            for d in range(1, 21):
                d_df = g_df[g_df["feature_days_since_ep"] == d]
                if len(d_df) == 0: continue
                
                y_true = d_df[f"y_{tgt}"]
                y_prob = d_df[f"prob_{tgt}"]
                if len(y_true.unique()) > 1:
                    auc = roc_auc_score(y_true, y_prob)
                else:
                    auc = np.nan
                    
                y_pred = (y_prob >= 0.5).astype(int)
                if sum(y_true) + sum(y_pred) > 0:
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
                else:
                    prec, fpr = np.nan, np.nan
                    
                days_res.append({"Day": d, "AUC": auc, "Precision": prec, "FPR": fpr, "BaseRate": y_true.mean()})
                
            df_res = pd.DataFrame(days_res).dropna(subset=["AUC"])
            if not df_res.empty:
                best_auc_idx = df_res["AUC"].idxmax()
                best_row = df_res.loc[best_auc_idx]
                avg_prec = df_res["Precision"].mean()
                avg_fpr = df_res["FPR"].mean()
                avg_base = df_res["BaseRate"].mean()
                print(f"  Target {tgt}: Best Day={int(best_row['Day'])} (AUC={best_row['AUC']:.3f}), "
                      f"Avg BaseRate={avg_base:.1%}, Avg Precision={avg_prec:.1%}, Avg FPR={avg_fpr:.1%}")

    # V1 comparison
    print("\n--- V1 vs V2 Comparison ---")
    df_v1 = pd.read_parquet("data/ml_datasets/amir_spec/dataset.parquet")
    df_v1["date_idx"] = pd.to_datetime(df_v1["entry_date"])
    df_v1_oos = df_v1[df_v1["date_idx"] >= pd.Timestamp("2023-01-01")].copy()
    
    # We want to compare V1 with V2 exactly on the days V1 represents.
    # We already established V1 entry_date corresponds mostly to V2 where date == entry_date (i.e. feature_days_since_ep=4..6)
    # Actually, V1's dataset has point-in-time features. We can merge on symbol and V1's "entry_date" which might map to V2's date.
    df_v1_oos["date"] = df_v1_oos["entry_date"]
    
    # Let's merge on symbol and date. The V2 date matching V1's "entry_date" is indeed day 4-6.
    merged = df_v1_oos.merge(df_v2_oos, on=["symbol", "date"], suffixes=("_v1", "_v2"))
    
    v1_model = joblib.load("data/models/amir_spec_xgboost_ordinal.pkl")
    features_v1 = [c for c in df_v1_oos.columns if c.startswith("feature_") and c not in list(masks.keys())]
    
    v1_data = merged[[c + "_v1" if c + "_v1" in merged.columns else c for c in features_v1]].copy()
    v1_data.columns = features_v1
    
    probs_v1 = v1_model.predict_proba(v1_data)
    merged["prob_50_v1"] = np.clip(probs_v1[:, 1:].sum(axis=1), 0, 1)
    
    for group in masks.keys():
        g_df = merged[merged[group] == True]
        if len(g_df) == 0 or len(g_df["y_50"].unique()) < 2: continue
        auc_v1 = roc_auc_score(g_df["y_50"], g_df["prob_50_v1"])
        auc_v2 = roc_auc_score(g_df["y_50"], g_df["prob_50"])
        print(f"Group: {group} (N = {len(g_df)}) -> V1 AUC: {auc_v1:.3f} | V2 AUC: {auc_v2:.3f}")

if __name__ == '__main__':
    main()
