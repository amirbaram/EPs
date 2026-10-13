import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

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
    # 1. Pinnacle Elite
    masks["Pinnacle Elite"] = (
        (~q["violated_48h"]) &
        (q["close_pos"] >= 0.65) &
        (q["rvol"] >= 2.5) &
        (q["gap_pct"] >= 5.0) &
        (q["sector"].isin(TOP_SECTORS) | q["theme"].isin(TOP_THEMES)) &
        ((q["sec_m1_pctile"] >= 50) | (q["thm_m1_pctile"] >= 50))
    )
    # 2. Emerging Leaders
    masks["Emerging Leaders"] = (
        (~q["violated_48h"]) &
        (q["close_pos"] >= 0.65) &
        (q["rvol"] >= 2.5) &
        ((q["sec_m1_pctile"] >= 65) | (q["thm_m1_pctile"] >= 65) | (q["sec_m3_pctile"] >= 65))
    )
    # 3. Institutional Sweet Spot
    masks["Institutional Sweet Spot"] = (
        (~q["violated_48h"]) & (q["close_pos"] >= 0.65) & (q["rvol"] >= 3.0) & (q["gap_pct"] >= 6.0)
    )
    # 4. Multi Quarter Leaders
    top_clusters = [
        "Computer Hardware", "Semiconductors", "Biotechnology", "Semiconductor Equipment",
        "Bitcoin Miners", "Quantum Computing", "Uranium & Nuclear", "Clean Energy",
        "Software - Application", "Software - Infrastructure", "Aerospace & Defense"
    ]
    masks["Multi Quarter Leaders"] = (
        (~q["violated_48h"]) & (q["sector"].isin(top_clusters) | q["theme"].isin(top_clusters))
    )
    # 5. Neglected Turnaround
    masks["Neglected Turnaround"] = (
        (q["neglect_6m"] <= -15.0) & (q["rvol"] >= 4.0) & (q["close_pos"] >= 0.70)
    )
    # 6. High Beta Momentum
    hb = ["Bitcoin Miners", "Quantum Computing", "Semiconductor Equipment", "Clean Energy", "Data Centers"]
    masks["High Beta Momentum"] = (
        (q["theme"].isin(hb)) & (q["rvol"] >= 3.5)
    )
    # 7. Idiosyncratic Alpha
    is_headwind = (
        ((q["sec_m1_pctile"] < 35) & (q["sec_m3_pctile"] < 35)) |
        (q["thm_m1_pctile"] < 35)
    )
    masks["Idiosyncratic Alpha"] = (
        is_headwind &
        (~q["violated_48h"]) &
        (q["close_pos"] >= 0.80) &
        (q["rvol"] >= 6.0) &
        (q["dvol_m"] >= 75.0)
    )
    return masks

def main():
    print("Loading data...")
    # Load EP base dataset to get the groups
    df_ep = pd.read_parquet("data/simulations/ep_combined_study_scored.parquet")
    df_ep["entry_date"] = pd.to_datetime(df_ep["date"]).dt.strftime("%Y-%m-%d")
    
    # Map groups
    masks = get_group_masks(df_ep)
    for group, m in masks.items():
        df_ep[group] = m

    df_groups = df_ep[["symbol", "entry_date"] + list(masks.keys())]
    
    # Load V2 rolling dataset
    df_v2 = pd.read_parquet("data/ml_datasets/amir_rolling/dataset_rolling.parquet")
    df_v2["date_idx"] = pd.to_datetime(df_v2["date"])
    
    # Merge groups into V2
    df_v2 = df_v2.merge(df_groups, on=["symbol", "entry_date"], how="inner")
    
    # Filter out of sample
    df_v2_oos = df_v2[df_v2["date_idx"] >= pd.Timestamp("2023-01-01")].copy()
    
    # Load V2 model
    v2_model = joblib.load("data/models/amir_rolling_xgboost_ordinal.pkl")
    features_v2 = [c for c in df_v2_oos.columns if c.startswith("feature_") and c not in list(masks.keys())]
    
    # Predict V2 probabilities
    probs_v2 = v2_model.predict_proba(df_v2_oos[features_v2])
    prob_50_v2 = np.clip(probs_v2[:, 1:].sum(axis=1), 0, 1)
    df_v2_oos["prob_50"] = prob_50_v2
    df_v2_oos["y_50"] = (df_v2_oos["target_50"] == 1).astype(int)
    
    print("\n--- V2 Day-by-Day Accuracy by Group (OOS 2023+) ---")
    group_results = {}
    for group in masks.keys():
        g_df = df_v2_oos[df_v2_oos[group] == True]
        if len(g_df) == 0:
            print(f"{group}: No events in OOS")
            continue
            
        print(f"\nGroup: {group} (N = {g_df['symbol'].nunique()} unique symbols)")
        days_res = []
        for d in range(1, 21):
            d_df = g_df[g_df["feature_days_since_ep"] == d]
            if len(d_df) == 0: continue
            
            y_true = d_df["y_50"]
            y_prob = d_df["prob_50"]
            
            if len(y_true.unique()) > 1:
                auc = roc_auc_score(y_true, y_prob)
            else:
                auc = np.nan
            brier = brier_score_loss(y_true, y_prob)
            y_pred = (y_prob >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
            
            days_res.append({
                "Day": d, "N": len(d_df), "AUC": auc, "Brier": brier, "Precision": precision,
                "Recall": recall, "FPR": fpr, "Base_Rate": y_true.mean()
            })
            
        group_results[group] = pd.DataFrame(days_res)
        
        # Display best day by AUC
        df_res = group_results[group]
        if not df_res.empty:
            best_day_auc = df_res.loc[df_res["AUC"].idxmax()] if not df_res["AUC"].isna().all() else None
            best_day_brier = df_res.loc[df_res["Brier"].idxmin()] if not df_res["Brier"].isna().all() else None
            print(f"  Best Day by AUC: Day {int(best_day_auc['Day'])} (AUC={best_day_auc['AUC']:.3f}, FPR={best_day_auc['FPR']:.1%})") if best_day_auc is not None else print("  No AUC computed")
            print(f"  Lowest False Positives: {df_res['FPR'].min():.1%}")
            
    # Load V1 and compare on matching dates
    print("\n--- V1 vs V2 Comparison ---")
    try:
        df_v1 = pd.read_parquet("data/ml_datasets/amir_spec/dataset.parquet")
        df_v1["date_idx"] = pd.to_datetime(df_v1["entry_date"])
        df_v1_oos = df_v1[df_v1["date_idx"] >= pd.Timestamp("2023-01-01")].copy()
        
        # Merge group logic
        # For V1, the EP date is entry_date - some days, but we can merge with df_v2 to get the group
        df_v2_oos_merged = df_v2_oos.copy()
        df_v1_oos["date"] = df_v1_oos["entry_date"]
        
        merged_v1_v2 = df_v1_oos.merge(df_v2_oos_merged, on=["symbol", "date"], suffixes=("_v1", "_v2"))
        
        v1_model = joblib.load("data/models/amir_spec_xgboost_ordinal.pkl")
        features_v1 = [c for c in df_v1_oos.columns if c.startswith("feature_") and c not in list(masks.keys())]
        
        # rename _v1 columns back to normal for prediction
        v1_data = merged_v1_v2[[c + "_v1" if c + "_v1" in merged_v1_v2.columns else c for c in features_v1]].copy()
        v1_data.columns = features_v1
        
        probs_v1 = v1_model.predict_proba(v1_data)
        merged_v1_v2["prob_50_v1"] = np.clip(probs_v1[:, 1:].sum(axis=1), 0, 1)
        # Using V2's ground truth
        merged_v1_v2["y_50"] = (merged_v1_v2["target_50_v2"] == 1).astype(int)
        
        for group in masks.keys():
            g_df = merged_v1_v2[merged_v1_v2[group] == True]
            if len(g_df) == 0 or len(g_df["y_50"].unique()) < 2:
                continue
                
            auc_v1 = roc_auc_score(g_df["y_50"], g_df["prob_50_v1"])
            brier_v1 = brier_score_loss(g_df["y_50"], g_df["prob_50_v1"])
            
            auc_v2 = roc_auc_score(g_df["y_50"], g_df["prob_50"]) # prob_50 from v2 prediction earlier
            brier_v2 = brier_score_loss(g_df["y_50"], g_df["prob_50"])
            
            print(f"Group: {group} (N = {len(g_df)})")
            print(f"  V1 (Point-in-time) -> AUC: {auc_v1:.3f}, Brier: {brier_v1:.4f}")
            print(f"  V2 (Rolling)       -> AUC: {auc_v2:.3f}, Brier: {brier_v2:.4f}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
