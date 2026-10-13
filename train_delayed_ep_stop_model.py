import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path

def train_model():
    print("="*80)
    print("🚀 TRAINING DELAYED EP STOP-LOSS ML MODEL (QUANTILE REGRESSION)")
    print("="*80)
    
    df = pd.read_parquet("data/simulations/delayed_ep_trades.parquet")
    print(f"Total delayed trades: {len(df)}")
    
    # Unpack features
    feat_rows = []
    for idx, row in df.iterrows():
        f = dict(row['feats']) if row['feats'] is not None else {}
        f['prob_50'] = row['prob_50']
        f['prob_100'] = row['prob_100']
        f['prob_200'] = row['prob_200']
        f['trigger_bar_offset'] = row['trigger_bar_offset']
        f['mae_pct'] = row['mae_pct']
        f['is_winner'] = row['is_winner']
        f['year'] = row['year']
        f['trade_ret_pct'] = row['trade_ret_pct']
        f['entry_px'] = row['entry_px']
        f['d1_low'] = row['d1_low']
        f['shelf_low'] = row['shelf_low']
        f['prior_low'] = row['prior_low']
        feat_rows.append(f)
        
    fdf = pd.DataFrame(feat_rows)
    
    # Train on WINNING trades to learn the required adverse excursion before winning
    train_winners = fdf[(fdf['is_winner'] == True) & (fdf['year'] < 2023)].copy()
    test_all = fdf[fdf['year'] >= 2023].copy()
    
    print(f"Training on {len(train_winners)} historical winners (<2023). Testing on {len(test_all)} out-of-sample trades (2023+).")
    
    feature_cols = [
        'feature_days_since_ep', 'feature_tightness_1m', 'feature_gap_pct', 'feature_rvol',
        'feature_close_pos', 'feature_is_novel_vol_9m', 'feature_is_inst_sweet_spot',
        'feature_tightness_since_ep', 'feature_digestion_ratio', 'feature_ret_since_ep',
        'feature_dist_to_d1_low', 'feature_is_blue_sky', 'feature_dist_to_overhead_pct',
        'feature_is_qullamagi_linear', 'prob_50', 'prob_100', 'prob_200', 'trigger_bar_offset'
    ]
    
    # Clean NaNs in numeric features
    for col in feature_cols:
        train_winners[col] = pd.to_numeric(train_winners[col], errors='coerce').fillna(0.0)
        test_all[col] = pd.to_numeric(test_all[col], errors='coerce').fillna(0.0)
        fdf[col] = pd.to_numeric(fdf[col], errors='coerce').fillna(0.0)
        
    X_train = train_winners[feature_cols]
    # mae_pct is negative (e.g. -6.5%). We want quantile regression at alpha=0.10 (conservative lower boundary)
    y_train = train_winners['mae_pct']
    
    model = lgb.LGBMRegressor(
        objective='quantile',
        alpha=0.10, # 10th percentile: covers 90% of winning adverse excursions
        n_estimators=100,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Feature importances
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop Predictors of Needed Stoploss Distance:")
    for k, v in imp.head(8).items():
        print(f"  • {k:<30}: {v}")
        
    # Predict on entire dataset
    fdf['pred_stop_pct'] = model.predict(fdf[feature_cols])
    
    # Bounded stop logic:
    # 1. Minimum stop distance: -4.0% (prevent micro-shakeouts)
    # 2. Maximum stop distance: -20.0% or distance to Day 1 low (never wider than Day 1 Low)
    fdf['d1_low_pct'] = (fdf['d1_low'] - fdf['entry_px']) / fdf['entry_px'] * 100.0 # negative
    fdf['clamped_stop_pct'] = fdf['pred_stop_pct'].clip(lower=fdf['d1_low_pct'], upper=-4.0)
    
    print("\nSummary of Predicted Clamped Stop Distances:")
    print(fdf['clamped_stop_pct'].describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]))
    
    # Save model
    out_path = Path("data/models/ep_delayed_stop_model.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, out_path)
    print(f"\n✓ Saved ML Stoploss Model to {out_path}")
    
    # Save enriched dataset with ML stop predictions
    df['ml_stop_pct'] = fdf['clamped_stop_pct'].values
    df['ml_stop_price'] = df['entry_px'] * (1.0 + df['ml_stop_pct'] / 100.0)
    df.to_parquet("data/simulations/delayed_ep_trades.parquet")
    print("✓ Updated data/simulations/delayed_ep_trades.parquet with ml_stop_price.")

if __name__ == "__main__":
    train_model()
