import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def train_dynamic_stop_model():
    print("🚀 TRAINING DYNAMIC STOP-LOSS MODEL (QUANTILE REGRESSION)")
    
    df = pd.read_parquet("data/ml_datasets/amir_spec/dataset.parquet")
    
    # Train ONLY on setups that worked (hit at least 50% target).
    df_winners = df[df["target_50"] == 1].copy()
    df_winners = df_winners.dropna(subset=["audit_mae_pct"])
    
    features = [c for c in df.columns if c.startswith("feature_")]
    num_features = [c for c in features if pd.api.types.is_numeric_dtype(df_winners[c])]
    cat_features = [c for c in features if c not in num_features]
    
    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ])
    
    X = df_winners[features]
    y = df_winners["audit_mae_pct"]
    
    lgb_model = lgb.LGBMRegressor(
        objective='quantile',
        alpha=0.05,
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", lgb_model)
    ])
    
    pipeline.fit(X, y)
    out_path = Path("data/models/ep_dynamic_stoploss_q05.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    print("  ✅ Saved Quantile Stop-Loss Pipeline")

if __name__ == "__main__":
    train_dynamic_stop_model()
