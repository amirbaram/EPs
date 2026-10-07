import pandas as pd
df1 = pd.read_parquet("data/simulations/ep_combined_study.parquet")
df2 = pd.read_parquet("data/ml_datasets/amir_spec/dataset.parquet")

print("df1 cols:", df1.columns.tolist())
print(df1[["symbol", "date"]].head(3))
print("---")
print(df2[["symbol", "entry_date"]].head(3))
