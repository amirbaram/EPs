import pandas as pd
import numpy as np
import datastore

df_all = pd.read_parquet("data/simulations/ep_combined_study_scored.parquet")
print(f"Total setups: {len(df_all)}")

# Filter for setups that meet DRE premise on Day 1:
# 1. Big Gap (>= 5%) or massive volume
# 2. RVOL >= 2.5x
# 3. Dollar volume >= $15M
# 4. Weak / messy Day 1 close (close_pos < 0.65) - i.e. NOT a classical clean EP close
# Let's inspect columns available:
print("Columns in df_all:", [c for c in df_all.columns if 'pos' in c or 'gap' in c or 'rvol' in c or 'dvol' in c or 'sub' in c])

