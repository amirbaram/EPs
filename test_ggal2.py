import datastore
sym = "GGAL"
bars = datastore.load_bars(sym)
d1_idx = bars.index.get_loc("2025-09-22")
print("D1 Low:", bars["low"].iloc[d1_idx])
print("Days 1-5 Lows:")
for i in range(d1_idx, d1_idx+5):
    print(bars.index[i].strftime("%Y-%m-%d"), bars["low"].iloc[i])
