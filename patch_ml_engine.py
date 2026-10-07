import re

with open("ep_ml_engine.py", "r") as f:
    code = f.read()

search = """        df = datastore.load_bars(sym)
        if df is None or len(df) <= d1_idx + 4:
            return None"""
replace = """        df = datastore.load_bars(sym)
        if df is None or len(df) <= d1_idx + (days_forward - 1):
            return None"""
code = code.replace(search, replace)

search2 = """        d5_idx = d1_idx + 4"""
replace2 = """        d5_idx = d1_idx + (days_forward - 1)"""
code = code.replace(search2, replace2)

with open("ep_ml_engine.py", "w") as f:
    f.write(code)

