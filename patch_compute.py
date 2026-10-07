import re

with open("serve_ep.py", "r") as f:
    code = f.read()

search = """    return {
        "sector_theme": {"""

replace = """    
    ai_scores = {}
    if ev_row:
        ai_scores["ml_50"] = float(ev_row.get("ml_50", 0)) if pd.notna(ev_row.get("ml_50")) else None
        ai_scores["ml_100"] = float(ev_row.get("ml_100", 0)) if pd.notna(ev_row.get("ml_100")) else None
        ai_scores["ml_150"] = float(ev_row.get("ml_150", 0)) if pd.notna(ev_row.get("ml_150")) else None
        ai_scores["ml_200"] = float(ev_row.get("ml_200", 0)) if pd.notna(ev_row.get("ml_200")) else None

    return {
        "ai_scores": ai_scores,
        "sector_theme": {"""

if search in code:
    code = code.replace(search, replace)
    print("compute_dossier patched!")
else:
    print("compute_dossier search failed!")

with open("serve_ep.py", "w") as f:
    f.write(code)
