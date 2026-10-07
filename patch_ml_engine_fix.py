with open("ep_ml_engine.py", "r") as f:
    code = f.read()

code = code.replace("def compute_features(self, sym: str, d1_idx: int) -> dict | None:", "def compute_features(self, sym: str, d1_idx: int, days_forward: int = 5) -> dict | None:")

with open("ep_ml_engine.py", "w") as f:
    f.write(code)
