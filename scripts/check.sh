#!/usr/bin/env bash
# Fast offline gate for EPs repository: byte-compile core modules + run regression suite (tests/).
# Run after EVERY change. Exits non-zero on first failure.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)"

echo "▸ py_compile (EP core modules)"
"$PY" - <<'PYEOF'
import py_compile, sys
CORE_MODULES = [
    "serve_ep.py", "serve_ep_tracker.py", "ep_ml_engine.py",
    "scanner_core.py", "indicators.py", "datastore.py", "labels.py",
    "thematic_engine.py", "generate_handbook_pdf.py"
]
bad = []
for f in CORE_MODULES:
    try:
        py_compile.compile(f, doraise=True)
    except Exception as e:
        bad.append(str(e))
if bad:
    print("\n".join(bad))
    sys.exit(1)
print(f"  ok — {len(CORE_MODULES)} core EP modules compiled cleanly")
PYEOF

echo "▸ regression suite (tests/)"
"$PY" -m unittest discover -s tests -p 'test_*.py' -v

echo "✓ EPs check passed"
