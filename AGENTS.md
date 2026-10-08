# AGENTS.md — Setup Scanner (`amirbaram/scanner`)

Full-US-market **EOD** stock scanner (Python 3.13). Detects ~10 setups (gappers, HVC,
delayed-HVC, flat base, high tight flag, higher-low@MA, undercut & rally, plus MTF
backburner & stairstep), each with a 0–100 quality score. Primary app is an
interactive dashboard + point-in-time backtester. Data: yfinance daily (adjusted).

## Run
```bash
.venv/bin/python serve.py          # PRIMARY app → http://127.0.0.1:8780 (loads bars to RAM, ~30-60s)
.venv/bin/python scan.py daily     # nightly: pull missing bars, scan, write output/
.venv/bin/python scan.py scan      # re-scan cached data, no download
.venv/bin/python scan.py init --max-tickers 300   # small first run (full init is ~30-60 min)
```
Deps: `pip install -r requirements.txt`. Use the `.venv/` interpreter for everything.

## The gate — run after EVERY change (non-negotiable)
```bash
bash scripts/check.sh              # byte-compiles core modules + runs tests/ (fast, offline, no deps)
```
- Green gate is required before calling any change done. It exits non-zero on first failure.
- **Every bug you fix adds a permanent regression test** in `tests/` — one that fails before your
  fix and passes after. Name the seed + date in the test's docstring.
- `tests/` is stdlib `unittest`, synthetic data only. `/api/*` endpoint smoke needs a live `:8780`
  server and is NOT in the fast gate — run it separately if you touch the server routes.

## How to work here (Amir's rules — follow them)
- **Discuss before implementing.** Propose any logic/algorithm/fix change and get approval BEFORE
  editing. Don't just start coding.
- **Ask before choosing an algorithm/method** (e.g. pivot vs zigzag). Don't default silently.
- **Answer questions first.** If Amir asks a question, answer it before deciding to change code.
- **Chart-first validation.** A new/changed *setup* must be verified on the chart BEFORE any long
  data/validation run. Don't kick off big runs on unvalidated detections.
- **Version file for wide-blast-radius changes.** For a change touching many call sites, create a
  new *versioned copy* of the file rather than editing in place.
- **Be specific.** Cite exact timestamps / prices / bar-counts in any chart or data claim — never
  "a few bars later". Gloss any codename (ST-4, SIG-9, …) in plain English on first use.
- **Plain-English UI.** Every surface states meaning + suggested action; no unexplained jargon.
- After a change that affects the running server, tell Amir whether he must **restart `serve.py`**
  or can just **reload the page**.
- Log bugs you find in `BUGS.md`; read it at the start of a work session.
- A validated/changed setup also needs: chart anatomy check → `build_setup_registry.py` rebuild →
  blindness check → docs update (the "setup graduation" checklist).

## Git discipline
- Branch is `main`, remote `origin`. **Do NOT merge or push to `main`.** Commit your work to your
  own branch only. Merging/pushing is the integrator's job, not a worker session's.
- Commit/push only when Amir asks.

## Architecture map
| Concern | File |
|---|---|
| CLI entry (init/daily/scan) | `scan.py` |
| Interactive server + backtester | `serve.py` |
| All default thresholds | `config.py` (runtime overrides → `data/settings.json` via `settings.py`) |
| Pivots + structure/regime (SwingsTimes port) | `trendlab.py` (ATR_FRACTION=0.7 daily) |
| Quality score (0–100) | `quality.py` (weights `Q_*` in config.py) |
| Consolidation shape classifier | `patterns.py` |
| Setup registry | `build_setup_registry.py` |
| Regression suite | `tests/` (see `tests/README.md`) |

## Config & data gotchas
- Thresholds live in `config.py`. The ⚙ settings panel overrides the **cache-safe** ones live and
  saves to `data/settings.json` (the nightly run honors them). `ATR_FRACTION` is intentionally NOT
  in the panel — it changes the cached pivot pass; edit `config.py` and reload.
- Liquidity is deliberately NOT filtered at scan time — filter in the dashboard (min $vol/price/ADR).
- `make_test_data.py` **OVERWRITES the data dir** — always run it isolated:
  `SCAN_DATA_DIR=/tmp/scan_t SCAN_OUTPUT_DIR=/tmp/scan_t .venv/bin/python make_test_data.py`
- Point `SCAN_DATA_DIR` / `SCAN_OUTPUT_DIR` at a throwaway dir to run an isolated instance without
  touching the primary cache. `NOTEST` must always stay clean (flag nothing).

See `README.md` (setups, scoring, regime gating in depth) and `DOCS.md` for detail.
