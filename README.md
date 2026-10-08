# Setup Scanner

Full-US-market EOD scanner for: gappers, high volume close, delayed high volume
close, flat base, high tight flag, higher low @ MA, undercut & rally (all daily),
plus two multi-timeframe setups that run on daily AND weekly: backburner (at/near
the all-time high after a strong uptrend, or retraced ≤50% of the up-move) and
stairstep (an orderly staircase down after a strong uptrend).

Every hit carries a 0–100 **quality score** (the dashboard's default sort) blending
ADR, prior-trend quality, consolidation tightness, EMA10/20 surfing and volume
dry-up. High tight flag classifies its consolidation shape (flag / pennant / rising
triangle / …). High volume close now requires a real prior uptrend.

## Install
    pip install -r requirements.txt

## First run (one-time, ~30-60 min for the full market)
    python scan.py init
    # or test small first:  python scan.py init --max-tickers 300

Downloads the full NASDAQ+NYSE+AMEX common-stock universe (~6-8k symbols)
and HISTORY_YEARS (default 3) of adjusted daily bars into data/bars/.
Resumable — re-run if it dies; already-cached tickers are skipped. To change
the depth, wipe data/bars/ and re-run init (the cache only extends forward).

## Every evening (after the close, ~3-6 min)
    python scan.py daily

Pulls only the missing bars, scans everything, writes to output/:
- dashboard_<date>.html  — open in a browser: per-setup tables (sortable,
  liquidity filters) + interactive chart per ticker with level/trigger lines
- hits_<date>.csv        — all hits with stats
- watchlist_<setup>_<date>.txt — paste-importable into a TradingView watchlist

Re-scan cached data without downloading:  python scan.py scan

## Interactive dashboard + backtester (the primary app)
    python serve.py        # loads bars into RAM, then http://127.0.0.1:8780

The live server re-runs detection on demand, so it hosts the interactive features:

- **Date picker / backtest** — pick any past date; the server re-scans as-of that
  date (every ticker truncated at the date — no look-ahead) and each chart shows the
  bars AFTER it, with an "as-of" marker + shaded outcome zone, so you can see how the
  setup resolved.
- **⚙ Settings panel** — tune any setup's thresholds (plus the quality weights and
  regime gates) and "Apply & Rescan" to see results update live. Saved to
  data/settings.json; the nightly `scan.py daily` honors them too. (ATR_FRACTION is
  intentionally not here — it changes the cached pivot pass; edit config.py + reload.)
- **↻ Update button** — pulls the latest EOD data in the background, rebuilds the
  in-RAM frames, and reloads to the new date. Status is polled while it runs.
- **Auto-update** — toggle it on (with a local HH:MM) in the settings panel and the
  server runs the daily update itself, once per day, while it's open.

First load reads all tickers into memory (~30-60s); set SERVE_MIN_DVOL_M=20 (env var)
to load only liquid names and start faster / scan quicker.

**Scan caching.** Full-universe detection is ~110s, so each scan is memoized by
(date, settings): the first scan of a date/settings combo pays the full cost, and
every revisit of it is ~instant (only the chart payload, ~2s, is rebuilt). Changing
the date or any setting is a new key (a deliberate recompute); the latest date is
warmed in the background at startup, and the cache is cleared + re-warmed after a data
update. Only the small hit-rows are cached (not the chart payloads), to keep RAM low.

## Regime-aware setups
Flat base, high tight flag, and higher-low-@-MA require a real prior uptrend,
detected by trendlab.py's chart segmentation (up/down/range with strength +
clarity), instead of the old two-point gain / rising-SMA proxies. Undercut &
rally requires a long-term uptrend too, but via the 50/200 SMA stack (price >
200 SMA, 50 SMA > 200 SMA) — more robust than segmentation for a setup whose
sharp undercut day distorts the swing structure. High volume close now requires a
strong prior uptrend too (current-regime strong_uptrend; toggle HVC_REQUIRE_UPTREND).
Gappers fire in any regime, and delayed-HVC's gap+base is itself the ignition so its
uptrend gate is optional (DHVC_REQUIRE_UPTREND, default off). Every hit carries
regime_dir + regime_clarity columns. Set REGIME_USE=False in config.py to fall back
to the legacy trend tests.

## Setup quality score
Every hit gets a 0–100 `quality` (the dashboard's default sort, color-coded),
computed in quality.py as a weighted blend of: ADR, prior up-move quality (net gain ×
clarity), consolidation tightness (tighter = better), "surfing" (closes at/above a
rising EMA10 & EMA20 through the base), and volume dry-up across the base. Each setup
scores only the factors that apply to it — event setups with no consolidation
(gapper, single-bar HVC, undercut) score on ADR + trend only, renormalized. Weights
and ramps live in config.py (Q_*) and are tunable in the settings panel.

## Consolidation patterns
patterns.py classifies a consolidation's shape via a least-squares fit of the upper
(highs) and lower (lows) trendlines, normalized to ATR/bar: flag / pennant /
rising_triangle / channel / other. It runs on EVERY setup that has a consolidation
window — high_tight_flag (the flag), flat_base, delayed_hvc, higher_low_ma (the
pullback), backburner (the pause since the ATH) and stairstep (the staircase) — and
each carries a `pattern` column. Single-bar events (gapper, hvc, undercut_rally) have
no consolidation, so no pattern. The dashboard has a **pat** dropdown (auto-shown only
on tabs that have patterns) to filter to one shape, which combines with the other
filters and the export button. High tight flag additionally gates at scan time on
HTF_ALLOWED_PATTERNS (the other setups always fire; pattern is informational there).

## Pivots & structure (trendlab.py)
Swings/pivots and structure come from a Python port of the "SwingsTimes" Pine
indicator — a single, causal pivot source shared by the regime engine AND the
pivot-using setups (higher-low, undercut). Pivots: an ATR-fraction swing tracker
(delta = ATR(14) × ATR_FRACTION) that confirms a pivot at the first opposing
bar-to-bar break (~1 bar later, no centered look-ahead) and tags it HH/LH/HL/LL/
DT/DB. Structure: a tag-driven FSM evaluated every bar (uptrend / downtrend /
rectangle / contracting / expanding + their breaks), firing intrabar without
waiting for the next pivot. ATR_FRACTION default is 0.7 (tuned for daily regime
gating); the Pine indicator's own default is 0.2 (finer visual zigzag). Lower it
for more, smaller segments; raise it for coarser ones.

## Dashboard filters
Beyond min $vol / price / ADR, each hit carries (and the dashboard filters on):
avg daily volume (K), % above the 52-week low, above-50-SMA, above-200-SMA, and
regime_dir / regime_clarity — i.e. a Minervini-style trend-template screen on top
of the pattern. Market-cap / EPS / sales-growth filters are not included (not in
yfinance daily bars), plus a **pat** dropdown to filter by consolidation shape
(shown only on tabs that have patterns). The filter inputs sit on the header row and
the chart shades the prior uptrend (green) that triggered the selected setup. Each hit
shows a `tf` column (1D / 1W); clicking a 1W hit renders a weekly chart. The **⬇ TV**
button downloads the current tab's filtered + sorted rows as a TradingView watchlist
(EXCHANGE:SYMBOL), so you can export exactly what you've screened down to.

## Weekly timeframe
Weekly bars are resampled from the daily cache (W-FRI) into data/bars_w/ and
rebuilt automatically on every `init` / `daily`. The two multi-timeframe setups
run on both 1D and 1W; the other six are daily-only.

Backburner knobs: BACKBURNER_MAX_RETRAC (25% — max pullback from the ATH, as % of
the up-move) and BACKBURNER_ATH_MAX_AGE (40 bars — the ATH must be recent; 0 = off).
Stairstep knobs: STAIRSTEP_MIN_BARS (4) consecutive bars each failing to make a
meaningful new high (high < previous high + STAIRSTEP_ATR_FRACTION×ATR), the run
ending on the last bar or one bar back, with a net-decline check.

## Tuning
All thresholds live in config.py (gap %, base depth, HTF pole gain, etc.) as the
DEFAULTS; the live server's ⚙ panel overrides the cache-safe ones at runtime and
saves them to data/settings.json (settings.py applies them on top of config for each
scan, the nightly run included). Liquidity is intentionally NOT filtered at scan time
— quiet tickers still get flagged; use the dashboard's min-$vol / min-price / min-ADR
filters. Point SCAN_DATA_DIR / SCAN_OUTPUT_DIR at a throwaway dir to run an isolated
instance (e.g. for the synthetic test) without touching the primary cache.

The dashboard embeds charts only for the top CHART_TOP_N hits per setup by
dollar volume (keeps the HTML small); every other row shows an
"open in TradingView" link instead.

Optional: set DISCORD_WEBHOOK in config.py to push the daily list.

## Scheduling (Windows)
Task Scheduler -> daily 23:30 (after US close, Israel time):
    python C:\path\to\scanner\scan.py daily

## Notes
- Data: yfinance (adjusted OHLC). Fine for live scanning; for backtesting
  you'd want survivorship-bias-free data (e.g. Norgate) later.
- Validation: make_test_data.py builds synthetic patterns (one per setup, incl.
  DHVCTEST for delayed-HVC, plus NOTEST which must stay clean). It OVERWRITES the
  data dir, so run it isolated:
      SCAN_DATA_DIR=/tmp/scan_t SCAN_OUTPUT_DIR=/tmp/scan_t python make_test_data.py
      SCAN_DATA_DIR=/tmp/scan_t SCAN_OUTPUT_DIR=/tmp/scan_t python scan.py scan
  Expect the core setups to fire (gapper, hvc, delayed_hvc, flat_base,
  high_tight_flag, higher_low_ma, undercut_rally) plus backburner/stairstep, with a
  populated quality column and NOTEST flagging nothing.
