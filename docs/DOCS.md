# Setup Scanner — Reference & Runbook

A full-US-market EOD stock scanner with an interactive backtesting app, multi-timeframe + intraday
EMA-Rider analysis, and a sector/theme classification layer. This document is the maintained
reference: the app, the setups, the config variables, and the data-maintenance processes.
(Quick-start lives in `README.md`; this is the deep reference.)

> The **Config reference** table near the bottom is generated — run `python gen_docs.py` after
> editing `config.py`. Everything else is hand-maintained; keep it current when features change.

---

## 1. Overview & module map

| Module | Role |
|---|---|
| `config.py` | All thresholds/defaults (the source of truth; the ⚙ panel overrides the cache-safe ones). |
| `universe.py` | Build the NASDAQ+NYSE+AMEX common-stock universe (+ curated ETFs); `active_symbols()` reads the liquid subset. |
| `datastore.py` | Per-ticker parquet bar cache (daily + weekly resample + 15m intraday); yfinance download / incremental update. |
| `indicators.py` | `add_indicators` — SMA/EMA/ATR/ADR/RVOL/$vol/returns/52w on a frame. |
| `setups.py` | Detectors (one per setup) → hit dicts; `DETECTORS` / `MULTI_TF_DETECTORS`. |
| `emarider.py` | EMA-streak engine (`current_streak`) + the visual lab. |
| `scan.py` | Nightly EOD scan (`scan_all`) + point-in-time `scan_asof` (the server's engine); `_base_row`. |
| `serve.py` | Flask app: the interactive dashboard/backtester + all `/api/*` endpoints + caches. |
| `report.py` | Dashboard HTML/JS, chart payloads, table columns. |
| `trendlab.py` / `regime.py` | Causal swing/pivot + structure segmentation (lazy segment metrics); regime gate used by several setups. |
| `framecache.py` | Per-frame memo cache keyed by object identity (avoids pandas `df.attrs` deep-copies); backs trendlab + ema-rider. |
| `quality.py` / `patterns.py` | 0–100 quality score; consolidation shape classifier. |
| `journal.py` | SQLite chart-study journal (per-ticker saved setups + comments). |
| `performance.py` | Equal-weighted sector/theme/universe/futures performance for the ▶ Perf drawer. |
| `labels.py` | Runtime sector + theme lookups (from `data/sectors.csv` / `data/themes.csv`). |
| `marketcap.py` | Cached market-cap lookup (yfinance fast_info). |
| `harvest.py` / `classify.py` / `build_labels.py` | Theme pipeline: cache profiles+holdings → classify → merge into `themes.csv`. |
| `build_active_universe.py` | Weekly: write the liquid `data/universe_active.csv`. |
| `mtf.py` / `build_mtf.py` | Multi-timeframe awareness state per ticker/TF; precompute the `data/mtf.parquet` matrix. |
| `market.py` | Top-down market situational-awareness score (long/out/short) from index trend + breadth + VIX + macro; dual close/live read. |
| `intraday.py` | Universe-wide intraday market internals from the 5m base (auto-inferred resolution): A/D, TRIN, net TICK, VWAP breadth, new intraday H/L. |
| `sr.py` | Support/resistance levels: swing-pivot zones clustered into strength-weighted levels, classified live (support below / resistance above the latest close). Works on any enriched frame (daily/weekly/intraday). |
| `ratios.py` | Ratio / relative-strength engine: synthetic A÷B OHLC frames (index legs on futures, member baskets for ETF-less groups) enriched + run through `mtf.tf_state` + `sr.sr_levels`. Powers the market risk/rotation block and group RS. |
| `narrative.py` | Natural-language **Playbook**: composes the awareness data (stance + rotation + group RS + member leaders/laggards + S/R) into hedged long/short candidate sentences. |
| `backfill_history.py` | One-time: deepen the daily cache to `HISTORY_YEARS` (active universe + futures). |
| `gen_docs.py` | Regenerate this file's config table. |

---

## 2. Apps & how to run

**Nightly EOD (CLI):**
- `python scan.py init` — one-time bulk download (HISTORY_YEARS of daily bars).
- `python scan.py daily` — pull missing bars (**active universe only**), scan, write `output/` (dashboard html, hits csv, TV watchlists).
- `python scan.py weekly` — **full-universe** tail refresh + re-evaluate active membership (`build_active_universe`). Run weekly so newly-liquid names can enter the universe.
- `python scan.py scan` — re-scan the cache without downloading.

**Interactive app (primary):** `python serve.py` → http://127.0.0.1:8780 (full universe).
- **Small-list app:** `SERVE_SYMBOLS_FILE="6M Winners.txt" PORT=8781 python serve.py` — loads only that
  watchlist (fast settings testing; intraday is intended here). Accepts TradingView `EXCHANGE:SYMBOL`,
  plain, or RTF lists.
- Toolbar: **date picker** (backtest as-of), **TF selector** (EMA Rider only: 1D/2D/3D/1W/2W/1M/3M/6M +
  intraday 15m…12h), **⚙** settings, **↻ Update** (EOD data, active universe only), **↻ Weekly**
  (full-universe refresh + re-evaluate active membership), **⬇ Intraday** (download 60-day 15m),
  **↻ Labels** (reload sector/theme CSVs, no restart), **▶ Perf** (sector/theme/universe/futures
  performance drawer), **📓 Journal** (chart study journal), **QM/FB/Best** presets, **⬇ TV** export.
  A **+** before each ticker saves it to the journal.
- The active-universe filter (§5) reduces what the full app loads; `SERVE_SYMBOLS_FILE` bypasses it.

**Restart rule:** code (`.py`) changes need a server restart. Threshold tweaks via ⚙ are live; label
edits are live via ↻ Labels.

---

## 3. Setups

Each detector emits a `state` and per-setup columns. Daily-only unless noted. Knobs are in `config.py`
(prefix in parens) and live-tunable in ⚙.

| Setup | Detects | States | Key knobs |
|---|---|---|---|
| Gapper | Big surprise gap on heavy volume | breakout | `GAP_*` |
| High Volume Close | Heavy-volume close near the high after a prior uptrend | breakout | `HVC_*` |
| Delayed HVC | Gap → tight hold → HVC breakout | breakout | `DHVC_*` |
| Flat Base | Flat-topped base near highs, multi-touch resistance, Stage-2 | building/breakout | `FB_*` |
| High Tight Flag | Explosive pole (90-120%+) + shallow tight flag (≤25% off high), volume breakout; pole-efficiency + strength evidence | building/breakout | `HTF_*` |
| QM Breakout | Qullamaggie continuation: big move → base → break a resistance in the breakout-level stack on volume | building/breakout | `QMB_*` |
| Higher Low @ MA | Pullback undercutting a rising 10/20/50 MA in an uptrend | building/breakout | `HL_*` |
| Undercut & Rally | Shakeout below a prior swing low, reclaim | building/breakout | `UR_*` |
| Uptrend / Downtrend | Current trendlab segment is up / down | — | (trendlab) |
| Backburner (1D+1W) | At/near ATH after a strong uptrend, or ≤ retrace | building/breakout | `BACKBURNER_*` |
| Stairstep (1D+1W) | Orderly staircase down after a strong uptrend | building/breakout | `STAIRSTEP_*` |
| **EMA Rider** bull/bear | A streak of closes above/below an EMA that keeps getting defended | **riding / touching / reversal** | `ER_*` |

**EMA Rider detail** (`emarider.py` + `setups._ema_rider`): a *riding* streak is N consecutive closes
on one side of `ER_EMA_LEN`; *touching* = the latest bar wicked into/through the EMA (within
`ER_ATR_FRAC`×ATR) but held; *reversal* = it was riding through the previous bar but the current bar
closed across the EMA. Fires when streak ≥ `ER_MIN_STREAK` with ≥ `ER_MIN_HOLDS` EMA touches. Runs on
the selected timeframe (daily-resampled 2D–6M, or intraday 15m–12h resampled from the Tiingo 5m base — yfinance 15m for futures).

Every hit also carries: `quality` (0–100, default sort), `state`, `tf`, `rs_rank`, liquidity columns,
`regime_dir/clarity`, `market_cap`, **`sector`** + **`themes`** (§6), and a `pattern` where applicable.

**Intraday breakout volume projection (QM + HTF).** A `breakout` state needs the popping bar to trade on
volume expansion (QM: ≥`QMB_BREAKOUT_VOL`× the 10-bar avg; HTF: `rvol` ≥ `HTF_BREAKOUT_RVOL`). While TODAY's
daily bar is still forming, its volume is only a fraction of a full session, so a live intraday breakout
could never clear that test until the close. With `BREAKOUT_VOL_PROJECT` on, `setups._project_last_volume`
scales the partial bar's volume up to a full-session estimate via a U-shaped intraday cumulative-volume
curve (`_VOL_CURVE`; ~13% in by 10:00, 45% by 12:30, …) — only when the latest bar is today in ET and the
market is open. Such breakouts are flagged **`provisional`** (a **~** marker on the state, with the session
fraction used in `vol_proj`) and settle to real volume at the close.

---

## 4. Daily data maintenance

- **Bars:** `datastore.init_history()` (bulk) / `update_daily(full=False)` (incremental, forward-only,
  **active universe only**) → per-ticker parquet in `data/bars/`. `update_daily(full=True)` (the weekly
  job) refreshes every name so membership can be re-evaluated. `build_weekly()` resamples W-FRI into
  `data/bars_w/` — rebuilt for the active set on update; out-of-universe weeklies rebuild lazily on load
  (`load_bars_w` mtime check).
- **In-app update:** the **↻ Update** button (or the auto-update scheduler at `AUTO_UPDATE_TIME`) runs
  `update_daily()` (active-only) in a background thread, rebuilds the RAM frames, clears caches, re-warms.
  Out-of-universe tickers are NOT downloaded during normal use — only the weekly `scan.py weekly` refreshes
  the full cache to let newly-liquid names join the universe.
- Adjusted OHLC (`auto_adjust=True`). HISTORY_YEARS controls depth (wipe `data/bars/` to deepen).
- **Split/dividend guard:** yfinance re-bases the WHOLE history on every adjustment event, so a plain
  tail-merge would seam two price scales (a fake overnight cliff — breaks structure + every detector).
  `update_daily` compares closes on the overlap days (`_adjustment_seam`); a re-based symbol gets its
  full history refetched instead of merged, and split-sized factors also back-adjust the raw Tiingo
  5-min store (`datastore.repair_tiingo_5m` — per-session alignment against the adjusted daily close).
  One-time repair scripts: `refetch_history.py` (resumable full re-pull of the active universe) and
  `repair_tiingo_5m.py` (sweep the whole 5-min store). After any such repair: rebuild the setup
  registries (`build_setup_registry.py --force` and `--sliding --force`) and clear `data/scan_cache/`.

## 4b. Data sources — who provides what (keep this current when adding sources; Amir 2026-07-03)

| Source | What we pull | Where in code | Store |
|---|---|---|---|
| **yfinance** | Daily OHLCV, split/dividend-ADJUSTED (init/update/refetch; the canonical price store) | `datastore.py` (`init_history`, `update_daily`, `_download_batch`), `refetch_history.py` | `data/bars/` (+ `bars_w/` weekly resample) |
| yfinance | 15-min intraday, rolling ~60d, full-replace. **FUTURES-ONLY as of the 5m migration (2026-07-06):** the 15 futures (ES=F/NQ=F/… trade 24h — no IEX/5m coverage) use this as their intraday base; equities moved to the Tiingo 5m base. | `datastore.fetch_intraday_15m` | `data/bars_15m/` |
| yfinance | Macro symbols (^VIX, ^VXN, DX-Y.NYB, ^TNX, TLT) | `datastore.update_macro` | `data/bars/` |
| yfinance | Company profiles (name/sector/industry/summary) + ETF top-10 holdings | `harvest.py` (`.info`, `funds_data.top_holdings`) | `data/profiles.json`, `data/etf_holdings.json` |
| yfinance | Market cap (`fast_info.market_cap`, one call/ticker) | `marketcap.py` | `data/marketcap.json` |
| yfinance | Earnings dates (`get_earnings_dates`, capped at 100/page) | `ep_news.py` | `data/news_cache/` |
| **Tiingo** (Power tier; key in `data/tiingo_key.txt`, gitignored — NEVER commit) | IEX 5-min intraday store (2017→present, RTH 09:30–15:55, 78 bars/day; RAW as-traded prices — split-adjusted locally by `datastore.repair_tiingo_5m`). **The intraday BASE for all EQUITIES since the 5m migration:** every intraday TF (15m–12h), all internals/awareness/charts/BB/rider/mtf-S-R/atr/rs4h/daytype/perf resample UP from this (`serve._intraday_base`, `datastore.load_bars_5m`, base auto-inferred). Volume is IEX-sample → calibrated ×`vol_factor` to consolidated scale. | `tiingo/` package (client/download/manifest budget governors), `datastore.load_bars_5m` | `data/tiingo/bars_5min/` |
| Tiingo | LIVE IEX quote polling (60s cadence: folds the forming 5-min bar into the live RAM store `serve.FRAMES5` — EP radar, internals, replay buffer). IEX volume ≈ 1–6% of consolidated tape → per-symbol calibration factors | `tiingo_live.py`, `ep_news.py` (radar) | `data/iex_vol_factor.json` (median ≈22×) |
| Tiingo | News API (archive floor ≈3 months — older windows silently return oldest stories; post-filtered) | `ep_news.news_for` | `data/news_cache/` |
| **SEC EDGAR** | 8-K filings + CIK ticker map (rate-limited, cached) | `ep_news.edgar_8k` | `data/edgar_cache/` |
| **Finviz** (static import) | Sector/industry group CSV used in theme/sector classification | `classify.py` via `data/sectors.csv` | `data/sectors.csv` |
| **TradingView** | OUTBOUND: chart links (`tvUrl`) + watchlist export (`⛶ TV` button). INBOUND — manual/ad-hoc extraction via the TradingView MCP (blob-download of a chart's history), NOT an automated feed: NYSE breadth USI:TRIN.NY / USI:ADD / USI:TICK.NY daily, NASDAQ breadth TRINQ / ADDQ / TICKQ daily, SPY consolidated ETH 5-min (~20k-bar plan cap) → `data/trin_daily.csv`, `data/tv/bars_daily_*.parquet`, `data/tv/bars_5min_eth_SPY.parquet` (local-only, gitignored) | `report.py`, `daytype_review_gen.py`, `dayfeatures.py`, `tv_breadth.py` (→ `market._breadth_block`) | — |
| **SqueezeMetrics** (free CSV) | Daily SPX GEX + DIX since 2011 (`squeezemetrics.com/monitor/static/DIX.csv`; lag 1 day = PIT) | `fetch_dayfeat_data.py` → `dayfeatures.py` | `data/features/squeeze_gex.csv` |
| **FRED** (free CSV) | HY OAS `BAMLH0A0HYM2` (fredgraph caps ~3y; full-history credit dial = HYG/IEF ratio via yfinance) | `fetch_dayfeat_data.py` → `dayfeatures.py` | `data/features/hy_oas.csv`, `hyg_ief.csv` |
| **CBOE** (free delayed JSON) | _SPX option chain → per-strike GEX / gamma-flip / walls, logged FORWARD daily (cron 16:35 ET) | `cboe_gex_logger.py` | `data/features/cboe_gex_daily.csv` + archived chains |

Rules of thumb: yfinance = settled EOD truth (adjusted); Tiingo = intraday/live layer (raw IEX);
never mix scales — the split guard (§4) keeps them aligned. When code gains a new source or a
new use of an existing one, ADD A ROW HERE (standing instruction in BUGS.md).

**TV-extraction dependency (removable — Amir 2026-07-06).** The TradingView INBOUND rows are the
only data we can NOT re-pull programmatically (they come from a manual MCP blob-download, not an
API). If we ever drop TV extractions, this is exactly what breaks and the graceful-degradation
path for each (every consumer already guards `if path.exists()`, so removal degrades — it does not
crash):
- `data/trin_daily.csv` (USI:TRIN.NY), `data/tv/bars_daily_{ADD,TICK,TRINQ,ADDQ,TICKQ}.parquet` —
  consumed by `dayfeatures.external_block` (feature columns `a_trin*`, `a_add_close`, `a_tick_*`,
  `a_trinq`, `a_addq_close`, `a_tickq_*`, `a_cumtickq_ch5`). These feed the **day-type models**
  (`daytype_prob2` / retrain) and the day-type review page — NOT the live market-health banner
  (`health_score.py` is TV-free: daily-bar trendlab census + %>20dma + MED + credit + rotation).
  Removal → those breadth features go NaN → the day-type models simply lose that block on the next
  retrain (they were fit with sparse coverage anyway). No live-banner impact.
- `data/tv/bars_5min_eth_SPY.parquet` — consumed by `daytype_review_gen.py` to overlay true
  consolidated extended-hours 5-min candles over IEX for the SPY review chart only. Removal → the
  review chart falls back to IEX 5-min (already the default for every other symbol).
- **CLOSE banner internals** (`tv_breadth.py` → `market._breadth_block`, when
  `config.USE_TV_BREADTH=True` — the default; Amir 2026-07-06). Uses `$ADD`/`$ADDQ` (net
  adv−dec) + `$TRIN`/`$TRINQ` for the daily/close breadth line **and** the breadth SCORE that
  feeds the market situational-awareness stance. `%>MA` and new-H/L have no free source, so they
  self-compute from our yfinance daily bars (the same census, which is also the fallback). A
  staleness guard (`TV_BREADTH_MAX_LAG_DAYS=7`) falls back to the sample if the manual extract
  lags the view date. Removal path: set `USE_TV_BREADTH=False` (or delete `data/tv/`) → the block
  reverts to the pure universe sample, no crash. The **LIVE intraday** internals (`intraday.py`)
  never use TV — TV has no realtime feed, so IEX-derived internals stand there.
So: TV extractions are a **day-type/review-page** dependency plus (default-on) the **CLOSE banner
breadth**. If a future component starts depending on them, ADD IT to this list.

## 5. Active universe (liquidity filter) — weekly

The app + nightly scan load only the **liquid subset** when `data/universe_active.csv` exists.
- **Floors** (config): `UNIVERSE_MIN_AVG_VOL_K`, `UNIVERSE_MIN_PRICE`, `UNIVERSE_MIN_DVOL_M`,
  `UNIVERSE_MIN_ADR_PCT`, `UNIVERSE_MIN_MCAP_M` (all must pass). Curated ETFs + futures are kept in the
  universe, but illiquid ETFs are excluded from every setup.
- **Rebuild weekly:** `python build_active_universe.py` → recomputes stats from the cache and writes the
  list (prints "N of M pass"). `universe.active_symbols()` reads it; `serve.load_frames` + `scan.scan_all`
  intersect with it. `SERVE_SYMBOLS_FILE` overrides (loads its list verbatim).
- Bars keep downloading for **all** tickers, so a name that later qualifies re-enters on the next rebuild
  with no re-download. Delete the csv to go back to the full universe.

## 6. Sector & theme labels

Every ticker can carry one **sector** (industry) and many **themes** (many-to-many). Stored in `data/`,
read by `labels.py`, attached to rows at response time (so edits are live via **↻ Labels** — no rescan).

Pipeline (free, no paid tokens):
1. `python harvest.py [--symbols FILE]` — cache yfinance business summaries (`data/profiles.json`) +
   ETF top-10 holdings (`data/etf_holdings.json`). Incremental; yfinance `.info` is rate-limited (~900/run)
   so re-run a few times to fill failures.
2. `python classify.py [--llm]` — off the cache: editable keyword rules (`data/theme_rules.csv`, one regex
   per theme) + the Finviz industry (`data/sectors.csv`) + ETF holdings → `data/themes_auto.csv`
   (each tag `source`-tagged) and `data/themes_review.csv` (unclassified queue). `--llm` asks a local
   Ollama model for the stragglers.
3. `python build_labels.py` — `themes.csv = (ETF ∪ auto ∪ manual) − exclude`; sectors from the universe
   snapshot. Hand-curate via `data/themes_manual.csv` (add) / `data/themes_exclude.csv` (remove).
4. Click **↻ Labels** in the app.

Market cap: `marketcap.ensure` fetches/caches `data/marketcap.json` (small lists fetch at startup).

## 7. Intraday & multi-timeframe

- **Intraday base (5m migration, 2026-07-06):** EQUITY intraday is the **Tiingo 5-min store** (`data/tiingo/bars_5min/`,
  RTH, calibrated → consolidated volume); every higher TF (15m–12h) resamples UP from it (1h–4h session-anchored,
  8h/12h continuous; base auto-inferred by `resample_intraday`). Live freshness comes from the Tiingo poller folding
  the forming 5m bar into `serve.FRAMES5` (the SINGLE live intraday feed since Phase 3c) — no per-symbol download
  needed. Equities without 5m data are dropped from live intraday. **FUTURES** (24h, no 5m — IEX carries none) keep
  the yfinance-15m path in a futures-only `FRAMES15`: **⬇ Intraday** downloads 60d of RTH 15m for the 15 futures
  only (`data/bars_15m/`, debounced `INTRADAY_REFRESH_COOLDOWN_MIN`; auto-refresh honors `INTRADAY_AUTO_REFRESH`).
  Snapshot only for futures 15m; equities replay from the deep 5m store.
- **Daily-resampled TFs** (2D–6M) honor the as-of date.
- **Intraday history context:** an intraday chart overlays a faint **daily-close line** (last
  `CHART_DAILY_CONTEXT_BARS` daily bars) left of the 60-day candles — zoom out to see where price came from.

## 8. Multi-timeframe awareness (`mtf.py`) + futures

A per-ticker, per-timeframe **situational-awareness** state (foundation for future scores/setups).
`mtf.tf_state(frame)` distills one TF into: price above/below 10EMA/20EMA/50SMA/150SMA/200SMA, MA
**stack** (−4..+4, +4 = perfect bull order), **tilt** (how many MAs rising), **ATR extension** from the
20/50/200 (× ATR), the **rider** direction+streak+touch (`emarider.current_streak`), and the **trend**
direction (`regime.current_regime`). `profile()` runs it over `config.MTF_TFS` (15m,1h,4h,1D,1W,1M).

- **On demand:** select a ticker → the **MTF grid** under the chart (`/api/mtf?symbol=&date=`) shows the
  state per TF, color-coded — e.g. *15m/1h riding down, 4h/1D/1W/1M riding up*.
- **Precompute:** `python build_mtf.py` → `data/mtf.parquet` (active∪futures × 6 TFs) — the queryable
  matrix; join to `labels` for sector/theme aggregates (the future market-awareness scores). Re-run after
  a data update / intraday download.
- **Futures** (`universe.FUTURES`: NQ/ES/RTY/YM/GC/SI/HG/PL/ZW/ZC/ZS/ZL/CL/NG/BTC `=F`) are kept in the
  universe (always-active) for macro context. They're 24h — intraday skips the RTH filter and uses
  continuous N-bar grouping. No sector/theme/market-cap.
- **Deeper history:** `HISTORY_YEARS=10`; `python backfill_history.py [--years N]` re-downloads the active
  universe + futures at depth (one-time) so 1M/3M/6M TFs have enough bars.

## 8b. Market situational awareness (`market.py`) — top-down long / out / short

A **market-wide** score complementing the per-ticker MTF grid. `market.market_score(as_of, frames)`
combines five weighted blocks — each returning a score in [−1,+1] and a plain-English **why** — into a
composite in [−100,+100] → a **5-band stance** (`MARKET_BANDS`): Strong Long · Cautious Long · Neutral
(Stay Out) · Cautious Short · Short.
- **Trend** (`MARKET_WEIGHTS.trend`): SPY/QQQ/IWM posture (daily + weekly) via `mtf.tf_state` (MA stack,
  tilt, above-MAs, clarity); ATR extension from the 50-MA (`MARKET_EXT_CAP`) trims conviction ("don't chase").
- **Breadth** (`.breadth`): one pass over the active universe — % above 50/200-SMA, advancers:decliners,
  new 52wk highs vs lows; plus a **NYSE vs NASDAQ advancer split** (via `universe.exchange_map()`) so a
  tech-listed drag shows up ("NYSE 62% vs NASDAQ 51% up").
- **Vol** (`.vol`): VIX/VXN level (`MARKET_VIX_LOW/HIGH`) + trend (calm & falling = risk-on); a wide
  **VXN−VIX spread** (≥6) is surfaced as a tech-specific risk term ("VXN 27 (+11 vs VIX — tech risk)").
- **Macro** (`.macro`): 10-yr yield (`^TNX`), long bonds (`TLT`), the dollar (`DX-Y.NYB`) trend —
  **continuous** (ATR-scaled `tanh`, not a hard sign, so a marginal EMA cross nudges); **damped** on a
  live/partial bar (`MARKET_PARTIAL_DAMP`).
- **Risk** (`.risk`, `market._risk_block` via `ratios.py`): a **risk-on/off rotation** basket
  (`config.RATIO_BASKET` — RTY/NQ, NQ/ES, SMH/SPY, XLY/XLP, HG/GC, KRE/SPY, …, index legs on futures). Each
  A/B ratio is built as a synthetic OHLC frame and scored by its `mtf.tf_state` posture × a risk-on sign;
  the mean is the block and the **why** names the leading risk-on / risk-off contributors — the
  index-level rotation the single-blob breadth read used to miss.
- **Data:** `config.MACRO_SYMBOLS` (`^VIX ^VXN DX-Y.NYB ^TNX TLT`) are fetched into `data/bars/` by
  `datastore.update_macro()` (hooked into the daily update). SPY/QQQ/IWM + sector ETFs are already curated.

**Dual close + LIVE intraday read (`intraday.py`).** `market.market_read` returns BOTH a settled **close**
score (daily bars — never lost) and, when the 15m cache is fresh for today, a **live** score (trend from
SPY/QQQ/IWM **intraday** 1h/4h posture; breadth from real **market internals**) plus the internals
themselves. `intraday.intraday_breadth` computes universe-wide **A/D, TRIN, net TICK, % up, % above VWAP,
new intraday highs/lows**, and a **cumulative A-D (TICK) line** (`cum_ad` + `cum_trend`) whose last-hour
slope tells if breadth is BUILDING or FADING — catching the "snapshot tick fine but cumulative rolling
over" divergence — and feeds the live score. All from the 15m bars (the internals a paid feed sells,
self-computed). Served as `{close, live, internals}`; `live=null` when 15m is stale/closed.
- **15m refresh:** the ⬇ Intraday button, or **auto-refresh** during US market hours (`INTRADAY_AUTO_REFRESH`
  + `INTRADAY_REFRESH_EVERY_MIN`, in serve's `_scheduler`). Fresh 15m rebuilds the RAM `FRAMES15` store and
  the live read.
- **UI:** always-visible top **banner** (`#marketbar`) — `CLOSE <stance> · ● LIVE <stance> · A/D · TRIN ·
  TICK · %>VWAP · H/L`, click to expand close + live blocks + internals. `/api/market?date=` (cached per
  as-of, invalidated on frame/15m reload). Weights/thresholds live-tune under ⚙ **market** / **intraday**.
- **Sector/theme leaderboard (`market.group_scores`):** scores every sector & theme by member breadth
  (%up, %>50/200-SMA) + the group ETF's posture (themes, via `build_labels.THEME_ETFS`) in **one pass over
  the universe** (O(universe), reuses the RAM frames; ~350 ms, cached). Also a structural **relative
  strength** per group — mean member `ret_1m`/`ret_3m` minus SPY's (`rs`, `rs_3m`) + an **`rs_trend`**
  (accel/fade: is the 1-month RS pace outrunning the 3-month) — surfaced with a ▲/▼ + hover in the
  leaderboard. Intraday when 15m is fresh, else daily. `/api/groups?date=` → ranked list; shown in the
  banner's expand ("which sectors/themes are in gear"). `GROUP_MIN_MEMBERS` floor.
- **Index S/R awareness:** the index posture why (`market._posture`) appends the nearest support/resistance
  in ATRs + strength (`sr.sr_levels`) — "SPY … 0.4 ATR under ×7 R · 0.7 over ×9 S".
- **Intraday replay slider ("time machine"):** a time slider in the banner scrubs the session (26 fifteen-min
  stops + a **LIVE** snap). Setting `t=HH:MM` recomputes the *whole* live picture — score, internals (incl.
  cum-A/D), leaderboard, Playbook, risk/rotation, and Perf-drawer since-open — **as of that moment**, by
  truncating the RAM 15m store (`serve._session_frames15` — a boolean mask, no disk) so each frame's latest
  session becomes the selected day up to `t`. Works on the current day (scrub back, snap to LIVE) and any past
  day within the ~60-session 15m window (older = CLOSE-only; slider disabled). The enabling change: `market_read`'s
  live gate is `latest_session(frames15) == as_of` (was `== today`), which also fixed a prior bug where a past
  date showed *today's* live tape. During any intraday/replay read the close anchor is the **prior** session
  (point-in-time; no look-ahead). Endpoints take `t`; results cache per `(date, t[, labels.sig])`.
- **Natural-language Playbook (`narrative.playbook`, v3):** the top of the banner's expand — a stance +
  rotation **summary**, then **unified ranked** long/short candidate lists. A **quality-gated,
  volatility-normalized conviction ranker** over whole-universe candidates (group leaders/laggards **+ the
  scanner's setup hits** via `_scan_cached`):
  1. **Hard tradeability gate** — `PLAYBOOK_MIN_PRICE` ($10), `_MIN_DVOL_M` ($50M), `_MIN_MCAP_B` ($2B; a
     `None` mcap passes on price+$vol so cache-missing large-caps aren't dropped). Fixes the "illiquid feel".
  2. **ADR-normalized momentum** — ranks on move ÷ ADR (not raw %), so a mega-cap's meaningful move competes
     with a small-cap's noise (raw-% momentum used to surface the smallest/most-volatile names).
  3. **Setup → side by type + state** (`PLAYBOOK_SETUP_WEIGHTS`; riders + trend weigh most): QM/HTF long-only;
     `uptrend`→long / `downtrend`→short; **bull rider `riding`→long, `touching`/`reversal`→short (breakdown)**;
     bear rider mirror (`reversal`→long); backburner→long (or fade a high-flier); flat_base de-weighted.
  4. **Rewards** liquidity/size (`W_LIQ·log $-vol`) + scan `quality`; a **≤`PLAYBOOK_MAX_PER_SECTOR` per-sector
     cap** keeps the list diversified; a light **regime tilt** by market stance.
  Candidates need no live setup — strong names are tagged **developing**. Two modes fall out of the data:
  **LIVE/replay** (intraday since-open + as-of scan) and **CLOSE/EOD** (settled daily + scan — next-day
  planning, no future data; `?mode=close`). Point-in-time throughout. Hand-set setup weights now → the
  follow-through backtest can make them data-driven later. `/api/playbook?date=&t=&mode=` cached per
  `(date, t, mode, labels.sig)`.
- **Planned:** per-timeframe S/R in the MTF grid; ratio charts as a chartable view.

## 8c. Situational awareness v2 — the Market Map (`sequence.py` / `awareness.py` / `calls.py`)

The composite score proved to have ~no next-day predictive value (179-day point-in-time study), and its
live path failed obvious days (2026-06-29: SPY +1.65% gap-and-go, old read "Neutral · Stay Out"). v2
rebuilds awareness around **relationships and sequences** — the map a discretionary trader actually reads.
The composite survives only as a small descriptive chip; the **banner headline is now the top ranked call**.

- **`sequence.py` — the sequence grammar.** `sequence_read(frame)` classifies ANY enriched frame (index,
  sector ETF, theme basket, ratio; daily or 1h) into a regime with the events that led to it, an
  invalidation level and one human phrase: `breakout/breakdown_fresh`, `impulse_up/down`, `trend_up/down`,
  `pullback_in_uptrend`, `bounce_in_downtrend`, `distribution`, `range`, `unclear`. Measurements anchor on
  the **major swing** (extreme trendlab pivot within `AWARE_MAJOR_LOOKBACK`), not the last micro-pivot;
  depth/retrace in ATRs (`atr_ref` = the daily ATR for 1h frames so budgets compare across TFs); flags
  `extended/late/at_support/at_resistance`; retest hold/fail events against `sr` bands; on intraday frames
  a session **gap beyond the prior session's extremes is itself a structural break**. Tuned until the label
  timeline matched the chart (Dec-23 breakout from a 38-bar base; Feb-12 "distribution risk, fresh lower
  low" before the March slide; Jun-12 "distribution risk, widening swings"; Jun "17-bar consolidation
  716.6–758.4"). All thresholds = `AWARE_*` knobs (⚙ **awareness**, `_NO_SCAN`).
- **`awareness.py` — instrument views + rotation.** Per instrument (4 indices, `AWARE_SECTOR_ETFS`,
  top-`AWARE_TOP_THEMES` themes ETF-else-basket, ~21 `AWARE_RATIOS` incl. index×index pairs): a settled
  **daily view** (posture + sequence + S/R ladder, cached per anchor) + a point-in-time **intraday overlay**
  (1h posture+sequence, **overnight gap** — pct, daily-ATRs, holding/faded/filled (needs ≥2 completed 15m
  bars; at 09:45 the gap is known but status `na`) — today's tape, session levels) + a descriptive **bias**
  badge. **Rotation map:** one returns-panel pass over the labeled universe → per group (11 sectors, 41
  themes, 145 industries) trailing `rs_1m` vs **today's rotation z-score** (`rot_z` vs the group's own
  63-session group-vs-SPY distribution, std-floored) → quadrants `leading_extending / leading_fading /
  lagging_improving / lagging_breaking`, plus member confirmation (**acting well/poorly**, `narrow` when the
  ETF is up but members aren't) and internals from the same pass (`intraday.aggregate_internals`).
- **`calls.py` — the rule table.** Explicit, config-thresholded rules over the payload emit ranked
  plain-language calls (scope market/index/group/ratio, side long/short/avoid, phrase + levels +
  what-next): `gap_go`, `gap_into_extension`, `gap_fade_reversal`, `gap_down_into_support`,
  `breakout/breakdown_follow` (close mode), `coil_at_highs`, `ratio_break` / `ratio_trend_fresh`,
  `group_leader_go` / `group_fade_warn` / `rotation_entry` / `group_break_short`, `risk_off_guard`.
  Each call displays its rule's **measured** stats from `data/rule_stats.json` (hit rate vs base rate, n,
  avg) — rules without ≥30 observations render **unvalidated** and rank low (Wilson-LB edge weighting);
  `AWARE_RULES_OFF` disables rules the validator kills. No unproven claims: every displayed edge is measured.
- **`pit.py` — shared point-in-time plumbing.** serve's 15m truncation lives here (bar labeled 09:30 closes
  09:45; strictly-completed bars only) + offline loaders + `assert_pit` guard, so the offline validator and
  the server replay share one implementation. Intraday reads anchor ALL daily data to the **prior settled
  session**.
- **UI:** 🧭 **Map** drawer — SUGGESTIONS (ranked calls + stats chips) · index cards (bias badge, gap chip,
  sequence phrase, S/R ladder; click → 1h chart) · rotation **Quadrant** (x=rs_1m, y=rot_z; solid=sector,
  hollow=theme; color=acting) / **Tiles** toggle (incl. industries) · ratio strip · sector/theme structure
  cards. Fully slider-aware (replay any date+time); close-mode = EOD planning view. `/api/awareness?date=&t=
  [&mode=close]`, cached per `(date, t, mode, aware_sig, labels.sig)` with a per-anchor daily-view LRU
  (scrubs recompute only overlays; warm read ~1–2s, no dependency on the 32s scan).
- **`validate_awareness.py` — the honesty loop.** Offline replay (no Flask, no scan): Tier-1 close-mode
  rules over years of daily history; Tier-2 intraday rules over the 60-day 15m window × several times of
  day. Logs every fired call point-in-time, joins forward outcomes (rest-of-day / 1d / 3d, absolute +
  vs-SPY), writes per-rule stats → `data/rule_stats.json` (feeds the UI chips); `--check-pit` audits
  look-ahead; `--kill-report` lists rules whose Wilson lower bound ≤ base rate.

## 9. Subsystems (brief)

- **◎ Fires (chart overlay of past setup triggers):** the ◎ Fires button on the chart toolbar opens a
  checklist of every setup that has a registry (`data/setup_registry/*.parquet`); checked setups get
  markers for ALL their past triggers on the charted symbol (one color per setup, shorts above-bar,
  EP marks show the subtype from the registry state `breakout:ep9m` etc.). All/none links; selection
  persists in browser localStorage and re-applies on every chart draw (daily TFs only — registries are
  daily). Server: `/api/fires?sym=` — per-symbol fire lists for every registry, mtime-cached
  (`_FIRES_CACHE`); unvalidated registries are listed but tagged. The EP DAY itself is also marked on
  any live/history EP chart by the detector (`marks` on the hit: "EP classic / EP 9M / EP big move").
- **Time-travel (history → whole-app as-of):** clicking a past fire date in ☰ History sets the app
  date (`#asof`) to that fire — scan tables, market banner, awareness/map, groups, playbook and the
  perf drawer all render as-of (they already took `?date=`). The chart request is dispatched FIRST and
  the rescan only after it lands (a cold rescan holds `_SCAN_LOCK` ~1min and would starve the chart).
  Amber **⏪ back to live** pill by the date picker returns to `window.liveDate` (updated after data
  updates). Cold dates compute once (~90s) into `scan_cache/{date}/` on disk and revisit in ~5s;
  dates before coverage return "no data for DATE — daily history starts ..." (`api_scan` floor
  check against SPY's first bar). Concurrent date loads are sequence-guarded (last click wins).
- **Extension Health Badge:** every place a ticker appears shows ● (current TF) and ◆ (worst TF)
  colored by ATR-multiple distance from the 50-MA, measured on the GOGGLES-bias side only (bull:
  stretch above the mean is chase-risk, below is a pullback; bear mirrored). Tiers per family
  (`EXT_BADGE_TIERS`): stock <4× healthy / 4–10× caution / >10× extreme; ETF/index/theme/ratio
  2.5 / 6. Server: `_ext_tfs` (per-TF ext, cached per sym/date; intraday TFs live-snapshot only),
  `/api/ext` batch back-fill for the map + history lists; scan rows / ticker card / group members /
  ratio chips embed it directly. Designed as the first of a family of health badges (hover = detail).
- **Regime / trendlab:** causal ATR-fraction swing tracker + structure FSM; several setups gate on a real
  prior uptrend. `ATR_FRACTION` tunes swing sensitivity (not in ⚙ — it keys the per-frame trendlab cache).
  Segment metrics (r2/clarity/ann/pole) are computed **lazily** (`Segment.ensure()`) — a scan reads only a
  couple of recent segments per ticker, so the ~450 per-ticker log-fits are deferred; the log-linear fit is
  the closed-form degree-1 regression (not `np.polyfit`/`lstsq`).
- **Quality / patterns:** weighted 0–100 score (`Q_*`); consolidation shape via trendline-slope fit.
- **Support/Resistance (`sr.py`):** reuses trendlab's causal ATR-fraction pivots (framecached), turns each
  into an OHLC zone, then **clusters** pivots whose extremes sit within `SR_CLUSTER_ATR × atr14` into one
  level (cluster span capped at the tol from the first key so a trend's pivot ladder doesn't chain into a
  giant band). **Strength** = member pivots + distinct later re-tests of the band (capped 25). Classified
  **live** by the latest close (below = support, above = resistance; inside-band judged by the level's
  center). Surfaces the nearest `SR_LIVE_COUNT` each side with `dist_atr`/`dist_pct` — the hooks the later
  index/sector/ratio awareness will consume. Drawn on the chart as dashed green/red strength-weighted
  `createPriceLine`s (`S×n`/`R×n`), toggled by the **⌇ S/R** button; computed per-TF and as-of-safe in
  `report._chart_payload` (no look-ahead in backtest charts).
- **Caching (serve.py):** per-setup `SETUP_CACHE` + shared `BASE_CACHE` (regime) → tuning one setup
  recomputes only it; `RIDER_CACHE` for per-TF riders; `BASE_CACHE`/labels keep label edits free (scan-table
  sector/themes are attached at response time, not baked in). `_MARKET_CACHE` (per date) and `_GROUPS_CACHE`
  (per **date + `labels.sig()`**) hold the market/leaderboard reads — the group scores depend on theme
  membership, so a theme edit changes `labels.sig` and auto-recomputes them (also cleared on
  `/api/reload_labels` and on frame/15m reload). All these caches are RAM-only (rebuilt on restart).
- **Performance / RAM:** frames are **float32** (`add_indicators`/`load_bars_15m`) — ~½ the RAM (a global
  numpy JSON provider in `serve.py` serializes the resulting `np.float32` for every endpoint). Per-frame
  trendlab/ema-rider results are memoized in **`framecache.py`** keyed by frame identity (NOT `df.attrs`,
  which pandas deep-copies on every op). The chart-side leak is fixed by disposing the lightweight-charts
  instance on each redraw; the `CHARTS` payload cache is bounded.
- **Journal (`journal.py`):** SQLite study log — the **+** by any ticker saves {symbol, date, setups (each
  with its own appeared-date, captured by clicking a chart bar), comments}; the **📓 Journal** drawer lists
  them by setup and jumps the chart to the saved date. `/api/journal` GET/POST/PUT/DELETE.
- **Perf drawer (`performance.py`):** equal-weighted, winsorized sector/theme/universe/futures performance
  across open/prev-close/1W/1M/3M, breadth %, ETF-backed themes use the ETF's own series. `/api/performance*`.

---

## 10. Roadmap & future plans

Planned work, recorded so day-to-day changes are made with the end state in mind. Nothing here is built
yet unless it also appears in a section above.

### Guiding principle — cache/persist correctly (store raw facts, version derived opinions)
Every cache key must include a **signature of every input** the result depends on, so a changed input
auto-invalidates only what it touched (as `settings.setup_sig` / `_regime_sig` / `labels.sig` already do).
Raw facts (prices, volumes, *what* triggered on *what* date, realized outcomes) never change; **derived**
classifications (theme scores, quality, regime, S/R strength) change when we retune. So anything we persist
for later (esp. the ML dataset below) must either stay **recomputable from raw facts** or be stamped with the
**version of the definitions** that produced it (`labels_version` / `config_sig`) so stale rows are
detectable and rebuildable. In-RAM caches (`_GROUPS_CACHE`, `_MARKET_CACHE`, `SETUP_CACHE`, …) are ephemeral
and rebuild on restart; only *persisted* derived data carries staleness risk.

### S/R levels — consume the engine (`sr.py` already runs on any frame)
- **Per-ticker timing/quality:** feed `nearest_support`/`nearest_resistance` (ATRs) into the quality score
  / as filters + columns; show reward:risk (distance to next resistance ÷ to next support); stops at nearest
  support, targets at nearest resistances.
- **Breakout conviction by level strength:** weight a QM/HTF breakout by the strength of the resistance it
  cleared (a ×25-tested level ≫ a ×2). The trigger *is* a resistance — annotate it.
- **Index & sector S/R → market awareness:** run `sr_levels` on SPY/QQQ + sector ETFs; "index 0.3 ATR under a
  ×15 resistance" = overhead supply (cap upside), "reclaimed a ×20 support" = dip zone. New term in
  `market._score_state` / group scores.
- **Confluence** (level aligns with a rising MA / index level / round number) scored higher; **"approaching a
  level"** watchlist (within X ATR of a high-strength level = bounce/breakout imminent).

### Ratio charts (build a synthetic A÷B frame, reuse trendlab + sr + setups)
- **Leadership rotation:** rank sectors by `ETF÷SPY` trend *structure* (upgrades the 1-day-breadth sector
  leaderboard to structural relative strength).
- **Risk-on/off block for the market score:** a basket — XLY/XLP, HYG/LQD, SMH/SPY, high-beta/low-vol,
  copper/gold — in aggregate uptrend = risk-on confirmation. New block in `market.py` beside
  trend/breadth/vol/macro.
- **Ratio S/R = rotation pivots** (`XLK÷SPY` breaking resistance = tech starting to lead); **relative-strength
  setups** = run the breakout detectors on `TICKER÷SPY` / `TICKER÷sector`.

### Setup improvements & new setups
Audit each existing setup against a reference definition (as done for QM/HTF), then add new ones. The
follow-through study below is the tool to decide which setups and parameters actually pay. (To be scoped.)

### Breakout follow-through study → structural-awareness component
"Of breakouts in the last week/month, how many followed through vs failed?" — efficient because
`_breakout_stack` already walks the whole consolidation and records breakout bars; extend it to **emit the
list of breakout dates**, then measure the forward outcome from the same in-RAM frame (cheap vectorized
slice). A `followthrough.py` walks each ticker **once** (O(universe × lookback), seconds–minute; no repeated
full scans), classifies success/fail (e.g. advanced ≥N·ATR before closing back under the trigger), and
aggregates by setup / level-strength / **market regime** / sector. "QM breakouts this month: 62%; ×10+
strength: 71%; when stance was Stay-Out: 38%" — the regime cut is the new awareness signal.

### Testing/eval harness + (later) ML — three layers, do NOT skip to layer 3
- **Layer 1 — point-in-time eval dataset (no ML):** for each historical day record the market score+stance,
  every setup's hits **with features** (quality, ADR, RS, pole efficiency, level strength, S/R distance,
  regime, sector RS) and the **realized forward outcomes** → versioned parquet. The one non-negotiable is
  **no look-ahead leakage** (we already enforce this: as-of scans, settled anchors, S/R as-of truncation).
- **Layer 2 — conditional base rates & calibration (still no ML):** "stance = Strong Long → next-day SPY up
  X%"; reliability diagrams. Interpretable, robust on limited data, delivers most of the value (the edge is
  mostly *filtering out low-base-rate conditions*).
- **Layer 3 — ML only if it beats layer 2:** gradient-boosted trees (LightGBM/XGBoost), **not** deep
  learning, predicting `P(follow-through)` / `P(next-day up)`, shown as a probability per setup. Mandatory:
  **walk-forward / purged time-series CV** (never random shuffles — outcomes overlap in time), OOS
  **calibration** reported, and a real baseline (must beat the base rate *and* our quality score). Watch
  regime non-stationarity (recent-weighted / rolling retrain) and survivorship bias (include delisted names).
- **Runner:** a **read-only research/eval agent** that never touches the live scanner — reads frames, runs
  the harness, writes datasets/reports, with fixed seeds + versioned dataset/config signatures for
  reproducibility. Realistic expectations: layers 1–2 high-value; a follow-through probability is a useful
  ranking/sizing aid, not a crystal ball; next-day index direction is genuinely hard (small, regime-driven
  edges). The classifications we already compute *are* the feature store this trains on.

---

## Config reference
<!--CONFIG_TABLE-->
| `BARS_DIR_W` | `DATA_DIR / "bars_w"` | weekly bars, resampled from daily (W-FRI) |
| `BARS_DIR_15M` | `DATA_DIR / "bars_15m"` | 60-day 15m intraday bars; higher intraday TFs built from these |
| `INTRADAY_REFRESH_COOLDOWN_MIN` | `30` | min minutes between intraday re-downloads of the same universe |
| `SETTINGS_FILE` | `DATA_DIR / "settings.json"` | live-tuned overrides (settings.py) |
| `HISTORY_YEARS` | `10` | initial download depth (deep enough for 1M/3M/6M timeframes) |
| `BATCH_SIZE` | `100` | tickers per yfinance request |
| `MIN_BARS` | `60` | ignore tickers with less history than this |

**pivots / swings (trendlab Pine "SwingsTimes" port — single pivot source)**

| Variable | Default | Description |
|---|---|---|
| `ATR_FRACTION` | `0.2` | swing reversal delta = ATR(14) * this. 0.2 = Pine's fine zigzag |
| `PIVOT_K` | `3` | (unused) old centered-fractal strength; superseded by ATR_FRACTION |

**1. gapper (tuned toward Qullamaggie episodic pivots — a big surprise gap on heavy volume)**

| Variable | Default | Description |
|---|---|---|
| `GAP_MIN_PCT` | `6.0` | open vs prior close, % (raise to 8-10 for only the biggest EPs) |
| `GAP_MIN_RVOL` | `3.0` | volume vs 50d avg |

**2. high volume close**

| Variable | Default | Description |
|---|---|---|
| `HVC_MIN_VOL_MULT` | `3.0` | volume vs 50d avg |
| `HVC_CLOSE_RANGE_POS` | `0.75` | close in top 25% of day range |
| `HVC_REQUIRE_UPTREND` | `True` | gate on a real prior uptrend (regime.current_regime strong_uptrend) |

**3. flat base — flat-topped consolidation; a horizontal resistance tested >=2x**

| Variable | Default | Description |
|---|---|---|
| `FB_MIN_BARS` | `15` | min base length (bars, ~3 weeks) |
| `FB_MAX_BARS` | `130` | max base length (bars, ~6 months) |
| `FB_TOP_TOL` | `4.0` | swing highs within this % count as the same (flat) resistance |
| `FB_MIN_TOUCHES` | `3` | resistance must be tested at least this many separate bars |
| `FB_MAX_DEPTH_PCT` | `12.0` | (top - base low) / top, % — tight base (elite cut: shallow only) |
| `FB_MAX_DIST_FROM_HIGH` | `12.0` | close must be within this % BELOW the top (coiling / actionable) |
| `FB_BREAKOUT_TOL` | `3.0` | close may be up to this % ABOVE the top and still flag (breakout day) |
| `FB_MAX_DRIFT` | `12.0` | \|close now vs close at base start\| — base must be sideways (flat), not trending |
| `FB_NEAR_HIGH` | `15.0` | base top must be within this % of the 52-week high (a high base, not mid-range) |
| `FB_MIN_TOUCH_SPAN` | `0.5` | resistance touches must span >= this fraction of the base (a held, flat top) |
| `FB_PRIOR_MODE` | `"advance"` | "advance"=require a real prior advance (leadership); "loose"=advance |
| `FB_PRIOR_LOOKBACK` | `130` | bars before the base used to measure the prior advance |
| `FB_PRIOR_GAIN_PCT` | `50.0` | prior advance: a strong rise (any MA) over FB_PRIOR_LOOKBACK qualifies (elite leadership) |
| `FB_REQUIRE_STAGE2` | `True` | require a Stage-2 uptrend: close > 50-SMA > 200-SMA, 200-SMA rising |
| `FB_VOL_CONTRACTION` | `1.0` | volume dry-up gate: base recent-half avg vol < this x earlier-half |

**4. high tight flag** (Traderlion/Deepvue)

Explosive **pole** (`HTF_MIN_GAIN_PCT`≈90–120%+ in `HTF_POLE_MAX_BARS`≈4–8wk) → a shallow **tight flag**
that pulls back ≤`HTF_MAX_PULLBACK_PCT` off the pole high (no specific shape) → **breakout** on volume.
The breakout LEVEL is the top of the shared **breakout-level stack** (`setups._breakout_stack`; see the QM
section): the pole high plus each intermediate resistance (swing-pivot highs + long-downswing highs), so
breaking each successive level on volume is a distinct, early breakout. **State machine:** `too_short`
(base below the optimal window), `optimal` (in the `HTF_OPTIMAL_MIN_BARS`..`HTF_FLAG_MAX_BARS` 3–5 week
window — green row background in the dashboard), and `breakout` — a volume-confirmed close popping the
stack top, allowed at **any length ≥`HTF_FLAG_MIN_BARS` (3)**. `breakouts`/`levels` count the pops so far
and the resistances remaining. The pole is **not** gated on trendlab by default (the official definition doesn't require
it); instead pole "cleanliness" is surfaced as `pole_efficiency` = net move ÷ path length (1.0 = perfectly
straight) — evidence-first, gated only if `HTF_MIN_EFFICIENCY>0`. The strength criteria (MA-stack, 52w-high
proximity, ADR) are computed + surfaced; each is gated by its own toggle so you can start loose and tighten
from results. RS leadership is the `rs_rank` dashboard filter (recommend ≥95), not gated in the detector.
Evidence columns: `pole_gain_pct, pole_bars, pole_efficiency, pullback_pct, flag_bars, adr_pct,
off_hi52_pct, ma_stack, vol_dryup, contraction, vol_x, breakouts, levels, pattern`. Chart shades pole/flag
+ draws the stack resistance levels (horizontal; top = active trigger), shared geometry with QM.

| Variable | Default | Description |
|---|---|---|
| `HTF_LOOKBACK` | `45` | bars to find the flag's reference high (pole high) |
| `HTF_MIN_GAIN_PCT` | `90.0` | pole gain (pole high / launch low − 1), % |
| `HTF_POLE_MAX_BARS` | `40` | pole must complete within ~8 weeks |
| `HTF_MAX_PULLBACK_PCT` | `25.0` | shallow flag: max pullback off the pole **high** (no shape req) |
| `HTF_FLAG_MIN_BARS` | `3` | min bars to register a base / allow a breakout (below = ignored) |
| `HTF_OPTIMAL_MIN_BARS` | `15` | start of the optimal 3–5wk window; below → `too_short`, in → `optimal` |
| `HTF_FLAG_MAX_BARS` | `25` | end of the optimal window (~5wk); above → dropped |
| `HTF_NEAR_HIGH` | `15.0` | gate: within this % of the 52-week high (0 = off) |
| `HTF_MIN_ADR` | `0.0` | gate: pole-period ADR% floor (evidence-first; set ~4 to enable) |
| `HTF_MIN_EFFICIENCY` | `0.0` | gate: pole directional-efficiency floor 0..1 (evidence-first; 0 = off) |
| `HTF_REQUIRE_MA_STACK` | `True` | gate: close > rising 50-SMA > rising 200-SMA |
| `HTF_RISING_MA_BARS` | `20` | window over which the 50/200-SMA must be rising |
| `HTF_BREAKOUT_RVOL` | `1.4` | breakout-bar volume vs 50d avg (~+40%) to confirm the `breakout` state |
| `HTF_REQUIRE_REGIME` | `False` | OFF by default: don't gate the pole on trendlab |
| `PATTERN_SLOPE_TOL` | `0.04` | \|slope\| (ATR/bar) below this = "flat"; classifies flag/pennant/triangle |

**5. higher low approaching MA**

| Variable | Default | Description |
|---|---|---|
| `HL_MA` | `20` | EMA the pullback should approach (Qullamaggie rides the 20-EMA) |
| `HL_TREND_SMA` | `50` | must be above & rising |
| `HL_ATR_DIST` | `1.0` | low within this many ATRs of the EMA |

**6. undercut & rally**

| Variable | Default | Description |
|---|---|---|
| `UR_LEVEL_LOOKBACK` | `40` | find prior pivot lows within this window |
| `UR_LEVEL_MIN_AGE` | `5` | pivot must be at least this many bars old |
| `UR_MAX_UNDERCUT_PCT` | `5.0` | don't count crashes through the level |

**7. backburner (multi-timeframe: 1D + 1W) — at/near ATH after a strong uptrend**

| Variable | Default | Description |
|---|---|---|
| `BACKBURNER_MAX_RETRAC` | `25.0` | max retracement from ATH, % of the prior up-move (top quarter) |
| `BACKBURNER_ATH_MAX_AGE` | `30` | ATH must be within this many bars (0 = off) — keeps it "at highs" |

**8. stairstep (multi-timeframe: 1D + 1W) — orderly stepdown after a swing high**

| Variable | Default | Description |
|---|---|---|
| `STAIRSTEP_MIN_BARS` | `6` | consecutive non-higher-high bars required |
| `STAIRSTEP_ATR_FRACTION` | `0.2` | threshold * ATR; a bar stays in the run while its high is |
| `STAIRSTEP_MAX_BREAKOUT_BARS` | `1` | how many of the most recent bars may break out above the |

**regime gate (uses trendlab.segment_chart; thresholds for zigzag live in trendlab.py)**

| Variable | Default | Description |
|---|---|---|
| `REGIME_USE` | `True` | master switch; False = legacy two-point / rising-SMA trend tests |
| `REGIME_PRIOR_GAIN` | `25.0` | prior up-segment must have gained at least this net % |
| `REGIME_PRIOR_CLARITY` | `0.30` | prior/current segment min clarity = (r2 + er) / 2 |

**report**

| Variable | Default | Description |
|---|---|---|
| `CHART_BARS` | `750` | bars embedded per flagged ticker (~3 yr) |
| `CHART_TOP_N` | `150` | embed charts only for the top-N hits per setup by $vol |
| `CHART_FWD_BARS` | `60` | outcome bars shown AFTER the as-of date in backtest mode |
| `CHART_ZOOM_BARS` | `150` | default visible window when a chart opens (recent N bars) |
| `CHART_ZOOM_FWD` | `8` | show this many bars PAST the as-of marker (small right margin) |
| `CHART_DAILY_CONTEXT_BARS` | `750` | daily-close history line drawn left of an intraday chart (where we came from) |
| `DISCORD_WEBHOOK` | `""` | optional: paste a webhook URL to get the daily list pushed |

**active universe (liquidity floors; build_active_universe.py writes data/universe_active.csv)**

`build_active_universe.py` keeps tickers passing ALL floors below. It runs in two passes: (1) the
bar-based liquidity floors (no network); (2) `marketcap.ensure()` fetches caps for the liquid
survivors, then the market-cap floor is applied (a ticker with **unknown** cap is kept — only
*known* sub-floor caps are dropped). Curated ETFs + futures are always kept in the universe (for the
Perf page / charts), BUT **illiquid ETFs generate no setup hits** — `scan._rows_from_enriched` gates
any ETF below the share-volume / $-volume floors out of every setup.

| Variable | Default | Description |
|---|---|---|
| `UNIVERSE_MIN_AVG_VOL_K` | `750` | mean 50d share volume floor (thousands) |
| `UNIVERSE_MIN_PRICE` | `5.0` | last close floor ($) |
| `UNIVERSE_MIN_DVOL_M` | `20.0` | mean 20d $-volume floor (millions) = price x avg daily volume |
| `UNIVERSE_MIN_ADR_PCT` | `2.0` | 20d average daily range floor (%) |
| `UNIVERSE_MIN_MCAP_M` | `150.0` | market-cap floor ($M); fetched for the liquid subset; unknown-cap kept |

**multi-timeframe awareness (mtf.py) — per-ticker state across timeframes**

| Variable | Default | Description |
|---|---|---|
| `MTF_TFS` | `["15m", "1h", "4h", "1D", "1W…` | the curated set the MTF profile/panel computes |
| `MTF_SLOPE_LOOKBACK` | `5` | bars back to measure each MA's slope (rising/falling tilt) |

**9. delayed high volume close — gap up in the last few days, consolidate, then an HVC**

| Variable | Default | Description |
|---|---|---|
| `DHVC_LOOKBACK` | `10` | search this many recent bars for the ignition gap |
| `DHVC_GAP_MIN_PCT` | `4.0` | ignition bar's gap (open vs prior close), % |
| `DHVC_GAP_MIN_RVOL` | `1.5` | ignition bar's volume vs 50d avg |
| `DHVC_MIN_BASE_BARS` | `2` | consolidation bars required between the gap and today |
| `DHVC_BASE_MAX_DEPTH` | `12.0` | consolidation (high-low)/high must be <= this % |
| `DHVC_GAP_HOLD_TOL` | `3.0` | consolidation low may dip at most this % below the gap day's close |
| `DHVC_REQUIRE_UPTREND` | `False` | the gap+base is the ignition; prior uptrend optional |

**10. QM breakout (Qullamaggie momentum continuation) — RS leader, big move, tightening**

Big move (30%+ over days–weeks, high within ~1–3 months) is the qualifying **context**. The actionable
**pivot is the top of the breakout-level STACK** (`setups._breakout_stack`), NOT a fitted trendline: the
pole high is the first level, and each intermediate resistance — a confirmed **swing-pivot high**
(`trendlab.swing_highs`) or, for a drawn-out ≥`BREAKOUT_STACK_DOWNSWING`-bar downswing with no pivot, the
prior bar's high — is pushed on top (nearest resistance first). A **volume-confirmed** close pops the top
(one bar can pop several); the **trigger** is the current top. So a consolidation yields a *sequence* of
breakouts (`breakouts`/`levels` count them). Through the base price
**surfs the rising 10/20-EMA**: pullbacks allowed down to the **50-MA**, **shakeouts** (an intraday
undercut bought back same-bar or reclaimed the next bar) are *bullish*; a **close below the 50-MA
rejects**, and **two consecutive closes below the 20-EMA** (an unrecovered breakdown) rejects. The
uptrend is gauged by the **50-MA rising** + the EMA stack (10>20). Lower lows allowed early, **higher
lows** into the breakout. **State:** `building` while coiling under the stack top → `breakout` the day a
level is popped **on volume ≥ `QMB_BREAKOUT_VOL`× the recent average**. The chart shades the pole (green)
+ consolidation (amber) and draws the stack levels (horizontal; top = active trigger). RS leadership
(top ~1–2% over 1/3/6-mo raw returns) is the `rs_rank` filter.

| Variable | Default | Description |
|---|---|---|
| `QMB_MOVE_LOOKBACK` | `63` | the move's high must be within ~this many bars (3 months) |
| `QMB_MIN_MOVE` | `30.0` | the move (recent low -> high) gained at least this % |
| `QMB_MOVE_MAX_BARS` | `35` | the move happened over at most this many bars (days-to-weeks) |
| `QMB_CONS_MIN` | `8` | consolidation length (pole high -> now) >= this (bars, ~1.5 weeks) |
| `QMB_CONS_MAX` | `45` | consolidation length <= this (bars, ~2 months) |
| `QMB_MAX_PULLBACK` | `50.0` | max pullback in the consolidation from the move high, % (less is better) |
| `QMB_SURF_BAND` | `3.0` | a close is "riding" the 20-EMA if >= EMA20*(1 - this%/100) |
| `QMB_MIN_RIDE_FRAC` | `0.5` | >= this fraction of consolidation bars riding at/above the 20-EMA |
| `QMB_CONTRACT_WIN` | `10` | recent window (bars) used as the breakout volume baseline |
| `QMB_BREAKOUT_VOL` | `1.3` | the breakout bar's volume must be >= this x the consolidation's avg volume |
| `BREAKOUT_STACK_DOWNSWING` | `8` | a downswing this many bars long (no pivot) pushes the prior bar's high as a synthetic level (shared with HTF) |

**11. EMA Rider (bullish & bearish) — a sustained streak of bars CLOSING above/below an**

| Variable | Default | Description |
|---|---|---|
| `ER_EMA_LEN` | `20` | the EMA the price streak rides |
| `ER_ATR_LEN` | `14` | ATR length for the "near EMA" proximity zone |
| `ER_ATR_FRAC` | `0.5` | an "EMA hold" = wick within ATR*this of the EMA (a wick through it counts) |
| `ER_MIN_STREAK` | `10` | current streak (consecutive closes one side of the EMA) must be >= this |
| `ER_MIN_HOLDS` | `1` | require at least this many EMA holds (touches after the 1st streak bar) |
| `ER_USE_STREAK_THRESH` | `False` | allow a close within the ATR zone past the EMA to still extend the streak |
| `ER_STREAK_ATR_FRAC` | `0.5` | streak tolerance = ATR*this (only used when ER_USE_STREAK_THRESH) |

**setup quality score (quality.py) — weights + min->max ramps per sub-score.**

| Variable | Default | Description |
|---|---|---|
| `Q_ADR_RAMP` | `(2.0, 10.0)` | adr_pct: this% -> 0, that% -> 1 |
| `Q_TREND_GAIN_RAMP` | `(25.0, 150.0)` | prior up-move net % -> 0..1 (x clarity) |
| `Q_TIGHT_RAMP` | `(20.0, 3.0)` | consolidation range %: loose -> 0, tight -> 1 (note: hi<lo = inverse) |
| `Q_EMASURF_BAND` | `2.0` | a close counts as "surfing" if >= EMA*(1 - this%/100) |
| `Q_VOLDRY_RAMP` | `(0.0, 0.5)` | volume reduction fraction across the base -> 0..1 |

**auto daily update (serve.py in-app scheduler) — defaults; live-tuned in settings.json**

| Variable | Default | Description |
|---|---|---|
| `AUTO_UPDATE` | `False` | master toggle for the background daily-update thread |
| `AUTO_UPDATE_TIME` | `"23:30"` | local HH:MM to run update_daily (after US close, Israel time) |

**live server (serve.py)**

| Variable | Default | Description |
|---|---|---|
| `SERVE_UNIVERSE_MIN_DVOL_M` | `0` | 0 = all cached tickers; >0 restricts server universe for speed |
<!--/CONFIG_TABLE-->

## Setup graduation checklist (once a setup is VALIDATED, or its detection logic changes)

Every chart-verified setup must ship with ALL of the following (Amir 2026-07-03):

1. **Pattern anatomy on charts** — the hit dict carries `marks` (component markers, e.g. rims /
   cup low / P1-P2 / LS-HEAD-RS), `segline` (necklines), `xlines` (measured-move target), plus
   the standard `level`/`trigger`. These render on the MAIN app chart for today's scan rows AND
   for history clicks (`/api/setup_hit` re-detects at the fire date to reconstruct the pattern).
2. **History registry rebuilt** — `build_setup_registry.py --setups <name> --force` after ANY
   detector change; registries are precomputed once to data/setup_registry/ and never computed
   at request time. Fast one-pass engines exist for event/state-machine setups; sliding replay
   (with the trigger-day prefilter) for structural templates.
   **Add the setup to the `VALIDATED` whitelist in build_setup_registry.py** — default builds
   refuse unvalidated setups (Amir 2026-07-03); a one-off build during a validation pass needs
   `--setups <name> --unvalidated`.
3. **Breakout-day blindness check** — if the trigger is "price crosses a level built from recent
   bars", the level's window must EXCLUDE the test bar (this bug shipped four times: flat_base,
   high_tight_flag, qm_breakout, cup_handle).
4. **SETUP_DOCS entry** — description + execution instructions in report.SETUP_DOCS ("? Info").
5. **Registered in all 6 integration points** — detector + SETUP_OF, config knobs, settings
   TUNABLE group, report SETUP_LABELS + EXTRA_COLS + the duplicate JS label map, playbook weight.

---

## 11. Episodic Pivot (EP) & Progressive Exposure Playbook

### 11.1 The 10-Year Quantitative Study (2016–2026)
A comprehensive backtest across the full 2,057-stock liquid US universe from July 2016 to October 2026
identified 6,501 raw EP events and 438 Pinnacle Elite trade executions (369 Base EP + 69 Yellow Re-Entries):
- **Base Expectancy:** +2.25 R EV per trade, 34.9% win rate, +983.1 R total strategy return across 10.25 years.
- **Asymmetric Skew:** Average win of +8.16 R vs strictly capped average loss of -0.93 R (8.7:1 skew ratio).
- **Hard Stop:** Day 1 Low Stop strictly invalidates failed moves and cuts early faders.

### 11.2 Progressive Exposure (Dynamic Heat Sizing)
Testing dynamic exposure adjustments based on recent success/failure sequences revealed profound capital efficiency gains:
- **Constant 1.0 R Baseline:** +983.1 R total P&L, -33.92 R maximum drawdown, 4.68 profit factor.
- **Stepped Streak Sizing (Recommended):**
  - Rule: After 2 consecutive losses, throttle heat to 0.5 R. After 3 consecutive losses, throttle to 0.25 R. After 1 win, reset to 1.0 R; after 2 consecutive wins, scale exposure to 1.35 R.
  - Result: **+979.3 R total P&L**, **-15.09 R maximum drawdown (55.5% drawdown reduction)**, **6.50 profit factor**, and a Return-to-Drawdown ratio of 64.89 (up +123% vs baseline).
- **Rolling 10-Trade Win Rate Heat:**
  - Rule: 0.5x heat if trailing 10-trade win rate < 30%; 1.35x heat if trailing win rate >= 45%.
  - Result: **+1,112.8 R total P&L (+130 R profit lift)**, **-21.67 R max drawdown (-36% reduction)**, 5.83 profit factor.

### 11.3 Multi-Timeframe Larsson Line Exits by Stock ADR (1D vs 2D vs 1W)
Empirical testing comparing 1-Day, 2-Day, and 1-Week Larsson Line exits across stock volatility (ADR) groups:
- **Low/Moderate ADR (<4.0%):** 1D Larsson EV = +0.25 R (+39.5 R total) vs 2D (-0.54 R) and 1W (-0.68 R).
- **High ADR (4.0% - 7.0%):** 1D Larsson EV = +0.22 R (+29.3 R total) vs 2D (-0.66 R) and 1W (-0.77 R).
- **Ultra-High ADR (>7.0%):** 1D Larsson EV = +0.73 R (+59.4 R total) vs 2D (-0.21 R) and 1W (-0.59 R).
- **Takeaway:** 1D Larsson line is mathematically superior across all ADR regimes. Multi-day and weekly aggregations lag trend exhaustion tops by 20% to 40%, giving back too much open profit. For volatile high-ADR runners, the optimal technique is **exiting on the 1D Blue Flip and re-entering on the Trade 2 Yellow Flip**.

### 11.4 Strategy PDF Documentation
Full visual handbook with high-resolution charts of OKLO, SEDG, and the progressive equity curve is available in `docs/EP_Pinnacle_Elite_Strategy_Handbook.pdf`.

### 11.5 Intermediate Continuation EMA Ribbon (Trade 2 & 3 Second-Leg Compounder)
To solve the "trend resumption after pullback" challenge without subjective discretion or lookahead bias, an exhaustive 10-year parameter sweep across 18 EMA ribbon combinations was conducted across 6,501 historical EP events (2016–2026).

#### Sweep Results & Winning Configurations:
1. **Ribbon (8, 12, 16, 21) — #1 in Total Net Profit (Primary Default):**
   - **Total Net R:** **+2,917.2 R** across 8,061 simulated trades
   - **Profit Factor:** **1.65**
   - **Expectancy (EV):** **+0.362 R per trade**
   - **Win Rate:** **32.8%** with multi-bagger right-tail compounding
   - **Role:** Fast, responsive intermediate trend alignment that captures early second-leg explosive moves.

2. **Ribbon (10, 15, 20, 30) — #1 in Expectancy:**
   - **Total Net R:** **+2,624.8 R** across 6,626 trades
   - **Profit Factor:** **1.65**
   - **Expectancy (EV):** **+0.396 R per trade**
   - **Win Rate:** **34.1%**
   - **Role:** Smoother trend envelope with fewer whipsaws; available via interactive dropdown in `serve_ep.py` and `/api/chart`.

#### Execution Rules (Strict Zero Lookahead):
- **Trigger:** Bullish ribbon alignment (EMA1 >= EMA2 >= EMA3 >= EMA4), transitioning from Gray or Blue to Yellow (`yellow_flip`).
- **Entry Timing:** Strictly executed at **`next_open`** (Open price of signal bar + 1). No intraday bar-close lookahead.
- **Stop Loss:** Pegged at the **5-day swing low** prior to the signal bar.
- **Realistic Gap-Down Fill:** If the entry bar opens below the stop price, the fill occurs at **`open`** (not theoretical stop price), preventing fictitious slip-free fills.
- **Day 1 Stop Loss Protection:** If the entry bar's low violates the stop price intraday on Day 1, the position is stopped out immediately on Day 1.
- **Exit Condition:** Held continuously as long as the ribbon stays Yellow or Gray (compression), and exits on the first reverse flip to Bearish Blue (`blue_flip`) or stop loss violation.

### 11.6 Dual-Track Continuation Architecture (Post-Trade 1 Failure Recovery)
To capture the large universe of institutional runners that experience an engineered stop-hunt or multi-week base digestion after Day 1, continuation setups are resolved into two distinct execution tracks:

#### Track 1: Institutional Undercut & Reclaim (U&R)
- **Problem Solved:** Catches fast bear-trap shakeouts where Trade 1 is stopped out in the first 1–10 sessions, but institutional accumulation immediately absorbs the float.
- **Controlled Undercut Guard:** The shakeout low must remain within 15% of Day 1 Low (`shakeout_low >= d1_low * 0.85`), automatically rejecting structural crashes.
- **Trigger:** A session closing back **ABOVE Day 1 Low** within 15 sessions of the initial stop-out.
- **Stop Loss:** Pegged at the absolute **shakeout swing low** (delivering tight 3.5%–5.5% risk).
- **Empirical Edge:** Captured **+328.3 R net profit** across 1,169 failed-breach recoveries with a 29.2% win rate and 1.44 Profit Factor, capturing monster multi-baggers like `BE` (+1,123.5%), `NVAX` (+673.1%), and `AMPX` (+320.5%).

#### Track 2: Corrected Dynamic Base Breakout (Intermediate Ribbon 8, 12, 16, 21)
- **Dynamic Anchor Reset:** Solves the historical anchor bug by resetting `cons_low` dynamically whenever a new swing peak is formed. Drawdown depth is strictly measured from the *active* consolidation peak, not an old low from weeks prior.
- **Persistent Yellow Power Continuation:** Eliminates the "must turn blue first" catch-22. If a leader pulls back >= 5% from its peak while the ribbon stays yellow (due to overwhelming institutional demand), re-entry triggers when price breaks above its 5-day high or reclaims 8 EMA.
- **Extended Base Search Window:** Allows up to **65 trading sessions** (13 weeks) of base building as long as price holds above its rising 50 SMA.
- **Empirical Edge:** Captured **396 delayed stage-2 base breakouts** on failed-breach setups (+45.4 R net) and **5,545 continuation trades across the full universe**, while reducing portfolio maximum drawdown by **+37.14 R** (-195.5 R down to -158.3 R).



