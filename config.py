"""Central configuration — every threshold lives here so tuning never means editing logic.

Thresholds in the "live-tunable" blocks below are overridable at runtime from the
settings panel (see settings.py); config.py holds the DEFAULTS. The data/output
directories honor SCAN_DATA_DIR / SCAN_OUTPUT_DIR env vars so an isolated test or
second instance can run without touching the primary cache (used by make_test_data)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("SCAN_DATA_DIR") or ROOT / "data")
BARS_DIR = DATA_DIR / "bars"
BARS_DIR_W = DATA_DIR / "bars_w"    # weekly bars, resampled from daily (W-FRI)
BARS_DIR_15M = DATA_DIR / "bars_15m"  # LEGACY name: now holds 60-day FUTURES 5m bars; higher TFs built from these
INTRADAY_REFRESH_COOLDOWN_MIN = 30   # min minutes between intraday re-downloads of the same universe
OUTPUT_DIR = Path(os.environ.get("SCAN_OUTPUT_DIR") or ROOT / "output")
UNIVERSE_FILE = DATA_DIR / "universe.csv"
SKIPLIST_FILE = DATA_DIR / "skiplist.txt"
SETTINGS_FILE = DATA_DIR / "settings.json"   # live-tuned overrides (settings.py)

HISTORY_YEARS = 10           # initial download depth (deep enough for 1M/3M/6M timeframes)
BATCH_SIZE = 100             # tickers per yfinance request
MIN_BARS = 60                # ignore tickers with less history than this

# ---- liquidity columns (attached to every hit; filter in the dashboard, not here)
ADR_WINDOW = 20
DOLLARVOL_WINDOW = 20
RVOL_WINDOW = 50

# ---- pivots / swings (trendlab Pine "SwingsTimes" port — single pivot source)
ATR_FRACTION = 0.2          # swing reversal delta = ATR(14) * this. 0.2 = Pine's fine zigzag
                            # (more, smaller swings/segments). 0.7 was the old coarser default
                            # tuned for regime gating — raise it back if setups get too choppy.
PIVOT_K = 3                  # (unused) old centered-fractal strength; superseded by ATR_FRACTION

# ---- 1. gapper (tuned toward Qullamaggie episodic pivots — a big surprise gap on heavy volume)
GAP_MIN_PCT = 6.0            # open vs prior close, % (raise to 8-10 for only the biggest EPs)
GAP_MIN_RVOL = 3.0           # volume vs 50d avg
GAP_LEADER_DVOL_M = 250.0    # liquid-leader relaxation (Amir 2026-07-07): a name trading >=$250M/day
GAP_LEADER_RVOL = 2.0        # can't 3x its average, so a hard gap on >=2x still qualifies (like EP leaders)

# ---- 2. high volume close
HVC_MIN_VOL_MULT = 3.0       # volume vs 50d avg
HVC_CLOSE_RANGE_POS = 0.75   # close in top 25% of day range
HVC_MIN_CHANGE_PCT = 2.0
HVC_REQUIRE_UPTREND = True   # gate on a real prior uptrend (regime.current_regime strong_uptrend)

# ---- 3. flat base — flat-topped consolidation; a horizontal resistance tested >=2x
# (pivot-cluster method). Breakout level = the resistance top. Works for continuation
# AND bottoming bases. Base shape and prior-trend are measured on separate windows.
FB_MIN_BARS = 25             # min base length (bars, ~5 weeks — O'Neill: <5-6wk bases unreliable)
FB_MAX_BARS = 130            # max base length (bars, ~6 months)
FB_TOP_TOL = 4.0             # swing highs within this % count as the same (flat) resistance
FB_MIN_TOUCHES = 3           # resistance must be tested at least this many separate bars
FB_MAX_DEPTH_PCT = 12.0      # (top - base low) / top, % — tight base (elite cut: shallow only)
FB_MAX_DIST_FROM_HIGH = 12.0 # close must be within this % BELOW the top (coiling / actionable)
FB_BREAKOUT_TOL = 3.0        # close may be up to this % ABOVE the top and still flag (breakout day)
FB_MAX_DRIFT = 12.0          # |close now vs close at base start| — base must be sideways (flat), not trending
FB_NEAR_HIGH = 15.0          # base top must be within this % of the 52-week high (a high base, not mid-range)
FB_MIN_TOUCH_SPAN = 0.5      # resistance touches must span >= this fraction of the base (a held, flat top)
FB_PRIOR_MODE = "advance"    # "advance"=require a real prior advance (leadership); "loose"=advance
                             # OR above the 200-SMA (broad, incl. bottoming); "off"=no prior gate
FB_PRIOR_LOOKBACK = 130      # bars before the base used to measure the prior advance
FB_PRIOR_GAIN_PCT = 30.0     # prior advance over FB_PRIOR_LOOKBACK (spec baseline: 30%)
FB_REQUIRE_STAGE2 = True     # require a Stage-2 uptrend: close > 50-SMA > 200-SMA, 200-SMA rising
FB_VOL_CONTRACTION = 1.0     # volume dry-up gate: base recent-half avg vol < this x earlier-half
FB_BREAKOUT_RVOL = 1.4       # breakout-day volume vs 50d avg (spec: 40-50% above average);
                             # a close above the top on less = state "breakout_lowvol", not "breakout"

# ---- 4. high tight flag (Traderlion/Deepvue): explosive pole + shallow tight flag + volume breakout
HTF_LOOKBACK = 45            # bars to find the flag's reference high (the pole high)
HTF_MIN_GAIN_PCT = 90.0      # pole gain (pole high / launch low - 1), % — ~90-120%+
HTF_POLE_MAX_BARS = 40       # pole must complete within ~8 weeks
HTF_MAX_PULLBACK_PCT = 25.0  # shallow flag: max pullback off the pole HIGH (official 10-25%); no shape req
# base-length state machine: [FLAG_MIN, OPTIMAL_MIN) = 'too_short'; [OPTIMAL_MIN, FLAG_MAX] = 'optimal'
# (highlighted); a volume breakout = 'breakout'. Bases longer than FLAG_MAX are dropped (no longer tight).
HTF_FLAG_MIN_BARS = 3        # min bars to register a forming base at all (below this = ignored)
HTF_OPTIMAL_MIN_BARS = 15    # start of the optimal 3-5 week window (below = 'too_short')
HTF_FLAG_MAX_BARS = 25       # end of the optimal window ~5 weeks (above = dropped)
HTF_NEAR_HIGH = 15.0         # gate: within this % of the 52-week high (0 = off)
HTF_MIN_ADR = 0.0            # gate: pole-period ADR% floor (evidence-first; set ~4 to enable)
HTF_MIN_EFFICIENCY = 0.0     # gate: pole directional-efficiency floor 0..1 (evidence-first; 0 = off)
HTF_REQUIRE_MA_STACK = True  # gate: close > rising 50-SMA > rising 200-SMA
HTF_RISING_MA_BARS = 20      # window over which the 50/200-SMA must be rising
HTF_BREAKOUT_RVOL = 1.4      # breakout-bar volume vs 50d avg (~+40%) to confirm the 'breakout' state
HTF_REQUIRE_REGIME = False   # OFF by default: don't gate the pole on trendlab (official def doesn't)
PATTERN_SLOPE_TOL = 0.04     # |slope| (ATR/bar) below this = "flat"; classifies flag/pennant/triangle

# ---- 5. higher low approaching MA
HL_MA = 20                   # EMA the pullback should approach (Qullamaggie rides the 20-EMA)
HL_TREND_SMA = 50            # must be above & rising
HL_ATR_DIST = 1.0            # low within this many ATRs of the EMA

# ---- 6. undercut & rally
UR_LEVEL_LOOKBACK = 40       # find prior pivot lows within this window
UR_LEVEL_MIN_AGE = 5         # pivot must be at least this many bars old
UR_MAX_UNDERCUT_PCT = 5.0    # don't count crashes through the level

# ---- 7. backburner (multi-timeframe: 1D + 1W) — at/near ATH after a strong uptrend
BACKBURNER_MAX_RETRAC = 25.0  # max retracement from ATH, % of the prior up-move (top quarter)
BACKBURNER_ATH_MAX_AGE = 30   # ATH must be within this many bars (0 = off) — keeps it "at highs"

# ---- 8. stairstep (multi-timeframe: 1D + 1W) — orderly stepdown after a swing high
STAIRSTEP_MIN_BARS = 10       # consecutive non-higher-high bars required (Amir 2026-07-03:
                              # 6 -> 8 -> 10 after the 8-bar verdict pass; a breakout bar at ANY
                              # point kills the count — the trailing-run walk already does this)
STAIRSTEP_ATR_FRACTION = 0.2  # threshold * ATR; a bar stays in the run while its high is
                              # below the previous high + threshold (no meaningful new high)
STAIRSTEP_MAX_BREAKOUT_BARS = 1  # how many of the most recent bars may break out above the
                                 # staircase (0 = must still be stepping through today)

# ---- regime gate (uses trendlab.segment_chart; thresholds for zigzag live in trendlab.py)
REGIME_USE = True            # master switch; False = legacy two-point / rising-SMA trend tests
REGIME_PRIOR_GAIN = 25.0     # prior up-segment must have gained at least this net %
REGIME_PRIOR_CLARITY = 0.30  # prior/current segment min clarity = (r2 + er) / 2

# ---- report
CHART_BARS = 750             # bars embedded per flagged ticker (~3 yr)
CHART_TOP_N = 150            # embed charts only for the top-N hits per setup by $vol
                             # (full market = thousands of hits -> a chart for every row
                             #  makes a 100MB+ HTML; other rows get a TradingView link)
CHART_FWD_BARS = 60          # outcome bars shown AFTER the as-of date in backtest mode
CHART_ZOOM_BARS = 150        # default visible window when a chart opens (recent N bars)
CHART_ZOOM_FWD = 8           # show this many bars PAST the as-of marker (small right margin)
CHART_DAILY_CONTEXT_BARS = 750  # daily-close history line drawn left of an intraday chart (where we came from)
DISCORD_WEBHOOK = ""         # optional: paste a webhook URL to get the daily list pushed

# ---- active universe (liquidity floors; build_active_universe.py writes data/universe_active.csv,
# the reduced set the app + nightly scan load. Curated ETFs are always kept. Re-run weekly.)
UNIVERSE_MIN_AVG_VOL_K = 750   # mean 50d share volume floor (thousands) — the PRIMARY share floor
UNIVERSE_DVOL_OR_MIN_VOL_K = 100  # OR-branch (Amir 2026-07-06): a high-priced name below the 750K
                               # share floor still qualifies if its $-volume clears UNIVERSE_MIN_DVOL_M
                               # AND its share volume is >= this — catches liquid-by-dollars names like
                               # ARGX ($366M/day, ~355K shares @ $940) while excluding truly thin (<100K) ones.
UNIVERSE_MIN_PRICE = 5.0       # last close floor ($)
UNIVERSE_MIN_DVOL_M = 20.0     # mean 20d $-volume floor (millions) = price x avg daily volume
UNIVERSE_MIN_ADR_PCT = 2.0     # 20d average daily range floor (%)
UNIVERSE_MIN_MCAP_M = 150.0    # market-cap floor ($M); fetched for the liquid subset in build_active_universe
UNIVERSE_EXCLUDE = {"INHD"}    # halted/dead symbols dropped everywhere (Amir 2026-07-05: INHD halted)

# trendlab swing fraction: ONE value for display AND calculations (Amir 2026-07-05: keep 0.2
# everywhere; the 0.1-vs-Pine parity notes live in BUGS under the outside-bar item).

# ---- multi-timeframe awareness (mtf.py) — per-ticker state across timeframes
MTF_TFS = ["15m", "1h", "4h", "1D", "1W", "1M"]   # the curated set the MTF profile/panel computes
MTF_SLOPE_LOOKBACK = 5         # bars back to measure each MA's slope (rising/falling tilt)

# ---- 9. delayed high volume close — gap up in the last few days, consolidate, then an HVC
DHVC_LOOKBACK = 10           # search this many recent bars for the ignition gap
DHVC_GAP_MIN_PCT = 4.0       # ignition bar's gap (open vs prior close), %
DHVC_GAP_MIN_RVOL = 1.5      # ignition bar's volume vs 50d avg
DHVC_MIN_BASE_BARS = 2       # consolidation bars required between the gap and today
DHVC_BASE_MAX_DEPTH = 12.0   # consolidation (high-low)/high must be <= this %
DHVC_GAP_HOLD_TOL = 3.0      # consolidation low may dip at most this % below the gap day's close
DHVC_REQUIRE_UPTREND = False # the gap+base is the ignition; prior uptrend optional

# ---- 10. QM breakout (Qullamaggie momentum continuation) — RS leader, big move, tightening
# flag that surfs the rising 10/20-EMA, then a range-expansion breakout. RS leadership (top
# 1-2% over 1/3/6mo) is the rs_rank column + a dashboard filter, not gated in the detector.
QMB_MOVE_LOOKBACK = 63       # the move's high must be within ~this many bars (3 months)
QMB_MIN_MOVE = 30.0          # the move (recent low -> high) gained at least this %
QMB_MOVE_MAX_BARS = 35       # the move happened over at most this many bars (days-to-weeks)
QMB_CONS_MIN = 8             # consolidation length >= this (bars, ~1.5 weeks)
QMB_CONS_MAX = 45            # consolidation length <= this (bars, ~2 months)
QMB_MAX_PULLBACK = 50.0      # max pullback in the consolidation from the move high, % (less is better;
                             # volatile momentum names can flush deep then re-tighten, e.g. FCEL ~46%)
QMB_SURF_BAND = 3.0          # a close is "riding" the EMA if >= EMA20*(1 - this%/100)
QMB_MIN_RIDE_FRAC = 0.5      # >= this fraction of consolidation bars riding at/above the 20-EMA
QMB_CONTRACT_WIN = 10        # recent window (bars before now) used as the breakout volume baseline
# the actionable breakout pivot is the consolidation's UPPER TRENDLINE (resistance), fit robustly so
# it ignores shakeout spikes — flat (box/ascending-triangle) or DESCENDING (pennant/symmetrical
# triangle). The lower boundary is the rising higher-lows; the two must converge (contract). Trigger =
# the upper line extrapolated to the bar. State: building (coiling under it) -> breakout (the day price
# first closes above it, on volume EXPANSION) -> dropped after. Failed breakouts re-fit next day (the
# poke-above becomes an ignored up-shakeout).
QMB_BREAKOUT_VOL = 1.3       # the breakout bar's volume must be >= this x the consolidation's avg volume

# ---- breakout-level stack (shared by the QM + HTF breakouts; see setups._breakout_stack)
BREAKOUT_STACK_DOWNSWING = 8  # a downswing this many bars long (no pivot) pushes the prior bar's high as a
                              # synthetic resistance level (consecutive synthetics replace each other)
# When the latest daily bar is TODAY's still-forming (partial) session, its volume is only a fraction of a
# full day's, so a real intraday breakout can't clear the volume test until the close. If True, project the
# partial bar's volume up to a full-session estimate via the U-shaped intraday volume curve (heavy at
# open/close) so intraday breakouts register PROVISIONALLY; they settle to real volume at EOD. QM + HTF.
BREAKOUT_VOL_PROJECT = True

# ---- 11. EMA Rider (bullish & bearish) — a sustained streak of bars CLOSING above/below an
# EMA that keeps getting defended at the line. Direct port of emaRider.pine (see emarider.py
# lab). A setup fires when the streak ending on the latest bar is long enough AND has held the
# EMA enough times. All knobs are live-tunable (they key a private per-frame streak cache).
ER_EMA_LEN = 20              # the EMA the price streak rides
ER_ATR_LEN = 14              # ATR length for the "near EMA" proximity zone
ER_ATR_FRAC = 0.5           # an "EMA hold" = wick within ATR*this of the EMA (a wick through it counts)
ER_MIN_STREAK = 10          # current streak (consecutive closes one side of the EMA) must be >= this
ER_MIN_HOLDS = 1            # require at least this many EMA holds (touches after the 1st streak bar)
ER_USE_STREAK_THRESH = False  # allow a close within the ATR zone past the EMA to still extend the streak
ER_STREAK_ATR_FRAC = 0.5    # streak tolerance = ATR*this (only used when ER_USE_STREAK_THRESH)
ER_USE_ARM_EXIT = True      # decisive-exit machine (emaRider.pine use_arm_exit): a wrong-side close ARMS an
                            # exit (ride keeps counting) -> resolves SAVED (back on-side) or BREAK (armed bar's
                            # extreme taken out -> flip, dated back to the arming bar). Off = legacy first-close flip.

# ---- setup quality score (quality.py) — weights + min->max ramps per sub-score.
# composite = 100 * sum(w_i * q_i) / sum(w_i) over the sub-scores that apply to the setup.
Q_WEIGHTS = {"adr": 0.20, "trend": 0.25, "tight": 0.20, "emasurf": 0.20, "voldry": 0.15, "catalyst": 0.35}
Q_ADR_RAMP = (2.0, 10.0)         # adr_pct: this% -> 0, that% -> 1
Q_TREND_GAIN_RAMP = (25.0, 150.0)  # prior up-move net % -> 0..1 (x clarity)
Q_TIGHT_RAMP = (20.0, 3.0)       # consolidation range %: loose -> 0, tight -> 1 (note: hi<lo = inverse)
Q_EMASURF_BAND = 2.0             # a close counts as "surfing" if >= EMA*(1 - this%/100)
Q_VOLDRY_RAMP = (0.0, 0.5)       # volume reduction fraction across the base -> 0..1
Q_EP_CPOS_RAMP = (0.65, 0.90)    # Day 1 close position: 0.65 -> 0, 0.90 -> 1
Q_EP_RVOL_RAMP = (3.0, 8.0)      # Day 1 RVOL: 3x -> 0, 8x+ -> 1

# ---- auto daily update (serve.py in-app scheduler) — defaults; live-tuned in settings.json
AUTO_UPDATE = False          # master toggle for the background daily-update thread
AUTO_UPDATE_TIME = "23:30"   # local HH:MM to run update_daily (after US close, Israel time)
INTRADAY_AUTO_REFRESH = True    # auto re-pull the FUTURES' 5m bars during US market hours (equities: 5m poller)
INTRADAY_REFRESH_EVERY_MIN = 5   # minutes between auto FUTURES-5m refreshes (matches the 5m bar cadence; yfinance feed delay ~8min)
INTRADAY_EQUITY_GAPFILL_MIN = 30  # minutes between the slower equity today-session gap-fills (safety net; the poller is the live source)

# ---- EP/RVOL significance grading + alerts (significance.py / alerts.py; plan-ep-significance-alerts)
SIG_NOTIFY_MACOS = True         # grade-A events also fire a native macOS notification (notify.py)
SIG_ALERT_GRADES = ("A",)       # which grades may alert (Amir 2026-07-08: A only)
SIG_REARM_MIN = 90              # minutes before a symbol may re-alert the same event kind
SIG_EXT_RED = {"stock": 10.0, "etf": 6.0}   # worst-TF ATR-ext at the RED badge tier -> grade capped at B
SIG_PM_DOLLAR_M = 3.0           # pre-market gapper alert: calibrated pre-market $vol threshold (M$)
RVOL_SANITY_MAX = 1000.0        # drop live cum_rvol above this — nothing legitimately trades 1000x its
#                                 ADV; a bad live day-volume print (HEI-A/HAE 2026-07-08) must not top
#                                 the leaderboard or feed the significance grade
DAYTYPE_PREP_TIME_ET = "07:40"  # ET time serve auto-runs daytype_live.prep() (today's pre-open bucket-A
#                                 features) once/weekday, so the live day-type read is ready at the open
#                                 (was a manual step). ET, not local — the scheduler triggers it in ET.

# ---- live session seed (serve._seed_today_session)
SEED_WORKERS = 8                # parallel Tiingo fetchers (2026-07-08: serial loop took ~17-20 min for
#                                 ~2040 names at the open; 8 workers cuts it to ~2-3 min, gentle enough
#                                 for the Power-tier rate limit — fetch-only parallelism, writes stay locked)

# ---- live server (serve.py)
SERVE_UNIVERSE_MIN_DVOL_M = 0  # 0 = all cached tickers; >0 restricts server universe for speed

# ---- true exchange breadth in the CLOSE banner internals (tv_breadth.py) — Amir 2026-07-06
# When on, the daily/CLOSE breadth block (market._breadth_block) uses REAL exchange internals
# (NYSE/NASDAQ net advancers−decliners + Arms/TRIN) from the manually-extracted TradingView
# daily series instead of the ~2000-name universe SAMPLE. %>MA and new-H/L have no free source,
# so they self-compute from our yfinance daily bars (the same census, doubling as the fallback).
# FLIP TO FALSE (or delete data/tv/) to remove ALL TradingView dependency — the block reverts to
# the pure universe-sample computation, no crash (every read is presence-guarded). The LIVE
# intraday internals (intraday.py) are untouched — TV has no realtime feed.
USE_TV_BREADTH = False
TV_ADD_SCALE = 1063.0        # ~1σ of NYSE net A−D ($ADD) → tanh scale for the breadth score term
TV_ADDQ_SCALE = 1225.0       # ~1σ of NASDAQ net A−D ($ADDQ)
TV_BREADTH_MAX_LAG_DAYS = 7  # if the manual TV extract lags the view date by more than this (calendar
                             # days), fall back to the self-computed sample instead of showing stale breadth

# Day-type FEATURES sourced from the manual TV breadth exports (TRIN/TICK/ADD + NASDAQ TRINQ/ADDQ/TICKQ)
# in dayfeatures.external_block. Measured (2026-07, frozen 2025-26 holdout): these add ~0 out-of-sample
# skill (drop TRIN+TICK −0.01; drop all incl ADD −0.05) AND they had NO lag guard, so the model + the LIVE
# read silently ate week-old internals whenever the export fell behind. OFF: drop them from train + live so
# the day-type read never goes stale and the manual TV export is no longer a day-type dependency. Self-
# computed adv/decl breadth is already in the matrix (universe_daily_block). Flip True to restore (+ retrain).
DAYTYPE_USE_TV_INTERNALS = False

# ---- market situational awareness (market.py) — top-down long/out/short score
# macro symbols fetched into data/bars/ alongside the universe (yfinance tickers)
MACRO_SYMBOLS = ["^VIX", "^VXN", "DX-Y.NYB", "^TNX", "TLT",
                 "^VIX9D", "^VIX3M",      # VIX term structure (day-feature study, 2026-07-04)
                 "BTC-USD", "^GDAXI", "^HSI"]  # MED block: 24/7 + foreign-session overnight reads
# composite = 100 * sum(w_i * block_i) / sum(w_i); block_i in [-1, +1]
MARKET_WEIGHTS = {"trend": 0.32, "breadth": 0.20, "vol": 0.16, "macro": 0.12, "risk": 0.20}
# ---- risk-on/off rotation block (market._risk_block via ratios.py). Each entry (a, b, sign, label): the
# A/B ratio's posture is scored to [-1,1] and multiplied by `sign` so +1 = risk-ON. A rising ratio here =
# higher risk appetite / broadening / cyclical leadership. Index legs use futures (match the internals desk).
RATIO_BASKET = [
    ("RTY=F", "ES=F", +1, "small-caps (RTY/ES)"),      # small-caps vs broad — higher beta = risk-on
    ("RTY=F", "NQ=F", +1, "small-caps>tech (RTY/NQ)"),  # broadening away from mega-cap tech
    ("NQ=F",  "ES=F", +1, "tech (NQ/ES)"),              # growth leadership
    ("SMH",   "SPY",  +1, "semis (SMH/SPY)"),           # semis = risk-on bellwether
    ("XLY",   "XLP",  +1, "discr>staples (XLY/XLP)"),   # offense over defense
    ("KRE",   "SPY",  +1, "regional banks (KRE/SPY)"),  # cyclical/credit appetite
    ("HG=F",  "GC=F", +1, "copper/gold (HG/GC)"),       # growth vs safe-haven
]
MARKET_EXT_CAP = 4.0          # |ATR extension from 50-MA| above this = overextended (trims conviction)
# Extension Health Badge tiers (ATR multiples from the 50-MA, measured on the goggles-bias side):
# below first = healthy, between = caution, above second = extreme. ETFs/indices/themes/ratios move
# less than single stocks (index 4x ext = historical pullback zone), so their caution starts earlier.
EXT_BADGE_TIERS = {"stock": [4.0, 10.0], "etf": [2.5, 6.0]}
# RSI Historical Extremes (port of Amir's rsi_extreme.pine, 2026-07-03): a symbol's OWN
# all-time / recent RSI extremes define its personal overbought/oversold zones.
RSIX_LEN = 14                 # Wilder RSI length
RSIX_BUF = 5.0                # "approaching" zone width in RSI points inside each extreme
RSIX_AT_TOL = 1.0             # within this of the ACTUAL extreme = AT it (badge/setup fire —
                              # Amir 2026-07-03 CORZ case: zone entry at buf-5 must NOT read "extreme")
RSIX_REC = 200                # rolling window (bars) for the "recent" extremes
RSIX_RECENT_BARS = 10         # an extreme event within this many bars still counts as "recent"
RSIX_TFS = ["5m", "15m", "1h", "4h", "12h", "1D", "1W"]   # store/badge timeframes
RSIX_MIN_HISTORY = 250        # bars before the expanding extreme is trusted (skip young series)
MARKET_VIX_LOW = 16.0         # VIX <= this = calm (risk-on); >= HIGH = fear (risk-off)
MARKET_VIX_HIGH = 26.0
# composite -> 5-band stance (upper edges; symmetric): >=+50 strong long .. <=-50 short
MARKET_BANDS = {"strong_long": 50.0, "cautious_long": 15.0, "cautious_short": -15.0, "short": -50.0}
MARKET_MACRO_ATR = 2.0       # macro leg (^TNX/TLT/DXY): ATRs from the 20-EMA at which it ~saturates (tanh);
                             # a marginal EMA cross now nudges the score instead of slamming it to ±1
MARKET_PARTIAL_DAMP = 0.5    # scale the macro block on a LIVE/partial (intraday) bar — least-settled data
GROUP_MIN_MEMBERS = 4        # min members for a sector/theme to get a score (leaderboard)
PLAYBOOK_TOP_N = 5           # natural-language Playbook: long/short candidate names surfaced per side
# Playbook v3 conviction blend (per-candidate, direction-aware). HARD tradeability gate first (drop, not
# rank), then: setup signals + group RS + VOLATILITY-NORMALIZED momentum (move ÷ ADR, not raw %) + liquidity
# reward + S/R room, with a per-sector diversification cap. Live-tunable.
PLAYBOOK_MIN_PRICE = 10.0    # gate: drop candidates under this share price (no low-priced/speculative names)
PLAYBOOK_MIN_DVOL_M = 50.0   # gate: min avg $-volume (M)
PLAYBOOK_MIN_MCAP_B = 2.0    # gate: min market cap ($B) — but only when mcap is KNOWN (None passes on price+$vol)
PLAYBOOK_MAX_PER_SECTOR = 2  # diversification: at most this many ideas per sector on each side
PLAYBOOK_W_SETUP = 1.0       # weight on the matched-setup signal (rider/trend weighted highest below)
PLAYBOOK_W_RS = 0.5          # weight on the candidate's group relative strength
PLAYBOOK_W_MOM = 1.0         # weight on ADR-normalized multi-window momentum (d1/wtd/w1, + open when live)
PLAYBOOK_W_INTRA = 0.8       # weight on today's ADR-normalized since-open (live/replay mode only)
PLAYBOOK_W_LIQ = 0.3         # reward more-liquid/larger names (log10 $-vol) so tradeability breaks ties
PLAYBOOK_W_ROOM = 0.4        # weight on S/R room (far from resistance for longs / support for shorts)
# per-setup base weight (× state/direction logic in narrative). Riders + trend give the best results; QM /
# HTF are long-only; flat base is de-weighted (not well configured yet). Data-driven weights later.
PLAYBOOK_SETUP_WEIGHTS = {
    "ema_rider_bull": 1.0, "ema_rider_bear": 1.0, "uptrend": 1.0, "downtrend": 1.0,
    "episodic_pivot": 0.4,   # starts neutral; the follow-through validator sets the real weight
    "backburner": 0.7, "qm_breakout": 0.7, "high_tight_flag": 0.7,
    "higher_low_ma": 0.5, "undercut_rally": 0.5, "gapper": 0.4, "hvc": 0.4,
    "delayed_hvc": 0.4, "stairstep": 0.4, "flat_base": 0.2,
    "cup_handle": 0.4, "double_top": 0.4, "head_shoulders": 0.4, "inverse_hs": 0.4,
    "rsi_extreme_revert": 0.4, "rsi_extreme_fade": 0.4,
    "_default": 0.4,
}

# ---- pattern geometry (geometry.py) — the pivot-template engine behind the pattern detectors.
# Templates match on a COARSER zigzag than the default 0.2 (which fragments cup rims / shoulders).
GEO_ATR_FRACTION = 0.5
CH_CUP_MIN = 20              # cup-with-handle: cup span (bars)
CH_CUP_MAX = 120
CH_RIM_TOL = 5.0             # left/right rim height tolerance (%)
CH_DEPTH_MIN = 12.0          # cup depth bounds (% of rim)
CH_DEPTH_MAX = 33.0
CH_HANDLE_MIN = 3            # handle span (bars)
CH_HANDLE_MAX = 15
CH_HANDLE_PULL_MIN = 3.0     # handle pullback off the right rim (%)
CH_HANDLE_PULL_MAX = 10.0
DT_LOOKBACK = 120            # double/triple top: pivot-high search window (bars)
DT_TOL = 2.0                 # peak height match tolerance (%)
DT_MIN_SPACING = 10          # min bars between matched peaks
DT_MIN_VALLEY = 8.0          # min valley depth between peaks (% of top)
DT_MAX_AGE = 20              # last peak must be at most this many bars old
HS_LOOKBACK = 140            # head & shoulders: same-side pivot window (bars)
HS_HEAD_MIN_ATR = 1.0        # head must exceed both shoulders by this many ATRs
HS_SHOULDER_TOL = 5.0        # shoulder height match tolerance (%)
HS_MAX_AGE = 25              # right shoulder at most this many bars old

# ---- support/resistance (sr.py) — swing-pivot zones clustered into strength-weighted levels
SR_LOOKBACK = 400            # bars of pivots to source S/R from
SR_CLUSTER_ATR = 0.75        # merge pivots whose extremes sit within this * ATR into one level
SR_LIVE_COUNT = 3            # nearest N support + N resistance levels to surface/draw
SR_MIN_STRENGTH = 1          # drop levels tested fewer than this many times

# ---- situational awareness v2 (sequence.py / awareness.py / calls.py) — the market map.
# Sequence grammar: regime labels from trendlab segments+swings, read point-in-time on any frame
# (index, sector ETF, theme basket, ratio). All thresholds live-tunable (settings "awareness").
AWARE_RANGE_MIN_BARS = 8       # a 'range' must span at least this many bars to be structure
AWARE_FRESH_BARS = 5           # a break is 'fresh' this many bars after the first close beyond it
AWARE_MAJOR_LOOKBACK = 45      # bars to search for the MAJOR swing high/low (depth/retrace anchor)
AWARE_CONSOL_MAX_ATR = 4.5     # max band height (ATRs) for a consolidation-after-advance read
AWARE_IMPULSE_SLOPE = 0.30     # ATRs/bar of net progress for an 'impulse' leg
AWARE_IMPULSE_MAX_BARS = 25    # an impulse older than this reads as steady trend, not impulse
AWARE_TREND_MIN_ER = 0.35      # min efficiency for a steady 'trend_up/down' leg read
AWARE_LEG_MIN_PCT = 3.0        # min prior leg (%) for a pullback/bounce read on DAILY (intraday /3)
AWARE_PB_MAX_ATR = 3.0         # max pullback depth (ATRs) before the pullback-in-uptrend read drops
AWARE_PB_MAX_RETRACE = 0.62    # max retrace of the prior leg (fraction)
AWARE_EXT_ATR = 2.5            # 'extended' flag: close this many ATRs beyond the 20-EMA
AWARE_RETEST_BARS = 7          # look for S/R retest events within this many recent bars
AWARE_AT_LEVEL_ATR = 0.5       # 'at_support/at_resistance' when the nearest level is this close (ATRs)
AWARE_LATE_BARS = 15           # impulse 'late' flag: leg age in bars ...
AWARE_LATE_CHAIN = 4           # ... or an HH/HL chain at least this long
AWARE_BASKET_ATR_FRACTION = 0.4  # coarser zigzag on close-only basket frames (ATR ≈ |Δclose|)
# gap read (equity ETFs only; futures are 24h): status needs >=2 completed 15m bars
AWARE_GAP_MIN_ATR = 0.5        # a gap smaller than this (in daily ATRs) is noise for the gap rules
AWARE_GAP_HOLD_ATR = 0.25      # 'holding' = last close >= max(open - this*ATR, prior close)
# instrument catalog: the sector row (all have daily + 15m data) and the ratio matrix
AWARE_SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
                     "SMH", "KRE"]
AWARE_TOP_THEMES = 15          # themes that get full instrument views (all still appear in rotation)
# (a, b, sign, label): sign=+1 when numerator leadership = risk-on (descriptive pairs use +1 too)
AWARE_RATIOS = [
    ("QQQ", "SPY", +1, "tech vs broad (NQ/ES)"),
    ("IWM", "SPY", +1, "small-caps vs broad (RTY/ES)"),
    ("QQQ", "IWM", +1, "mega-tech vs small-caps"),
    ("DIA", "SPY", -1, "defensives vs broad (YM/ES)"),
    ("RTY=F", "ES=F", +1, "futures: RTY/ES"),
    ("NQ=F", "ES=F", +1, "futures: NQ/ES"),
    ("HG=F", "GC=F", +1, "copper/gold"),
    ("SMH", "SPY", +1, "semis vs broad"),
    ("XLY", "XLP", +1, "discretionary vs staples"),
    ("KRE", "SPY", +1, "regional banks vs broad"),
] + [(etf, "SPY", +1, f"{etf} vs SPY") for etf in
     ("XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC")]
# rotation quadrant: trailing RS (x) vs today's rotation z-score (y)
AWARE_ROT_WINDOW = 63          # settled sessions of group-vs-SPY daily relative returns
AWARE_ROT_Z_FLOOR = 0.15       # std floor (%) so dead-quiet groups can't print fake 5-sigma days
AWARE_ROT_EXT = 0.75           # |rot z| beyond this = extending / fading today
AWARE_ROT_LEAD = 1.0           # trailing 1m RS (%) beyond ± this = leading / lagging
AWARE_IND_ROWS = 15            # top/bottom industries (by |rot z|) included in the payload

# geometry v2 (Amir's verification pass #1): context gates + lifecycle
CH_PRIOR_ADV = 30.0          # cup&handle: min prior advance (%) into the LEFT rim (continuation)
CH_MAX_RETRACE = 0.5         # cup depth as a fraction of the prior advance (1/3 ideal, 1/2 max)
DT_PRIOR_ADV = 25.0          # double top: min advance (%) into peak 1
HS_PRIOR_MOVE = 25.0         # H&S: min prior advance into the head (mirror decline for iH&S)
HS_NECK_SLOPE_ATR = 0.08     # max neckline slope (ATR/bar) — steeper = too complex to trust
HS_ARM_MAX = 45              # H&S: shoulders must sit within this many bars of the head
HS_SYM_RATIO = 2.5           # H&S: max left-arm/right-arm span asymmetry

# ---- 12. episodic pivot (Bonde) — catalyst gap or objective volume event + delayed-reaction watch
EP_MIN_GAP = 6.0             # event: gap % (classic catalyst EP)
EP_MIN_RVOL = 3.0            # event: volume vs 50d avg (classic)
EP_MIN_SHARES = 9_000_000    # event: EP-9M objective filter — raw shares traded in one day
EP_VOL_MULT = 3.0            # EP-9M: rvol multiple required alongside the share count
EP_MIN_PRICE = 1.0           # penny-safety floor only — the real liquidity guard is dollar volume
EP_MIN_DVOL_M = 10.0         # event-day dollar volume floor, $M (replaces Bonde's $3 price gate)
EP_BIGMOVE_CHG = 10.0        # big-move path: day change % that is an EP even with NO gap
EP_LEADER_DVOL_B = 1.0       # liquid leader: avg daily dollar volume, $B — mega-caps can't 3x volume
EP_LEADER_RVOL = 2.0         # relaxed rvol requirement for liquid leaders (all three paths)
EP_GAP_RETAIN_MIN = 25.0     # gap % above which the retention test applies (big-gap events)
EP_GAP_RETAIN = 0.5          # big gaps must CLOSE retaining >= this fraction of the gap (Amir verdict:
                             # SDOT kept 44% / BULL 31% = fades & split artifacts, not EPs)
EP_MIN_BASE_DVOL_M = 0.5     # median $vol over the prior 60 bars, $M — kills dead-shell pumps that
                             # only trade on the event day (event-day dollar volume can't see this)
EP_WATCH_BARS = 21           # delayed-reaction window after the event (~1 month)
EP_BREAKOUT_RVOL = 1.5       # volume confirmation for a delayed-reaction breakout (stack pop)
EP9M_MIN_CHG = 4.0           # EP-9M also needs a real move, % — kills rebalance-day volume spikes
EP_MIN_CLOSE_POS = 0.65       # Day 1 quality gate: close in upper 35% of daily range (eliminates 47% of traps)
EP_DEBOUNCE_BARS = 10         # debounce cooldown: bars required between distinct EP Day 1 ignitions

# --- LIVE intraday RVOL/EP-radar $vol floors (M$) — lowered + per-setup (Amir 2026-07-06): the old
# blanket 5M excluded thin-name EPs (BLZE ~3M$/day before its catalyst). First-pass defaults — tune.
# EP radar hunts EPs directly (move+gap+pace gates filter noise) -> low blanket floor. RVOL leaders
# scan the whole universe -> a name IN a flagged setup qualifies at its per-setup floor below, else the
# generic default. Replay stays maximally permissive (1M) and ignores these.
LIVE_EP_MIN_DVOL_M = 1.5             # EP radar blanket floor
LIVE_RVOL_MIN_DVOL_M = 2.0           # RVOL leaders default (names not in a flagged setup); was 5.0
SETUP_MIN_DVOL_M = {                 # per-setup override for the RVOL-leaders floor (thin-name-friendly first)
    "episodic_pivot": 1.0, "gapper": 1.0, "high_tight_flag": 1.0,
    "hvc": 1.5, "delayed_hvc": 1.5, "backburner": 1.5,
    "flat_base": 3.0, "qm_breakout": 3.0, "cup_handle": 3.0,
    "higher_low_ma": 3.0, "undercut_rally": 2.5, "stairstep": 3.0,
}

# backburner RSI stage machine (Dan/TCG oversold-bounce: first-touch per leg, re-armed on new ATH)
BB_RSI_ENTRY1 = 30.0         # first entry: leg's first RSI dip below this
BB_RSI_ENTRY2 = 20.0         # second entry: the same episode deepening to this
BB_OS_MAX_RETRAC = 60.0      # entry1/entry2 may retrace this much of the up-move (RSI<30 implies deep)
BB_MIN_ADV = 30.0            # OS entries need a swing-scale advance (%) into the ATH
BB_OS_ATH_MAX_AGE = 60       # OS entries: the ATH may be up to this many bars back (RSI needs time)
LIVE_POLL_SEC = 60           # Tiingo live poll cadence (bulk IEX quotes -> forming 15m bar)
FEED_FREEZE_CYCLES = 2       # RTH cycles the reference basket's IEX volume can stall before we flag the
#                              trade-data feed as frozen/degraded. Weekend sweep 2026-07-11 on both tapes:
#                              N=2 flags the real 07-10 freeze at 09:33:39 (~75s earlier than N=3, ~2x the
#                              flagged coverage) with ZERO false positives across 67 healthy RTH cycles
#                              (incl. the sampled afternoon and all of 07-09) — the 8-name basket never
#                              stalled even one cycle outside the true freeze.
