"""Live-tunable settings: apply UI overrides on top of the config.py defaults.

Detectors, regime and quality all read `config.*` globals directly, so rather than
thread a Settings object through every signature we override the globals in place
for the duration of a scan, guarded by a lock (serve.py is threaded). config.py
stays the single source of DEFAULTS; overrides persist to data/settings.json.

Only the cache-safe TUNABLE names are exposed. ATR_FRACTION is deliberately NOT
tunable here: it changes the pivot pass that trendlab memoizes per frame
(framecache by frame identity), so a live change would read stale caches — keep it
in config.py and rebuild frames if you change it.
"""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager

import config

# Cache-safe thresholds the settings panel may override (grouped for the UI).
TUNABLE: dict[str, list[str]] = {
    "gapper": ["GAP_MIN_PCT", "GAP_MIN_RVOL"],
    "hvc": ["HVC_MIN_VOL_MULT", "HVC_CLOSE_RANGE_POS", "HVC_MIN_CHANGE_PCT", "HVC_REQUIRE_UPTREND"],
    "flat_base": ["FB_MIN_BARS", "FB_MAX_BARS", "FB_TOP_TOL", "FB_MIN_TOUCHES", "FB_MAX_DEPTH_PCT",
                  "FB_MAX_DIST_FROM_HIGH", "FB_BREAKOUT_TOL", "FB_MAX_DRIFT", "FB_NEAR_HIGH", "FB_MIN_TOUCH_SPAN",
                  "FB_PRIOR_MODE", "FB_PRIOR_LOOKBACK", "FB_PRIOR_GAIN_PCT", "FB_REQUIRE_STAGE2",
                  "FB_VOL_CONTRACTION", "FB_BREAKOUT_RVOL"],
    "high_tight_flag": ["HTF_LOOKBACK", "HTF_MIN_GAIN_PCT", "HTF_POLE_MAX_BARS", "HTF_MAX_PULLBACK_PCT",
                        "HTF_FLAG_MIN_BARS", "HTF_OPTIMAL_MIN_BARS", "HTF_FLAG_MAX_BARS", "HTF_NEAR_HIGH", "HTF_MIN_ADR",
                        "HTF_MIN_EFFICIENCY", "HTF_REQUIRE_MA_STACK", "HTF_RISING_MA_BARS",
                        "HTF_BREAKOUT_RVOL", "BREAKOUT_STACK_DOWNSWING", "BREAKOUT_VOL_PROJECT",
                        "HTF_REQUIRE_REGIME"],
    "higher_low_ma": ["HL_TREND_SMA", "HL_ATR_DIST"],
    "undercut_rally": ["UR_LEVEL_LOOKBACK", "UR_LEVEL_MIN_AGE", "UR_MAX_UNDERCUT_PCT"],
    "backburner": ["BACKBURNER_MAX_RETRAC", "BACKBURNER_ATH_MAX_AGE", "BB_RSI_ENTRY1", "BB_RSI_ENTRY2", "BB_OS_MAX_RETRAC", "BB_MIN_ADV", "BB_OS_ATH_MAX_AGE"],
    "episodic_pivot": ["EP_MIN_GAP", "EP_MIN_RVOL", "EP_MIN_SHARES", "EP_VOL_MULT", "EP_MIN_PRICE",
                       "EP_MIN_DVOL_M", "EP_BIGMOVE_CHG", "EP_LEADER_DVOL_B", "EP_LEADER_RVOL",
                       "EP_GAP_RETAIN_MIN", "EP_GAP_RETAIN", "EP_MIN_BASE_DVOL_M",
                       "EP_WATCH_BARS", "EP_BREAKOUT_RVOL", "EP9M_MIN_CHG", "EP_MIN_CLOSE_POS",
                       "EP_DEBOUNCE_BARS"],
    "stairstep": ["STAIRSTEP_MIN_BARS", "STAIRSTEP_ATR_FRACTION", "STAIRSTEP_MAX_BREAKOUT_BARS"],
    "cup_handle": ["GEO_ATR_FRACTION", "CH_CUP_MIN", "CH_CUP_MAX", "CH_RIM_TOL", "CH_DEPTH_MIN",
                   "CH_DEPTH_MAX", "CH_HANDLE_MIN", "CH_HANDLE_MAX", "CH_HANDLE_PULL_MIN",
                   "CH_HANDLE_PULL_MAX", "CH_PRIOR_ADV", "CH_MAX_RETRACE"],
    "double_top": ["DT_LOOKBACK", "DT_TOL", "DT_MIN_SPACING", "DT_MIN_VALLEY", "DT_MAX_AGE",
                   "DT_PRIOR_ADV"],
    "head_shoulders": ["HS_LOOKBACK", "HS_HEAD_MIN_ATR", "HS_SHOULDER_TOL", "HS_MAX_AGE",
                       "HS_PRIOR_MOVE", "HS_NECK_SLOPE_ATR", "HS_ARM_MAX", "HS_SYM_RATIO"],
    "delayed_hvc": ["DHVC_LOOKBACK", "DHVC_GAP_MIN_PCT", "DHVC_GAP_MIN_RVOL", "DHVC_MIN_BASE_BARS",
                    "DHVC_BASE_MAX_DEPTH", "DHVC_GAP_HOLD_TOL", "DHVC_REQUIRE_UPTREND"],
    "qm_breakout": ["QMB_MOVE_LOOKBACK", "QMB_MIN_MOVE", "QMB_MOVE_MAX_BARS", "QMB_CONS_MIN",
                    "QMB_CONS_MAX", "QMB_MAX_PULLBACK", "QMB_SURF_BAND", "QMB_MIN_RIDE_FRAC",
                    "QMB_CONTRACT_WIN", "QMB_BREAKOUT_VOL", "BREAKOUT_STACK_DOWNSWING",
                    "BREAKOUT_VOL_PROJECT"],
    "ema_rider": ["ER_EMA_LEN", "ER_ATR_LEN", "ER_ATR_FRAC", "ER_MIN_STREAK", "ER_MIN_HOLDS",
                  "ER_USE_STREAK_THRESH", "ER_STREAK_ATR_FRAC", "ER_USE_ARM_EXIT"],
    "rsi_extremes": ["RSIX_LEN", "RSIX_BUF", "RSIX_AT_TOL", "RSIX_REC", "RSIX_RECENT_BARS",
                     "RSIX_MIN_HISTORY"],
    "regime": ["REGIME_USE", "REGIME_PRIOR_GAIN", "REGIME_PRIOR_CLARITY"],
    "quality": ["Q_WEIGHTS", "Q_ADR_RAMP", "Q_TREND_GAIN_RAMP", "Q_TIGHT_RAMP",
                "Q_EMASURF_BAND", "Q_VOLDRY_RAMP"],
    "auto_update": ["AUTO_UPDATE", "AUTO_UPDATE_TIME"],
    "intraday": ["INTRADAY_AUTO_REFRESH", "INTRADAY_REFRESH_EVERY_MIN"],
    "market": ["MARKET_WEIGHTS", "MARKET_EXT_CAP", "MARKET_VIX_LOW", "MARKET_VIX_HIGH", "MARKET_BANDS",
               "MARKET_MACRO_ATR", "MARKET_PARTIAL_DAMP"],
    "support_resistance": ["SR_LOOKBACK", "SR_CLUSTER_ATR", "SR_LIVE_COUNT", "SR_MIN_STRENGTH"],
    "playbook": ["PLAYBOOK_TOP_N", "PLAYBOOK_MIN_PRICE", "PLAYBOOK_MIN_DVOL_M", "PLAYBOOK_MIN_MCAP_B",
                 "PLAYBOOK_MAX_PER_SECTOR", "PLAYBOOK_SETUP_WEIGHTS", "PLAYBOOK_W_SETUP", "PLAYBOOK_W_RS",
                 "PLAYBOOK_W_MOM", "PLAYBOOK_W_INTRA", "PLAYBOOK_W_LIQ", "PLAYBOOK_W_ROOM"],
    "awareness": ["AWARE_RANGE_MIN_BARS", "AWARE_FRESH_BARS", "AWARE_MAJOR_LOOKBACK",
                  "AWARE_CONSOL_MAX_ATR", "AWARE_IMPULSE_SLOPE", "AWARE_IMPULSE_MAX_BARS",
                  "AWARE_TREND_MIN_ER", "AWARE_LEG_MIN_PCT", "AWARE_PB_MAX_ATR", "AWARE_PB_MAX_RETRACE",
                  "AWARE_EXT_ATR", "AWARE_RETEST_BARS", "AWARE_AT_LEVEL_ATR", "AWARE_LATE_BARS",
                  "AWARE_LATE_CHAIN", "AWARE_BASKET_ATR_FRACTION", "AWARE_GAP_MIN_ATR",
                  "AWARE_GAP_HOLD_ATR", "AWARE_TOP_THEMES", "AWARE_ROT_WINDOW", "AWARE_ROT_Z_FLOOR",
                  "AWARE_ROT_EXT", "AWARE_ROT_LEAD", "AWARE_IND_ROWS"],
}
_NAMES = {n for group in TUNABLE.values() for n in group}

# Which TUNABLE group holds each SETUP's own knobs. A knob exclusive to one setup can be
# cached per-setup (changing it only busts that setup); everything else (regime, quality,
# PATTERN_SLOPE_TOL, ...) is GLOBAL — changing it busts every setup's cache.
_SETUP_GROUP = {
    "gapper": "gapper", "episodic_pivot": "episodic_pivot",
    "hvc": "hvc", "delayed_hvc": "delayed_hvc", "flat_base": "flat_base",
    "high_tight_flag": "high_tight_flag", "higher_low_ma": "higher_low_ma",
    "undercut_rally": "undercut_rally", "qm_breakout": "qm_breakout",
    "backburner": "backburner", "stairstep": "stairstep",
    "ema_rider_bull": "ema_rider", "ema_rider_bear": "ema_rider",
    "uptrend": None, "downtrend": None,        # no own knobs — driven by global regime only
    "cup_handle": "cup_handle", "double_top": "double_top",
    "head_shoulders": "head_shoulders", "inverse_hs": "head_shoulders",
    "rsi_extreme_revert": "rsi_extremes", "rsi_extreme_fade": "rsi_extremes",
}
_SHARED = {"PATTERN_SLOPE_TOL"}                # lives in a setup group but classifies many setups
_NO_SCAN = (set(TUNABLE["auto_update"]) | set(TUNABLE["support_resistance"])   # don't affect scan output at all:
            | set(TUNABLE["market"]) | set(TUNABLE["playbook"])   # S/R = chart-render; market/rotation/playbook = awareness panel only
            | set(TUNABLE["awareness"]))                          # awareness map = its own cache (aware_sig)
# settings that belong exclusively to ONE setup (safe to isolate from the others)
_EXCLUSIVE = {n for grp in _SETUP_GROUP.values() if grp for n in TUNABLE[grp]} - _SHARED


def setup_sig(setup: str) -> str:
    """Signature of just the settings that affect THIS setup's output: its own exclusive knobs
    plus everything global. Excludes other setups' exclusive knobs, so tuning one setup leaves
    the others' cache keys unchanged."""
    own = set(TUNABLE.get(_SETUP_GROUP.get(setup) or "", [])) - _SHARED
    other_exclusive = _EXCLUSIVE - own
    cur = current()
    relevant = {k: v for k, v in cur.items() if k not in other_exclusive and k not in _NO_SCAN}
    return json.dumps(relevant, sort_keys=True, default=str)

def aware_sig() -> str:
    """Signature of the awareness-map knobs only — keys the /api/awareness caches so tuning the
    grammar re-reads the map without touching any scan cache."""
    cur = current()
    return json.dumps({k: cur[k] for k in TUNABLE["awareness"] if k in cur},
                      sort_keys=True, default=str)


_LOCK = threading.RLock()   # serializes config-override scans across server threads


def defaults() -> dict:
    """The full set of effective default values, straight from config.py."""
    return {n: getattr(config, n) for n in _NAMES}


def load() -> dict:
    """Saved overrides only (subset of TUNABLE). Missing/invalid file -> no overrides."""
    try:
        raw = json.loads(config.SETTINGS_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return {}
    return {k: v for k, v in raw.items() if k in _NAMES}


def current() -> dict:
    """Effective values shown in the UI = defaults with saved overrides applied."""
    return {**defaults(), **load()}


def _coerce(name, value):
    """Coerce an incoming JSON value to the type of the config default (best effort)."""
    cur = getattr(config, name)
    if isinstance(cur, bool):
        return bool(value) if not isinstance(value, str) else value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(cur, int) and not isinstance(cur, bool):
        return int(value)
    if isinstance(cur, float):
        return float(value)
    return value   # str / list / dict pass through


def save(overrides: dict) -> dict:
    """Validate + persist overrides; returns the new effective settings."""
    clean = {}
    for k, v in overrides.items():
        if k in _NAMES:
            try:
                clean[k] = _coerce(k, v)
            except (TypeError, ValueError):
                pass   # skip anything that won't coerce; keep the default
    config.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.SETTINGS_FILE.write_text(json.dumps(clean, indent=2))
    return current()


@contextmanager
def apply(overrides: dict | None = None):
    """Temporarily set config globals to `overrides` (default: saved overrides),
    restoring them afterward. Held under a lock so concurrent scans don't race."""
    ov = load() if overrides is None else overrides
    with _LOCK:
        saved = {}
        try:
            for k, v in ov.items():
                if k in _NAMES:
                    saved[k] = getattr(config, k)
                    setattr(config, k, v)
            yield
        finally:
            for k, v in saved.items():
                setattr(config, k, v)
