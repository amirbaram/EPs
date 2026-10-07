"""dcs_v5_config — shared threshold defaults for the Structure Engine v5 feature layer (Stage 2, PY).

Every value here is a DEFAULT, not a constant baked into detector logic — per the master plan's
standing rule #9 (base rates drift with regime; hard-coding today's numbers bakes in decay). Pine v5
mirrors these names/defaults in its own `input.*` block (contract: docs/contract-structure-v5.md);
this file is the Python side only. Kept as its own module rather than a new block in the shared
config.py to avoid collision risk with other concurrent sessions (Amir approved at G1, 2026-07-15).

Source: FeaturesStructures/evidence-brief-implementation.md §2 (defaults line), §7 (deepening-touch
count), and FeaturesStructures/execution-plan-agents.md §6.4 (contract threshold config).
"""
from __future__ import annotations

# ---- level-interaction events (F-E1-E5) ----
RECLAIM_K = 3                       # closed bars allowed for a spring/upthrust penetration to reclaim
QUIET_TEST_RATIO = 0.6              # post-spring/upthrust test rvol <= this x spring-bar rvol -> confirmed
ACCEPTANCE_CLOSES = 2               # consecutive closes beyond a level -> acceptance (role-flip candidate)
EQ_FRAC = 0.25                      # equal-pivot tolerance, x ATR (v4 default 0 disables flat regimes and
                                     # under-fires range events; v5/PY use a nonzero default, sweepable)
MIN_BOUNDARY_AGE_BARS = 5           # spring/upthrust gate (F-E1/E2 ONLY as of 2026-07-16 — see below):
                                     # the range boundary (nearest confirmed pivot) must be at least
                                     # this many bars old before a penetration counts — without it, the
                                     # "boundary" re-anchors on every fresh pivot (median same-side
                                     # pivot gap measured at ~7 bars on NQ=F 2h) and springs over-fire
                                     # ~100/yr instead of "rare". Briefly also a hard gate on F-E3
                                     # (test-quality) the same day it was added here; Amir removed that
                                     # specific use (via FeaturesStructures) once round 2 below showed
                                     # what it was actually discarding — see dcs_levels_events.py's
                                     # _detect_test_quality docstring. F-E3 now emits every touch and
                                     # persists round_trip/boundary_age as dedicated fields instead of
                                     # gating at detection; this constant still feeds F-E1/E2 only.
                                     #
                                     # STILL UNVALIDATED (Session B, chart-review-lab, 2026-07-16, two
                                     # rounds — the second PARTIALLY RETRACTS the first, read this not
                                     # the commit history): (1) boundary age at this TF has median 4 /
                                     # p90 11 / p99 19 bars — age>=10 is roughly the 86th percentile of
                                     # a distribution that rarely gets old, so this gate is NOT "wait
                                     # for the level to establish", it's selecting a rare tail — that
                                     # reframing stands. (2) Round 1 found forward-resolved reject rate
                                     # DECREASING monotonically with age and read that as the gate
                                     # keeping the worse bucket. Round 2 tested the planning session's
                                     # better construct — did an OPPOSITE-side pivot confirm since the
                                     # boundary was set (a genuine "round trip", causal via core.pivots
                                     # .conf) — and found the round-1 decay was COMPOSITION, not age:
                                     # every single age-0 touch is a non-round-trip by construction (no
                                     # opposite pivot CAN confirm in an empty interval), those touches
                                     # are still inside the leg that created the level, and they alone
                                     # explain nearly all of round 1's apparent decay. Conditioned on
                                     # round-trip-completed, the age decay collapses to a much smaller
                                     # residual (~20pp -> ~6pp on ES). Net: the age gate is doing
                                     # ROUGHLY THE RIGHT THING for the WRONG STATED REASON — age
                                     # correlates with round-trip completion, but is a costly proxy for
                                     # it (age>=10 keeps only 348/4822 = ~7% of genuine round-trip
                                     # touches on ES 2h; a direct round-trip gate would keep ~14x more
                                     # of them, with comparable-or-better forward behavior). A
                                     # round-trip construct ("has an opposite-side pivot confirmed
                                     # since this boundary was set") is the stronger candidate — that's
                                     # what F-E3 (touches) now records instead of gating on. Do not
                                     # re-tune this value blind.
                                     #
                                     # VALUE CHANGE 10->5 (Amir via FeaturesStructures, 2026-07-16):
                                     # 10 was measured too restrictive on live-length windows — checked
                                     # directly against a real 303-bar Pine v5 fixture, only 2 of 20 raw
                                     # low-side penetration candidates passed age>=10 (springs 0 fired
                                     # vs Pine's 11). 5 is a JUDGEMENT CALL, NOT A VALIDATED VALUE —
                                     # same unvalidated status as 10 was, just less restrictive. Nothing
                                     # above has been re-swept at 5; Session B's round-trip analysis was
                                     # measured on TOUCHES (F-E3), not springs, and does not transfer
                                     # automatically — springs' penetration+reclaim requirement is
                                     # itself round-trip-shaped, so the age-0 circularity that dominated
                                     # touches (6059 non-round-trips, ZERO round-trips) may be far
                                     # weaker or absent here. That's untested, not assumed either way —
                                     # see round_trip/boundary_age on spring/upthrust rows (added the
                                     # same day, Amir-approved) for the mechanism to test it. THAT
                                     # recording is itself CENSORED at age>=5 (the gate still runs
                                     # before a row exists) — ages 0-4, the bucket touches showed ALL
                                     # their circularity in, remain invisible to it. Flagged, not
                                     # silently worked around — see _detect_springs_upthrusts'
                                     # docstring for the full caveat.
REF_WINDOW_BARS = 3                 # F-E3 reference-bar fix (Amir/PINE Q3.3, 2026-07-16): the
                                     # doctrine reference is the climax EPISODE, not the single bar
                                     # the extreme printed on (the heaviest bar is frequently
                                     # adjacent to the extreme, not on it — classic selling-climax
                                     # shape). ref_rvol/ref_spread/ref_vol_z = max over
                                     # [pivot_bar, min(confirm_bar, pivot_bar+REF_WINDOW_BARS)].
                                     # Matches PINE's DCEvents window size — coordinate before
                                     # changing.

# ---- bar anatomy (F-B1) ----
SPREAD_PCTILE_WINDOW = 100          # rolling lookback for bar_spread_pctile (feeds the climax gate too)

# ---- climax & divergence composites (F-C1, F-C2) ----
CLIMAX_VOL_Z = 2.0                  # bar climax gate: vol_z >= this
CLIMAX_SPREAD_PCTILE = 80           # AND spread percentile >= this, at an N-bar extreme
DIVERGENCE_WINDOW_M = 20            # rolling window for effort-result divergence count (F-C2)
DIVERGENCE_RVOL_MIN = 1.5           # F-C2 "high-rvol" bar threshold
DIVERGENCE_SPREAD_PCTILE_MAX = 40   # F-C2 "narrow-spread" bar threshold
DIVERGENCE_COUNT_THRESHOLD = 3      # F-C2 ledger event fires when the rolling count first reaches
                                     # this within the M-bar window (not specified numerically in the
                                     # brief beyond the window itself — first-pass estimate, sweepable)
CLIMAX_EXTREME_WINDOW = 100         # F-C1 "N-bar extreme" — reuses SPREAD_PCTILE_WINDOW's scale

# ---- per-leg / range volume aggregates (F-L1-L5) ----
LEG_HISTORY_WINDOW = 8              # rolling window of prior impulse legs (matches dcs_character's
                                     # existing 8-leg buffer — F-L1/F-L3 ratios compare against this)
UPDOWN_VOL_WINDOW_BARS = 20         # F-L2's always-on up-volume/down-volume ratio (works without an
                                     # alive trend, unlike the 8-leg with-trend/counter-trend version)

# ---- EMA-touch ledger (F-M1-M3) ----
DEEPENING_TOUCH_COUNT = 2           # consecutive-deepening (quality-declining) touches -> ema_touch_degraded
EMA_STRADDLE_WINDOW = 20            # F-M2: trailing bars checked for "range contains the EMA" fraction
EMA_FLATNESS_WINDOW = 100           # F-M2: rolling window for the slope-flatness percentile rank
# EMA target CONFIRMED (2026-07-15, cross-referenced against a second session's source check): the
# ledger extends dcs_character.rider()'s existing 20-period rider (mtf.char.rd) — Pine erLen=20,
# dcs_character default length=20, live config.ER_EMA_LEN=20, AND emarider.py's own ER_EMA_LEN=20 all
# agree. Do NOT reimplement the rider state machine — read mtf.char.rd directly (already computed,
# already tested). Do NOT use emarider.py here — that's the ST-lab/live-app's separate port with a
# documented SAVED-vs-BREAK ordering divergence from dcs_character.rider() (see dcs_character.py's
# module docstring); this module stays internally consistent with the dcs_* engine it feeds.

# ---- per-swing price geometry (dcs_swing_features.py) ----
SWING_RATIO_EPS_ATR_FRAC = 0.05      # convexity/half-ratio/MAE denominators (a leg's net move) below
                                      # this x ATR are treated as "too small to ratio meaningfully" and
                                      # return NaN instead of an exploding ratio — a fixed 1e-10 price-
                                      # unit guard is meaningless once prices are in the thousands (NQ)

# ---- volume normalization (F-B2) ----
# rvol window is INTENTIONALLY kept separate per side, not unified (Amir, G1 2026-07-15):
#   - Python: config.RVOL_WINDOW (existing lab convention) — dcs_volume.py imports it directly,
#     NOT redefined here.
#   - Pine v4/v5: rvol SMA window, currently 20 (`dc_pip_structure_v4.pine`) — documented here only;
#     Pine itself reads its own `input.int`, this is not wired to Pine code.
RVOL_WINDOW_PINE_DOCUMENTED = 20
VOL_Z_TRAILING_DAYS = 20            # trailing days for the minute-of-session-matched median baseline
                                     # (F-B2); falls back to a rolling SMA where session mapping is
                                     # unavailable (non-Databento feeds)

# ---- rvol RESPEC (Amir addendum, 2026-07-16 — supersedes the plain-rolling-mean vol_rvol) ----
# baseline(slot) = MEDIAN (not mean — one FOMC spike poisons a mean for two weeks) over the trailing
# RVOL_BASELINE_TRAILING_N same-slot occurrences, pooled across the RVOL_KERNEL_HALF_WIDTH nearest
# neighboring slots each side (kernel smoothing — approximates a fitted diurnal curve where slots are
# thin: overnight Globex, half-days) — reuses VOL_Z_TRAILING_DAYS as the "~20 sessions" count, same
# number the spec asks for both places.
RVOL_BASELINE_TRAILING_N = VOL_Z_TRAILING_DAYS
RVOL_KERNEL_HALF_WIDTH = 2           # neighboring slots pooled each side of the bar's own slot
RVOL_WINSORIZE_PCTILE = 1.0          # winsorize vol_rvol/vol_log_rvol at the [p, 100-p] percentile
                                      # band for the feature matrix — first-pass, sweepable
RVOL_ROLL_WINDOW_TRADING_DAYS = 8    # ES/NQ quarterly-contract roll window: trading days before each
                                      # Mar/Jun/Sep/Dec 3rd-Friday expiration excluded from baseline
                                      # population — "matters more than the 14-vs-20-day choice" per
                                      # Amir; volume migrates between contracts in this window
# Scheduled-event handling has TWO sides, not one (Amir, explicit): baseline-side EXCLUDE (these
# session-days never CONTRIBUTE to any slot's trailing baseline, though bars ON them still get a
# computed rvol using the surviving population) vs. feature-side KEEP-COARSE (a separate
# vol_event_type column, never a per-event-type volume model — no evidence base, pure overfit risk).
# Scoped to the addendum's NAMED high-impact releases only (FOMC/CPI/PPI/EMPLOYMENT) — JOLTS/ISM are
# in the calendar but not named for exclusion, left in the baseline population on purpose.
RVOL_BASELINE_EXCLUDE_RELEASES = ("FOMC", "CPI", "PPI", "EMPLOYMENT")
RVOL_EVENT_TYPE_MAP = {"FOMC": "rate_decision", "CPI": "inflation_print", "PPI": "inflation_print",
                        "EMPLOYMENT": "employment"}   # opex is computed separately (calendar rule,
                                                       # not in econ_calendar.parquet)

# ---- EMA rider ride_active gate (Amir addendum §6, 2026-07-16) ----
MIN_RIDE_BARS = 10                  # signed rider streak must reach this before ride_active gates
                                     # open — during ranges price crosses the EMA constantly and the
                                     # arm/save/deep counters accumulate garbage; plausible range
                                     # 8-15 per the addendum, sweepable per timeframe (config here is
                                     # the shared default; per-TF overrides are a lab sweep, not yet
                                     # built — flagged, not silently assumed one-size-fits-all)
