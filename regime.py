"""Regime context for the setup detectors, built on trendlab's segment_chart.

Replaces the old weak trend proxies (flat-base two-point gain, higher-low's
"SMA50 rising over 5 bars") with real up/down/range segmentation. Segments are
computed once per frame and cached on the frame's .attrs, so several gated
detectors hitting the same ticker share one segment_chart call.

Causality: callers pass a frame already truncated at the as-of bar, so
segment_chart sees no future data — its labels are valid point-in-time.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import config
import trendlab


@dataclass
class TrendCtx:
    direction: str | None    # 'up' | 'down' | None (no trend segment found)
    net_pct: float
    clarity: float
    is_pole: bool
    start: int = -1          # bar positions of the matched segment (for chart shading)
    end: int = -1

    @property
    def is_uptrend(self) -> bool:
        return self.direction == "up"

    @property
    def strong_uptrend(self) -> bool:
        return (self.is_uptrend
                and self.net_pct >= config.REGIME_PRIOR_GAIN
                and self.clarity >= config.REGIME_PRIOR_CLARITY)


_NONE = TrendCtx(None, 0.0, 0.0, False)


def _segments(d: pd.DataFrame):
    """Segments for the frame. trendlab memoizes the full pivot+structure pass on
    d.attrs, so this (and the setups' swing_lows) share one computation per frame."""
    return trendlab.segment_chart(d)[0]


def _ctx(seg) -> TrendCtx:
    seg.ensure()                                   # lazy trendlab metrics — compute on access
    return TrendCtx(seg.kind, seg.net_pct, seg.clarity, "pole" in seg.tags,
                    seg.start, seg.end)


def prior_trend_context(d: pd.DataFrame, ref_pos: int) -> TrendCtx:
    """Most recent up/down segment that began before bar position ref_pos
    (i.e. the trend leading INTO a base/flag starting at ref_pos)."""
    for seg in reversed(_segments(d)):
        if seg.start < ref_pos and seg.kind in ("up", "down"):
            return _ctx(seg)
    return _NONE


def current_regime(d: pd.DataFrame) -> TrendCtx:
    """Most recent up/down segment in the frame — ignores a trailing
    range/transition so a shallow pullback tail doesn't mask the uptrend."""
    for seg in reversed(_segments(d)):
        if seg.kind in ("up", "down"):
            return _ctx(seg)
    return _NONE


def prior_uptrend(d: pd.DataFrame, ref_pos: int) -> bool:
    return prior_trend_context(d, ref_pos).strong_uptrend
