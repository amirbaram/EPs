"""dcs_mtf — real parent-timeframe structure stacks (plan P3).

Runs the dcs_core + dcs_character engines on each parent frame from dcs_data.parents() and aligns
parent per-bar state onto the child timeline. Unlike Pine's request.security, parent history here is
UNLIMITED (daily/weekly stores) — a designed improvement over the chart-span limit proven live.

TWO alignments, chosen per use:
- CAUSAL (default, for research/events): child bar i sees the state of the last parent bar whose
  CLOSE time <= the child bar's open time — the same availability rule as Pine's lookahead_off on
  historical bars, and the only alignment allowed to feed the events ledger / outcome studies
  (point-in-time discipline; see the intraday look-ahead lessons in this repo).
- DEVELOPING (display only, mirrors the Pine lookahead_on table): child bar i sees the state of the
  parent bar CONTAINING it, as computed at that parent bar's close — leaks the rest of the parent
  bar by construction; provisional exactly like the live chart's developing parent row.

Bar-close conventions handled per frame type: intraday parents are START-labeled (closed when the
next parent bar starts); 1D is date-labeled (closed at the next midnight); 1W/1M are END-labeled by
datastore (W-FRI / period end; closed after that date ends).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import dcs_character
import dcs_core
import dcs_data

END_LABELED = {"1W", "1M", "2D", "3D"}


def _ns(index: pd.DatetimeIndex) -> np.ndarray:
    """int64 NANOSECONDS regardless of the index's storage unit (pandas 3 builds string-constructed
    indexes at microsecond resolution; mixing units silently breaks searchsorted alignment)."""
    return index.as_unit("ns").asi8

# relationship codes (child kind vs nearest defined parent kind)
REL_NONE, REL_ALIGNED, REL_PULLBACK_IN, REL_BOUNCE_IN, REL_RANGING_IN, REL_LOCAL = 0, 1, 2, 3, 4, 5
REL_TXT = {REL_NONE: "", REL_ALIGNED: "aligned with the parent trend",
           REL_PULLBACK_IN: "pullback inside the parent uptrend",
           REL_BOUNCE_IN: "bounce inside the parent downtrend",
           REL_RANGING_IN: "ranging inside the parent trend",
           REL_LOCAL: "parent has no trend — the move is local"}


def _closed_times(index: pd.DatetimeIndex, tf: str) -> np.ndarray:
    """When each parent bar's state becomes KNOWABLE (int64 ns)."""
    t = _ns(index)
    if tf == "1D" or tf in END_LABELED:
        return _ns((index + pd.Timedelta(days=1)).normalize())
    out = np.empty(len(t), dtype=np.int64)
    out[:-1] = t[1:]
    out[-1] = np.iinfo(np.int64).max          # last bar: never closed within this dataset
    return out


def _containing(index: pd.DatetimeIndex, tf: str, child_t: np.ndarray) -> np.ndarray:
    """Index of the parent bar containing each child time (-1 before the first)."""
    if tf in END_LABELED:
        # end-labeled: containing period = first label >= child date
        idx = _ns(index).searchsorted(child_t, side="left")
        idx[idx >= len(index)] = len(index) - 1
        return idx
    return np.maximum(_ns(index).searchsorted(child_t, side="right") - 1, -1)


@dataclass
class ParentView:
    tf: str
    tfr: dcs_data.TFFrame
    core: dcs_core.CoreState
    char: dcs_character.CharState
    map_causal: np.ndarray      # child bar i -> parent bar idx (last CLOSED; -1 none)
    map_dev: np.ndarray         # child bar i -> containing parent bar idx (developing view)

    def state(self, i: int, developing: bool = False) -> dict | None:
        k = int((self.map_dev if developing else self.map_causal)[i])
        if k < 0:
            return None
        c = self.core
        return {"tf": self.tf, "bar": k, "time": c.index[k],
                "regime": c.regime[k], "dev_regime": c.dev_regime[k],
                "kind": int(c.kind[k]), "legs": int(c.legs_disp[k]),
                "dev_swing": int(c.dev_swing[k]), "prot": float(c.prot[k]),
                "origin_price": float(c.origin_price[k]), "len_atr": float(c.len_atr[k]),
                "character": self.char.label[k], "clim": bool(self.char.clim[k]),
                "atr": float(c.atr[k])}


@dataclass
class MTFState:
    symbol: str
    tf: str
    child: dcs_data.TFFrame
    core: dcs_core.CoreState
    char: dcs_character.CharState
    parents: list = field(default_factory=list)   # list[ParentView]
    rel: np.ndarray = None                        # causal relationship code per child bar

    def snapshot(self, i: int, developing: bool = False) -> dict:
        """One events-ledger row: child state + every parent state at child bar i."""
        c = self.core
        row = {"symbol": self.symbol, "tf": self.tf, "time": c.index[i],
               "regime": c.regime[i], "kind": int(c.kind[i]), "legs": int(c.legs_disp[i]),
               "dev_swing": int(c.dev_swing[i]), "prot": float(c.prot[i]),
               "character": self.char.label[i], "rel": int(self.rel[i])}
        for pv in self.parents:
            ps = pv.state(i, developing=developing)
            key = f"p_{pv.tf}"
            row[key] = None if ps is None else {k: ps[k] for k in
                                                ("regime", "kind", "legs", "dev_swing", "character")}
        return row


def relationship(child_kind: int, parent_kind: int) -> int:
    if parent_kind == 1:
        return REL_PULLBACK_IN if child_kind == -1 else REL_ALIGNED if child_kind == 1 else REL_RANGING_IN
    if parent_kind == -1:
        return REL_BOUNCE_IN if child_kind == 1 else REL_ALIGNED if child_kind == -1 else REL_RANGING_IN
    return REL_LOCAL


def stack(symbol: str, tf: str = "5m", n_parents: int = 3, mult: float = 1.0,
          eq_frac: float = 0.0, pine_compat: bool = False) -> MTFState | None:
    """Child + parents, engines run, alignments built. None when the child frame is missing;
    parents missing from the store are skipped (logged in .parents order)."""
    child = dcs_data.frame(symbol, tf)
    if child is None:
        return None
    core = dcs_core.run(child, mult=mult, eq_frac=eq_frac, pine_compat=pine_compat)
    char = dcs_character.run(child, core, pine_compat=pine_compat)
    st = MTFState(symbol=symbol, tf=tf, child=child, core=core, char=char)
    child_t = _ns(child.df.index)
    for ptf in dcs_data.parents(tf, child.futures, n=n_parents):
        ptfr = dcs_data.frame(symbol, ptf)
        if ptfr is None or len(ptfr.df) < dcs_core.ATR_LEN + 2:
            continue
        pcore = dcs_core.run(ptfr, mult=mult, eq_frac=eq_frac, pine_compat=pine_compat)
        pchar = dcs_character.run(ptfr, pcore, pine_compat=pine_compat)
        closed = _closed_times(ptfr.df.index, ptf)
        map_causal = closed.searchsorted(child_t, side="right") - 1
        map_dev = _containing(ptfr.df.index, ptf, child_t)
        st.parents.append(ParentView(ptf, ptfr, pcore, pchar,
                                     map_causal.astype(np.int64), map_dev.astype(np.int64)))
    # causal relationship vs the nearest DEFINED parent (first parent with an alive trend, else first)
    n = core.n
    rel = np.zeros(n, np.int8)
    if st.parents:
        for i in range(n):
            code = REL_NONE
            for pv in st.parents:
                k = int(pv.map_causal[i])
                if k < 0:
                    continue
                pk = int(pv.core.kind[k])
                code = relationship(int(core.kind[i]), pk)
                if pk != 0:
                    break                      # nearest trending parent wins
            rel[i] = code
    st.rel = rel
    return st
