"""Institutional Trading Calendar & Mechanical Volume Anomaly Engine.

Detects scheduled exchange calendar events that cause artificial, mechanical volume spikes
unrelated to fundamental catalyst accumulation:
1. Quad Witching: Third Friday of March, June, September, and December (index futures,
   index options, stock options, and single-stock futures expire simultaneously).
2. Standard Monthly Options Expiration: Third Friday of every month (or preceding Thursday
   if Friday is a market holiday, e.g. Good Friday).
3. Index Reconstitution Days: Annual Russell Reconstitution (fourth Friday of June).
4. Quarter-End & Month-End: Final trading day of each quarter/month (institutional window
   dressing and portfolio rebalancing closing crosses).
"""

from __future__ import annotations

import datetime
from pathlib import Path
import pandas as pd

# Path to SPY daily bars to ground the exact exchange trading sessions
ROOT = Path(__file__).resolve().parent
SPY_PATH = ROOT / "data" / "bars" / "SPY.parquet"

_TRADING_DAYS_SET: set[str] | None = None
_MONTH_ENDS_SET: set[str] | None = None
_QUARTER_ENDS_SET: set[str] | None = None
_OPEX_SET: set[str] | None = None
_QUAD_WITCHING_SET: set[str] | None = None
_RUSSELL_RECON_SET: set[str] | None = None


def _load_calendar_cache():
    global _TRADING_DAYS_SET, _MONTH_ENDS_SET, _QUARTER_ENDS_SET
    global _OPEX_SET, _QUAD_WITCHING_SET, _RUSSELL_RECON_SET

    if _TRADING_DAYS_SET is not None:
        return

    trading_days = []
    if SPY_PATH.exists():
        try:
            spy = pd.read_parquet(SPY_PATH)
            trading_days = list(pd.to_datetime(spy.index).tz_localize(None).sort_values())
        except Exception:
            trading_days = []

    # Supplement / extend trading days with official US equity exchange calendar up to 2035
    from pandas.tseries.holiday import (
        AbstractHolidayCalendar, Holiday, nearest_workday,
        USMartinLutherKingJr, USPresidentsDay, USMemorialDay,
        USLaborDay, USThanksgivingDay, GoodFriday
    )

    class USTradingCalendar(AbstractHolidayCalendar):
        rules = [
            Holiday('New Years Day', month=1, day=1, observance=nearest_workday),
            USMartinLutherKingJr,
            USPresidentsDay,
            GoodFriday,
            USMemorialDay,
            Holiday('Juneteenth', month=6, day=19, start_date='2022-01-01', observance=nearest_workday),
            Holiday('July 4th', month=7, day=4, observance=nearest_workday),
            USLaborDay,
            USThanksgivingDay,
            Holiday('Christmas', month=12, day=25, observance=nearest_workday),
        ]

    cal = USTradingCalendar()
    start_date = "2015-01-01"
    if trading_days:
        last_spy_dt = trading_days[-1]
        start_date = (last_spy_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    b_days = pd.bdate_range(start=start_date, end="2035-12-31")
    holidays = cal.holidays(start=start_date, end="2035-12-31")
    holidays_set = set(holidays)
    future_trading_days = [d for d in b_days if d not in holidays_set]
    trading_days.extend(future_trading_days)

    td_series = pd.Series(trading_days).drop_duplicates().sort_values()
    _TRADING_DAYS_SET = set(td_series.dt.strftime("%Y-%m-%d"))

    # Month-ends and quarter-ends
    me_series = td_series.groupby([td_series.dt.year, td_series.dt.month]).last()
    _MONTH_ENDS_SET = set(me_series.dt.strftime("%Y-%m-%d"))

    qe_series = td_series[td_series.dt.month.isin([3, 6, 9, 12])].groupby([td_series.dt.year, td_series.dt.month]).last()
    _QUARTER_ENDS_SET = set(qe_series.dt.strftime("%Y-%m-%d"))

    # Options Expirations and Quad Witching (2015 - 2030)
    _OPEX_SET = set()
    _QUAD_WITCHING_SET = set()

    years = sorted(list(set(td_series.dt.year)))
    for y in years:
        for m in range(1, 13):
            # 3rd Friday falls between day 15 and day 21 inclusive
            for d in range(15, 22):
                dt = pd.Timestamp(y, m, d)
                if dt.weekday() == 4:  # Friday
                    dt_str = dt.strftime("%Y-%m-%d")
                    if dt_str in _TRADING_DAYS_SET:
                        exp_str = dt_str
                    else:
                        # Market closed on 3rd Friday -> move to preceding Thursday
                        exp_str = (dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

                    _OPEX_SET.add(exp_str)
                    if m in [3, 6, 9, 12]:
                        _QUAD_WITCHING_SET.add(exp_str)
                    break

    # Russell Reconstitution: 4th Friday of June
    _RUSSELL_RECON_SET = set()
    for y in years:
        # 4th Friday of June falls between June 22 and June 28 inclusive
        for d in range(22, 29):
            dt = pd.Timestamp(y, 6, d)
            if dt.weekday() == 4:
                dt_str = dt.strftime("%Y-%m-%d")
                if dt_str in _TRADING_DAYS_SET:
                    recon_str = dt_str
                else:
                    recon_str = (dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                _RUSSELL_RECON_SET.add(recon_str)
                break


def _to_str(dt: str | datetime.date | pd.Timestamp) -> str:
    if isinstance(dt, (datetime.date, pd.Timestamp)):
        return dt.strftime("%Y-%m-%d")
    return str(dt)[:10]


def is_trading_day(dt: str | datetime.date | pd.Timestamp) -> bool:
    """Returns True if the date is an exchange trading day (present in _TRADING_DAYS_SET)."""
    _load_calendar_cache()
    assert _TRADING_DAYS_SET is not None
    return _to_str(dt) in _TRADING_DAYS_SET


def is_monthly_opex(dt: str | datetime.date | pd.Timestamp) -> bool:
    """Returns True if the date is a Standard Monthly Options Expiration day (3rd Friday or holiday-shifted Thursday)."""
    _load_calendar_cache()
    return _to_str(dt) in _OPEX_SET


def is_quad_witching(dt: str | datetime.date | pd.Timestamp) -> bool:
    """Returns True if the date is Quadruple Witching (3rd Friday of March, June, September, December)."""
    _load_calendar_cache()
    return _to_str(dt) in _QUAD_WITCHING_SET


def is_russell_reconstitution(dt: str | datetime.date | pd.Timestamp) -> bool:
    """Returns True if the date is the Annual Russell Reconstitution (4th Friday of June)."""
    _load_calendar_cache()
    return _to_str(dt) in _RUSSELL_RECON_SET


def is_month_end(dt: str | datetime.date | pd.Timestamp) -> bool:
    """Returns True if the date is the final exchange trading session of the calendar month."""
    _load_calendar_cache()
    return _to_str(dt) in _MONTH_ENDS_SET


def is_quarter_end(dt: str | datetime.date | pd.Timestamp) -> bool:
    """Returns True if the date is the final exchange trading session of Q1, Q2, Q3, or Q4."""
    _load_calendar_cache()
    return _to_str(dt) in _QUARTER_ENDS_SET


def is_mechanical_volume_date(dt: str | datetime.date | pd.Timestamp) -> bool:
    """Returns True if the date belongs to ANY mechanical volume anomaly category."""
    _load_calendar_cache()
    d_str = _to_str(dt)
    return (
        d_str in _OPEX_SET or
        d_str in _RUSSELL_RECON_SET or
        d_str in _MONTH_ENDS_SET
    )


def get_calendar_anomaly_type(dt: str | datetime.date | pd.Timestamp) -> str | None:
    """Returns human-readable description of the mechanical calendar event, or None."""
    _load_calendar_cache()
    d_str = _to_str(dt)
    reasons = []
    if d_str in _QUAD_WITCHING_SET:
        reasons.append("Quad Witching")
    elif d_str in _OPEX_SET:
        reasons.append("Monthly OpEx")

    if d_str in _RUSSELL_RECON_SET:
        reasons.append("Russell Reconstitution")

    if d_str in _QUARTER_ENDS_SET:
        reasons.append("Quarter-End")
    elif d_str in _MONTH_ENDS_SET:
        reasons.append("Month-End")

    return " & ".join(reasons) if reasons else None


def annotate_calendar_events(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Annotates DataFrame with boolean indicator columns for all mechanical calendar events."""
    _load_calendar_cache()
    df = df.copy()
    dates = df[date_col].astype(str).str[:10]
    df["is_opex"] = dates.isin(_OPEX_SET)
    df["is_quad_witching"] = dates.isin(_QUAD_WITCHING_SET)
    df["is_russell_recon"] = dates.isin(_RUSSELL_RECON_SET)
    df["is_month_end"] = dates.isin(_MONTH_ENDS_SET)
    df["is_quarter_end"] = dates.isin(_QUARTER_ENDS_SET)
    df["is_mechanical_volume_date"] = df["is_opex"] | df["is_russell_recon"] | df["is_month_end"]
    df["calendar_anomaly"] = [get_calendar_anomaly_type(d) for d in dates]
    return df
