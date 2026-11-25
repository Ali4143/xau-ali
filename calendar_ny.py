"""DST-aware America/New_York (17:00 NY rollover) market calendar for XAUUSD (24/5)."""
from __future__ import annotations

import json
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import config

NY_ZONE = ZoneInfo(config.NY_TZ)


def to_ny(index_utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index_utc.tz is None:
        index_utc = index_utc.tz_localize("UTC")
    return index_utc.tz_convert(NY_ZONE)


def forex_trading_day(index_utc: pd.DatetimeIndex) -> pd.Series:
    """Map each UTC bar to its Forex/Gold trading date using 17:00 America/New_York boundary.

    Bars at or after 17:00 NY belong to the NEXT calendar date's forex day.
    DST transitions are handled automatically by zoneinfo.
    """
    ny = to_ny(index_utc)
    # Shift forward by (24 - 17) = 7 hours so 17:00 NY becomes 00:00 of the next date
    shifted = ny + pd.Timedelta(hours=24 - config.FOREX_DAY_ROLLOVER_HOUR_NY)
    return pd.Series(shifted.date, index=index_utc, name="forex_day")


def detect_calendar_from_data(df_m1: pd.DataFrame, save: bool = True) -> dict:
    """Derive weekend closes and daily maintenance gaps strictly from observed 1M data."""
    if df_m1.empty or len(df_m1) < 2:
        return {"total_bars": len(df_m1), "gap_count": 0, "largest_gap_minutes": 0.0}

    diffs = df_m1.index.to_series().diff().dropna()
    gap_mins = diffs.dt.total_seconds() / 60.0
    big_gaps = gap_mins[gap_mins >= config.MAINTENANCE_GAP_MINUTES]
    weekend_gaps = big_gaps[big_gaps >= 24 * 60]
    maint_gaps = big_gaps[(big_gaps >= config.MAINTENANCE_GAP_MINUTES) & (big_gaps < 24 * 60)]

    cal_info = {
        "timezone_internal": "UTC",
        "forex_day_boundary": f"{config.FOREX_DAY_ROLLOVER_HOUR_NY:02d}:00 {config.NY_TZ}",
        "total_bars": int(len(df_m1)),
        "start_utc": str(df_m1.index.min()),
        "end_utc": str(df_m1.index.max()),
        "total_gaps_ge_45m": int(len(big_gaps)),
        "weekend_gaps_count": int(len(weekend_gaps)),
        "maintenance_gaps_count": int(len(maint_gaps)),
        "largest_gap_minutes": round(float(gap_mins.max()), 2),
        "median_maintenance_gap_minutes": round(float(maint_gaps.median()), 2) if len(maint_gaps) else 0.0,
        "missing_1m_bars_excl_gaps": int((gap_mins[(gap_mins > 1.5) & (gap_mins < config.MAINTENANCE_GAP_MINUTES)] - 1).sum()),
    }
    if save:
        config.CALENDAR_FILE.write_text(json.dumps(cal_info, indent=2), encoding="utf-8")
    return cal_info


def gap_mask_before(index_utc: pd.DatetimeIndex) -> np.ndarray:
    """Boolean array of length N: True at position i if a market gap (>= 45m) occurred right before bar i."""
    if len(index_utc) == 0:
        return np.array([], dtype=bool)
    diffs = np.diff(index_utc.asi8) / 1e9 / 60.0  # minutes
    mask = np.zeros(len(index_utc), dtype=bool)
    mask[1:] = diffs >= config.MAINTENANCE_GAP_MINUTES
    return mask


def trading_horizon_mask(
    index_utc: pd.DatetimeIndex,
    origin_pos: int,
    max_h: int = config.HORIZON_BARS,
) -> tuple[np.ndarray, int, int]:
    """Return valid trading bar positions after origin_pos without crossing a closed-market gap.

    Returns:
      - positions: int array of length <= max_h
      - valid_len: number of continuous trading bars before hitting a gap or data end
      - eos_step: 1-based step index of the first 17:00 NY boundary (or valid_len if earlier)
    """
    n = len(index_utc)
    if origin_pos >= n - 1:
        return np.array([], dtype=int), 0, 0

    fday = forex_trading_day(index_utc).to_numpy()
    origin_fday = fday[origin_pos]
    gmask = gap_mask_before(index_utc)

    end_limit = min(n, origin_pos + 1 + max_h)
    valid = []
    eos_step = 0
    for pos in range(origin_pos + 1, end_limit):
        if gmask[pos]:
            # Stop at closed market / maintenance gap (never cross or forward-fill across closed market)
            break
        valid.append(pos)
        if eos_step == 0 and fday[pos] == origin_fday:
            pass
        elif eos_step == 0 and fday[pos] != origin_fday:
            eos_step = len(valid) - 1

    valid_len = len(valid)
    if eos_step <= 0:
        eos_step = valid_len
    return np.array(valid, dtype=int), valid_len, eos_step


def session_flags(index_utc: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute session flags and cyclical UTC/DOW encodings (flags, not filters)."""
    utc = index_utc.tz_convert("UTC") if index_utc.tz is not None else index_utc.tz_localize("UTC")
    hr = utc.hour + utc.minute / 60.0
    dow = utc.dayofweek.astype(float)

    df = pd.DataFrame(index=index_utc)
    df["sess_asia"] = ((hr >= 0.0) & (hr < 9.0)).astype(np.float32)
    df["sess_london"] = ((hr >= 7.0) & (hr < 16.0)).astype(np.float32)
    df["sess_ny"] = ((hr >= 12.0) & (hr < 21.0)).astype(np.float32)
    df["sess_overlap"] = ((hr >= 12.0) & (hr < 16.0)).astype(np.float32)
    df["kz_london"] = ((hr >= 7.0) & (hr < 9.0)).astype(np.float32)
    df["kz_ny"] = ((hr >= 12.5) & (hr < 14.5)).astype(np.float32)
    df["utc_sin"] = np.sin(2.0 * np.pi * hr / 24.0).astype(np.float32)
    df["utc_cos"] = np.cos(2.0 * np.pi * hr / 24.0).astype(np.float32)
    df["dow_sin"] = np.sin(2.0 * np.pi * dow / 5.0).astype(np.float32)
    df["dow_cos"] = np.cos(2.0 * np.pi * dow / 5.0).astype(np.float32)
    return df
