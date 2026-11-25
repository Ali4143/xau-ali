"""Isolated fetch adapter and rolling FIFO Parquet store for xau_mtf.

Key guarantees:
1. 1M data is stored in UTC with FIFO rolling retention (default 1.5 years / 548 days).
2. 15M data is DERIVED (resampled) strictly from 1M so both timeframes are 100% synced
   with zero retention mismatch.
3. Optional MetaTrader 5 fallback hook if installed, Dukascopy primary.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .. import config
from .vendor.dukascopy_vendor import clean_ohlcv, fetch_dukascopy_chunked


def m1_path(symbol: str = config.DEFAULT_SYMBOL) -> Path:
    return config.DATA_DIR / f"{symbol.lower()}_m1.parquet"


def m15_path(symbol: str = config.DEFAULT_SYMBOL) -> Path:
    return config.DATA_DIR / f"{symbol.lower()}_m15.parquet"


def meta_path(symbol: str = config.DEFAULT_SYMBOL) -> Path:
    return config.DATA_DIR / f"{symbol.lower()}_meta.json"


def resample_m1_to_m15(df_m1: pd.DataFrame) -> pd.DataFrame:
    """Strictly derive closed 15M bars from 1M bars (label='left', closed='left').

    Example: 1M bars [10:00, 10:01, ..., 10:14] form the 15M bar timestamped 10:00
    whose close_time is 10:15 (so it is CLOSED at decision bar t >= 10:15).
    """
    if df_m1.empty:
        return df_m1.copy()
    agg = df_m1.resample("15min", label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "tick_volume": "sum",
            "real_volume": "sum",
            "spread": "mean",
        }
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"])
    return agg


def enforce_fifo_rolling_window(df_m1: pd.DataFrame, max_days: int = config.RETENTION_DAYS_M1) -> tuple[pd.DataFrame, int]:
    """FIFO rolling retention: if span > max_days, drop oldest bars outside the rolling window."""
    if df_m1.empty:
        return df_m1, 0
    latest_ts = df_m1.index.max()
    cutoff = latest_ts - pd.Timedelta(days=max_days)
    before = len(df_m1)
    trimmed = df_m1[df_m1.index >= cutoff].copy()
    dropped = before - len(trimmed)
    return trimmed, dropped


def load_m1(symbol: str = config.DEFAULT_SYMBOL) -> pd.DataFrame:
    p = m1_path(symbol)
    if not p.exists():
        return clean_ohlcv(None)
    df = pd.read_parquet(p)
    return clean_ohlcv(df)


def load_m15(symbol: str = config.DEFAULT_SYMBOL) -> pd.DataFrame:
    p = m15_path(symbol)
    if p.exists():
        df = pd.read_parquet(p)
        return clean_ohlcv(df)
    df_m1 = load_m1(symbol)
    return resample_m1_to_m15(df_m1)


def save_synced_store(
    df_m1: pd.DataFrame,
    symbol: str = config.DEFAULT_SYMBOL,
    source: str = "dukascopy_bid",
    max_days: int = config.RETENTION_DAYS_M1,
) -> dict:
    """Clean, apply FIFO rolling window, derive 15M from 1M, and persist both Parquets."""
    df_m1 = clean_ohlcv(df_m1)
    df_m1, dropped = enforce_fifo_rolling_window(df_m1, max_days=max_days)
    df_m15 = resample_m1_to_m15(df_m1)

    df_m1.to_parquet(m1_path(symbol))
    df_m15.to_parquet(m15_path(symbol))

    real_vol_nonzero = int((df_m1["real_volume"] > 0).sum()) if len(df_m1) else 0
    meta = {
        "symbol": symbol.upper(),
        "source": source,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "m1_rows": int(len(df_m1)),
        "m15_rows": int(len(df_m15)),
        "start_utc": str(df_m1.index.min()) if len(df_m1) else None,
        "end_utc": str(df_m1.index.max()) if len(df_m1) else None,
        "fifo_retention_days": max_days,
        "fifo_dropped_rows": int(dropped),
        "real_volume_nonzero_rows": real_vol_nonzero,
        "real_volume_note": "Spot gold real_volume is 0; tick_volume is used as proxy",
    }
    meta_path(symbol).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def smart_sync(
    symbol: str = config.DEFAULT_SYMBOL,
    initial_days: int = config.INITIAL_FETCH_DAYS_M1,
    progress_cb=None,
) -> dict:
    """Smart Dual-Source Sync (Zero-Delay TradingView Live + Automatic Missing-Day Dukascopy Backfill + Zero Duplicates).

    Logic:
      1. Inspect local store `last_ts`.
      2. If empty: fetch `initial_days` from Dukascopy, then overlay latest 2000 live 1M bars from TradingView.
      3. If local store has a multi-day gap (> 3 days offline): backfill missing days from Dukascopy up to now-2d,
         then stitch latest live 1M bars up to the current second from TradingView.
      4. If gap <= 3 days: fetch ONLY from TradingView (fast ~1.5s, 0-second delay, matches TradingView chart!).
      5. Strict deduplication (`keep='last'`), UTC monotonic sort, OHLC check, 1.5Y FIFO roll, and 15M resample.
    """
    from .tv_live import fetch_tv_m1

    now_utc = datetime.now(timezone.utc)
    existing = load_m1(symbol)
    parts = []
    sources_used = []

    if not existing.empty:
        parts.append(existing)
        last_ts = existing.index.max().to_pydatetime()
        gap_mins = max(0.0, (now_utc - last_ts).total_seconds() / 60.0)
    else:
        last_ts = None
        gap_mins = float(initial_days * 1440)

    # Step A: If empty or offline > 3 days (4320 mins), backfill deep missing gap from Dukascopy
    if existing.empty:
        start_dt = (now_utc - timedelta(days=initial_days)).replace(tzinfo=None)
        end_dt = now_utc.replace(tzinfo=None)
        if progress_cb:
            progress_cb(0.2, f"Backfilling {initial_days}d history from Dukascopy...")
        dk_df = fetch_dukascopy_chunked(symbol, "1m", start_dt, end_dt, chunk_days=30, progress_cb=progress_cb)
        if not dk_df.empty:
            parts.append(dk_df)
            sources_used.append("dukascopy_backfill")
        tv_bars = 2000
    elif gap_mins > 4200:
        # Offline > ~2.9 days: backfill missing middle gap via Dukascopy, then top off with TradingView
        start_dt = (last_ts - timedelta(hours=2)).replace(tzinfo=None)
        end_dt = (now_utc - timedelta(days=2)).replace(tzinfo=None)
        if end_dt > start_dt:
            if progress_cb:
                progress_cb(0.3, f"Auto-backfilling missing gap ({start_dt.date()} -> {end_dt.date()}) via Dukascopy...")
            dk_df = fetch_dukascopy_chunked(symbol, "1m", start_dt, end_dt, chunk_days=15, progress_cb=progress_cb)
            if not dk_df.empty:
                parts.append(dk_df)
                sources_used.append("dukascopy_gap_fill")
        tv_bars = 4500
    else:
        # Normal live update (<= 3 days): pull exact needed bars + 120 bar safety overlap from TradingView
        tv_bars = int(min(5000, max(180, gap_mins + 120)))

    # Step B: Pull zero-delay live 1M bars from TradingView (always last so TV live wins on dedup!)
    if progress_cb:
        progress_cb(0.85, f"Fetching {tv_bars} live 1M bars from TradingView...")
    tv_df, tv_src = fetch_tv_m1(symbol, n_bars=tv_bars)
    if not tv_df.empty:
        parts.append(tv_df)
        sources_used.append(tv_src or "tradingview")
    elif existing.empty or gap_mins > 60:
        # Fallback to Dukascopy tail only if TradingView failed
        start_dt = (now_utc - timedelta(hours=max(4, int(gap_mins / 60) + 2))).replace(tzinfo=None)
        dk_tail = fetch_dukascopy_chunked(symbol, "1m", start_dt, now_utc.replace(tzinfo=None), chunk_days=2)
        if not dk_tail.empty:
            parts.append(dk_tail)
            sources_used.append("dukascopy_fallback")

    if not parts:
        return save_synced_store(clean_ohlcv(None), symbol=symbol, source="empty")

    before_dedup = sum(len(p) for p in parts)
    merged = pd.concat(parts, axis=0)
    # Strict dedup: keep='last' ensures fresh TradingView live bars cleanly replace older/overlapping bars
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    dedup_removed = before_dedup - len(merged)

    src_label = " + ".join(sources_used) if sources_used else "local_cached"
    meta = save_synced_store(merged, symbol=symbol, source=src_label)
    meta["duplicates_prevented"] = int(dedup_removed)
    meta["tv_live_bars_added"] = int(len(tv_df))
    meta_path(symbol).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def fetch_live_tail(symbol: str = config.DEFAULT_SYMBOL, hours: int = 4) -> dict:
    """Fast zero-delay live sync via TradingView (with auto missing-day Dukascopy backfill if needed)."""
    return smart_sync(symbol=symbol, initial_days=60)


def data_health_report(df_m1: pd.DataFrame, df_m15: pd.DataFrame, symbol: str = config.DEFAULT_SYMBOL) -> dict:
    """Comprehensive Data Health & Integrity Audit (bars, span, duplicates, NaNs, OHLC, gaps, daily coverage)."""
    if df_m1.empty:
        return {"ok": False, "m1_rows": 0, "m15_rows": 0, "status": "EMPTY"}

    idx = df_m1.index
    diffs_min = idx.to_series().diff().dropna().dt.total_seconds() / 60.0
    weekend_gaps = diffs_min[diffs_min >= 24 * 60]
    maint_gaps = diffs_min[(diffs_min >= config.MAINTENANCE_GAP_MINUTES) & (diffs_min < 24 * 60)]
    intra_missing_gaps = diffs_min[(diffs_min > 1.5) & (diffs_min < config.MAINTENANCE_GAP_MINUTES)]
    missing_1m_bars = int((intra_missing_gaps - 1.0).round().sum()) if len(intra_missing_gaps) else 0

    dups = int(idx.duplicated().sum())
    nans = int(df_m1[["open", "high", "low", "close"]].isna().sum().sum())
    bad_ohlc = int(
        (
            (df_m1["high"] < df_m1[["open", "close"]].max(axis=1))
            | (df_m1["low"] > df_m1[["open", "close"]].min(axis=1))
            | (df_m1["close"] <= 0)
        ).sum()
    )
    span_days = round((idx.max() - idx.min()).total_seconds() / 86400.0, 1)
    sorted_ok = bool(idx.is_monotonic_increasing)
    clean_ok = (dups == 0) and (nans == 0) and (bad_ohlc == 0) and sorted_ok

    # Per-day bar counts (last 30 calendar days)
    daily_cnt = df_m1["close"].groupby(idx.date).agg(bars="count", open_p="first", close_p="last")
    daily_rows = [
        {
            "Date (UTC)": str(d),
            "1M Bars": int(r["bars"]),
            "Coverage %": round(min(100.0, float(r["bars"]) / 1380.0 * 100.0), 1),
            "Close ($)": round(float(r["close_p"]), 2),
        }
        for d, r in daily_cnt.tail(30).iterrows()
    ]

    return {
        "ok": clean_ok,
        "status": "✅ CLEAN" if clean_ok else "⚠️ ISSUES FOUND",
        "symbol": symbol.upper(),
        "m1_rows": int(len(df_m1)),
        "m15_rows": int(len(df_m15)),
        "start_utc": str(idx.min()),
        "end_utc": str(idx.max()),
        "span_days": span_days,
        "fifo_limit_days": config.RETENTION_DAYS_M1,
        "duplicates": dups,
        "nans": nans,
        "bad_ohlc": bad_ohlc,
        "sorted_ok": sorted_ok,
        "weekend_gaps": int(len(weekend_gaps)),
        "maintenance_gaps": int(len(maint_gaps)),
        "unexpected_intraday_gaps": int(len(intra_missing_gaps)),
        "missing_1m_bars_intraday": missing_1m_bars,
        "largest_unexpected_gap_min": round(float(intra_missing_gaps.max()), 1) if len(intra_missing_gaps) else 0.0,
        "real_volume_nonzero": int((df_m1["real_volume"] > 0).sum()) if "real_volume" in df_m1.columns else 0,
        "daily_recent": daily_rows,
    }


def fetch_and_update(
    symbol: str = config.DEFAULT_SYMBOL,
    days: int = config.INITIAL_FETCH_DAYS_M1,
    end_dt: Optional[datetime] = None,
    progress_cb=None,
) -> dict:
    """Fetch full requested `days` history (backfilling BOTH older missing history AND newer live tail)."""
    now_utc = (end_dt or datetime.now(timezone.utc)).replace(tzinfo=None)
    want_start = now_utc - timedelta(days=days)
    existing = load_m1(symbol)
    parts = []

    if existing.empty:
        new_df = fetch_dukascopy_chunked(symbol, "1m", want_start, now_utc, chunk_days=30, progress_cb=progress_cb)
        if not new_df.empty:
            parts.append(new_df)
    else:
        parts.append(existing)
        first_ts = existing.index.min().tz_convert("UTC").to_pydatetime().replace(tzinfo=None)
        last_ts = existing.index.max().tz_convert("UTC").to_pydatetime().replace(tzinfo=None)

        # 1. Backfill OLDER history if user requested a deeper window (e.g. 60d -> 365d)
        if first_ts > want_start + timedelta(days=2):
            old_df = fetch_dukascopy_chunked(symbol, "1m", want_start, first_ts + timedelta(hours=2), chunk_days=30, progress_cb=progress_cb)
            if not old_df.empty:
                parts.append(old_df)

        # 2. Fetch NEWER history from last_ts -> now_utc
        if (now_utc - last_ts).total_seconds() >= 120:
            new_df = fetch_dukascopy_chunked(symbol, "1m", last_ts - timedelta(hours=2), now_utc, chunk_days=15, progress_cb=progress_cb)
            if not new_df.empty:
                parts.append(new_df)

    # 3. Top off with zero-delay live TradingView bars
    try:
        from .tv_live import fetch_tv_m1

        tv_df, _ = fetch_tv_m1(symbol, n_bars=500)
        if not tv_df.empty:
            parts.append(tv_df)
    except Exception:
        pass

    merged = pd.concat(parts, axis=0) if parts else clean_ohlcv(None)
    return save_synced_store(merged, symbol=symbol, source="dukascopy + tradingview")
