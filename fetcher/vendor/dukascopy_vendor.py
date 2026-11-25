"""Standalone vendored Dukascopy fetcher (adapted cleanly from D:\\python fetcher\\app.py, version 2026-09-22).

Self-contained inside xau_mtf/fetcher/vendor/ so xau_mtf has ZERO external path dependency.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

VENDOR_SOURCE_NOTE = "Adapted from D:\\python fetcher\\app.py (dukascopy_python OFFER_SIDE_BID)"

# Standard Dukascopy string instrument IDs (no top-level dukascopy_python import required on Streamlit Cloud!)
INSTRUMENTS = {
    "XAUUSD": "XAU/USD",
    "XAGUSD": "XAG/USD",
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "AUDUSD": "AUD/USD",
    "USDCAD": "USD/CAD",
    "USDCHF": "USD/CHF",
    "NZDUSD": "NZD/USD",
    "EURJPY": "EUR/JPY",
    "GBPJPY": "GBP/JPY",
    "EURGBP": "EUR/GBP",
    "AUDJPY": "AUD/JPY",
    "EURAUD": "EUR/AUD",
}


def resolve_instrument(symbol: str) -> str:
    """Resolve known symbol or auto-format custom 6-letter forex pair (e.g. CADJPY -> CAD/JPY)."""
    sym = symbol.upper().strip()
    if sym in INSTRUMENTS:
        return INSTRUMENTS[sym]
    if "/" in sym:
        return sym
    if len(sym) == 6 and sym.isalpha():
        return f"{sym[:3]}/{sym[3:]}"
    raise ValueError(f"Unsupported symbol {sym}. Pass a 6-letter forex pair (e.g. EURJPY) or one of {list(INSTRUMENTS)}")


def _dukascopy_interval(dukascopy_python, key: str):
    return {
        "1m": dukascopy_python.INTERVAL_MIN_1,
        "5m": dukascopy_python.INTERVAL_MIN_5,
        "15m": dukascopy_python.INTERVAL_MIN_15,
    }[key]


def clean_ohlcv(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Clean OHLCV DataFrame: UTC index, deduplicate, sort ascending, OHLC consistency."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "tick_volume", "real_volume", "spread"])
    out = df.copy()
    out.columns = [str(c).lower().strip() for c in out.columns]
    if not isinstance(out.index, pd.DatetimeIndex):
        if "timestamp" in out.columns:
            out.index = pd.to_datetime(out["timestamp"], utc=True)
            out = out.drop(columns=["timestamp"])
        else:
            out.index = pd.to_datetime(out.index, utc=True)
    elif out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")

    out.index.name = "timestamp"
    out = out[~out.index.duplicated(keep="last")].sort_index()

    # Map volume -> tick_volume explicitly; spot gold real_volume is 0.0
    if "tick_volume" not in out.columns:
        out["tick_volume"] = out["volume"].astype(float) if "volume" in out.columns else 0.0
    if "real_volume" not in out.columns:
        out["real_volume"] = 0.0
    if "spread" not in out.columns:
        out["spread"] = 0.25  # Default XAUUSD spread proxy when bid-only feed

    for c in ("open", "high", "low", "close"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out[out[c] > 0]
    out = out.dropna(subset=["open", "high", "low", "close"])

    # Enforce OHLC consistency: low <= min(open, close) <= max(open, close) <= high
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)

    return out[["open", "high", "low", "close", "tick_volume", "real_volume", "spread"]].astype(float)


def fetch_dukascopy_chunked(
    symbol: str,
    interval_key: str,
    start_dt: datetime,
    end_dt: datetime,
    chunk_days: int = 30,
    progress_cb=None,
) -> pd.DataFrame:
    """Fetch Dukascopy historical bars in chunks to avoid timeout/memory spikes."""
    try:
        import dukascopy_python
    except Exception as e:
        print(f"[dukascopy_vendor] dukascopy_python unavailable: {e}")
        return clean_ohlcv(None)

    inst = resolve_instrument(symbol)
    interval = _dukascopy_interval(dukascopy_python, interval_key)

    cur = start_dt.replace(tzinfo=None)
    end = end_dt.replace(tzinfo=None)
    parts = []
    total_days = max(1, (end - cur).days)
    done_days = 0

    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        try:
            df_chunk = dukascopy_python.fetch(
                instrument=inst,
                interval=interval,
                offer_side=dukascopy_python.OFFER_SIDE_BID,
                start=cur,
                end=nxt,
            )
            if df_chunk is not None and not df_chunk.empty:
                parts.append(df_chunk)
        except Exception as e:
            print(f"[dukascopy_vendor] Chunk {cur.date()} -> {nxt.date()} warning: {e}")
        done_days += (nxt - cur).days
        if progress_cb:
            progress_cb(min(1.0, done_days / total_days), f"Fetched {cur.date()} -> {nxt.date()}")
        cur = nxt

    if not parts:
        return clean_ohlcv(None)
    merged = pd.concat(parts, axis=0)
    return clean_ohlcv(merged)
