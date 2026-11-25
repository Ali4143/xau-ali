"""Zero-delay live 1-minute TradingView connector for XAUUSD (TVC:GOLD) and Forex pairs."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .vendor.dukascopy_vendor import clean_ohlcv

# Exact TradingView symbol + exchange mapping (matches user's TradingView TVC:GOLD chart)
TV_SYMBOL_MAP: dict[str, list[tuple[str, str]]] = {
    "XAUUSD": [("XAUUSD", "OANDA"), ("GOLD", "TVC"), ("XAUUSD", "FX_IDC")],
    "XAGUSD": [("SILVER", "TVC"), ("XAGUSD", "OANDA")],
    "EURUSD": [("EURUSD", "FX_IDC"), ("EURUSD", "OANDA")],
    "GBPUSD": [("GBPUSD", "FX_IDC"), ("GBPUSD", "OANDA")],
    "USDJPY": [("USDJPY", "FX_IDC"), ("USDJPY", "OANDA")],
    "AUDUSD": [("AUDUSD", "FX_IDC"), ("AUDUSD", "OANDA")],
    "USDCAD": [("USDCAD", "FX_IDC"), ("USDCAD", "OANDA")],
    "USDCHF": [("USDCHF", "FX_IDC"), ("USDCHF", "OANDA")],
    "NZDUSD": [("NZDUSD", "FX_IDC"), ("NZDUSD", "OANDA")],
}


def _candidates_for(symbol: str) -> list[tuple[str, str]]:
    sym = symbol.upper().replace("/", "").strip()
    if sym in TV_SYMBOL_MAP:
        return TV_SYMBOL_MAP[sym]
    return [(sym, "FX_IDC"), (sym, "OANDA"), (sym, "TVC")]


def fetch_tv_m1(symbol: str = "XAUUSD", n_bars: int = 500) -> tuple[pd.DataFrame, Optional[str]]:
    """Fetch up to `n_bars` (max 5000) real-time 1-minute bars from TradingView in UTC.

    Returns (cleaned_df, source_tag) or (empty_df, None) on failure.
    """
    from tvDatafeed import Interval, TvDatafeed

    n_req = max(10, min(int(n_bars), 5000))
    tv = TvDatafeed()
    for tv_sym, exch in _candidates_for(symbol):
        try:
            raw = tv.get_hist(symbol=tv_sym, exchange=exch, interval=Interval.in_1_minute, n_bars=n_req)
            if raw is None or raw.empty:
                continue
            df = raw.copy()
            # tvDatafeed returns naive timestamps in local machine timezone or UTC depending on version.
            # Normalize strictly to UTC by aligning the latest bar to current UTC minute!
            if df.index.tz is None:
                now_utc = pd.Timestamp.now(tz="UTC").floor("min")
                latest_naive = pd.Timestamp(df.index.max())
                # Detect exact whole-hour offset between tvDatafeed naive clock and UTC
                offset_hrs = round((latest_naive - now_utc.tz_localize(None)).total_seconds() / 3600.0)
                df.index = (df.index - pd.Timedelta(hours=offset_hrs)).tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            if "volume" in df.columns:
                df["tick_volume"] = df["volume"].astype(float)
            df = clean_ohlcv(df)
            if not df.empty:
                return df, f"tradingview:{exch}:{tv_sym}"
        except Exception as e:
            print(f"[tv_live] {exch}:{tv_sym} failed: {e}")
            continue
    return clean_ohlcv(None), None
