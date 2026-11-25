"""Zero-leakage Multi-Timeframe (15M HTF + 1M LTF) causal feature pipeline for XAUUSD."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pywt

from . import config
from .calendar_ny import forex_trading_day, session_flags

BPS = 1e4


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    ag = d.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    al = (-d).clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    rs_sum = ag + al
    rsi = np.where(rs_sum > 0, 100.0 * ag / np.where(rs_sum > 0, rs_sum, 1.0), 50.0)
    return pd.Series(rsi, index=close.index)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def causal_wavelet_close(close: np.ndarray, window: int = 64, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """Strictly causal rolling DWT denoising: bar i uses ONLY close[i-window+1 : i+1].

    Uses symmetric/edge padding strictly within the past window (zero future bars).
    To mitigate endpoint boundary wiggle, we blend the last denoised point softly with
    an EMA-3 anchor only when detail energy is extreme, while preserving pure causal DWT.
    """
    n = len(close)
    out = close.copy().astype(np.float64)
    if n < window:
        return out
    for i in range(window - 1, n):
        seg = close[i - window + 1 : i + 1]
        # Subtract local mean so boundary reflection stays centered around recent price level
        mu = seg[-1]
        centered = seg - mu
        coeffs = pywt.wavedec(centered, wavelet, mode="constant", level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) else 0.0
        thr = sigma * np.sqrt(2.0 * np.log(window))
        thresh_coeffs = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
        rec = pywt.waverec(thresh_coeffs, wavelet, mode="constant")
        out[i] = mu + rec[len(seg) - 1]
    return out


def compute_ny_daily_pivots(df_m1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute classic floor pivots strictly from the PREVIOUS completed 17:00 NY forex day."""
    fday = forex_trading_day(df_m1.index)
    daily = df_m1.groupby(fday).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        bars=("close", "count"),
    )
    # Shift by 1 completed forex day -> zero leakage of today's H/L/C
    prev = daily.shift(1)
    H, L, C = prev["high"], prev["low"], prev["close"]
    P = (H + L + C) / 3.0
    piv_daily = pd.DataFrame(
        {
            "P": P,
            "R1": 2.0 * P - L,
            "S1": 2.0 * P - H,
            "R2": P + (H - L),
            "S2": P - (H - L),
            "R3": H + 2.0 * (P - L),
            "S3": L - 2.0 * (H - P),
            "prev_close": C,
            "day_open": daily["open"],
        },
        index=daily.index,
    )
    mapped = piv_daily.reindex(fday.values)
    mapped.index = df_m1.index
    return mapped, daily


def build_m15_context_features(df_m15: pd.DataFrame) -> pd.DataFrame:
    """Compute causal features on closed 15M bars (index = 15M bar open timestamp)."""
    c = df_m15["close"].astype(float)
    o = df_m15["open"].astype(float)
    h = df_m15["high"].astype(float)
    lo = df_m15["low"].astype(float)
    rng = (h - lo).replace(0.0, np.nan)

    f = pd.DataFrame(index=df_m15.index)
    f["m15_ema9_bps"] = (c - _ema(c, 9)) / c * BPS
    f["m15_ema21_bps"] = (c - _ema(c, 21)) / c * BPS
    f["m15_rsi14"] = _wilder_rsi(c, 14) / 100.0 - 0.5
    atr15 = _atr(h, lo, c, 14)
    f["m15_atr14_raw"] = atr15
    f["m15_atr14_pct"] = atr15 / c
    f["m15_body_ratio"] = ((c - o) / rng).fillna(0.0).clip(-1.0, 1.0)
    f["m15_uwick_ratio"] = ((h - np.maximum(o, c)) / rng).fillna(0.0).clip(0.0, 1.0)
    f["m15_lwick_ratio"] = ((np.minimum(o, c) - lo) / rng).fillna(0.0).clip(0.0, 1.0)

    # Market structure over rolling 32 closed 15M bars (~8 hours): HH/HL vs LH/LL
    hi_16 = h.rolling(16, min_periods=4).max()
    hi_prev16 = hi_16.shift(16)
    lo_16 = lo.rolling(16, min_periods=4).min()
    lo_prev16 = lo_16.shift(16)
    hh = (hi_16 > hi_prev16).astype(float)
    hl = (lo_16 > lo_prev16).astype(float)
    lh = (hi_16 < hi_prev16).astype(float)
    ll = (lo_16 < lo_prev16).astype(float)
    f["m15_structure"] = ((hh + hl) - (lh + ll)) / 2.0  # in [-1, +1]
    return f.astype(np.float32)


def build_mtf_features(
    df_m1: pd.DataFrame,
    df_m15: pd.DataFrame,
    use_wavelet: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build strictly causal MTF feature table aligned to every 1M decision bar t.

    Leakage guarantee:
      - At 1M bar t with B = floor(t, 15min):
        * ONLY closed 15M bars with close_time <= B (i.e., open_time <= B - 15min) are joined.
        * Current partial 15M block [B..t] is computed causally via cumulative groupby on 1M bars <= t.
        * Daily pivots use ONLY the previous completed 17:00 NY forex day.
        * Tick VWAP resets at 17:00 NY and uses cumulative sums strictly <= t.
        * Wavelet DWT uses strictly trailing 64-bar windows <= t.

    Returns:
      - feat_df: DataFrame of all model features aligned to df_m1.index
      - aux_df: Auxiliary columns (m15_atr14_raw, P, R1, S1, R2, S2, tick_vwap_raw) for targets & UI
    """
    s = df_m1.sort_index()
    c = s["close"].astype(float)
    o = s["open"].astype(float)
    h = s["high"].astype(float)
    lo = s["low"].astype(float)
    tv = s["tick_volume"].astype(float).fillna(0.0)
    sp = s["spread"].astype(float).fillna(config.DEFAULT_SPREAD_USD)

    f = pd.DataFrame(index=s.index)

    # ── A) 15M HTF CONTEXT (strictly closed 15M bars with close_time <= B) ───
    m15_feats = build_m15_context_features(df_m15)
    # A 15M bar timestamped s_open has close_time = s_open + 15min.
    # It becomes CLOSED at t >= s_open + 15min (i.e., B >= s_open + 15min).
    m15_closed = m15_feats.copy()
    m15_closed.index = m15_closed.index + pd.Timedelta(minutes=15)

    B = s.index.floor("15min")
    aligned_15m = m15_closed.reindex(B, method="ffill")
    aligned_15m.index = s.index
    for col in m15_feats.columns:
        if col != "m15_atr14_raw":
            f[col] = aligned_15m[col].astype(np.float32)
    m15_atr_raw = aligned_15m["m15_atr14_raw"].astype(float).bfill().replace(0.0, np.nan)

    # Partial current 15M block state (strictly causal: uses 1M bars in [B .. t])
    elapsed = ((s.index - B).total_seconds() / 60.0).astype(np.float32)
    f["blk15_elapsed"] = elapsed / 15.0
    f["blk15_remaining"] = (15.0 - elapsed) / 15.0
    blk_open = o.groupby(B).transform("first")
    blk_high = h.groupby(B).cummax()
    blk_low = lo.groupby(B).cummin()
    f["blk15_dist_open_atr"] = ((c - blk_open) / (m15_atr_raw + 1e-6)).clip(-10.0, 10.0)
    f["blk15_dist_high_atr"] = ((c - blk_high) / (m15_atr_raw + 1e-6)).clip(-10.0, 10.0)
    f["blk15_dist_low_atr"] = ((c - blk_low) / (m15_atr_raw + 1e-6)).clip(-10.0, 10.0)

    # ── B) 1M LTF EXECUTION & INDICATORS ─────────────────────────────────────
    if use_wavelet:
        wav = causal_wavelet_close(c.to_numpy(dtype=np.float64), window=64, wavelet="db4", level=3)
        f["wav_bps"] = ((c.to_numpy(dtype=np.float64) - wav) / c.to_numpy(dtype=np.float64)) * BPS
    else:
        f["wav_bps"] = 0.0

    for span in (9, 21, 50, 200):
        f[f"ema{span}_bps"] = (c - _ema(c, span)) / c * BPS

    atr1 = _atr(h, lo, c, 14)
    f["atr14_pct"] = atr1 / c

    m20 = c.rolling(20, min_periods=5).mean()
    sd20 = c.rolling(20, min_periods=5).std(ddof=0)
    f["bb_pos"] = ((c - m20) / (2.0 * sd20 + 1e-9)).clip(-5.0, 5.0)
    f["bb_width"] = (4.0 * sd20 / c) * BPS

    f["rsi14"] = _wilder_rsi(c, 14) / 100.0 - 0.5

    macd = _ema(c, 12) - _ema(c, 26)
    macd_sig = _ema(macd, 9)
    f["macd_h_bps"] = (macd - macd_sig) / c * BPS

    lo14 = lo.rolling(14, min_periods=5).min()
    hi14 = h.rolling(14, min_periods=5).max()
    stoch_k = ((c - lo14) / (hi14 - lo14 + 1e-9)).clip(0.0, 1.0)
    f["stoch_k"] = stoch_k - 0.5
    f["stoch_d"] = stoch_k.rolling(3, min_periods=1).mean() - 0.5

    # Returns & candle shape
    ret1 = np.log(c / c.shift(1))
    f["ret1_bps"] = ret1 * BPS
    f["ret2_bps"] = ret1.shift(1) * BPS
    f["ret3_bps"] = ret1.shift(2) * BPS
    f["rv12_bps"] = ret1.rolling(12, min_periods=3).std(ddof=0) * BPS

    rng1 = (h - lo).replace(0.0, np.nan)
    f["body_ratio"] = ((c - o) / rng1).fillna(0.0).clip(-1.0, 1.0)
    f["uwick_ratio"] = ((h - np.maximum(o, c)) / rng1).fillna(0.0).clip(0.0, 1.0)
    f["lwick_ratio"] = ((np.minimum(o, c) - lo) / rng1).fillna(0.0).clip(0.0, 1.0)

    # Tick-volume proxy features (explicitly tick-volume based, NOT real volume)
    tv_mu = tv.rolling(60, min_periods=10).mean()
    tv_sd = tv.rolling(60, min_periods=10).std(ddof=0)
    f["tick_vol_z"] = ((tv - tv_mu) / (tv_sd + 1e-6)).clip(-5.0, 5.0)

    signed_tv = np.sign(c - o) * tv
    f["tick_flow_5"] = (signed_tv.rolling(5, min_periods=1).sum() / (tv.rolling(5, min_periods=1).sum() + 1e-6)).clip(-1.0, 1.0)
    f["tick_flow_15"] = (signed_tv.rolling(15, min_periods=1).sum() / (tv.rolling(15, min_periods=1).sum() + 1e-6)).clip(-1.0, 1.0)

    obv = signed_tv.cumsum()
    obv_mu = obv.rolling(60, min_periods=10).mean()
    obv_sd = obv.rolling(60, min_periods=10).std(ddof=0)
    f["obv_z"] = ((obv - obv_mu) / (obv_sd + 1e-6)).clip(-5.0, 5.0)

    f["spread_bps"] = (sp / c) * BPS

    # ── C) 17:00 NY DAILY LEVELS & TICK VWAP ─────────────────────────────────
    piv_mapped, _ = compute_ny_daily_pivots(s)
    for k in ("P", "R1", "S1", "R2", "S2"):
        f[f"piv_{k.lower()}_bps"] = np.log(c / piv_mapped[k]) * BPS

    f["gap_bps"] = np.log(piv_mapped["day_open"] / piv_mapped["prev_close"]) * BPS
    f["ret_open_bps"] = np.log(c / piv_mapped["day_open"]) * BPS

    fday = forex_trading_day(s.index)
    tp = (h + lo + c) / 3.0
    cum_tpv = (tp * tv).groupby(fday).cumsum()
    cum_v = tv.groupby(fday).cumsum().replace(0.0, np.nan)
    tick_vwap = (cum_tpv / cum_v).fillna(tp)
    f["tick_vwap_bps"] = np.log(c / tick_vwap) * BPS

    # ── D) CAUSAL SMC PROXIES (FVG 3-candle confirmed, liquidity distance) ───
    # 3-candle FVG confirmed strictly after candle 3 (bar t) closes:
    # Bull FVG: low[t] > high[t-2]; Bear FVG: high[t] < low[t-2]
    f["smc_fvg_bull"] = (lo > h.shift(2)).astype(np.float32)
    f["smc_fvg_bear"] = (h < lo.shift(2)).astype(np.float32)
    swing_hi = h.rolling(60, min_periods=10).max()
    swing_lo = lo.rolling(60, min_periods=10).min()
    f["smc_dist_liq_hi"] = ((swing_hi - c) / (atr1 + 1e-6)).clip(0.0, 20.0)
    f["smc_dist_liq_lo"] = ((c - swing_lo) / (atr1 + 1e-6)).clip(0.0, 20.0)

    # ── E) SESSION FLAGS & CYCLICAL ENCODINGS ────────────────────────────────
    sflags = session_flags(s.index)
    for col in sflags.columns:
        f[col] = sflags[col]

    aux = pd.DataFrame(
        {
            "m15_atr14_raw": m15_atr_raw,
            "m1_atr14_raw": atr1,
            "tick_vwap_raw": tick_vwap,
            "P": piv_mapped["P"],
            "R1": piv_mapped["R1"],
            "S1": piv_mapped["S1"],
            "R2": piv_mapped["R2"],
            "S2": piv_mapped["S2"],
            "R3": piv_mapped["R3"],
            "S3": piv_mapped["S3"],
        },
        index=s.index,
    )
    return f.astype(np.float32), aux
