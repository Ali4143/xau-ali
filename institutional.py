"""Institutional Footprint, Cross-Market Macro (DXY/XAGUSD/US10Y) & Causal Regime Engine.

All feature calculations in `compute_institutional_features()` are strictly causal
(bar `t` uses ONLY data <= `t`) so they can be safely fed into ML models with zero leakage.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .calendar_ny import forex_trading_day

CROSS_CACHE_PATH = config.DATA_DIR / "cross_market_cache.json"

INST_FEATURE_COLS = [
    "vpin_proxy",
    "cvd_20",
    "vsa_absorption",
    "dist_pdh_atr",
    "dist_pdl_atr",
    "dist_asian_hi_atr",
    "dist_asian_lo_atr",
    "dist_london_hi_atr",
    "dist_london_lo_atr",
    "liq_sweep",
    "liq_sweep_decay",
    "fvg_imbalance",
    "dist_round10_atr",
    "dist_round50_atr",
    "efficiency_ratio",
    "vol_z_60",
    "regime_state",
]


def detect_candle_patterns_and_ladder(
    df_m1: pd.DataFrame,
    inst_df: pd.DataFrame,
    min_conf: float = 70.0,
    cooldown_bars: int = 8,
) -> dict[str, Any]:
    """Causal candle-pattern detector (Doji, Hammer, Shooting Star, Engulfing, Impulse)
    + strict >=70% confidence signal filter (de-clustered) + auto-stepping T1->T2->T3->T4->T5
    dynamic target ladder with fast-momentum acceleration.
    """
    if df_m1 is None or len(df_m1) < 25 or inst_df is None or inst_df.empty:
        return {"signals": [], "last_candle_pattern": "Normal", "active_ladder": None}

    s = _ensure_utc_index(df_m1).tail(240)
    ins = inst_df.reindex(s.index)
    o = s["open"].to_numpy(dtype=np.float64)
    h = s["high"].to_numpy(dtype=np.float64)
    l = s["low"].to_numpy(dtype=np.float64)
    c = s["close"].to_numpy(dtype=np.float64)
    v = s["volume"].to_numpy(dtype=np.float64) if "volume" in s.columns else np.ones(len(s), dtype=np.float64)
    v = np.maximum(v, 1.0)
    atr = ins["_atr1m"].ffill().bfill().to_numpy(dtype=np.float64)
    atr = np.maximum(atr, 0.25)
    sw = ins["liq_sweep"].fillna(0.0).to_numpy(dtype=np.float64)
    sw_dec = ins["liq_sweep_decay"].fillna(0.0).to_numpy(dtype=np.float64)
    vsa = ins["vsa_absorption"].fillna(0.0).to_numpy(dtype=np.float64)
    cvd = ins["cvd_20"].fillna(0.0).to_numpy(dtype=np.float64)
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy(dtype=np.float64)
    vol_ma = pd.Series(v).rolling(20, min_periods=5).mean().bfill().to_numpy(dtype=np.float64)
    vol_rel = v / np.maximum(vol_ma, 1.0)

    def _classify_bar(i: int) -> tuple[str, int, float]:
        """Return (pattern_name, direction [+1/-1/0], base_score)."""
        rng = max(h[i] - l[i], 1e-6)
        body = abs(c[i] - o[i])
        body_r = body / rng
        up_w = (h[i] - max(o[i], c[i])) / rng
        lo_w = (min(o[i], c[i]) - l[i]) / rng
        is_bull = c[i] > o[i]
        is_bear = c[i] < o[i]

        # 1) Bullish / Bearish Engulfing (requires previous bar)
        if i >= 1:
            prev_bull = c[i - 1] > o[i - 1]
            prev_bear = c[i - 1] < o[i - 1]
            prev_body = abs(c[i - 1] - o[i - 1])
            if prev_bear and is_bull and c[i] > o[i - 1] and o[i] <= c[i - 1] and body > prev_body * 1.05:
                return ("Bull Engulfing", 1, 42.0)
            if prev_bull and is_bear and c[i] < o[i - 1] and o[i] >= c[i - 1] and body > prev_body * 1.05:
                return ("Bear Engulfing", -1, 42.0)

        # 2) Doji variants
        if body_r <= 0.14:
            if lo_w >= 0.58:
                return ("Dragonfly Doji", 1, 36.0)
            if up_w >= 0.58:
                return ("Gravestone Doji", -1, 36.0)
            return ("Doji (Indecision)", 0, 0.0)

        # 3) Hammer / Shooting Star (Pin Bars)
        if body_r <= 0.36 and lo_w >= 0.56 and up_w <= 0.18:
            return ("Hammer Pin Bar", 1, 40.0)
        if body_r <= 0.36 and up_w >= 0.56 and lo_w <= 0.18:
            return ("Shooting Star", -1, 40.0)

        # 4) Strong Institutional Momentum Marubozu
        if body_r >= 0.72 and rng >= 1.25 * atr[i]:
            return ("Strong Bull Impulse", 1, 38.0) if is_bull else ("Strong Bear Impulse", -1, 38.0)

        # 5) Inside Bar
        if i >= 1 and h[i] <= h[i - 1] and l[i] >= l[i - 1]:
            return ("Inside Bar", 0, 0.0)

        return ("Bull Candle" if is_bull else "Bear Candle", 0, 0.0)

    signals: list[dict[str, Any]] = []
    last_sig_i = -999
    last_bar_pat = "Normal"

    for i in range(1, len(s)):
        pat_name, p_dir, base_sc = _classify_bar(i)
        if i == len(s) - 1:
            last_bar_pat = pat_name
        if p_dir == 0 or base_sc <= 0.0:
            continue

        # Institutional confluence score (strictly causal at bar i)
        conf = base_sc
        has_sweep = (sw[i] * p_dir > 0.5) or (sw_dec[i] * p_dir > 0.18)
        if has_sweep:
            conf += 22.0
        if vol_rel[i] >= 1.35:
            conf += 14.0
        elif vol_rel[i] >= 1.15:
            conf += 7.0
        if vsa[i] * p_dir > 0.5:
            conf += 12.0
        if cvd[i] * p_dir > 0.12:
            conf += 10.0
        if (p_dir == 1 and c[i] >= ema20[i]) or (p_dir == -1 and c[i] <= ema20[i]):
            conf += 8.0

        conf = min(96.0, round(conf, 1))
        if conf >= min_conf and (i - last_sig_i) >= cooldown_bars:
            last_sig_i = i
            signals.append(
                {
                    "idx": i,
                    "time_utc": s.index[i],
                    "price": float(c[i]),
                    "low": float(l[i]),
                    "high": float(h[i]),
                    "dir": p_dir,
                    "pattern": pat_name + (" + Sweep" if has_sweep else ""),
                    "confidence": conf,
                    "atr": float(atr[i]),
                }
            )

    # Build Dynamic Auto-Stepping Target Ladder (T1 -> T2 -> T3 -> T4 -> T5) from latest >=70% signal
    active_ladder = None
    if signals:
        latest = signals[-1]
        age_bars = (len(s) - 1) - latest["idx"]
        if age_bars <= 65:
            p_sig = latest["price"]
            d = latest["dir"]
            step = round(max(2.2 * latest["atr"], 2.20), 2)
            levels_t = {k: round(p_sig + d * k * step, 2) for k in range(1, 6)}
            inval_px = round(p_sig - d * 1.6 * step, 2)

            # Walk forward strictly through realized bars after signal
            highest_hit = 0
            invalidated = False
            for j in range(latest["idx"] + 1, len(s)):
                if (d == 1 and l[j] <= inval_px) or (d == -1 and h[j] >= inval_px):
                    invalidated = True
                    break
                for k in range(1, 6):
                    if (d == 1 and h[j] >= levels_t[k]) or (d == -1 and l[j] <= levels_t[k]):
                        highest_hit = max(highest_hit, k)

            if not invalidated:
                # Check fast momentum acceleration over last 3 closed candles
                mom3 = d * (c[-1] - c[-4]) if len(c) >= 4 else 0.0
                fast_accel = mom3 >= 1.8 * atr[-1]
                base_next = min(5, highest_hit + 1)
                active_k = min(5, max(base_next, highest_hit + 2, 3)) if fast_accel else base_next
                active_ladder = {
                    "signal_time_utc": latest["time_utc"].isoformat(),
                    "pattern": latest["pattern"],
                    "confidence": latest["confidence"],
                    "dir": d,
                    "side_label": "LONG ▲" if d == 1 else "SHORT ▼",
                    "entry_price": round(p_sig, 2),
                    "invalidation_price": inval_px,
                    "step_usd": step,
                    "targets": levels_t,
                    "highest_hit": highest_hit,
                    "active_target_num": active_k,
                    "active_target_price": levels_t[active_k],
                    "fast_momentum": bool(fast_accel),
                    "status_text": (
                        f"🚀 FAST MOMENTUM — Auto-Extended to T{active_k} (${levels_t[active_k]:,.2f})"
                        if fast_accel
                        else (
                            f"✅ T{highest_hit} Hit → Next: T{active_k} (${levels_t[active_k]:,.2f})"
                            if highest_hit > 0
                            else f"🎯 Active T1 (${levels_t[1]:,.2f}) · T2 (${levels_t[2]:,.2f})"
                        )
                    ),
                }

    return {
        "signals": signals,
        "last_candle_pattern": last_bar_pat,
        "active_ladder": active_ladder,
    }


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out = df.copy()
    out.index = idx
    return out.sort_index()


def compute_institutional_features(df_m1: pd.DataFrame) -> pd.DataFrame:
    """Compute 100% causal institutional microstructure, liquidity pool, sweep & regime features."""
    if df_m1 is None or df_m1.empty:
        return pd.DataFrame(columns=INST_FEATURE_COLS)

    s = _ensure_utc_index(df_m1)
    o = s["open"].astype(np.float64)
    h = s["high"].astype(np.float64)
    l = s["low"].astype(np.float64)
    c = s["close"].astype(np.float64)
    v = s["volume"].astype(np.float64).clip(lower=1.0) if "volume" in s.columns else pd.Series(1.0, index=s.index)

    # Causal 1M ATR (14-bar) for scale-free distances
    prev_c = c.shift(1).fillna(c)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean().clip(lower=0.05)

    # 1. Order-Flow Toxicity (VPIN proxy via Bulk Volume Classification) & Net CVD(20)
    rng = (h - l).clip(lower=1e-6)
    buy_frac = (((c - o) / rng).clip(-1.0, 1.0) * 0.5 + 0.5)
    buy_vol = v * buy_frac
    sell_vol = v * (1.0 - buy_frac)
    imbal = (buy_vol - sell_vol).abs()
    tot_vol_30 = v.rolling(30, min_periods=5).sum().clip(lower=1.0)
    vpin_proxy = (imbal.rolling(30, min_periods=5).sum() / tot_vol_30).clip(0.0, 1.0).fillna(0.5)
    tot_vol_20 = v.rolling(20, min_periods=5).sum().clip(lower=1.0)
    cvd_20 = ((buy_vol - sell_vol).rolling(20, min_periods=5).sum() / tot_vol_20).clip(-1.0, 1.0).fillna(0.0)

    # 2. VSA Absorption (High Volume + Small Body + Long Rejection Wick)
    vol_rel = v / v.rolling(30, min_periods=5).mean().clip(lower=1.0)
    body_ratio = (c - o).abs() / rng
    lower_wick = (np.minimum(o, c) - l) / rng
    upper_wick = (h - np.maximum(o, c)) / rng
    bull_abs = (vol_rel > 1.4) & (body_ratio < 0.38) & (lower_wick > 0.55)
    bear_abs = (vol_rel > 1.4) & (body_ratio < 0.38) & (upper_wick > 0.55)
    vsa_absorption = pd.Series(0.0, index=s.index, dtype=np.float64)
    vsa_absorption[bull_abs] = 1.0
    vsa_absorption[bear_abs] = -1.0

    # 3. Causal Liquidity Pools: PDH/PDL (17:00 NY rollover), Asian (00-07 UTC), London (07-13 UTC)
    day_id = forex_trading_day(s.index)
    day_hi = h.groupby(day_id).max()
    day_lo = l.groupby(day_id).min()
    # Strictly previous completed NY day (shift 1) -> zero leakage!
    prev_day_hi = day_hi.shift(1)
    prev_day_lo = day_lo.shift(1)
    pdh = day_id.map(prev_day_hi).astype(np.float64)
    pdl = day_id.map(prev_day_lo).astype(np.float64)
    # Fallback for the very first day in dataset using expanding causal high/low
    pdh = pdh.fillna(h.expanding(min_periods=1).max().shift(1).fillna(h))
    pdl = pdl.fillna(l.expanding(min_periods=1).min().shift(1).fillna(l))

    utc_hr = s.index.hour
    # Asian session causal running high/low (00:00..06:59 UTC), carried forward for rest of NY day
    in_asian = (utc_hr >= 0) & (utc_hr < 7)
    asian_hi_raw = h.where(in_asian).groupby(day_id).cummax()
    asian_lo_raw = l.where(in_asian).groupby(day_id).cummin()
    asian_hi = asian_hi_raw.groupby(day_id).ffill().shift(1).fillna(pdh)
    asian_lo = asian_lo_raw.groupby(day_id).ffill().shift(1).fillna(pdl)

    # London session causal running high/low (07:00..12:59 UTC), carried forward for rest of NY day
    in_london = (utc_hr >= 7) & (utc_hr < 13)
    london_hi_raw = h.where(in_london).groupby(day_id).cummax()
    london_lo_raw = l.where(in_london).groupby(day_id).cummin()
    london_hi = london_hi_raw.groupby(day_id).ffill().shift(1).fillna(asian_hi)
    london_lo = london_lo_raw.groupby(day_id).ffill().shift(1).fillna(asian_lo)

    dist_pdh_atr = ((pdh - c) / atr).clip(-25.0, 25.0)
    dist_pdl_atr = ((c - pdl) / atr).clip(-25.0, 25.0)
    dist_asian_hi_atr = ((asian_hi - c) / atr).clip(-25.0, 25.0)
    dist_asian_lo_atr = ((c - asian_lo) / atr).clip(-25.0, 25.0)
    dist_london_hi_atr = ((london_hi - c) / atr).clip(-25.0, 25.0)
    dist_london_lo_atr = ((c - london_lo) / atr).clip(-25.0, 25.0)

    # 4. Causal Stop-Hunt / Liquidity Sweep Detector
    # Wick pierces a prior liquidity pool & closes back inside
    prior_60_hi = h.shift(1).rolling(60, min_periods=15).max()
    prior_60_lo = l.shift(1).rolling(60, min_periods=15).min()
    bear_sweep = (
        ((h > pdh) & (c < pdh))
        | ((h > asian_hi) & (c < asian_hi))
        | ((h > london_hi) & (c < london_hi))
        | ((h > prior_60_hi) & (c < prior_60_hi) & (vol_rel > 1.25))
    )
    bull_sweep = (
        ((l < pdl) & (c > pdl))
        | ((l < asian_lo) & (c > asian_lo))
        | ((l < london_lo) & (c > london_lo))
        | ((l < prior_60_lo) & (c > prior_60_lo) & (vol_rel > 1.25))
    )
    liq_sweep = pd.Series(0.0, index=s.index, dtype=np.float64)
    liq_sweep[bull_sweep & ~bear_sweep] = 1.0
    liq_sweep[bear_sweep & ~bull_sweep] = -1.0
    liq_sweep_decay = liq_sweep.ewm(halflife=10, adjust=False).mean()

    # 5. Causal Fair Value Gap (3-bar institutional imbalance)
    bull_fvg = (l - h.shift(2)).clip(lower=0.0)
    bear_fvg = (l.shift(2) - h).clip(lower=0.0)
    fvg_imbalance = ((bull_fvg - bear_fvg) / atr).clip(-5.0, 5.0).fillna(0.0)

    # 6. Round-Number Psychological Magnets ($10 and $50 levels)
    r10 = np.round(c / 10.0) * 10.0
    r50 = np.round(c / 50.0) * 50.0
    dist_round10_atr = ((r10 - c) / atr).clip(-10.0, 10.0)
    dist_round50_atr = ((r50 - c) / atr).clip(-25.0, 25.0)

    # 7. Causal 3-State Volatility & Efficiency Regime Classifier (Forward-Only, 3-bar persistence)
    ret1 = np.log(c / prev_c).fillna(0.0)
    rv15 = ret1.rolling(15, min_periods=5).std().fillna(0.0)
    rv_mean = rv15.rolling(240, min_periods=30).mean().fillna(rv15)
    rv_std = rv15.rolling(240, min_periods=30).std().clip(lower=1e-7).fillna(1e-4)
    vol_z_60 = ((rv15 - rv_mean) / rv_std).clip(-4.0, 6.0).fillna(0.0)

    # Kaufman Efficiency Ratio (30-min net displacement / path length)
    net_move = (c - c.shift(30)).abs().fillna(0.0)
    path_len = c.diff().abs().rolling(30, min_periods=5).sum().clip(lower=1e-6)
    efficiency_ratio = (net_move / path_len).clip(0.0, 1.0).fillna(0.0)

    # Raw causal state: 2 = HIGH_VOL_SHOCK, 1 = CLEAN_TREND, 0 = CHOP_RANGE
    raw_state = np.where(vol_z_60 > 2.0, 2, np.where(efficiency_ratio >= 0.28, 1, 0))
    # 3-bar causal persistence filter (eliminates whipsaw flips)
    persisted = np.zeros(len(raw_state), dtype=np.float64)
    cur_st = int(raw_state[0]) if len(raw_state) else 0
    streak = 0
    cand = cur_st
    for i, st_i in enumerate(raw_state):
        st_i = int(st_i)
        if st_i == cur_st:
            streak = 0
            cand = cur_st
        else:
            if st_i == cand:
                streak += 1
            else:
                cand = st_i
                streak = 1
            if streak >= 3:
                cur_st = cand
                streak = 0
        persisted[i] = float(cur_st)

    out = pd.DataFrame(
        {
            "vpin_proxy": vpin_proxy,
            "cvd_20": cvd_20,
            "vsa_absorption": vsa_absorption,
            "dist_pdh_atr": dist_pdh_atr,
            "dist_pdl_atr": dist_pdl_atr,
            "dist_asian_hi_atr": dist_asian_hi_atr,
            "dist_asian_lo_atr": dist_asian_lo_atr,
            "dist_london_hi_atr": dist_london_hi_atr,
            "dist_london_lo_atr": dist_london_lo_atr,
            "liq_sweep": liq_sweep,
            "liq_sweep_decay": liq_sweep_decay,
            "fvg_imbalance": fvg_imbalance,
            "dist_round10_atr": dist_round10_atr,
            "dist_round50_atr": dist_round50_atr,
            "efficiency_ratio": efficiency_ratio,
            "vol_z_60": vol_z_60,
            "regime_state": persisted,
            # Auxiliary raw price levels for chart plotting (NOT fed as raw non-stationary ML inputs)
            "_pdh": pdh,
            "_pdl": pdl,
            "_asian_hi": asian_hi,
            "_asian_lo": asian_lo,
            "_london_hi": london_hi,
            "_london_lo": london_lo,
            "_atr1m": atr,
        },
        index=s.index,
    )
    return out


def fetch_cross_market_snapshot(
    df_xau_m1: pd.DataFrame | None = None,
    max_age_sec: int = 55,
    allow_network: bool = True,
) -> dict[str, Any]:
    """Fetch live 1M DXY, XAGUSD (Silver), and US10Y via TradingView & compute SMT divergence."""
    now_ts = time.time()
    if CROSS_CACHE_PATH.exists():
        try:
            cached = json.loads(CROSS_CACHE_PATH.read_text(encoding="utf-8"))
            if (now_ts - float(cached.get("fetched_epoch", 0))) < max_age_sec or not allow_network:
                return cached
        except Exception:
            pass

    if not allow_network:
        return {"ok": False, "source": "offline", "assets": {}, "macro_bias_score": 0.0, "smt_status": "N/A (offline)"}

    assets: dict[str, dict[str, Any]] = {}
    tv_specs = [
        ("DXY", "TVC", -1.0),     # Inverse to Gold
        ("XAGUSD", "OANDA", 1.0), # Direct with Gold
        ("US10Y", "TVC", -1.0),   # Inverse to Gold
    ]
    series_map: dict[str, pd.Series] = {}
    try:
        from tvDatafeed import Interval, TvDatafeed

        tv = TvDatafeed()
        for sym, exch, sign in tv_specs:
            try:
                d = tv.get_hist(symbol=sym, exchange=exch, interval=Interval.in_1_minute, n_bars=75)
                if d is not None and not d.empty:
                    cl = d["close"].astype(float)
                    last_p = float(cl.iloc[-1])
                    p15 = float(cl.iloc[-min(16, len(cl))])
                    p60 = float(cl.iloc[-min(61, len(cl))])
                    chg15_pct = (last_p / p15 - 1.0) * 100.0 if p15 > 0 else 0.0
                    chg60_pct = (last_p / p60 - 1.0) * 100.0 if p60 > 0 else 0.0
                    assets[sym] = {
                        "exchange": exch,
                        "price": round(last_p, 4),
                        "chg15_pct": round(chg15_pct, 4),
                        "chg60_pct": round(chg60_pct, 4),
                        "gold_impact": "BULLISH" if (chg15_pct * sign) > 0.01 else ("BEARISH" if (chg15_pct * sign) < -0.01 else "NEUTRAL"),
                        "gold_sign": sign,
                    }
                    series_map[sym] = cl
            except Exception:
                continue
    except Exception:
        pass

    # Macro Gold Bias Score (-100 to +100) weighted by DXY (50%), Silver (30%), US10Y (20%)
    score = 0.0
    if "DXY" in assets:
        # 0.10% move in DXY over 15m is strong
        score += float(np.clip(-assets["DXY"]["chg15_pct"] / 0.10, -1.0, 1.0)) * 50.0
    if "XAGUSD" in assets:
        # 0.25% move in Silver over 15m is strong
        score += float(np.clip(assets["XAGUSD"]["chg15_pct"] / 0.25, -1.0, 1.0)) * 30.0
    if "US10Y" in assets:
        # 0.30% relative yield move over 15m
        score += float(np.clip(-assets["US10Y"]["chg15_pct"] / 0.30, -1.0, 1.0)) * 20.0

    # SMT Divergence check over last 30 bars (XAUUSD vs DXY & XAGUSD)
    smt_status = "➖ Aligned (No Divergence)"
    smt_signal = 0
    if df_xau_m1 is not None and len(df_xau_m1) >= 35:
        xau_c = df_xau_m1["close"].astype(float).tail(35).to_numpy()
        xau_hi_recent = xau_c[-10:].max() > xau_c[-30:-10].max()
        xau_lo_recent = xau_c[-10:].min() < xau_c[-30:-10].min()
        if "DXY" in series_map and len(series_map["DXY"]) >= 35:
            dxy_c = series_map["DXY"].tail(35).to_numpy()
            dxy_hi_recent = dxy_c[-10:].max() > dxy_c[-30:-10].max()
            dxy_lo_recent = dxy_c[-10:].min() < dxy_c[-30:-10].min()
            if dxy_hi_recent and not xau_lo_recent:
                smt_status = "🟢 BULLISH SMT (DXY Higher High, Gold Refused Lower Low — Accumulation)"
                smt_signal = 1
            elif dxy_lo_recent and not xau_hi_recent:
                smt_status = "🔴 BEARISH SMT (DXY Lower Low, Gold Failed Higher High — Distribution)"
                smt_signal = -1
        if smt_signal == 0 and "XAGUSD" in series_map and len(series_map["XAGUSD"]) >= 35:
            xag_c = series_map["XAGUSD"].tail(35).to_numpy()
            xag_hi_recent = xag_c[-10:].max() > xag_c[-30:-10].max()
            xag_lo_recent = xag_c[-10:].min() < xag_c[-30:-10].min()
            if xag_hi_recent and not xau_hi_recent:
                smt_status = "🟢 BULLISH SILVER LEAD (XAGUSD Broke High, Gold Lagging)"
                smt_signal = 1
            elif xag_lo_recent and not xau_lo_recent:
                smt_status = "🔴 BEARISH SILVER LEAD (XAGUSD Broke Low, Gold Lagging)"
                smt_signal = -1

    res = {
        "ok": bool(assets),
        "fetched_epoch": now_ts,
        "fetched_utc": pd.Timestamp.utcnow().isoformat(),
        "assets": assets,
        "macro_bias_score": round(score, 1),
        "smt_status": smt_status,
        "smt_signal": smt_signal,
    }
    try:
        CROSS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CROSS_CACHE_PATH.write_text(json.dumps(res, indent=2), encoding="utf-8")
    except Exception:
        pass
    return res


def build_institutional_summary(
    df_m1: pd.DataFrame,
    active_fc: dict[str, Any] | None,
    cross_snap: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine causal institutional footprints + cross-market macro + TFT bias + 1:2 RR plan."""
    if df_m1 is None or df_m1.empty:
        return {}

    inst_df = compute_institutional_features(df_m1.tail(2500))
    last = inst_df.iloc[-1]
    p_now = float(df_m1["close"].iloc[-1])
    atr1m = float(last["_atr1m"])
    # Approximate 15m ATR (~sqrt(15) * 1m ATR, bounded cleanly)
    atr15 = float(np.clip(atr1m * np.sqrt(15.0), 2.5, 35.0))

    st_code = int(last["regime_state"])
    regime_map = {
        0: ("CHOP / RANGE", "#FBBF24", "Sideways chop — sirf Liquidity Sweep par entry dekhein"),
        1: ("CLEAN TREND", "#34D399", "Institutional trend active — TFT + Macro direction follow karein"),
        2: ("HIGH-VOL SHOCK", "#F87171", "News / Panic spike — NO TRADE / Risk-Off!"),
    }
    reg_lbl, reg_col, reg_note = regime_map.get(st_code, regime_map[0])

    # Check recent liquidity sweeps in last 15 bars
    recent_sw = inst_df["liq_sweep"].tail(15)
    bull_sw_n = int((recent_sw > 0.5).sum())
    bear_sw_n = int((recent_sw < -0.5).sum())
    if bull_sw_n > bear_sw_n:
        sweep_lbl, sweep_dir = f"🟢 BULL STOP-HUNT SWEEP ({bull_sw_n}x in 15m)", 1
    elif bear_sw_n > bull_sw_n:
        sweep_lbl, sweep_dir = f"🔴 BEAR STOP-HUNT SWEEP ({bear_sw_n}x in 15m)", -1
    else:
        sweep_lbl, sweep_dir = "➖ No Sweep (Last 15m)", 0

    macro_score = float((cross_snap or {}).get("macro_bias_score", 0.0))
    smt_sig = int((cross_snap or {}).get("smt_signal", 0))
    cvd = float(last["cvd_20"])
    vpin = float(last["vpin_proxy"])

    # TFT / Active Forecast direction
    tft_dir = 0
    if active_fc and active_fc.get("q50"):
        q50_end = float(active_fc["q50"][min(59, len(active_fc["q50"]) - 1)])
        p0_fc = float(active_fc.get("P0", p_now))
        if q50_end - p0_fc > 0.35:
            tft_dir = 1
        elif q50_end - p0_fc < -0.35:
            tft_dir = -1

    # Combined Institutional Confluence Score (-100 to +100)
    inst_score = (
        0.35 * macro_score
        + 25.0 * smt_sig
        + 20.0 * sweep_dir
        + 15.0 * cvd
        + 15.0 * tft_dir
    )
    inst_score = float(np.clip(inst_score, -100.0, 100.0))

    if st_code == 2:
        verdict, v_col, side = "⛔ HIGH-VOL SHOCK — NO TRADE", "#F87171", 0
    elif st_code == 0 and sweep_dir == 0 and abs(inst_score) < 35:
        verdict, v_col, side = "🟡 CHOP / TRAP — WAIT FOR SWEEP", "#FBBF24", 0
    elif inst_score >= 22:
        verdict, v_col, side = "🟢 INSTITUTIONAL LONG SETUP", "#34D399", 1
    elif inst_score <= -22:
        verdict, v_col, side = "🔴 INSTITUTIONAL SHORT SETUP", "#F87171", -1
    else:
        verdict, v_col, side = "🟡 MIXED / NEUTRAL — WAIT", "#FBBF24", 0

    # Fixed 1:2 Risk-to-Reward ATR Plan
    sl_dist = round(1.5 * atr15, 2)
    tp_dist = round(3.0 * atr15, 2)
    if side == 1:
        sl_px, tp_px = round(p_now - sl_dist, 2), round(p_now + tp_dist, 2)
    elif side == -1:
        sl_px, tp_px = round(p_now + sl_dist, 2), round(p_now - tp_dist, 2)
    else:
        sl_px, tp_px = None, None

    return {
        "p_now": p_now,
        "atr15": round(atr15, 2),
        "regime_state": st_code,
        "regime_label": reg_lbl,
        "regime_color": reg_col,
        "regime_note": reg_note,
        "vpin_proxy": round(vpin, 3),
        "cvd_20": round(cvd, 3),
        "efficiency_ratio": round(float(last["efficiency_ratio"]), 3),
        "vol_z_60": round(float(last["vol_z_60"]), 2),
        "sweep_label": sweep_lbl,
        "sweep_dir": sweep_dir,
        "macro_bias_score": round(macro_score, 1),
        "smt_status": (cross_snap or {}).get("smt_status", "➖ Aligned"),
        "inst_score": round(inst_score, 1),
        "verdict": verdict,
        "verdict_color": v_col,
        "side": side,
        "sl_dist": sl_dist,
        "tp_dist": tp_dist,
        "sl_price": sl_px,
        "tp_price": tp_px,
        "levels": {
            "PDH": round(float(last["_pdh"]), 2),
            "PDL": round(float(last["_pdl"]), 2),
            "Asian_H": round(float(last["_asian_hi"]), 2),
            "Asian_L": round(float(last["_asian_lo"]), 2),
            "London_H": round(float(last["_london_hi"]), 2),
            "London_L": round(float(last["_london_lo"]), 2),
        },
        "inst_tail": inst_df.tail(240),
        "patterns": detect_candle_patterns_and_ladder(df_m1, inst_df, min_conf=70.0, cooldown_bars=8),
    }
