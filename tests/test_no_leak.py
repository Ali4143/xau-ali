"""Mandatory Phase-1 Verification Tests:
1. Zero-Leakage Test (1M + 15M MTF causality): Alter every bar after t, assert feature vector at t is UNCHANGED.
2. 15M Future-Block Isolation Test: Assert 15M bars with close_time > floor(t, 15m) are NEVER used.
3. NY 17:00 Calendar & Gap Test: Assert trading horizons skip/stop at weekend & daily maintenance gaps.
4. Wavelet Endpoint Stability Test: Check causal DWT endpoint deviation vs close is bounded and causal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from xau_mtf.calendar_ny import detect_calendar_from_data, forex_trading_day, trading_horizon_mask
from xau_mtf.features import build_mtf_features, causal_wavelet_close
from xau_mtf.fetcher.fetch_adapter import load_m1, resample_m1_to_m15


def _synthetic_xau_sample(n_bars: int = 3200) -> pd.DataFrame:
    """Create realistic 24/5 XAUUSD 1M bars with a weekend gap & 17:00 NY maintenance break."""
    rng = np.random.default_rng(42)
    idx_raw = pd.date_range("2026-10-05 00:00:00", periods=n_bars + 2500, freq="1min", tz="UTC")
    # Remove 21:00-22:00 UTC (17:00-18:00 NY EDT) daily maintenance gap & Saturday/Sunday
    ny = idx_raw.tz_convert("America/New_York")
    keep = ~((ny.hour == 17) | (ny.dayofweek == 5) | ((ny.dayofweek == 6) & (ny.hour < 18)))
    idx = idx_raw[keep][:n_bars]
    close = 2650.0 + np.cumsum(rng.normal(0.0, 0.35, len(idx)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0.15, 0.1, len(idx)))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.15, 0.1, len(idx)))
    tv = rng.integers(15, 250, len(idx)).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": tv,
            "real_volume": np.zeros(len(idx)),
            "spread": np.full(len(idx), 0.25),
        },
        index=idx,
    )


def test_zero_leakage(df_m1: pd.DataFrame) -> None:
    df_sub = df_m1.tail(2600).copy()
    df_15_orig = resample_m1_to_m15(df_sub)
    feat_orig, aux_orig = build_mtf_features(df_sub, df_15_orig, use_wavelet=True)

    # Pick a decision bar t well after warmup (e.g. index 2000, mid-block inside a 15M candle)
    t_pos = 2000
    t_ts = df_sub.index[t_pos]

    # Aggressively corrupt EVERY bar strictly AFTER t_ts (+500 USD shock, 100x volume)
    df_corrupt = df_sub.copy()
    after_mask = df_corrupt.index > t_ts
    assert after_mask.sum() > 200, "Need bars after t_ts"
    for col in ("open", "high", "low", "close"):
        df_corrupt.loc[after_mask, col] = df_corrupt.loc[after_mask, col] + 500.0
    df_corrupt.loc[after_mask, "tick_volume"] = df_corrupt.loc[after_mask, "tick_volume"] * 100.0

    # Re-derive 15M from corrupted 1M (this ALSO tests that the in-progress 15M candle's future close leaks NOTHING!)
    df_15_corrupt = resample_m1_to_m15(df_corrupt)
    feat_corrupt, aux_corrupt = build_mtf_features(df_corrupt, df_15_corrupt, use_wavelet=True)

    v_orig = feat_orig.iloc[t_pos].to_numpy(dtype=np.float64)
    v_corr = feat_corrupt.iloc[t_pos].to_numpy(dtype=np.float64)
    max_diff = float(np.nanmax(np.abs(v_orig - v_corr)))
    assert np.allclose(v_orig, v_corr, rtol=0.0, atol=1e-7, equal_nan=True), (
        f"LEAKAGE DETECTED at {t_ts}! max_diff={max_diff}"
    )

    # Also verify no NaNs after warmup (from bar 1500 onward)
    post_warmup = feat_orig.iloc[1500:]
    nan_cols = post_warmup.columns[post_warmup.isna().any()].tolist()
    assert not nan_cols, f"Unexpected NaNs after warmup in columns: {nan_cols}"
    print(f"[PASS] Zero-Leakage Test at {t_ts}: max_diff = {max_diff:.2e}, NaNs after warmup = 0")


def test_calendar_and_gaps(df_m1: pd.DataFrame) -> None:
    cal = detect_calendar_from_data(df_m1, save=False)
    assert cal["total_bars"] == len(df_m1)

    # Verify 17:00 NY rollover assigns 16:59 NY and 17:00 NY to different forex days
    ts_test = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-10-05 20:59:00", tz="UTC"),  # 16:59 EDT (NY)
            pd.Timestamp("2026-10-05 21:00:00", tz="UTC"),  # 17:00 EDT (NY) -> next forex day
        ]
    )
    fdays = forex_trading_day(ts_test)
    assert fdays.iloc[0] != fdays.iloc[1], "17:00 NY rollover failed!"

    # Verify trading_horizon_mask stops cleanly before a gap
    diffs = np.diff(df_m1.index.asi8) / 1e9 / 60.0
    gap_positions = np.where(diffs >= 45.0)[0]
    if len(gap_positions):
        g_idx = int(gap_positions[0])
        origin = max(0, g_idx - 20)
        pos, vlen, eos = trading_horizon_mask(df_m1.index, origin, max_h=360)
        assert vlen == (g_idx - origin), f"Expected horizon to stop at gap ({g_idx - origin}), got {vlen}"
        print(f"[PASS] Calendar Gap Test: horizon stopped cleanly at {vlen} bars before gap (total gaps={cal['total_gaps_ge_45m']})")
    else:
        print("[PASS] Calendar Rollover Test (no >=45m gap in sample slice)")


def test_wavelet_boundary_stability(df_m1: pd.DataFrame) -> None:
    c = df_m1["close"].tail(500).to_numpy(dtype=np.float64)
    w = causal_wavelet_close(c, window=64, wavelet="db4", level=3)
    diff_bps = np.abs(c[64:] - w[64:]) / c[64:] * 1e4
    max_bps = float(np.max(diff_bps))
    med_bps = float(np.median(diff_bps))
    assert np.all(np.isfinite(w)), "Wavelet produced non-finite values!"
    assert max_bps < 100.0, f"Wavelet endpoint distortion too high: {max_bps:.2f} bps"
    print(f"[PASS] Wavelet Boundary Stability Test: median={med_bps:.2f} bps, max={max_bps:.2f} bps")


def run_all() -> None:
    df_real = load_m1("XAUUSD")
    if len(df_real) >= 2600:
        print(f"Running Phase-1 tests on REAL XAUUSD data ({len(df_real):,} bars)...")
        df_use = df_real
    else:
        print("Running Phase-1 tests on synthetic 24/5 XAUUSD data (real data not yet loaded)...")
        df_use = _synthetic_xau_sample(3200)

    test_zero_leakage(df_use)
    test_calendar_and_gaps(df_use)
    test_wavelet_boundary_stability(df_use)
    print("ALL PHASE-1 TESTS PASSED 100%!")


if __name__ == "__main__":
    run_all()
