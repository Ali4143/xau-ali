"""Chronological MTF dataset builder with 360-bar embargo and Train-only scaling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .calendar_ny import forex_trading_day, gap_mask_before
from .features import build_mtf_features

STATIC_CTX_COLS = [
    "sess_asia",
    "sess_london",
    "sess_ny",
    "sess_overlap",
    "kz_london",
    "kz_ny",
    "utc_sin",
    "utc_cos",
    "dow_sin",
    "dow_cos",
]


@dataclass
class Samples:
    X: np.ndarray          # [N, L, F] scaled causal sequence
    C: np.ndarray          # [N, C_dim] static session/time context
    y: np.ndarray          # [N, H] target_delta / ATR14_15M
    m: np.ndarray          # [N, H] valid trading-bar mask (0 or 1)
    P0: np.ndarray         # [N] origin close price
    atr15: np.ndarray      # [N] origin 15M ATR14 (for converting back to USD points)
    eos: np.ndarray        # [N] 1-based EOS step index
    times: np.ndarray      # [N] origin UTC timestamps (int64 ns)
    spread: np.ndarray     # [N] origin spread USD

    def __len__(self) -> int:
        return len(self.P0)


class RobustTrainScaler:
    """StandardScaler fitted strictly on TRAINING rows, clipping extreme outliers to [-8, 8]."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.cols_: list[str] = []

    def fit(self, df: pd.DataFrame) -> "RobustTrainScaler":
        self.cols_ = list(df.columns)
        arr = df.to_numpy(dtype=np.float64)
        self.mean_ = np.nanmean(arr, axis=0)
        self.std_ = np.nanstd(arr, axis=0)
        self.std_ = np.where((~np.isfinite(self.std_)) | (self.std_ < 1e-6), 1.0, self.std_)
        self.mean_ = np.where(np.isfinite(self.mean_), self.mean_, 0.0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        arr = df[self.cols_].to_numpy(dtype=np.float64)
        z = (arr - self.mean_) / self.std_
        z = np.where(np.isfinite(z), z, 0.0)
        return np.clip(z, -8.0, 8.0).astype(np.float32)

    def save(self, path: Path) -> None:
        joblib.dump({"mean": self.mean_, "std": self.std_, "cols": self.cols_}, path)

    @classmethod
    def load(cls, path: Path) -> "RobustTrainScaler":
        d = joblib.load(path)
        s = cls()
        s.mean_, s.std_, s.cols_ = d["mean"], d["std"], d["cols"]
        return s


def _build_samples_for_indices(
    X_scaled: np.ndarray,
    C_arr: np.ndarray,
    close: np.ndarray,
    atr15: np.ndarray,
    spread: np.ndarray,
    gmask: np.ndarray,
    fday: np.ndarray,
    ts_ns: np.ndarray,
    origins: np.ndarray,
    L: int = config.LOOKBACK_M1,
    H: int = config.HORIZON_BARS,
) -> Samples:
    n = len(origins)
    F = X_scaled.shape[1]
    X_out = np.empty((n, L, F), dtype=np.float32)
    C_out = np.empty((n, C_arr.shape[1]), dtype=np.float32)
    y_out = np.zeros((n, H), dtype=np.float32)
    m_out = np.zeros((n, H), dtype=np.float32)
    P0_out = np.empty(n, dtype=np.float64)
    atr_out = np.empty(n, dtype=np.float64)
    eos_out = np.empty(n, dtype=np.int32)
    t_out = np.empty(n, dtype=np.int64)
    sp_out = np.empty(n, dtype=np.float64)

    total_bars = len(close)
    for idx_i, pos in enumerate(origins):
        X_out[idx_i] = X_scaled[pos - L + 1 : pos + 1]
        C_out[idx_i] = C_arr[pos]
        p0 = close[pos]
        a0 = max(float(atr15[pos]), 1e-4)
        P0_out[idx_i] = p0
        atr_out[idx_i] = a0
        t_out[idx_i] = ts_ns[pos]
        sp_out[idx_i] = spread[pos]

        # Build horizon strictly stopping at any closed-market gap
        end_p = min(total_bars, pos + 1 + H)
        orig_day = fday[pos]
        eos_step = 0
        valid_cnt = 0
        for step, fut in enumerate(range(pos + 1, end_p)):
            if gmask[fut]:
                break
            y_out[idx_i, step] = (close[fut] - p0) / a0
            m_out[idx_i, step] = 1.0
            valid_cnt = step + 1
            if eos_step == 0 and fday[fut] != orig_day:
                eos_step = step if step > 0 else 1
        eos_out[idx_i] = eos_step if eos_step > 0 else max(1, valid_cnt)

    return Samples(
        X=X_out,
        C=C_out,
        y=y_out,
        m=m_out,
        P0=P0_out,
        atr15=atr_out,
        eos=eos_out,
        times=t_out,
        spread=sp_out,
    )


def build_chronological_splits(
    df_m1: pd.DataFrame,
    df_m15: pd.DataFrame,
    stride: int = 15,
    use_wavelet: bool = True,
    include_institutional: bool = False,
    L: int = config.LOOKBACK_M1,
    H: int = config.HORIZON_BARS,
) -> tuple[Samples, Samples, Samples, RobustTrainScaler, pd.DataFrame, pd.DataFrame]:
    """Build strictly chronological Train (70%) / Val (15%) / Test (15%) splits with 360-bar embargo."""
    feat_df, aux_df = build_mtf_features(
        df_m1, df_m15, use_wavelet=use_wavelet, include_institutional=include_institutional
    )

    warmup = max(L + 200, 1500)
    n_total = len(df_m1)
    if n_total <= warmup + H + 2 * config.EMBARGO_BARS + 100:
        warmup = max(L + 60, 400)

    usable_start = warmup
    usable_end = n_total - 2  # at least 1 future bar
    usable_len = usable_end - usable_start

    tr_end = usable_start + int(usable_len * config.TRAIN_FRAC)
    va_start = tr_end + config.EMBARGO_BARS
    va_end = va_start + int(usable_len * config.VAL_FRAC)
    te_start = va_end + config.EMBARGO_BARS
    te_end = usable_end

    # Fit scaler STRICTLY on training bars only (<= tr_end)
    scaler = RobustTrainScaler().fit(feat_df.iloc[usable_start:tr_end])
    X_scaled = scaler.transform(feat_df)
    C_arr = feat_df[STATIC_CTX_COLS].to_numpy(dtype=np.float32)

    close = df_m1["close"].to_numpy(dtype=np.float64)
    atr15 = aux_df["m15_atr14_raw"].ffill().bfill().to_numpy(dtype=np.float64)
    spread = df_m1["spread"].fillna(config.DEFAULT_SPREAD_USD).to_numpy(dtype=np.float64)
    gmask = gap_mask_before(df_m1.index)
    fday = forex_trading_day(df_m1.index).to_numpy()
    ts_ns = df_m1.index.asi8

    tr_orig = np.arange(usable_start, max(usable_start + 1, tr_end - H), stride, dtype=int)
    va_orig = np.arange(va_start, max(va_start + 1, va_end - H), stride, dtype=int)
    te_orig = np.arange(te_start, max(te_start + 1, te_end - H), stride, dtype=int)

    # Filter out origins that have < 15 valid future trading bars (e.g. right before weekend close)
    def _filter_valid(origs):
        keep = []
        for p in origs:
            if p + 15 < n_total and not gmask[p + 1 : p + 16].any():
                keep.append(p)
        return np.array(keep, dtype=int)

    tr_orig = _filter_valid(tr_orig)
    va_orig = _filter_valid(va_orig)
    te_orig = _filter_valid(te_orig)

    tr = _build_samples_for_indices(X_scaled, C_arr, close, atr15, spread, gmask, fday, ts_ns, tr_orig, L, H)
    va = _build_samples_for_indices(X_scaled, C_arr, close, atr15, spread, gmask, fday, ts_ns, va_orig, L, H)
    te = _build_samples_for_indices(X_scaled, C_arr, close, atr15, spread, gmask, fday, ts_ns, te_orig, L, H)
    return tr, va, te, scaler, feat_df, aux_df


def build_live_sample(
    df_m1: pd.DataFrame,
    df_m15: pd.DataFrame,
    scaler: RobustTrainScaler,
    origin_pos: int | None = None,
    use_wavelet: bool = True,
    include_institutional: bool = False,
    L: int = config.LOOKBACK_M1,
) -> tuple[Samples, pd.DataFrame, pd.DataFrame]:
    """Build a single-row Samples object at origin_pos (defaults to latest closed 1M bar)."""
    # Use tail slice of ~3500 bars so warmup (EMA200, pivots, 32x15M) is exact and super fast
    s_m1 = df_m1.iloc[: (origin_pos + 1)] if origin_pos is not None else df_m1
    s_m1 = s_m1.tail(3500).copy()
    from .fetcher.fetch_adapter import resample_m1_to_m15

    s_m15 = resample_m1_to_m15(s_m1)
    feat_df, aux_df = build_mtf_features(
        s_m1, s_m15, use_wavelet=use_wavelet, include_institutional=include_institutional
    )
    X_scaled = scaler.transform(feat_df)
    C_arr = feat_df[STATIC_CTX_COLS].to_numpy(dtype=np.float32)

    pos = len(s_m1) - 1
    X_1 = X_scaled[pos - L + 1 : pos + 1][None, :, :]
    C_1 = C_arr[pos][None, :]
    p0 = float(s_m1["close"].iloc[pos])
    a0 = max(float(aux_df["m15_atr14_raw"].ffill().bfill().iloc[pos]), 1e-4)
    sp = float(s_m1["spread"].iloc[pos]) if "spread" in s_m1.columns else config.DEFAULT_SPREAD_USD

    sample = Samples(
        X=X_1,
        C=C_1,
        y=np.zeros((1, config.HORIZON_BARS), dtype=np.float32),
        m=np.ones((1, config.HORIZON_BARS), dtype=np.float32),
        P0=np.array([p0], dtype=np.float64),
        atr15=np.array([a0], dtype=np.float64),
        eos=np.array([config.HORIZON_BARS], dtype=np.int32),
        times=np.array([s_m1.index.asi8[pos]], dtype=np.int64),
        spread=np.array([sp], dtype=np.float64),
    )
    return sample, feat_df, aux_df
