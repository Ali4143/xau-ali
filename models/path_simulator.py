"""Model 7: PathSimulator (9-Quantile Monte Carlo Markov Zig-Zag Path Generator for XAUUSD)."""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .. import config
from ..calendar_ny import gap_mask_before

LEVELS = np.array(config.SIM_LEVELS, dtype=np.float64)
BPS = 1e4
WARMUP = 600


class _SimState:
    """Vectorized recursive causal indicator state machine for N simulated paths."""

    def __init__(self, c0: np.ndarray):
        c0 = np.asarray(c0, dtype=np.float64)
        n = len(c0)
        self.C = c0.copy()
        self.e = {k: c0.copy() for k in (9, 12, 21, 26, 50)}
        self.sig = np.zeros(n)
        self.ag = np.zeros(n)
        self.al = np.zeros(n)
        self.win = np.tile(c0[:, None], (1, 20))
        self.rets = np.zeros((n, 12))

    def step(self, c: np.ndarray) -> None:
        c = np.asarray(c, dtype=np.float64)
        r = np.log(np.maximum(c, 1e-6) / np.maximum(self.C, 1e-6))
        for k in self.e:
            self.e[k] = self.e[k] + (2.0 / (k + 1.0)) * (c - self.e[k])
        macd = self.e[12] - self.e[26]
        self.sig = self.sig + 0.2 * (macd - self.sig)
        d = c - self.C
        self.ag = self.ag + (np.maximum(d, 0.0) - self.ag) / 14.0
        self.al = self.al + (np.maximum(-d, 0.0) - self.al) / 14.0
        self.win = np.concatenate([self.win[:, 1:], c[:, None]], axis=1)
        self.rets = np.concatenate([self.rets[:, 1:], r[:, None]], axis=1)
        self.C = c

    def tile(self, n: int) -> "_SimState":
        s = _SimState.__new__(_SimState)
        s.C = np.repeat(self.C, n)
        s.e = {k: np.repeat(v, n) for k, v in self.e.items()}
        s.sig, s.ag, s.al = np.repeat(self.sig, n), np.repeat(self.ag, n), np.repeat(self.al, n)
        s.win, s.rets = np.repeat(self.win, n, axis=0), np.repeat(self.rets, n, axis=0)
        return s

    def features(self, utc_hr: float) -> np.ndarray:
        C = self.C
        tot = self.ag + self.al
        rsi = np.where(tot > 0, self.ag / np.where(tot > 0, tot, 1.0), 0.5) - 0.5
        m20, s20 = self.win.mean(1), self.win.std(1)
        macd = self.e[12] - self.e[26]
        n = len(C)
        f = np.empty((n, 12), dtype=np.float32)
        f[:, 0] = self.rets[:, -1] * BPS
        f[:, 1] = self.rets[:, -2] * BPS
        f[:, 2] = self.rets[:, -3] * BPS
        f[:, 3] = (C - self.e[9]) / C * BPS
        f[:, 4] = (C - self.e[21]) / C * BPS
        f[:, 5] = (C - self.e[50]) / C * BPS
        f[:, 6] = rsi
        f[:, 7] = (macd - self.sig) / C * BPS
        f[:, 8] = (C - m20) / (2.0 * s20 + 1e-9)
        f[:, 9] = self.rets.std(1) * BPS
        f[:, 10] = np.sin(2.0 * np.pi * utc_hr / 24.0)
        f[:, 11] = np.cos(2.0 * np.pi * utc_hr / 24.0)
        return f


def _pick_best_path(lr: np.ndarray, tgt: np.ndarray) -> int:
    H = lr.shape[1]
    fh = min(H, config.H_1H)
    w = np.ones(H, dtype=np.float64)
    w[:fh] = 3.0
    d = np.abs(lr - tgt[None, :])
    score = (d * w[None, :]).sum(axis=1) / w.sum() + 0.5 * d.max(axis=1)
    return int(np.argmin(score))


def compute_path_targets(paths: np.ndarray, P0: float, pivots: dict | None = None) -> dict:
    """Extract Target High (median max), Target Low (median min), close quantiles, and pivot touch probs."""
    hi = paths.max(axis=1)
    lo = paths.min(axis=1)
    q = lambda arr: [float(v) for v in np.quantile(arr, [0.10, 0.50, 0.90])]  # noqa: E731
    out = {
        "n_paths": int(len(paths)),
        "day_high": q(hi),
        "day_low": q(lo),
        "close": q(paths[:, -1]),
        "up_prob": float((paths[:, -1] > P0).mean()),
        "touch": {},
    }
    if pivots:
        for k in ("R3", "R2", "R1", "P", "S1", "S2", "S3"):
            v = pivots.get(k)
            if v is None or not np.isfinite(v):
                continue
            p = float((hi >= v).mean()) if v >= P0 else float((lo <= v).mean())
            out["touch"][k] = {"level": float(v), "prob": round(p, 4), "side": "up" if v >= P0 else "down"}
    return out


class PathSimulator:
    name = "path_simulator"

    def __init__(self, params: dict | None = None):
        self.params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 80,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "n_jobs": 1,
            "verbose": -1,
            "random_state": 42,
        }
        if params:
            self.params.update(params)
        self.models: list[lgb.Booster] = []

    def fit(self, df_m1: pd.DataFrame, train_end_idx: int, val_end_idx: int, progress_cb=None) -> "PathSimulator":
        s = df_m1.sort_index()
        close = s["close"].to_numpy(dtype=np.float64)
        gmask = gap_mask_before(s.index)
        hrs = (s.index.hour + s.index.minute / 60.0).to_numpy(dtype=np.float64)

        st = _SimState(close[:1])
        X_list, y_list, idx_list = [], [], []
        # Step through series up to val_end_idx (never touch Test!)
        limit = min(val_end_idx, len(s) - 1)
        for i in range(1, limit):
            st.step(close[i : i + 1])
            if i < 200 or gmask[i + 1]:
                continue
            X_list.append(st.features(hrs[i])[0])
            y_list.append(np.log(close[i + 1] / close[i]) * BPS)
            idx_list.append(i)

        X = np.asarray(X_list, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.float64)
        idx_arr = np.asarray(idx_list, dtype=int)
        tr_m = idx_arr < train_end_idx
        va_m = (idx_arr >= train_end_idx + config.EMBARGO_BARS) & (idx_arr < val_end_idx)

        self.models = []
        X_tr_s = np.ascontiguousarray(X[tr_m], dtype=np.float64)
        y_tr_s = np.ascontiguousarray(y[tr_m], dtype=np.float64)
        X_va_s = np.ascontiguousarray(X[va_m], dtype=np.float64) if va_m.sum() > 50 else None
        y_va_s = np.ascontiguousarray(y[va_m], dtype=np.float64) if va_m.sum() > 50 else None
        dtr = lgb.Dataset(X_tr_s, label=y_tr_s, free_raw_data=False)
        dva = lgb.Dataset(X_va_s, label=y_va_s, reference=dtr, free_raw_data=False) if X_va_s is not None else None
        n_rounds = int(self.params.get("n_estimators", 300))
        base_p = {
            "learning_rate": float(self.params.get("learning_rate", 0.05)),
            "num_leaves": int(self.params.get("num_leaves", 31)),
            "min_child_samples": int(self.params.get("min_child_samples", 80)),
            "subsample": float(self.params.get("subsample", 0.8)),
            "subsample_freq": 1,
            "colsample_bytree": float(self.params.get("colsample_bytree", 0.8)),
            "num_threads": 1,
            "verbose": -1,
            "seed": 42,
        }
        for qi, q in enumerate(LEVELS):
            p_q = {**base_p, "objective": "quantile", "alpha": float(q), "metric": "quantile"}
            cbs = [lgb.early_stopping(25, verbose=False)] if dva is not None else []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                booster = lgb.train(
                    p_q,
                    dtr,
                    num_boost_round=n_rounds,
                    valid_sets=[dva] if dva is not None else None,
                    callbacks=cbs,
                )
            self.models.append(booster)
            if progress_cb:
                progress_cb(f"path_simulator quantile {qi + 1}/{len(LEVELS)} ({q:.2f}) done")
        return self

    def _quantiles(self, F: np.ndarray) -> np.ndarray:
        F64 = np.ascontiguousarray(F, dtype=np.float64)
        Q = np.column_stack([b.predict(F64, num_threads=1) for b in self.models])
        return np.sort(Q, axis=1)

    @staticmethod
    def _sample(Q: np.ndarray, u: np.ndarray) -> np.ndarray:
        k = np.clip(np.searchsorted(LEVELS, u) - 1, 0, len(LEVELS) - 2)
        w = (u - LEVELS[k]) / (LEVELS[k + 1] - LEVELS[k])
        rows = np.arange(len(Q))
        return Q[rows, k] + w * (Q[rows, k + 1] - Q[rows, k])

    def simulate(
        self,
        df_upto: pd.DataFrame,
        horizon: int = config.HORIZON_BARS,
        n_paths: int = config.SIM_PATHS,
        target_price_q50: np.ndarray | None = None,
        pivots: dict | None = None,
        seed: int | None = None,
    ) -> dict:
        s = df_upto.sort_index().tail(WARMUP)
        close = s["close"].to_numpy(dtype=np.float64)
        origin_ts = s.index[-1]
        P0 = float(close[-1])

        st = _SimState(close[:1])
        for c_val in close[1:]:
            st.step(np.array([c_val]))

        if seed is None:
            seed = int(hashlib.sha256(str(origin_ts).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        S = st.tile(n_paths)

        H = int(horizon)
        paths = np.empty((n_paths, H), dtype=np.float64)
        hr0 = origin_ts.hour + origin_ts.minute / 60.0
        for j in range(H):
            hr_j = (hr0 + (j + 1) / 60.0) % 24.0
            F = S.features(hr_j)
            r = self._sample(self._quantiles(F), rng.uniform(0.01, 0.99, n_paths)) / BPS
            S.step(S.C * np.exp(r))
            paths[:, j] = S.C

        lr = np.log(paths / P0)
        if target_price_q50 is not None and len(target_price_q50) >= H:
            tgt_lr = np.log(np.maximum(np.asarray(target_price_q50[:H], dtype=np.float64), 1e-6) / P0)
            lr_eff = lr + (tgt_lr - np.median(lr, axis=0))[None, :]
            best = _pick_best_path(lr_eff, tgt_lr)
            res = lr_eff[best] - tgt_lr
            res_dt = res - np.linspace(0.0, float(res[-1]), H)
            chosen_lr = tgt_lr + res_dt
            chosen_path = P0 * np.exp(chosen_lr)
            paths_eff = P0 * np.exp(lr_eff)
        else:
            tgt_lr = np.median(lr, axis=0)
            best = _pick_best_path(lr, tgt_lr)
            chosen_path = paths[best]
            paths_eff = paths

        qs = np.quantile(paths_eff, [0.10, 0.50, 0.90], axis=0)
        return {
            "P0": P0,
            "path": chosen_path.tolist(),
            "sim_q10": qs[0].tolist(),
            "sim_q50": qs[1].tolist(),
            "sim_q90": qs[2].tolist(),
            "n_paths": n_paths,
            "seed": seed,
            "targets": compute_path_targets(paths_eff, P0, pivots=pivots),
        }

    def save(self, d: Path) -> None:
        d = Path(d) / "path_simulator"
        d.mkdir(parents=True, exist_ok=True)
        for i, b in enumerate(self.models):
            b.save_model(str(d / f"q{i}.txt"))
        (d / "meta.json").write_text(
            json.dumps({"levels": LEVELS.tolist(), "params": self.params}, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, d: Path) -> "PathSimulator":
        d = Path(d) / "path_simulator"
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        obj = cls(params=meta.get("params"))
        obj.models = [lgb.Booster(model_file=str(d / f"q{i}.txt")) for i in range(len(meta["levels"]))]
        return obj
