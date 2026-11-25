"""Model 2: LightGBM Multi-Step Quantile Regressors (pinball loss across horizon knots)."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .. import config
from ..dataset import Samples

LGB_LAST_TAPS = (1, 5, 15, 30, 60, 120)  # Multi-scale causal lags from the 120-bar lookback


def flatten_mtf_lags(S: Samples) -> np.ndarray:
    """Extract multi-scale lagged slices from S.X [N, L, F] and concatenate with static context S.C."""
    L = S.X.shape[1]
    taps = [L - t for t in LGB_LAST_TAPS if t <= L]
    parts = [S.X[:, idx, :] for idx in taps]
    parts.append(S.C)
    return np.concatenate(parts, axis=1).astype(np.float32)


class LGBQuantileForecaster:
    name = "lightgbm"

    def __init__(self, steps=config.LGB_STEPS, params: dict | None = None):
        self.steps = tuple(steps)
        self.params = {
            "n_estimators": 600,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 60,
            "subsample": 0.75,
            "subsample_freq": 1,
            "colsample_bytree": 0.70,
            "reg_lambda": 1.0,
            "n_jobs": 1,
            "verbose": -1,
            "random_state": 42,
        }
        if params:
            self.params.update(params)
        self.models: dict[tuple[int, int], lgb.Booster] = {}
        self.H: int = config.HORIZON_BARS

    def fit(self, tr: Samples, va: Samples, workdir: Path | None = None, progress_cb=None) -> "LGBQuantileForecaster":
        self.H = tr.y.shape[1]
        Xtr, Xva = flatten_mtf_lags(tr), flatten_mtf_lags(va)
        self.models = {}

        for idx_s, s in enumerate(self.steps):
            k = s - 1
            if k >= self.H:
                continue
            mt = tr.m[:, k] > 0
            mv = va.m[:, k] > 0
            if mt.sum() < 40:
                continue
            Xtr_k = np.ascontiguousarray(Xtr[mt], dtype=np.float64)
            ytr_k = np.ascontiguousarray(tr.y[mt, k], dtype=np.float64)
            Xva_k = np.ascontiguousarray(Xva[mv], dtype=np.float64) if mv.sum() >= 20 else None
            yva_k = np.ascontiguousarray(va.y[mv, k], dtype=np.float64) if mv.sum() >= 20 else None
            dtr = lgb.Dataset(Xtr_k, label=ytr_k, free_raw_data=False)
            dva = lgb.Dataset(Xva_k, label=yva_k, reference=dtr, free_raw_data=False) if Xva_k is not None else None
            n_rounds = int(self.params.get("n_estimators", 600))
            base_p = {
                "learning_rate": float(self.params.get("learning_rate", 0.03)),
                "num_leaves": int(self.params.get("num_leaves", 31)),
                "min_child_samples": int(self.params.get("min_child_samples", 60)),
                "subsample": float(self.params.get("subsample", 0.75)),
                "subsample_freq": int(self.params.get("subsample_freq", 1)),
                "colsample_bytree": float(self.params.get("colsample_bytree", 0.70)),
                "reg_lambda": float(self.params.get("reg_lambda", 1.0)),
                "num_threads": 1,
                "verbose": -1,
                "seed": int(self.params.get("random_state", 42)),
            }
            for qi, q in enumerate(config.QUANTILES):
                p_q = {**base_p, "objective": "quantile", "alpha": float(q), "metric": "quantile"}
                cbs = [lgb.early_stopping(30, verbose=False)] if dva is not None else []
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    booster = lgb.train(
                        p_q,
                        dtr,
                        num_boost_round=n_rounds,
                        valid_sets=[dva] if dva is not None else None,
                        callbacks=cbs,
                    )
                self.models[(s, qi)] = booster
            if progress_cb:
                progress_cb(f"lightgbm knot {idx_s + 1}/{len(self.steps)} (step {s}) done")
        return self

    def predict(self, S: Samples) -> np.ndarray:
        n = len(S)
        out = np.zeros((n, self.H, 3), dtype=np.float32)
        if not n or not self.models:
            return out
        X = np.ascontiguousarray(flatten_mtf_lags(S), dtype=np.float64)
        steps = sorted({s for s, _ in self.models})
        grid = np.arange(1, self.H + 1, dtype=float)
        xs = np.array([0.0] + [float(s) for s in steps], dtype=float)

        for qi in range(3):
            pts = np.column_stack([self.models[(s, qi)].predict(X, num_threads=1) for s in steps])
            for i in range(n):
                out[i, :, qi] = np.interp(grid, xs, np.concatenate([[0.0], pts[i]]))
        return np.sort(out, axis=-1)

    def save(self, d: Path) -> None:
        d = Path(d) / "lightgbm"
        d.mkdir(parents=True, exist_ok=True)
        for (s, qi), booster in self.models.items():
            booster.save_model(str(d / f"s{s}_q{qi}.txt"))
        meta = {
            "steps": list(self.steps),
            "keys": [list(k) for k in self.models],
            "H": self.H,
            "params": self.params,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, d: Path) -> "LGBQuantileForecaster":
        d = Path(d) / "lightgbm"
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        obj = cls(steps=meta["steps"], params=meta.get("params"))
        obj.H = int(meta.get("H", config.HORIZON_BARS))
        for s, qi in meta["keys"]:
            obj.models[(int(s), int(qi))] = lgb.Booster(model_file=str(d / f"s{s}_q{qi}.txt"))
        return obj
