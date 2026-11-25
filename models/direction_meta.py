"""Model 6: DirectionMeta (Stacked LightGBM classifier with calibration & Brier score gating)."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .. import config
from ..dataset import Samples

HORIZONS = {"1h": config.H_1H, "3h": config.H_3H, "EOS": None}


def _step_indices(key: str, eos: np.ndarray, H: int) -> np.ndarray:
    st = HORIZONS[key]
    if st is None:
        k = np.asarray(eos, dtype=int) - 1
    else:
        k = np.full(len(eos), st - 1, dtype=int)
    return np.clip(k, 0, H - 1)


def _meta_features(pred: np.ndarray, k: np.ndarray, S: Samples) -> np.ndarray:
    n = len(pred)
    r = np.arange(n)
    kmax = np.clip(S.eos - 1, 0, pred.shape[1] - 1)
    q10, q50, q90 = pred[r, k, 0], pred[r, k, 1], pred[r, k, 2]
    w = np.maximum(q90 - q10, 1e-6)
    xlast = S.X[:, -1, :]
    return np.column_stack(
        [q50, q10, q90, w, q50 / w, pred[:, 0, 1], pred[r, kmax, 1], S.C, xlast]
    ).astype(np.float32)


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    return float(np.mean((p_pred - y_true) ** 2))


def _dir_shift(p_up: float, alpha: float = 0.60, thr: float = 0.03) -> float:
    edge = float(p_up) - 0.5
    if abs(edge) < thr:
        return 0.0
    s_val = float(np.sign(edge) * ((abs(edge) - thr) / max(1e-6, 0.5 - thr)))
    return float(np.clip(alpha * s_val, -0.85, 0.85))


class DirectionMeta:
    name = "direction_meta"

    def __init__(self, params: dict | None = None):
        self.params = {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "num_leaves": 7,
            "min_child_samples": 40,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 5.0,
            "n_jobs": 1,
            "verbose": -1,
            "random_state": 42,
        }
        if params:
            self.params.update(params)
        self.models: dict[str, lgb.LGBMClassifier] = {}
        self.calibrators: dict[str, IsotonicRegression] = {}
        self.val_info: dict[str, dict] = {}
        self.use_steering: bool = False

    def fit(self, pred_va: np.ndarray, va: Samples) -> "DirectionMeta":
        """Fit strictly on OUT-OF-SAMPLE validation predictions (split 60/40 inside val for isotonic calibration)."""
        H = pred_va.shape[1]
        n = len(va)
        split = int(n * 0.65)
        r = np.arange(n)

        improved_any = False
        for h_key in HORIZONS:
            k = _step_indices(h_key, va.eos, H)
            ok = (va.m[r, k] > 0) & (va.y[r, k] != 0)
            X_all = _meta_features(pred_va, k, va)
            lab_all = (va.y[r, k] > 0).astype(int)

            ok_tr = ok.copy()
            ok_tr[split:] = False
            ok_cal = ok.copy()
            ok_cal[:split] = False

            if ok_tr.sum() < 80 or ok_cal.sum() < 40:
                self.val_info[h_key] = {"skip": True, "n_tr": int(ok_tr.sum()), "n_cal": int(ok_cal.sum())}
                continue
            y_tr, y_cal = lab_all[ok_tr], lab_all[ok_cal]
            if y_tr.min() == y_tr.max() or y_cal.min() == y_cal.max():
                self.val_info[h_key] = {"skip": True, "reason": "single class"}
                continue

            mcs = int(max(15, min(self.params["min_child_samples"], len(y_tr) // 15)))
            p_bin = {
                "objective": "binary",
                "metric": "binary_logloss",
                "learning_rate": float(self.params.get("learning_rate", 0.03)),
                "num_leaves": int(self.params.get("num_leaves", 7)),
                "min_child_samples": mcs,
                "subsample": float(self.params.get("subsample", 0.8)),
                "subsample_freq": 1,
                "colsample_bytree": float(self.params.get("colsample_bytree", 0.8)),
                "reg_lambda": float(self.params.get("reg_lambda", 5.0)),
                "num_threads": 1,
                "verbose": -1,
                "seed": 42,
            }
            dtr = lgb.Dataset(np.ascontiguousarray(X_all[ok_tr], dtype=np.float64), label=np.ascontiguousarray(y_tr, dtype=np.float64))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mdl = lgb.train(p_bin, dtr, num_boost_round=int(self.params.get("n_estimators", 200)))
                raw_p = mdl.predict(np.ascontiguousarray(X_all[ok_cal], dtype=np.float64))

            iso = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds="clip")
            iso.fit(raw_p, y_cal)
            cal_p = iso.predict(raw_p)

            bs_base = brier_score(y_cal, np.full_like(y_cal, y_tr.mean(), dtype=float))
            bs_raw = brier_score(y_cal, raw_p)
            bs_cal = brier_score(y_cal, cal_p)
            improves = bs_cal < bs_base
            if improves:
                improved_any = True

            self.models[h_key] = mdl
            self.calibrators[h_key] = iso
            self.val_info[h_key] = {
                "n_tr": int(ok_tr.sum()),
                "n_cal": int(ok_cal.sum()),
                "brier_baseline": round(bs_base, 5),
                "brier_raw": round(bs_raw, 5),
                "brier_calibrated": round(bs_cal, 5),
                "improves_brier": bool(improves),
            }

        self.use_steering = improved_any
        return self

    def predict(self, pred: np.ndarray, S: Samples) -> dict[str, np.ndarray]:
        H = pred.shape[1]
        out = {}
        for h_key, mdl in self.models.items():
            k = _step_indices(h_key, S.eos, H)
            X = _meta_features(pred, k, S)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_p = mdl.predict(np.ascontiguousarray(X, dtype=np.float64), num_threads=1)
            cal = self.calibrators.get(h_key)
            out[h_key] = cal.predict(raw_p) if cal is not None else raw_p
        return out

    def predict_live_dict(self, pred_1: np.ndarray, S_1: Samples) -> dict:
        probs = self.predict(pred_1, S_1)
        res = {}
        for h_key, arr in probs.items():
            pu = float(arr[0])
            conf = max(pu, 1.0 - pu)
            res[h_key] = {
                "p_up": round(pu, 4),
                "side": "up" if pu >= 0.5 else "down",
                "conf": round(conf, 4),
                "calibrated": h_key in self.calibrators,
            }
        return res

    def adjust_median(self, q_1: np.ndarray, dirc_live: dict | None, alpha: float = 0.60, thr: float = 0.03) -> np.ndarray:
        """Steer q50 inside [q10, q90] ONLY if DirectionMeta improved Brier score on validation."""
        q50 = np.asarray(q_1[:, 1], dtype=float).copy()
        H = len(q50)
        if not self.use_steering or not dirc_live or H <= 1:
            return q50
        pts = {0: 0.0}
        for key, kk in (("1h", min(config.H_1H, H) - 1), ("3h", min(config.H_3H, H) - 1), ("EOS", H - 1)):
            entry = dirc_live.get(key)
            if isinstance(entry, dict) and "p_up" in entry and kk > 0:
                pts[kk] = _dir_shift(entry["p_up"], alpha=alpha, thr=thr)
        if len(pts) <= 1:
            return q50
        ak = np.array(sorted(pts), dtype=float)
        ash = np.array([pts[int(k)] for k in ak], dtype=float)
        sh = np.interp(np.arange(H, dtype=float), ak, ash)
        up_room = np.maximum(0.0, q_1[:, 2] - q50)
        dn_room = np.maximum(0.0, q50 - q_1[:, 0])
        return q50 + np.where(sh >= 0.0, sh * up_room, sh * dn_room)

    def save(self, d: Path) -> None:
        d = Path(d) / "direction_meta"
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "models": self.models,
                "calibrators": self.calibrators,
                "val_info": self.val_info,
                "use_steering": self.use_steering,
                "params": self.params,
            },
            d / "bundle.joblib",
        )
        (d / "val_info.json").write_text(json.dumps(self.val_info, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, d: Path) -> "DirectionMeta":
        d = Path(d) / "direction_meta"
        b = joblib.load(d / "bundle.joblib")
        obj = cls(params=b.get("params"))
        obj.models = b["models"]
        obj.calibrators = b["calibrators"]
        obj.val_info = b["val_info"]
        obj.use_steering = bool(b.get("use_steering", False))
        return obj
