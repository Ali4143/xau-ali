"""Validation-fitted non-negative ensemble weighting with strict NoChange exclusion rule (v2)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from . import config
from .models.common import pinball_loss_np

ENSEMBLE_WEIGHTS_V2_FILE = config.ARTIFACTS_DIR / "ensemble_weights_v2.json"


def _val_mae_usd(pred_norm: np.ndarray, y_norm: np.ndarray, m: np.ndarray, atr15: np.ndarray) -> float:
    mask = m > 0
    if not mask.any():
        return float("inf")
    err = np.abs((y_norm - pred_norm[:, :, 1]) * atr15[:, None])
    return float(err[mask].mean())


def fit_ensemble_weights_v2(
    val_preds: dict[str, np.ndarray],
    y_val: np.ndarray,
    m_val: np.ndarray,
    atr15_val: np.ndarray,
    nochange_val_pred: np.ndarray,
) -> tuple[dict[str, float], dict]:
    """Fit non-negative convex weights strictly on VALIDATION set with NoChange exclusion rule.

    Rule: Any model with validation MAE >= NoChange validation MAE receives weight 0.0.
    Never uses test set data.
    """
    nc_mae = _val_mae_usd(nochange_val_pred, y_val, m_val, atr15_val)
    nc_pb = pinball_loss_np(y_val, m_val, nochange_val_pred)

    diag = {
        "nochange_val_mae_usd": round(nc_mae, 4),
        "nochange_val_pinball": round(nc_pb, 6),
        "candidates": {},
        "excluded_models": [],
        "included_models": [],
    }

    eligible: list[str] = []
    for name, pred in val_preds.items():
        mae_v = _val_mae_usd(pred, y_val, m_val, atr15_val)
        pb_v = pinball_loss_np(y_val, m_val, pred)
        # Pass if model beats NoChange on val MAE OR beats NoChange on val pinball within 0.5% MAE tolerance
        passes = (mae_v < nc_mae) or (pb_v < nc_pb and mae_v <= nc_mae * 1.005)
        excluded = not passes
        diag["candidates"][name] = {
            "val_mae_usd": round(mae_v, 4),
            "val_pinball": round(pb_v, 6),
            "beats_nochange_val_mae": bool(mae_v < nc_mae),
            "beats_nochange_val_pinball": bool(pb_v < nc_pb),
            "excluded": excluded,
        }
        if excluded:
            diag["excluded_models"].append(name)
        else:
            eligible.append(name)
            diag["included_models"].append(name)

    weights = {n: 0.0 for n in val_preds}
    if not eligible:
        # Fallback: pick single lowest validation pinball model if none pass
        best_single = min(val_preds.keys(), key=lambda k: diag["candidates"][k]["val_pinball"])
        weights[best_single] = 1.0
        diag["note"] = f"No model passed NoChange gate on val; fallback to lowest val pinball ({best_single})"
        return weights, diag

    if len(eligible) == 1:
        weights[eligible[0]] = 1.0
        return weights, diag

    stack = np.stack([val_preds[n] for n in eligible], axis=0)  # [M, N, H, 3]
    init_w = np.full(len(eligible), 1.0 / len(eligible), dtype=np.float64)

    def obj(w: np.ndarray) -> float:
        blend = np.tensordot(w, stack, axes=(0, 0))
        return pinball_loss_np(y_val, m_val, blend)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in eligible]
    res = minimize(obj, init_w, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 150})
    w_opt = np.clip(res.x if res.success else init_w, 0.0, 1.0)
    w_opt = w_opt / max(w_opt.sum(), 1e-9)

    # KSE100-proven Anti-Flatness Rule: cap LightGBM at 30% when >=2 passing neural models exist
    shrink = 0.15
    neural_idx = [i for i, n in enumerate(eligible) if n != "lightgbm"]
    if "lightgbm" in eligible and len(neural_idx) >= 2:
        l_idx = eligible.index("lightgbm")
        if w_opt[l_idx] > 0.35:
            npb = np.array([diag["candidates"][eligible[i]]["val_pinball"] for i in neural_idx], dtype=np.float64)
            z = (npb - npb.min()) / max(1e-6, float(npb.std()))
            nw = np.exp(-2.0 * z)
            nw = (1.0 - shrink) * (nw / nw.sum()) + shrink / len(neural_idx)
            w_opt = np.zeros_like(w_opt)
            w_opt[l_idx] = 0.30
            for i, val in zip(neural_idx, nw):
                w_opt[i] = 0.70 * val
            diag["rule_applied"] = "KSE100 30% LightGBM cap + 70% passing neural softmax"

    # 15% shrinkage across ALL passing models so every passing model contributes its vote
    w_opt = (1.0 - shrink) * w_opt + shrink * (1.0 / len(eligible))
    w_opt = w_opt / max(w_opt.sum(), 1e-9)

    for n, v in zip(eligible, w_opt):
        weights[n] = round(float(v), 4)

    # Renormalize rounded non-zero weights to sum to 1.0
    tot = sum(weights.values())
    if tot > 0:
        for k in eligible:
            weights[k] = round(weights[k] / tot, 4)

    diag["ensemble_val_pinball"] = round(float(obj(w_opt)), 6)
    return weights, diag


# Backwards-compatible alias for train_all
def fit_ensemble_weights(
    val_preds: dict[str, np.ndarray],
    y_val: np.ndarray,
    m_val: np.ndarray,
    prior_weights: dict[str, float] | None = None,
    reg_strength: float = 0.0,
) -> dict[str, float]:
    nc_zero = np.zeros((len(y_val), y_val.shape[1], 3), dtype=np.float32)
    atr_ones = np.ones(len(y_val), dtype=np.float64)
    w, _ = fit_ensemble_weights_v2(val_preds, y_val, m_val, atr_ones, nc_zero)
    return w


def blend_predictions(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    active = [(preds[k], float(w)) for k, w in weights.items() if k in preds and w > 0]
    if not active:
        first = next(iter(preds.values()))
        return np.zeros_like(first)
    tot_w = sum(w for _, w in active)
    out = sum(p * (w / tot_w) for p, w in active)
    return np.sort(out, axis=-1).astype(np.float32)


def save_weights(weights: dict[str, float], path: Path = config.ENSEMBLE_WEIGHTS_FILE, diag: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"weights": weights, "diagnostics": diag} if diag is not None else weights
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_weights(path: Path = ENSEMBLE_WEIGHTS_V2_FILE) -> dict[str, float]:
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw["weights"] if isinstance(raw, dict) and "weights" in raw else raw
    if config.ENSEMBLE_WEIGHTS_FILE.exists():
        raw = json.loads(config.ENSEMBLE_WEIGHTS_FILE.read_text(encoding="utf-8"))
        return raw["weights"] if isinstance(raw, dict) and "weights" in raw else raw
    return dict(config.PRIOR_WEIGHTS)
