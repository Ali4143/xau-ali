"""Dedicated Optuna Bayesian Tuner (strictly Train/Validation only, never touches Test)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .. import config
from ..dataset import Samples
from ..models.common import pinball_loss_np
from ..models.lgb_quantile import LGBQuantileForecaster
from ..models.tcn import TCNForecaster


def load_best_params() -> dict:
    if config.BEST_PARAMS_FILE.exists():
        try:
            return json.loads(config.BEST_PARAMS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_best_params(kind: str, params: dict, val_pinball: float, n_trials: int) -> dict:
    cur = load_best_params()
    cur[kind] = {
        "params": params,
        "val_pinball": round(float(val_pinball), 6),
        "n_trials": int(n_trials),
        "tuned_utc": datetime.now(timezone.utc).isoformat(),
    }
    config.BEST_PARAMS_FILE.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    return cur


def tune_lightgbm(tr: Samples, va: Samples, n_trials: int = 12, progress_cb=None) -> dict:
    """Tune LightGBM quantile regressors on validation pinball loss; auto-saves best after every trial."""
    import gc
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fast_steps = (12, 60, 180, 360)
    best_state = {"loss": float("inf"), "params": None, "done": 0}

    def objective(trial) -> float:
        p = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 25, 150),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.95),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 5.0, log=True),
            "n_estimators": 400,
        }
        mdl = LGBQuantileForecaster(steps=fast_steps, params=p).fit(tr, va)
        pred = mdl.predict(va)
        loss = float(pinball_loss_np(va.y, va.m, pred))
        best_state["done"] += 1
        if loss < best_state["loss"]:
            best_state["loss"] = loss
            best_state["params"] = dict(trial.params, n_estimators=600)
            save_best_params("lightgbm", best_state["params"], loss, best_state["done"])
        if progress_cb:
            progress_cb(f"Optuna LightGBM trial {trial.number + 1}/{n_trials}: val_pinball={loss:.5f} (best={best_state['loss']:.5f} saved)")
        del mdl, pred
        gc.collect()
        return loss

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    try:
        study.optimize(objective, n_trials=n_trials)
    except (KeyboardInterrupt, MemoryError, RuntimeError, Exception) as exc:
        if best_state["params"] is not None:
            if progress_cb:
                progress_cb(f"Optuna LightGBM stopped early ({exc}); best trial ({best_state['loss']:.5f}) ALREADY saved!")
            return {"model": "lightgbm", "best_val_pinball": best_state["loss"], "best_params": best_state["params"], "partial": True}
        raise
    best_p = best_state["params"] or dict(study.best_params, n_estimators=600)
    save_best_params("lightgbm", best_p, best_state["loss"], n_trials)
    return {"model": "lightgbm", "best_val_pinball": best_state["loss"], "best_params": best_p}


def tune_tcn(tr: Samples, va: Samples, n_trials: int = 6, epochs: int = 20, progress_cb=None) -> dict:
    """Tune Causal Dilated TCN on validation pinball loss; auto-saves best after every trial."""
    import gc
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    best_state = {"loss": float("inf"), "params": None, "done": 0}

    def objective(trial) -> float:
        hp = {
            "filters": trial.suggest_categorical("filters", [24, 32, 48]),
            "kernel": trial.suggest_categorical("kernel", [2, 3]),
            "levels": trial.suggest_int("levels", 5, 6),
            "dense": trial.suggest_categorical("dense", [48, 64]),
            "dropout": trial.suggest_float("dropout", 0.10, 0.35),
            "lr": trial.suggest_float("lr", 5e-4, 3e-3, log=True),
            "batch": 64,
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True),
        }
        mdl = TCNForecaster(hp=hp, seeds=(11,), epochs=epochs).fit(tr, va, workdir=config.ARTIFACTS_DIR / "optuna_tmp")
        pred = mdl.predict(va)
        loss = float(pinball_loss_np(va.y, va.m, pred))
        best_state["done"] += 1
        if loss < best_state["loss"]:
            best_state["loss"] = loss
            best_state["params"] = dict(trial.params, batch=64)
            save_best_params("tcn", best_state["params"], loss, best_state["done"])
        if progress_cb:
            progress_cb(f"Optuna TCN trial {trial.number + 1}/{n_trials}: val_pinball={loss:.5f} (best={best_state['loss']:.5f} saved)")
        del mdl, pred
        gc.collect()
        return loss

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
    )
    try:
        study.optimize(objective, n_trials=n_trials)
    except (KeyboardInterrupt, MemoryError, RuntimeError, Exception) as exc:
        if best_state["params"] is not None:
            if progress_cb:
                progress_cb(f"Optuna TCN stopped early ({exc}); best trial ({best_state['loss']:.5f}) ALREADY saved!")
            return {"model": "tcn", "best_val_pinball": best_state["loss"], "best_params": best_state["params"], "partial": True}
        raise
    best_hp = best_state["params"] or dict(study.best_params, batch=64)
    save_best_params("tcn", best_hp, best_state["loss"], n_trials)
    return {"model": "tcn", "best_val_pinball": best_state["loss"], "best_params": best_hp}
