"""Non-destructive v2 evaluation runner:
Loads existing trained model artifacts (NEVER overwrites them!), applies v2 validation exclusion rule,
saves ensemble_weights_v2.json, runs v2 TEST metrics + walk-forward folds + heuristic correlation,
and writes xau_mtf/reports/v2/metrics_v2.{json,md}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config
from .dataset import RobustTrainScaler, build_chronological_splits
from .ensemble import ENSEMBLE_WEIGHTS_V2_FILE, blend_predictions, fit_ensemble_weights_v2, save_weights
from .evaluation import (
    compute_directional_accuracy,
    evaluate_model_on_test,
    evaluate_target_high_low,
    evaluate_walk_forward_folds,
    save_metrics_report,
)
from .fetcher.fetch_adapter import load_m1, load_m15, meta_path
from .models.baselines import HistoricalProfileBaseline, NoChangeBaseline
from .models.direction_meta import DirectionMeta
from .pipeline import CORE_MODEL_CLASSES


def run_v2_evaluation(symbol: str = config.DEFAULT_SYMBOL, commission_usd: float = 0.0) -> dict:
    model_dir = config.MODELS_DIR / symbol.lower()
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    stride = int(manifest.get("stride", 15))
    use_wavelet = bool(manifest.get("use_wavelet", True))

    df_m1 = load_m1(symbol)
    df_m15 = load_m15(symbol)
    tr, va, te, _, _, _ = build_chronological_splits(df_m1, df_m15, stride=stride, use_wavelet=use_wavelet)

    nc = NoChangeBaseline.load(model_dir)
    hp = HistoricalProfileBaseline.load(model_dir)
    va_nc, te_nc = nc.predict(va), nc.predict(te)
    te_hp = hp.predict(te)

    va_preds, te_preds = {}, {}
    for m_name in CORE_MODEL_CLASSES:
        if m_name in manifest.get("models_trained", []):
            mdl = CORE_MODEL_CLASSES[m_name].load(model_dir)
            va_preds[m_name] = mdl.predict(va)
            te_preds[m_name] = mdl.predict(te)

    # Fit v2 weights strictly on VALIDATION set with NoChange exclusion rule
    weights_v2, weight_diag = fit_ensemble_weights_v2(va_preds, va.y, va.m, va.atr15, va_nc)
    save_weights(weights_v2, path=ENSEMBLE_WEIGHTS_V2_FILE, diag=weight_diag)

    te_ens_v2 = blend_predictions(te_preds, weights_v2)

    results = [
        evaluate_model_on_test("no_change", te_nc, te, bench_norm=None, commission_usd=commission_usd),
        evaluate_model_on_test("hist_profile", te_hp, te, bench_norm=te_nc, commission_usd=commission_usd),
    ]
    for m_name, p_te in te_preds.items():
        results.append(evaluate_model_on_test(m_name, p_te, te, bench_norm=te_nc, commission_usd=commission_usd))

    ens_res = evaluate_model_on_test("ensemble", te_ens_v2, te, bench_norm=te_nc, commission_usd=commission_usd)
    results.append(ens_res)

    wf_res = evaluate_walk_forward_folds(te_ens_v2, te_nc, te)
    hl_eval = evaluate_target_high_low(te_ens_v2, te)

    # Evaluate Heuristic Score predictive value (correlation with next-hour direction on TEST set)
    k_1h = min(config.H_1H, te.y.shape[1]) - 1
    ok_1h = te.m[:, k_1h] > 0
    y_1h = te.y[ok_1h, k_1h] * te.atr15[ok_1h]
    q50_1h = te_ens_v2[ok_1h, k_1h, 1] * te.atr15[ok_1h]
    # Reconstruct heuristic momentum proxy from last feature bar (tick_flow_15, rsi14) + q50_1h
    # Feature correlation with realized 1h return
    corr_q50 = float(np.corrcoef(q50_1h, y_1h)[0, 1]) if np.std(q50_1h) > 1e-6 and np.std(y_1h) > 1e-6 else 0.0
    da_heur = compute_directional_accuracy(y_1h, q50_1h)
    heur_eval = {
        "note": "HEURISTIC SCORE, not validated probability. Evaluated on TEST set 1h horizon.",
        "q50_1h_corr_with_actual_1h": round(0.0 if np.isnan(corr_q50) else corr_q50, 4),
        "q50_1h_da_strict": da_heur,
    }

    dir_brier = {}
    if (model_dir / "direction_meta" / "val_info.json").exists():
        dir_brier = json.loads((model_dir / "direction_meta" / "val_info.json").read_text(encoding="utf-8"))

    data_meta = json.loads(meta_path(symbol).read_text(encoding="utf-8")) if meta_path(symbol).exists() else {}
    v2_dir = config.REPORTS_DIR / "v2"
    report = save_metrics_report(
        results=results,
        ensemble_res=ens_res,
        target_hl_res=hl_eval,
        dir_brier=dir_brier,
        weights=weights_v2,
        meta=data_meta,
        out_dir=v2_dir,
        wf_res=wf_res,
        weight_diag=weight_diag,
        heuristic_eval=heur_eval,
        filename_stem="metrics_v2",
    )
    return report


if __name__ == "__main__":
    rep = run_v2_evaluation("XAUUSD")
    print("V2 GATE VERDICT:", json.dumps(rep["deployment_gate"], indent=2))
    print("V2 WEIGHTS:", json.dumps(rep["ensemble_weights"], indent=2))
