"""Out-of-sample TEST evaluation (v2):
- Fixed Directional Accuracy (excludes flat forecasts pred==0 and zero-change y==0; NoChange = N/A)
- HAC Newey-West DM (Bartlett kernel, lag = h-1) + Non-Overlapping DM (every h bars) with min n >= 30 gate
- Holm-Bonferroni multiple-comparison correction (raw + adjusted p-values)
- Strict 5-condition Deployment Gate (DEPLOYABLE vs NOT VERIFIED with failed condition list)
- Position-aware Cost Model (cost applied ONLY when position != 0; flat = $0.00 cost)
- Walk-forward rolling fold evaluation & heuristic score correlation check
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config
from .dataset import Samples
from .models.common import pinball_loss_np

MIN_DM_SAMPLES = 30
POSITION_ATR_THRESHOLD = 0.10  # Open +/-1 position only if |q50[+60]| > 0.10 * ATR14_15M


def compute_directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> dict:
    """Strict Directional Accuracy excluding flat forecasts (sign(pred)==0) and zero-change truth (sign(y)==0)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if valid_mask is not None:
        m = np.asarray(valid_mask, dtype=bool)
        yt, yp = yt[m], yp[m]

    finite = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[finite], yp[finite]
    n_total = int(len(yt))

    s_pred = np.sign(yp)
    s_true = np.sign(yt)
    excl_flat = int(np.sum(s_pred == 0))
    excl_zero_y = int(np.sum((s_pred != 0) & (s_true == 0)))

    keep = (s_pred != 0) & (s_true != 0)
    n_eval = int(np.sum(keep))
    if n_eval == 0:
        return {
            "da_pct": None,
            "label": "N/A (flat forecast)" if excl_flat == n_total else "N/A",
            "n_total": n_total,
            "n_eval": 0,
            "excluded_flat": excl_flat,
            "excluded_zero_y": excl_zero_y,
        }
    da = round(float(np.mean(s_pred[keep] == s_true[keep]) * 100.0), 2)
    return {
        "da_pct": da,
        "label": f"{da:.2f}%",
        "n_total": n_total,
        "n_eval": n_eval,
        "excluded_flat": excl_flat,
        "excluded_zero_y": excl_zero_y,
    }


def hac_dm_test(loss_model: np.ndarray, loss_bench: np.ndarray, lag: int = 24) -> dict:
    """One-sided Diebold-Mariano test with Newey-West (Bartlett kernel) HAC variance.

    H1: model has strictly lower loss than benchmark (mean(loss_bench - loss_model) > 0).
    """
    d = np.asarray(loss_bench - loss_model, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = int(len(d))
    if n < 5:
        return {
            "dm_stat": 0.0,
            "p_value": 1.0,
            "p_two_sided": 1.0,
            "n": n,
            "lag": 0,
            "var_naive": 0.0,
            "var_hac": 0.0,
            "sufficient_sample": False,
            "status": "INSUFFICIENT SAMPLE",
        }

    mean_d = float(np.mean(d))
    dc = d - mean_d
    eff_lag = max(0, min(int(lag), n - 2))

    gamma0 = float(np.dot(dc, dc) / n)
    var_hac = gamma0
    for k in range(1, eff_lag + 1):
        w = 1.0 - k / (eff_lag + 1.0)
        gamma_k = float(np.dot(dc[k:], dc[:-k]) / n)
        var_hac += 2.0 * w * gamma_k

    # Guard against tiny negative HAC variance due to finite-sample Bartlett truncation
    var_hac = max(var_hac, gamma0 * 0.1, 1e-12)
    dm_stat = mean_d / np.sqrt(var_hac / n)
    p_one = float(1.0 - norm.cdf(dm_stat))
    p_two = float(2.0 * (1.0 - norm.cdf(abs(dm_stat))))
    suff = n >= MIN_DM_SAMPLES
    return {
        "dm_stat": round(float(dm_stat), 4),
        "p_value": round(p_one, 5),
        "p_two_sided": round(p_two, 5),
        "n": n,
        "lag": eff_lag,
        "var_naive": round(gamma0, 6),
        "var_hac": round(var_hac, 6),
        "sufficient_sample": suff,
        "status": "OK" if suff else f"INSUFFICIENT SAMPLE (n={n} < {MIN_DM_SAMPLES})",
    }


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Apply Holm-Bonferroni step-down multiple-comparison correction to a dict of p-values."""
    if not p_values:
        return {}
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running_max = 0.0
    for i, (key, p) in enumerate(items):
        adj = min(1.0, float(p) * (m - i))
        running_max = max(running_max, adj)
        adjusted[key] = round(running_max, 5)
    return adjusted


def _da_at_horizon(y_usd: np.ndarray, m: np.ndarray, pred_usd: np.ndarray, step: Optional[int], eos: np.ndarray) -> dict:
    n, H = y_usd.shape
    r = np.arange(n)
    k = np.clip(eos - 1, 0, H - 1) if step is None else np.full(n, min(step, H) - 1, dtype=int)
    return compute_directional_accuracy(y_usd[r, k], pred_usd[r, k], valid_mask=m[r, k] > 0)


def _select_non_overlapping_indices(times_ns: np.ndarray, horizon_minutes: int) -> np.ndarray:
    """Select strictly non-overlapping origin indices spaced by at least `horizon_minutes`."""
    if len(times_ns) == 0:
        return np.array([], dtype=int)
    gap_ns = int(horizon_minutes) * 60 * 1_000_000_000
    selected = [0]
    last_t = int(times_ns[0])
    for i in range(1, len(times_ns)):
        if int(times_ns[i]) - last_t >= gap_ns:
            selected.append(i)
            last_t = int(times_ns[i])
    return np.array(selected, dtype=int)


def evaluate_model_on_test(
    name: str,
    pred_norm: np.ndarray,
    te: Samples,
    bench_norm: Optional[np.ndarray] = None,
    commission_usd: float = 0.0,
) -> dict:
    """Compute v2 metrics on TEST set with strict DA, HAC + Non-Overlap DM, and position-aware Net Proxy."""
    atr = te.atr15[:, None]
    y_usd = te.y * atr
    q10_usd = pred_norm[:, :, 0] * atr
    q50_usd = pred_norm[:, :, 1] * atr
    q90_usd = pred_norm[:, :, 2] * atr
    m = te.m > 0

    abs_err = np.abs(y_usd - q50_usd)
    sq_err = (y_usd - q50_usd) ** 2

    mae_all = float(abs_err[m].mean()) if m.any() else float("nan")
    rmse_all = float(np.sqrt(sq_err[m].mean())) if m.any() else float("nan")
    cov80 = float(((y_usd >= q10_usd) & (y_usd <= q90_usd))[m].mean() * 100.0) if m.any() else float("nan")
    pb_loss = pinball_loss_np(te.y, te.m, pred_norm)

    da_1h = _da_at_horizon(y_usd, te.m, q50_usd, config.H_1H, te.eos)
    da_3h = _da_at_horizon(y_usd, te.m, q50_usd, config.H_3H, te.eos)
    da_eos = _da_at_horizon(y_usd, te.m, q50_usd, None, te.eos)

    # Horizon buckets
    buckets = {}
    for lbl, a, b in (("1-15", 0, 15), ("16-60", 15, 60), ("61-180", 60, 180), ("181-360", 180, te.y.shape[1])):
        mb = m[:, a:b]
        buckets[lbl] = {
            "MAE_USD": round(float(abs_err[:, a:b][mb].mean()), 3) if mb.any() else None,
            "RMSE_USD": round(float(np.sqrt(sq_err[:, a:b][mb].mean())), 3) if mb.any() else None,
        }

    # Explicit position-aware 1h Directional Proxy (NEVER charges cost on flat positions!)
    k_1h = min(config.H_1H, te.y.shape[1]) - 1
    ok_1h = m[:, k_1h]
    thr_usd = POSITION_ATR_THRESHOLD * te.atr15
    pos_1h = np.where(q50_usd[:, k_1h] > thr_usd, 1.0, np.where(q50_usd[:, k_1h] < -thr_usd, -1.0, 0.0))
    active_trades = ok_1h & (pos_1h != 0)
    cost_usd = te.spread + float(commission_usd)

    n_trades = int(np.sum(active_trades))
    gross_pnl = float((pos_1h[active_trades] * y_usd[active_trades, k_1h]).mean()) if n_trades > 0 else 0.0
    net_pnl = float((pos_1h[active_trades] * y_usd[active_trades, k_1h] - cost_usd[active_trades]).mean()) if n_trades > 0 else 0.0

    out = {
        "model": name,
        "test_samples": int(len(te)),
        "pinball_loss": round(pb_loss, 5),
        "MAE_USD": round(mae_all, 3),
        "RMSE_USD": round(rmse_all, 3),
        "coverage_80_pct": round(cov80, 2),
        "DA_1h": da_1h,
        "DA_3h": da_3h,
        "DA_EOS": da_eos,
        "DA_1h_pct": da_1h["da_pct"],
        "DA_3h_pct": da_3h["da_pct"],
        "DA_EOS_pct": da_eos["da_pct"],
        "horizon_buckets": buckets,
        "directional_proxy_1h": {
            "note": "Directional proxy, not a trading backtest.",
            "rule": f"Long if q50[+60] > +{POSITION_ATR_THRESHOLD} ATR15, Short if < -{POSITION_ATR_THRESHOLD} ATR15, else Flat",
            "n_trades": n_trades,
            "n_flat_skipped": int(np.sum(ok_1h) - n_trades),
            "gross_usd_per_trade": round(gross_pnl, 4),
            "net_usd_per_trade": round(net_pnl, 4),
            "commission_usd": float(commission_usd),
        },
        "trading_proxy_1h_gross_usd": round(gross_pnl, 4),
        "trading_proxy_1h_net_usd": round(net_pnl, 4),
    }

    if bench_norm is not None:
        bench_q50_usd = bench_norm[:, :, 1] * atr
        bench_mae = float(np.abs(y_usd - bench_q50_usd)[m].mean())
        skill = (bench_mae - mae_all) / max(bench_mae, 1e-9) * 100.0

        m_f = te.m.astype(np.float64)
        denom = np.maximum(m_f.sum(axis=1), 1.0)
        l_mod_ae = (np.abs(y_usd - q50_usd) * m_f).sum(axis=1) / denom
        l_ben_ae = (np.abs(y_usd - bench_q50_usd) * m_f).sum(axis=1) / denom
        l_mod_se = (((y_usd - q50_usd) ** 2) * m_f).sum(axis=1) / denom
        l_ben_se = (((y_usd - bench_q50_usd) ** 2) * m_f).sum(axis=1) / denom

        # Estimate origin spacing in minutes to set horizon-matched HAC lag = (h / stride_min) - 1
        diffs_min = np.diff(te.times) / 1e9 / 60.0 if len(te.times) > 1 else np.array([15.0])
        stride_min = max(1.0, float(np.median(diffs_min[diffs_min > 0]))) if len(diffs_min) else 15.0
        lag_full = max(1, int(round(config.HORIZON_BARS / stride_min)) - 1)

        dm_hac_ae = hac_dm_test(l_mod_ae, l_ben_ae, lag=lag_full)
        dm_hac_se = hac_dm_test(l_mod_se, l_ben_se, lag=lag_full)

        # Non-overlapping DM on strictly spaced 360-min windows
        no_idx = _select_non_overlapping_indices(te.times, config.HORIZON_BARS)
        dm_no_ae = hac_dm_test(l_mod_ae[no_idx], l_ben_ae[no_idx], lag=0)

        # Horizon-specific HAC & Non-Overlapping DM (1h=60, 3h=180)
        dm_horizons = {}
        for h_lbl, h_steps in (("1h", config.H_1H), ("3h", config.H_3H)):
            kh = min(h_steps, te.y.shape[1]) - 1
            okh = m[:, kh]
            lm_h = np.abs(y_usd[okh, kh] - q50_usd[okh, kh])
            lb_h = np.abs(y_usd[okh, kh] - bench_q50_usd[okh, kh])
            lag_h = max(1, int(round(h_steps / stride_min)) - 1)
            idx_no_h = _select_non_overlapping_indices(te.times[okh], h_steps)
            dm_horizons[h_lbl] = {
                "hac": hac_dm_test(lm_h, lb_h, lag=lag_h),
                "non_overlapping": hac_dm_test(lm_h[idx_no_h], lb_h[idx_no_h], lag=0),
            }

        out["skill_edge_vs_nochange_pct"] = round(float(skill), 3)
        out["dm_hac"] = dm_hac_ae
        out["dm_hac_se"] = dm_hac_se
        out["dm_non_overlapping"] = dm_no_ae
        out["dm_horizons"] = dm_horizons

    return out


def evaluate_walk_forward_folds(
    ens_pred_norm: np.ndarray,
    bench_pred_norm: np.ndarray,
    te: Samples,
    fold_days: int = 30,
    step_days: int = 15,
) -> dict:
    """Evaluate rolling folds across the out-of-sample period (adaptive if span < 30 days)."""
    ts = pd.to_datetime(te.times, utc=True)
    if len(ts) == 0:
        return {"folds": [], "n_folds": 0, "positive_folds": 0, "positive_fold_pct": 0.0}

    t_start, t_end = ts.min(), ts.max()
    total_days = max(1.0, (t_end - t_start).total_seconds() / 86400.0)
    # If out-of-sample test span is shorter than 30 days, use 3-day folds stepping 2 days and note explicitly
    if total_days < fold_days:
        eff_fold_days = max(2, int(total_days // 2))
        eff_step_days = max(1, eff_fold_days // 2)
        note = f"Test span is {total_days:.1f} days (< {fold_days}d); used rolling {eff_fold_days}-day folds stepping {eff_step_days}d."
    else:
        eff_fold_days, eff_step_days = fold_days, step_days
        note = f"Rolling {eff_fold_days}-day folds stepping {eff_step_days} days across {total_days:.1f}d out-of-sample period."

    folds = []
    cur = t_start
    fold_id = 1
    while cur < t_end:
        nxt = cur + pd.Timedelta(days=eff_fold_days)
        sel = (ts >= cur) & (ts < nxt)
        if sel.sum() >= 20:
            sub = Samples(
                X=te.X[sel], C=te.C[sel], y=te.y[sel], m=te.m[sel],
                P0=te.P0[sel], atr15=te.atr15[sel], eos=te.eos[sel],
                times=te.times[sel], spread=te.spread[sel],
            )
            r = evaluate_model_on_test("ensemble", ens_pred_norm[sel], sub, bench_norm=bench_pred_norm[sel])
            folds.append(
                {
                    "fold": fold_id,
                    "start_utc": str(ts[sel].min()),
                    "end_utc": str(ts[sel].max()),
                    "n_origins": int(sel.sum()),
                    "n_non_overlapping_360m": int(r["dm_non_overlapping"]["n"]),
                    "MAE_USD": r["MAE_USD"],
                    "DA_1h_pct": r["DA_1h_pct"],
                    "skill_pct": r["skill_edge_vs_nochange_pct"],
                    "hac_dm_p": r["dm_hac"]["p_value"],
                    "non_overlap_dm_p": r["dm_non_overlapping"]["p_value"],
                    "beats_nochange": bool(r["skill_edge_vs_nochange_pct"] > 0),
                }
            )
            fold_id += 1
        if nxt >= t_end:
            break
        cur = cur + pd.Timedelta(days=eff_step_days)

    skills = [f["skill_pct"] for f in folds]
    maes = [f["MAE_USD"] for f in folds]
    pos_cnt = sum(1 for f in folds if f["beats_nochange"])
    return {
        "note": note,
        "n_folds": len(folds),
        "positive_folds": pos_cnt,
        "positive_fold_pct": round(100.0 * pos_cnt / max(1, len(folds)), 2),
        "mean_mae_usd": round(float(np.mean(maes)), 3) if maes else None,
        "std_mae_usd": round(float(np.std(maes)), 3) if maes else None,
        "mean_skill_pct": round(float(np.mean(skills)), 3) if skills else None,
        "std_skill_pct": round(float(np.std(skills)), 3) if skills else None,
        "folds": folds,
    }


def evaluate_target_high_low(pred_norm: np.ndarray, te: Samples) -> dict:
    """Evaluate Target High/Low reach rate against actual future High/Low AND a naive ATR baseline."""
    atr = te.atr15[:, None]
    y_usd = te.y * atr
    m = te.m > 0
    q10_usd = pred_norm[:, :, 0] * atr
    q50_usd = pred_norm[:, :, 1] * atr
    q90_usd = pred_norm[:, :, 2] * atr

    hit_hi, hit_lo, in_rng = [], [], []
    naive_hi, naive_lo = [], []
    for i in range(len(te)):
        mi = m[i]
        if mi.sum() < 15:
            continue
        act_hi = float(y_usd[i, mi].max())
        act_lo = float(y_usd[i, mi].min())
        tgt_hi = 0.5 * (float(q50_usd[i, mi].max()) + float(q90_usd[i, mi].max()))
        tgt_lo = 0.5 * (float(q50_usd[i, mi].min()) + float(q10_usd[i, mi].min()))
        hit_hi.append(act_hi >= tgt_hi * 0.85 if tgt_hi > 0 else True)
        hit_lo.append(act_lo <= tgt_lo * 0.85 if tgt_lo < 0 else True)
        in_rng.append((act_hi <= float(q90_usd[i, mi].max())) and (act_lo >= float(q10_usd[i, mi].min())))
        # Naive symmetric baseline: +/- 2.0 * ATR15
        nb = 2.0 * float(te.atr15[i])
        naive_hi.append(act_hi >= nb)
        naive_lo.append(act_lo <= -nb)

    return {
        "n_evaluated": len(hit_hi),
        "target_high_reach_rate_pct": round(float(np.mean(hit_hi) * 100.0), 2) if hit_hi else None,
        "target_low_reach_rate_pct": round(float(np.mean(hit_lo) * 100.0), 2) if hit_lo else None,
        "full_range_containment_pct": round(float(np.mean(in_rng) * 100.0), 2) if in_rng else None,
        "naive_2atr_high_reach_pct": round(float(np.mean(naive_hi) * 100.0), 2) if naive_hi else None,
        "naive_2atr_low_reach_pct": round(float(np.mean(naive_lo) * 100.0), 2) if naive_lo else None,
    }


def evaluate_strict_deployment_gate(
    results: list[dict],
    ensemble_res: dict,
    wf_res: dict,
) -> dict:
    """Apply strict 5-condition v2 Deployment Gate with Holm-Bonferroni adjusted p-values."""
    # Holm-Bonferroni correction across all non-baseline models vs NoChange
    hac_p_map = {r["model"]: r["dm_hac"]["p_value"] for r in results if "dm_hac" in r}
    no_p_map = {r["model"]: r["dm_non_overlapping"]["p_value"] for r in results if "dm_non_overlapping" in r}
    hac_holm = holm_bonferroni(hac_p_map)
    no_holm = holm_bonferroni(no_p_map)

    for r in results:
        nm = r["model"]
        if "dm_hac" in r:
            r["dm_hac"]["p_holm"] = hac_holm.get(nm, 1.0)
        if "dm_non_overlapping" in r:
            r["dm_non_overlapping"]["p_holm"] = no_holm.get(nm, 1.0)

    failed = []

    # 1. Ensemble MAE < NoChange MAE (skill > 0)
    skill = float(ensemble_res.get("skill_edge_vs_nochange_pct", 0.0))
    if skill <= 0.0:
        failed.append(f"Cond 1 FAILED: Ensemble skill ({skill:+.2f}%) <= 0% vs NoChange")

    # 2. Ensemble MAE <= MAE of EVERY single member model
    member_names = {"tft", "lightgbm", "tcn", "wavelet_bilstm", "timemixer"}
    ens_mae = float(ensemble_res["MAE_USD"])
    better_members = [
        f"{r['model']} (${r['MAE_USD']:.3f})"
        for r in results
        if r["model"] in member_names and float(r["MAE_USD"]) < ens_mae
    ]
    if better_members:
        failed.append(f"Cond 2 FAILED: Ensemble MAE (${ens_mae:.3f}) worse than single member(s): {', '.join(better_members)}")

    # 3. HAC DM p_holm < 0.05 (n >= 30) AND Non-Overlap DM p_holm < 0.05 (n >= 30)
    hac_d = ensemble_res.get("dm_hac", {})
    no_d = ensemble_res.get("dm_non_overlapping", {})
    if not hac_d.get("sufficient_sample", False) or float(hac_d.get("p_holm", 1.0)) >= 0.05:
        failed.append(
            f"Cond 3a FAILED: HAC DM Holm p={hac_d.get('p_holm', 1.0):.4f} (n={hac_d.get('n', 0)}, status={hac_d.get('status')})"
        )
    if not no_d.get("sufficient_sample", False) or float(no_d.get("p_holm", 1.0)) >= 0.05:
        failed.append(
            f"Cond 3b FAILED: Non-Overlap DM Holm p={no_d.get('p_holm', 1.0):.4f} (n={no_d.get('n', 0)}, status={no_d.get('status')})"
        )

    # 4. 80% tunnel coverage in [70%, 90%]
    cov = float(ensemble_res.get("coverage_80_pct", 0.0))
    if not (70.0 <= cov <= 90.0):
        failed.append(f"Cond 4 FAILED: 80% tunnel coverage ({cov:.1f}%) outside [70%, 90%]")

    # 5. Walk-forward positive on >= 60% of folds
    wf_pct = float(wf_res.get("positive_fold_pct", 0.0))
    if wf_res.get("n_folds", 0) == 0 or wf_pct < 60.0:
        failed.append(f"Cond 5 FAILED: Walk-forward positive folds {wf_pct:.1f}% < 60% ({wf_res.get('positive_folds', 0)}/{wf_res.get('n_folds', 0)})")

    deployable = len(failed) == 0
    return {
        "verdict": "DEPLOYABLE" if deployable else "NOT VERIFIED",
        "dashboard_label": "DEPLOYABLE" if deployable else "Research output, not verified.",
        "deployable": deployable,
        "failed_conditions": failed,
        "reason": "All 5 strict v2 gate conditions passed." if deployable else f"{len(failed)} condition(s) failed: " + " | ".join(failed),
    }


def save_metrics_report(
    results: list[dict],
    ensemble_res: dict,
    target_hl_res: dict,
    dir_brier: dict,
    weights: dict[str, float],
    meta: dict,
    out_dir: Path = config.REPORTS_DIR,
    wf_res: Optional[dict] = None,
    weight_diag: Optional[dict] = None,
    heuristic_eval: Optional[dict] = None,
    filename_stem: str = "metrics",
) -> dict:
    """Apply Strict v2 Deployment Gate and write JSON + Markdown reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wf = wf_res or {"n_folds": 0, "positive_folds": 0, "positive_fold_pct": 0.0, "folds": []}
    gate = evaluate_strict_deployment_gate(results, ensemble_res, wf)

    report = {
        "deployment_gate": gate,
        "data_meta": meta,
        "ensemble_weights": weights,
        "ensemble_weight_diagnostics": weight_diag,
        "models": results,
        "walk_forward": wf,
        "target_high_low_eval": target_hl_res,
        "direction_meta_brier": dir_brier,
        "heuristic_score_eval": heuristic_eval,
    }
    (out_dir / f"{filename_stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# XAUUSD Multi-Timeframe 1-Minute Forecast Engine — Evaluation Report (v2)",
        "",
        f"> **Deployment Gate Verdict:** **{gate['verdict']}** (`{gate['dashboard_label']}`)",
        "",
    ]
    if gate["failed_conditions"]:
        lines.append("### Failed Gate Conditions:")
        for fc in gate["failed_conditions"]:
            lines.append(f"- ❌ {fc}")
        lines.append("")

    lines += [
        f"- **Data Range:** `{meta.get('start_utc')}` to `{meta.get('end_utc')}` (`{meta.get('m1_rows', 0):,}` 1M bars)",
        f"- **Frozen Validation Ensemble Weights (v2):** `{json.dumps(weights)}`",
        "",
        "## Model Comparison (Out-of-Sample TEST Set Only — Raw & Holm-Bonferroni Adjusted)",
        "",
        "| Model | MAE (USD) | RMSE (USD) | 80% Cov | DA 1h (excl flat) | DA 3h | DA EOS | Skill | HAC p (Holm) | Non-Ovlp p (Holm, n) | 1h Net Proxy* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        sk = r.get("skill_edge_vs_nochange_pct")
        sk_s = f"{sk:+.2f}%" if sk is not None else "—"
        if "dm_hac" in r:
            hp = f"{r['dm_hac']['p_value']:.4f} ({r['dm_hac'].get('p_holm', 1.0):.4f})"
            np_d = r["dm_non_overlapping"]
            np_s = f"{np_d['p_value']:.4f} ({np_d.get('p_holm', 1.0):.4f}, n={np_d['n']})"
        else:
            hp, np_s = "—", "—"
        da1 = r["DA_1h"]
        da1_s = f"{da1['label']} ({da1['excluded_flat']})"
        da3_s = r["DA_3h"]["label"]
        dae_s = r["DA_EOS"]["label"]
        net_s = f"${r['trading_proxy_1h_net_usd']:+.3f} ({r['directional_proxy_1h']['n_trades']}t)"
        lines.append(
            f"| **{r['model']}** | {r['MAE_USD']:.3f} | {r['RMSE_USD']:.3f} | {r['coverage_80_pct']:.1f}% | "
            f"{da1_s} | {da3_s} | {dae_s} | {sk_s} | {hp} | {np_s} | {net_s} |"
        )

    lines += [
        "",
        "*`1h Net Proxy`: Directional proxy, not a trading backtest. Cost applied ONLY on non-flat positions (`|q50[+60]| > 0.10 ATR15`).",
        "",
        "## Walk-Forward Folds",
        f"- {wf.get('note', '')}",
        f"- Positive Folds: **{wf.get('positive_folds', 0)}/{wf.get('n_folds', 0)} ({wf.get('positive_fold_pct', 0):.1f}%)** | "
        f"Mean MAE: `${wf.get('mean_mae_usd')}` (±{wf.get('std_mae_usd')}) | Mean Skill: `{wf.get('mean_skill_pct')}%`",
        "",
        "| Fold | Start UTC | End UTC | Origins | Non-Ovlp 360m (n) | MAE (USD) | DA 1h | Skill % | HAC p | Non-Ovlp p |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in wf.get("folds", []):
        lines.append(
            f"| {f['fold']} | {f['start_utc'][:16]} | {f['end_utc'][:16]} | {f['n_origins']} | "
            f"{f['n_non_overlapping_360m']} | {f['MAE_USD']:.3f} | {f['DA_1h_pct']} | {f['skill_pct']:+.2f}% | "
            f"{f['hac_dm_p']:.4f} | {f['non_overlap_dm_p']:.4f} |"
        )

    lines += [
        "",
        "## Target High / Low Reach vs Naive 2-ATR Baseline",
        f"- Model Target High Reach: **{target_hl_res.get('target_high_reach_rate_pct')}%** (Naive +2 ATR: {target_hl_res.get('naive_2atr_high_reach_pct')}%)",
        f"- Model Target Low Reach: **{target_hl_res.get('target_low_reach_rate_pct')}%** (Naive -2 ATR: {target_hl_res.get('naive_2atr_low_reach_pct')}%)",
        f"- Full Range Containment (q10-q90): **{target_hl_res.get('full_range_containment_pct')}%**",
        "",
        "## Known Limitations",
        "- The zig-zag path is a **plausible simulated Monte Carlo path**, not a guaranteed price prediction.",
        "- Spot Gold `real_volume` is 0; `tick_volume` is used strictly as a proxy.",
        "- Odds% and Confluence% are **HEURISTIC SCORES, not validated probabilities**.",
        "- Macro news events (CPI, NFP, Fed) cannot be predicted from price history alone.",
    ]
    (out_dir / f"{filename_stem}.md").write_text("\n".join(lines), encoding="utf-8")
    return report
