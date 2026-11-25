# XAUUSD Multi-Timeframe 1-Minute Forecast Engine — Evaluation Report (v2)

> **Deployment Gate Verdict:** **NOT VERIFIED** (`Research output, not verified.`)

### Failed Gate Conditions:
- ❌ Cond 1 FAILED: Ensemble skill (-1.35%) <= 0% vs NoChange
- ❌ Cond 2 FAILED: Ensemble MAE ($13.627) worse than single member(s): lightgbm ($13.577), tft ($13.351), wavelet_bilstm ($13.527)
- ❌ Cond 3a FAILED: HAC DM Holm p=1.0000 (n=3422, status=OK)
- ❌ Cond 3b FAILED: Non-Overlap DM Holm p=1.0000 (n=151, status=OK)
- ❌ Cond 5 FAILED: Walk-forward positive folds 0.0% < 60% (0/3)

- **Data Range:** `2025-11-25 17:09:00+00:00` to `2026-11-25 19:25:00+00:00` (`354,520` 1M bars)
- **Frozen Validation Ensemble Weights (v2):** `{"lightgbm": 0.5908, "tcn": 0.4092, "tft": 0.0, "wavelet_bilstm": 0.0, "timemixer": 0.0}`

## Model Comparison (Out-of-Sample TEST Set Only — Raw & Holm-Bonferroni Adjusted)

| Model | MAE (USD) | RMSE (USD) | 80% Cov | DA 1h (excl flat) | DA 3h | DA EOS | Skill | HAC p (Holm) | Non-Ovlp p (Holm, n) | 1h Net Proxy* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **no_change** | 13.446 | 19.645 | 78.9% | N/A (flat forecast) (3311) | N/A (flat forecast) | N/A (flat forecast) | — | — | — | $+0.000 (0t) |
| **hist_profile** | 13.443 | 19.623 | 77.9% | 50.65% (0) | 50.73% | 50.79% | +0.02% | 0.6458 (1.0000) | 0.9351 (1.0000, n=151) | $+1.778 (604t) |
| **lightgbm** | 13.577 | 19.840 | 78.3% | 47.45% (0) | 47.64% | 43.37% | -0.97% | 0.9988 (1.0000) | 0.8884 (1.0000, n=151) | $+0.000 (0t) |
| **tcn** | 13.738 | 20.045 | 79.2% | 47.36% (0) | 51.96% | 42.34% | -2.17% | 0.9999 (1.0000) | 0.9990 (1.0000, n=151) | $-0.742 (636t) |
| **tft** | 13.351 | 19.585 | 80.1% | 54.79% (0) | 56.50% | 54.73% | +0.71% | 0.0785 (0.5492) | 0.3461 (1.0000, n=151) | $+0.566 (2729t) |
| **wavelet_bilstm** | 13.527 | 19.731 | 79.2% | 50.14% (0) | 41.57% | 45.53% | -0.60% | 0.9999 (1.0000) | 0.9958 (1.0000, n=151) | $-0.590 (199t) |
| **timemixer** | 13.809 | 20.085 | 78.4% | 45.78% (0) | 45.62% | 40.53% | -2.70% | 1.0000 (1.0000) | 0.9999 (1.0000, n=151) | $-0.496 (1666t) |
| **ensemble** | 13.627 | 19.904 | 79.6% | 47.30% (0) | 48.44% | 40.79% | -1.35% | 0.9997 (1.0000) | 0.9871 (1.0000, n=151) | $-0.151 (37t) |

*`1h Net Proxy`: Directional proxy, not a trading backtest. Cost applied ONLY on non-flat positions (`|q50[+60]| > 0.10 ATR15`).

## Walk-Forward Folds
- Rolling 30-day folds stepping 15 days across 51.6d out-of-sample period.
- Positive Folds: **0/3 (0.0%)** | Mean MAE: `$13.95` (±2.285) | Mean Skill: `-1.297%`

| Fold | Start UTC | End UTC | Origins | Non-Ovlp 360m (n) | MAE (USD) | DA 1h | Skill % | HAC p | Non-Ovlp p |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-10-04 22:49 | 2026-11-03 21:34 | 1999 | 88 | 11.301 | 47.31 | -0.85% | 0.9873 | 0.9182 |
| 2 | 2026-10-19 22:49 | 2026-11-18 21:34 | 1999 | 88 | 13.671 | 48.6 | -1.23% | 0.9980 | 0.9876 |
| 3 | 2026-11-03 23:04 | 2026-11-25 13:19 | 1423 | 63 | 16.877 | 47.28 | -1.82% | 0.9966 | 0.9648 |

## Target High / Low Reach vs Naive 2-ATR Baseline
- Model Target High Reach: **39.22%** (Naive +2 ATR: 33.4%)
- Model Target Low Reach: **59.88%** (Naive -2 ATR: 51.69%)
- Full Range Containment (q10-q90): **64.44%**

## Known Limitations
- The zig-zag path is a **plausible simulated Monte Carlo path**, not a guaranteed price prediction.
- Spot Gold `real_volume` is 0; `tick_volume` is used strictly as a proxy.
- Odds% and Confluence% are **HEURISTIC SCORES, not validated probabilities**.
- Macro news events (CPI, NFP, Fed) cannot be predicted from price history alone.