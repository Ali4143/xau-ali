# XAUUSD Multi-Timeframe 1-Minute Forecast Engine — Evaluation Report (v2)

> **Deployment Gate Verdict:** **NOT VERIFIED** (`Research output, not verified.`)

### Failed Gate Conditions:
- ❌ Cond 1 FAILED: Ensemble skill (-1.53%) <= 0% vs NoChange
- ❌ Cond 2 FAILED: Ensemble MAE ($13.626) worse than single member(s): lightgbm ($13.523), tft ($13.306), wavelet_bilstm ($13.503)
- ❌ Cond 3a FAILED: HAC DM Holm p=1.0000 (n=3421, status=OK)
- ❌ Cond 3b FAILED: Non-Overlap DM Holm p=1.0000 (n=151, status=OK)
- ❌ Cond 5 FAILED: Walk-forward positive folds 0.0% < 60% (0/0)

- **Data Range:** `2025-11-25 17:09:00+00:00` to `2026-11-25 19:24:00+00:00` (`354,519` 1M bars)
- **Frozen Validation Ensemble Weights (v2):** `{"lightgbm": 0.5649, "tcn": 0.4351, "tft": 0.0, "wavelet_bilstm": 0.0, "timemixer": 0.0}`

## Model Comparison (Out-of-Sample TEST Set Only — Raw & Holm-Bonferroni Adjusted)

| Model | MAE (USD) | RMSE (USD) | 80% Cov | DA 1h (excl flat) | DA 3h | DA EOS | Skill | HAC p (Holm) | Non-Ovlp p (Holm, n) | 1h Net Proxy* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **no_change** | 13.420 | 19.597 | 78.9% | N/A (flat forecast) (3310) | N/A (flat forecast) | N/A (flat forecast) | — | — | — | $+0.000 (0t) |
| **hist_profile** | 13.416 | 19.573 | 77.8% | 51.12% (0) | 50.63% | 51.07% | +0.03% | 0.6508 (1.0000) | 0.9527 (1.0000, n=151) | $+1.709 (604t) |
| **lightgbm** | 13.523 | 19.810 | 78.5% | 49.21% (0) | 51.53% | 45.72% | -0.77% | 0.9829 (1.0000) | 0.6468 (1.0000, n=151) | $+0.000 (0t) |
| **tcn** | 13.813 | 20.114 | 78.7% | 47.22% (0) | 47.38% | 41.86% | -2.93% | 1.0000 (1.0000) | 0.9931 (1.0000, n=151) | $-1.004 (1144t) |
| **tft** | 13.306 | 19.507 | 79.6% | 54.41% (0) | 56.74% | 55.25% | +0.84% | 0.0452 (0.3163) | 0.1153 (0.8070, n=151) | $+0.544 (2705t) |
| **wavelet_bilstm** | 13.503 | 19.685 | 79.9% | 50.00% (0) | 42.07% | 46.21% | -0.62% | 0.9999 (1.0000) | 0.9938 (1.0000, n=151) | $-0.271 (214t) |
| **timemixer** | 13.860 | 20.176 | 78.3% | 46.04% (0) | 43.26% | 41.74% | -3.28% | 1.0000 (1.0000) | 0.9940 (1.0000, n=151) | $-0.665 (1856t) |
| **ensemble** | 13.626 | 19.913 | 79.4% | 47.19% (0) | 46.48% | 41.83% | -1.53% | 0.9995 (1.0000) | 0.9357 (1.0000, n=151) | $-1.256 (153t) |

*`1h Net Proxy`: Directional proxy, not a trading backtest. Cost applied ONLY on non-flat positions (`|q50[+60]| > 0.10 ATR15`).

## Walk-Forward Folds
- 
- Positive Folds: **0/0 (0.0%)** | Mean MAE: `$None` (±None) | Mean Skill: `None%`

| Fold | Start UTC | End UTC | Origins | Non-Ovlp 360m (n) | MAE (USD) | DA 1h | Skill % | HAC p | Non-Ovlp p |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|

## Target High / Low Reach vs Naive 2-ATR Baseline
- Model Target High Reach: **38.91%** (Naive +2 ATR: 33.7%)
- Model Target Low Reach: **60.16%** (Naive -2 ATR: 51.01%)
- Full Range Containment (q10-q90): **63.81%**

## Known Limitations
- The zig-zag path is a **plausible simulated Monte Carlo path**, not a guaranteed price prediction.
- Spot Gold `real_volume` is 0; `tick_volume` is used strictly as a proxy.
- Odds% and Confluence% are **HEURISTIC SCORES, not validated probabilities**.
- Macro news events (CPI, NFP, Fed) cannot be predicted from price history alone.