# XAUUSD Multi-Timeframe 1-Minute Forecast Engine — Evaluation Report (v2)

> **Deployment Gate Verdict:** **NOT VERIFIED** (`Research output, not verified.`)

### Failed Gate Conditions:
- ❌ Cond 1 FAILED: Ensemble skill (-0.07%) <= 0% vs NoChange
- ❌ Cond 2 FAILED: Ensemble MAE ($13.691) worse than single member(s): tft ($13.628), wavelet_bilstm ($13.644)
- ❌ Cond 3a FAILED: HAC DM Holm p=1.0000 (n=1730, status=OK)
- ❌ Cond 3b FAILED: Non-Overlap DM Holm p=1.0000 (n=151, status=OK)
- ❌ Cond 5 FAILED: Walk-forward positive folds 0.0% < 60% (0/0)

- **Data Range:** `2025-11-25 17:09:00+00:00` to `2026-11-25 22:59:00+00:00` (`354,734` 1M bars)
- **Frozen Validation Ensemble Weights (v2):** `{"lightgbm": 0.2925, "tcn": 0.0707, "tft": 0.1152, "wavelet_bilstm": 0.5216, "timemixer": 0.0}`

## Model Comparison (Out-of-Sample TEST Set Only — Raw & Holm-Bonferroni Adjusted)

| Model | MAE (USD) | RMSE (USD) | 80% Cov | DA 1h (excl flat) | DA 3h | DA EOS | Skill | HAC p (Holm) | Non-Ovlp p (Holm, n) | 1h Net Proxy* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **no_change** | 13.682 | 20.031 | 78.3% | N/A (flat forecast) (1656) | N/A (flat forecast) | N/A (flat forecast) | — | — | — | $+0.000 (0t) |
| **hist_profile** | 13.689 | 20.033 | 76.8% | 53.14% (0) | 51.46% | 50.92% | -0.05% | 0.8070 (1.0000) | 0.9727 (1.0000, n=151) | $+0.888 (454t) |
| **lightgbm** | 13.817 | 20.269 | 78.6% | 51.51% (0) | 49.27% | 45.95% | -0.99% | 0.9700 (1.0000) | 0.7794 (1.0000, n=151) | $-0.068 (344t) |
| **tcn** | 13.852 | 20.214 | 71.9% | 49.40% (0) | 46.82% | 42.49% | -1.24% | 0.9996 (1.0000) | 0.9863 (1.0000, n=151) | $-0.157 (103t) |
| **tft** | 13.628 | 19.970 | 79.0% | 49.94% (0) | 54.05% | 51.33% | +0.39% | 0.0236 (0.1651) | 0.0254 (0.1777, n=151) | $-0.768 (111t) |
| **wavelet_bilstm** | 13.644 | 19.997 | 81.8% | 46.14% (0) | 59.42% | 50.17% | +0.27% | 0.0322 (0.1933) | 0.2006 (1.0000, n=151) | $+1.086 (309t) |
| **timemixer** | 14.156 | 20.554 | 75.0% | 48.19% (0) | 39.92% | 43.76% | -3.47% | 1.0000 (1.0000) | 1.0000 (1.0000, n=151) | $-0.637 (1009t) |
| **ensemble** | 13.691 | 20.062 | 80.7% | 48.73% (0) | 56.37% | 44.80% | -0.07% | 0.6656 (1.0000) | 0.4998 (1.0000, n=151) | $+0.956 (55t) |

*`1h Net Proxy`: Directional proxy, not a trading backtest. Cost applied ONLY on non-flat positions (`|q50[+60]| > 0.10 ATR15`).

## Walk-Forward Folds
- 
- Positive Folds: **0/0 (0.0%)** | Mean MAE: `$None` (±None) | Mean Skill: `None%`

| Fold | Start UTC | End UTC | Origins | Non-Ovlp 360m (n) | MAE (USD) | DA 1h | Skill % | HAC p | Non-Ovlp p |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|

## Target High / Low Reach vs Naive 2-ATR Baseline
- Model Target High Reach: **39.77%** (Naive +2 ATR: 34.22%)
- Model Target Low Reach: **57.63%** (Naive -2 ATR: 50.06%)
- Full Range Containment (q10-q90): **67.63%**

## Known Limitations
- The zig-zag path is a **plausible simulated Monte Carlo path**, not a guaranteed price prediction.
- Spot Gold `real_volume` is 0; `tick_volume` is used strictly as a proxy.
- Odds% and Confluence% are **HEURISTIC SCORES, not validated probabilities**.
- Macro news events (CPI, NFP, Fed) cannot be predicted from price history alone.