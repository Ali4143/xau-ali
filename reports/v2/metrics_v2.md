# XAUUSD Multi-Timeframe 1-Minute Forecast Engine — Evaluation Report (v2)

> **Deployment Gate Verdict:** **NOT VERIFIED** (`Research output, not verified.`)

### Failed Gate Conditions:
- ❌ Cond 2 FAILED: Ensemble MAE ($17.850) worse than single member(s): tft ($17.721), wavelet_bilstm ($17.836)
- ❌ Cond 3a FAILED: HAC DM Holm p=0.0923 (n=497, status=OK)
- ❌ Cond 3b FAILED: Non-Overlap DM Holm p=0.1808 (n=23, status=INSUFFICIENT SAMPLE (n=23 < 30))

- **Data Range:** `2026-09-27 22:00:00+00:00` to `2026-11-25 14:39:00+00:00` (`58,899` 1M bars)
- **Frozen Validation Ensemble Weights (v2):** `{"lightgbm": 0.2059, "tcn": 0.0, "tft": 0.0745, "wavelet_bilstm": 0.7196, "timemixer": 0.0}`

## Model Comparison (Out-of-Sample TEST Set Only — Raw & Holm-Bonferroni Adjusted)

| Model | MAE (USD) | RMSE (USD) | 80% Cov | DA 1h (excl flat) | DA 3h | DA EOS | Skill | HAC p (Holm) | Non-Ovlp p (Holm, n) | 1h Net Proxy* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **no_change** | 18.569 | 25.356 | 79.6% | N/A (flat forecast) (479) | N/A (flat forecast) | N/A (flat forecast) | — | — | — | $+0.000 (0t) |
| **hist_profile** | 18.395 | 25.315 | 75.9% | 44.89% (0) | 53.65% | 50.92% | +0.93% | 0.3726 (0.7452) | 0.4424 (0.8849, n=23) | $-1.885 (304t) |
| **lightgbm** | 18.105 | 24.860 | 78.2% | 54.70% (0) | 67.35% | 71.03% | +2.50% | 0.0245 (0.1223) | 0.1655 (0.5731, n=23) | $+3.779 (24t) |
| **tcn** | 18.061 | 24.795 | 82.6% | 56.16% (0) | 67.12% | 58.35% | +2.73% | 0.0677 (0.2030) | 0.1433 (0.5731, n=23) | $+1.913 (262t) |
| **tft** | 17.721 | 24.424 | 78.5% | 61.59% (0) | 67.35% | 56.54% | +4.57% | 0.0128 (0.0897) | 0.0960 (0.4801, n=23) | $+2.899 (441t) |
| **wavelet_bilstm** | 17.836 | 24.742 | 80.7% | 61.38% (0) | 66.21% | 58.55% | +3.95% | 0.0276 (0.1223) | 0.0258 (0.1808, n=23) | $+2.033 (360t) |
| **timemixer** | 21.731 | 29.013 | 72.6% | 40.08% (0) | 45.43% | 43.86% | -17.03% | 0.9997 (0.9997) | 0.9578 (0.9578, n=23) | $-3.486 (449t) |
| **ensemble** | 17.850 | 24.700 | 81.1% | 61.17% (0) | 67.12% | 62.78% | +3.87% | 0.0154 (0.0923) | 0.0265 (0.1808, n=23) | $+2.015 (329t) |

*`1h Net Proxy`: Directional proxy, not a trading backtest. Cost applied ONLY on non-flat positions (`|q50[+60]| > 0.10 ATR15`).

## Walk-Forward Folds
- Test span is 7.5 days (< 30d); used rolling 3-day folds stepping 1d.
- Positive Folds: **5/6 (83.3%)** | Mean MAE: `$19.51` (±2.428) | Mean Skill: `2.604%`

| Fold | Start UTC | End UTC | Origins | Non-Ovlp 360m (n) | MAE (USD) | DA 1h | Skill % | HAC p | Non-Ovlp p |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-11-17 21:07 | 2026-11-20 20:52 | 273 | 12 | 18.290 | 65.15 | +6.14% | 0.0163 | 0.0363 |
| 2 | 2026-11-18 21:07 | 2026-11-20 21:37 | 185 | 9 | 19.701 | 63.07 | +4.98% | 0.1027 | 0.1819 |
| 3 | 2026-11-19 21:07 | 2026-11-20 21:37 | 94 | 5 | 24.709 | 61.36 | +2.78% | 0.3368 | 0.4308 |
| 4 | 2026-11-20 21:07 | 2026-11-23 20:52 | 91 | 5 | 18.735 | 46.59 | -0.08% | 0.4840 | 0.8312 |
| 5 | 2026-11-22 23:07 | 2026-11-24 20:52 | 179 | 8 | 18.287 | 55.11 | +0.97% | 0.1830 | 0.2430 |
| 6 | 2026-11-22 23:07 | 2026-11-25 08:37 | 221 | 10 | 17.336 | 56.28 | +0.83% | 0.1846 | 0.1175 |

## Target High / Low Reach vs Naive 2-ATR Baseline
- Model Target High Reach: **26.76%** (Naive +2 ATR: 27.36%)
- Model Target Low Reach: **63.98%** (Naive -2 ATR: 60.36%)
- Full Range Containment (q10-q90): **65.79%**

## Known Limitations
- The zig-zag path is a **plausible simulated Monte Carlo path**, not a guaranteed price prediction.
- Spot Gold `real_volume` is 0; `tick_volume` is used strictly as a proxy.
- Odds% and Confluence% are **HEURISTIC SCORES, not validated probabilities**.
- Macro news events (CPI, NFP, Fed) cannot be predicted from price history alone.