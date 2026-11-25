# XAUUSD Multi-Timeframe 1-Minute Forecast Engine — Test Set Evaluation Report

> **Deployment Gate Verdict:** **DEPLOYABLE** — Ensemble beats NoChange on test MAE (skill +3.09%, HAC DM p=0.0164 < 0.05)

- **Data Range:** `2026-09-27 22:00:00+00:00` to `2026-11-25 14:39:00+00:00` (`58,899` 1M bars)
- **Learned Ensemble Weights (Validation):** `{"lightgbm": 0.1779, "tcn": 0.0, "tft": 0.1268, "wavelet_bilstm": 0.6035, "timemixer": 0.0918}`

## Model Comparison (Out-of-Sample TEST Set Only)

| Model | MAE (USD) | RMSE (USD) | 80% Cover | DA 1h | DA 3h | DA EOS | Skill vs NoChange | HAC DM p | 1h Net Proxy (USD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **no_change** | 18.569 | 25.356 | 79.6% | 61.6% | 67.3% | 71.4% | — | — | -0.3000 |
| **hist_profile** | 18.395 | 25.315 | 75.9% | 44.9% | 53.6% | 50.7% | +0.93% | 0.3629 | -1.8883 |
| **lightgbm** | 18.105 | 24.860 | 78.2% | 54.7% | 67.3% | 71.0% | +2.50% | 0.0161 | +1.1389 |
| **tcn** | 18.061 | 24.795 | 82.6% | 56.2% | 67.1% | 58.4% | +2.73% | 0.0480 | +1.3891 |
| **tft** | 17.721 | 24.424 | 78.5% | 61.6% | 67.3% | 56.5% | +4.57% | 0.0076 | +2.7388 |
| **wavelet_bilstm** | 17.836 | 24.742 | 80.7% | 61.4% | 66.2% | 58.5% | +3.95% | 0.0168 | +2.6079 |
| **timemixer** | 21.731 | 29.013 | 72.6% | 40.1% | 45.4% | 43.9% | -17.03% | 1.0000 | -2.8737 |
| **ensemble** | 17.995 | 24.852 | 81.1% | 59.5% | 66.4% | 54.1% | +3.09% | 0.0164 | +2.2131 |

## Target High / Low Evaluation (TEST Set)
- Target High Reach Rate: **24.55%**
- Target Low Reach Rate: **64.99%**
- Full Range Containment (q10-q90): **66.0%**

## Known Limitations
- Spot Gold `real_volume` is 0; `tick_volume` is used strictly as a proxy.
- 6-hour directional accuracy on Gold is naturally close to 50%; treat outputs as research telemetry.
- Scheduled macro news events (CPI, NFP, FOMC) cannot be predicted from price history alone.