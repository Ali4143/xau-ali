# TimeMixer Root-Cause Diagnosis (`v2`)

## Observed Issue
- Out-of-sample TEST Skill vs NoChange: **-17.03%** (MAE `$21.731` vs NoChange `$18.569`).
- One-sided HAC DM $p$-value: **1.0000**.

## Root-Cause Checks
1. **Output Head Sign Error?** — **No.** `ordered_quantile_head` uses the exact same `q10 = q50 - softplus`, `q90 = q50 + softplus` formula as TFT, TCN, and Wavelet-BiLSTM.
2. **Target Normalization Mismatch?** — **No.** TimeMixer receives the exact same normalized target `(close(t+h) - close(t)) / ATR14_15M(t)` as all other models.
3. **DM Formula Sign?** — **No.** Our one-sided DM test tests $H_1: \text{MAE}_{\text{NoChange}} > \text{MAE}_{\text{model}}$. Because TimeMixer's MAE (`$21.73`) is substantially worse than NoChange (`$18.57`), $dm\_stat \ll 0$, so $1 - \Phi(dm\_stat) = 1.0000$.
4. **Actual Root Cause — Severe Parameter-to-Sample Overfitting (Epoch 1 Collapse):**
   - From `xau_mtf/artifacts/models/xauusd/timemixer.json`:
     - `epochs_run: 9`, **`best_epoch: 1`**, `best_val_loss: 0.65482` (worst of all 5 neural/tree models).
   - `TimeMixerForecaster` uses `Flatten()(xs[i])` across 3 scales (`120x32 + 60x32 + 30x32 = 6,720` flattened units) feeding directly into `Dense(64) -> Dense(1080)`, creating >500,000 trainable parameters on only 2,627 training samples. It overfit immediately after Epoch 1.
   - Because v1 `ensemble.py` lacked a validation exclusion rule and regularized weights toward priors, TimeMixer still received `9.18%` ensemble weight despite having worse validation MAE than `NoChange`.

## Resolution Applied in v2
- Added a strict **Validation Exclusion Rule** in `xau_mtf/ensemble.py`: any member model whose **validation MAE $\ge$ NoChange validation MAE** is automatically excluded (`weight = 0.0`) before fitting non-negative convex weights on the validation set.
- TimeMixer is therefore assigned `0.0` weight in `ensemble_weights_v2.json` until its head is redesigned (e.g., GlobalAveragePooling1D instead of Flatten) and verified on a full 1-year dataset.
