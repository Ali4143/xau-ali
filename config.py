"""Central configuration for xau_mtf (100% isolated from KSE100_Interpol)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ROOT / "reports"
FORECASTS_DIR = ROOT / "forecasts"

for _d in (DATA_DIR, CONFIG_DIR, ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR, FORECASTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Primary instrument (extensible dictionary, Phase 1-2 focused strictly on XAUUSD)
DEFAULT_SYMBOL = "XAUUSD"

# Rolling FIFO retention: 15M is derived/resampled strictly from 1M so both share the exact same window.
INITIAL_FETCH_DAYS_M1 = 365       # 1.0 year initial fetch
RETENTION_DAYS_M1 = 548           # 1.5 years max rolling window (FIFO: oldest day dropped when exceeded)

# Time & Forex Day Anchor (America/New_York handles US DST automatically)
NY_TZ = "America/New_York"
FOREX_DAY_ROLLOVER_HOUR_NY = 17   # 17:00 America/New_York = Forex/Gold day close & open
MAINTENANCE_GAP_MINUTES = 45      # Gaps >= 45m are flagged as daily maintenance / weekend close

# Multi-Timeframe Lookbacks & Horizon
LOOKBACK_M15 = 32                 # 32 closed 15M bars (~8 hours context)
LOOKBACK_M1 = 120                 # 120 closed 1M bars (~2 hours execution)
HORIZON_BARS = 360                # 360 trading bars (up to 6 hours)
H_1H = 60
H_3H = 180

# Quantiles & Model Config
QUANTILES = (0.10, 0.50, 0.90)
LGB_STEPS = (6, 12, 24, 36, 48, 60, 90, 120, 180, 240, 300, 360)
SEEDS = (11,)                     # Default 1 seed for fast verification; configurable to (11, 22)
MAX_EPOCHS = 60
ES_PATIENCE = 8
LR_PATIENCE = 4

# Splits & Embargo (chronological only)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
EMBARGO_BARS = 360

# Costs (Gold XAUUSD typical spread & commission)
DEFAULT_SPREAD_USD = 0.25
DEFAULT_COMMISSION_USD = 0.05

# Path Simulator
SIM_PATHS = 300
SIM_LEVELS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95)

# Ensemble prior weights (starting point only; final weights learned on validation set)
PRIOR_WEIGHTS = {
    "tft": 0.3716,
    "lightgbm": 0.3000,
    "tcn": 0.2538,
    "wavelet_bilstm": 0.0459,
    "timemixer": 0.0287,
}

BEST_PARAMS_FILE = CONFIG_DIR / "best_params.json"
ENSEMBLE_WEIGHTS_FILE = ARTIFACTS_DIR / "ensemble_weights.json"
CALENDAR_FILE = CONFIG_DIR / "calendar.json"
FORECAST_LOG_FILE = FORECASTS_DIR / "forecast_log.jsonl"
LOCK_LOG_FILE = FORECASTS_DIR / "session_locks.jsonl"
