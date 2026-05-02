"""
config.py
=========
Single source of truth for all shared constants.
Import from here in every other module — never duplicate these values.
"""

from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Project root (where api/, src/, models/ live)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Paths – all directories relative to ROOT_DIR
# ---------------------------------------------------------------------------
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

BUFFER_PATH   = MODEL_DIR / "sensor_buffer.pkl"
META_PATH     = MODEL_DIR / "metro_model_meta.pkl"
REGISTRY_PATH = MODEL_DIR / "model_registry.json"
LIVE_MODEL_PATH = MODEL_DIR / "metro_capacity_model_live.txt"   # single definition

# ---------------------------------------------------------------------------
# Station topology
# ---------------------------------------------------------------------------
INTERCHANGE_STATIONS: frozenset[int] = frozenset(
    [119, 120, 122, 208, 209, 211, 215, 313, 319, 320, 324, 332, 335]
)

STATION_LINE_MAP: dict[int, int] = {
    **{sid: 1 for sid in range(100, 200)},
    **{sid: 2 for sid in range(200, 300)},
    **{sid: 3 for sid in range(300, 400)},
}

ALL_STATION_IDS: list[int] = list(range(100, 140)) + list(range(200, 220)) + list(range(300, 340))

# ---------------------------------------------------------------------------
# Capacity thresholds
# ---------------------------------------------------------------------------
MAX_CAPACITY  = 100
SEAT_CAPACITY = 40

# ---------------------------------------------------------------------------
# Sensor validation rules
# ---------------------------------------------------------------------------
SENSOR_SCHEMA: dict = {
    "station_id":   {"type": int,   "min": 100,  "max": 399},
    "people_count": {"type": (int, float), "min": 0, "max": 300},
}
SENSOR_DROPOUT_THRESHOLD = 5

# ---------------------------------------------------------------------------
# Buffer / rolling window
# ---------------------------------------------------------------------------
ROLLING_WINDOW_DAYS = 30
LAG_BUFFER_SIZE     = 16

# ---------------------------------------------------------------------------
# Retraining triggers
# ---------------------------------------------------------------------------
RETRAIN_EVERY_N_ROWS   = 10_000
RETRAIN_EVERY_HOURS    = 6
MAE_DRIFT_THRESHOLD    = 8.0

# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------
LGBM_PARAMS: dict = {
    "objective":        "regression",
    "metric":           "mae",
    "boosting_type":    "gbdt",
    "num_leaves":       63,
    "learning_rate":    0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "verbose":          -1,
    "n_jobs":           -1,
    "min_data_in_leaf": 50,
    "max_depth":        -1,
}

CATEGORICAL_FEATURES: list[str] = ["station_id", "line_number"]
TRAIN_RATIO: float = 0.8
NUM_BOOST_ROUND: int = 1000
EARLY_STOPPING_ROUNDS: int = 50

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST: str = "0.0.0.0"
API_PORT: int = 8000

# ---------------------------------------------------------------------------
# API keys – placeholder values (must be replaced in production)
# Use environment variables or a .env file for real deployments.
# ---------------------------------------------------------------------------
SENSOR_API_KEYS: set[str] = {
    "sensor-key-change-me-1",
    "sensor-key-change-me-2",
}

APP_API_KEYS: set[str] = {
    "app-key-change-me-1",
}

ALL_API_KEYS: set[str] = SENSOR_API_KEYS | APP_API_KEYS

# Rate limiting (requests per minute per API key)
RATE_LIMIT_SENSOR: int = 120
RATE_LIMIT_APP:    int = 60
