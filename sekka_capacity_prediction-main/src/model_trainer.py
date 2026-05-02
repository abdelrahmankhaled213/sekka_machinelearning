# src/model_trainer.py
"""
model_trainer.py
================
Responsible for:
  - Feature engineering (shared between initial + incremental training)
  - Initial full training from a JSON data dump
  - Incremental warm-start retraining from the live SensorBuffer
  - Model versioning — every trained version is saved and logged
  - Drift detection — evaluate recent MAE before deciding to promote a new model

Model versioning
----------------
Every trained model is saved as:
    models/metro_capacity_model_v<N>.txt

The "live" model (used by CapacityPredictor) is always:
    models/metro_capacity_model_live.txt  ← copy of best version

A human-readable log is kept in:
    models/model_registry.json
"""

import json
import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Relative imports – because this file is inside src/
from .config import (
    CATEGORICAL_FEATURES,
    EARLY_STOPPING_ROUNDS,
    INTERCHANGE_STATIONS,
    LGBM_PARAMS,
    LIVE_MODEL_PATH,
    MAE_DRIFT_THRESHOLD,
    META_PATH,
    MODEL_DIR,
    NUM_BOOST_ROUND,
    REGISTRY_PATH,
    TRAIN_RATIO,
)
from .sensor_buffer import SensorBuffer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature engineering  (used by both initial training and incremental retrain)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the full feature matrix from a DataFrame that has at minimum:
      station_id, people_count, timestamp, station_type, zone, line_number

    Returns
    -------
    df           : DataFrame with all engineered columns
    feature_cols : ordered list of feature column names (the X columns)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # --- time ---
    df["hour"]         = df["timestamp"].dt.hour
    df["minute"]       = df["timestamp"].dt.minute
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = df["day_of_week"].isin([4, 5]).astype(int)  # Fri-Sat

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # --- peak / prayer ---
    df["is_morning_peak"]  = ((df["hour"] >= 7)  & (df["hour"] <= 9)).astype(int)
    df["is_evening_peak"]  = ((df["hour"] >= 17) & (df["hour"] <= 19)).astype(int)
    df["is_peak"]          = (df["is_morning_peak"] | df["is_evening_peak"]).astype(int)
    df["is_friday_prayer"] = (
        (df["day_of_week"] == 4) & (df["hour"] >= 12) & (df["hour"] <= 14)
    ).astype(int)

    # --- station ---
    df["station_id"]     = df["station_id"].astype("category")
    df["line_number"]    = df["line_number"].astype("category")
    df["is_interchange"] = df["station_id"].isin(INTERCHANGE_STATIONS).astype(int)

    # --- one-hot encode station_type and zone ---
    type_dummies = pd.get_dummies(df["station_type"], prefix="type")
    zone_dummies = pd.get_dummies(df["zone"],         prefix="zone")
    df = pd.concat([df, type_dummies, zone_dummies], axis=1)

    # --- vectorised lag & rolling (no Python loop over stations) ---
    df = df.sort_values(["station_id", "timestamp"])
    grouped = df.groupby("station_id", observed=True)["people_count"]

    df["people_count_lag_1"] = grouped.shift(1)
    df["people_count_lag_3"] = grouped.shift(3)
    df["people_count_lag_8"] = grouped.shift(8)
    df["rolling_1h_mean"]    = grouped.transform(lambda x: x.rolling(8,  min_periods=1).mean())
    df["rolling_2h_mean"]    = grouped.transform(lambda x: x.rolling(16, min_periods=1).mean())

    # --- historical averages ---
    df["day_hour_avg"]     = df.groupby(["day_of_week", "hour"], observed=True)["people_count"].transform("mean")
    df["station_hour_avg"] = df.groupby(["station_id", "hour"], observed=True)["people_count"].transform("mean")

    cat_cols = df.select_dtypes(include="category").columns.tolist()
    non_cat  = df.columns.difference(cat_cols)
    df[non_cat] = df[non_cat].bfill().ffill().fillna(30)

    # --- feature list (sorted one-hots ensure stable column order) ---
    type_cols = sorted(c for c in df.columns if c.startswith("type_"))
    zone_cols = sorted(c for c in df.columns if c.startswith("zone_"))

    feature_cols = [
        "hour", "minute", "day_of_week", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "is_morning_peak", "is_evening_peak", "is_peak",
        "is_friday_prayer", "is_interchange",
        "people_count_lag_1", "people_count_lag_3", "people_count_lag_8",
        "rolling_1h_mean", "rolling_2h_mean",
        "day_hour_avg", "station_hour_avg",
        "station_id", "line_number",
        *type_cols,
        *zone_cols,
    ]

    return df, feature_cols


# ---------------------------------------------------------------------------
# Low-level train helper
# ---------------------------------------------------------------------------

def _run_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    init_model: Optional[lgb.Booster] = None,
) -> tuple[lgb.Booster, dict]:
    """
    Run LightGBM training and return (model, metrics).

    Parameters
    ----------
    init_model : if provided, warm-start from this booster (incremental mode).
                 If None, trains from scratch.
    """
    for col in CATEGORICAL_FEATURES:
        X_train[col] = X_train[col].astype("category")
        X_test[col]  = X_test[col].astype("category")

    train_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES)
    valid_ds = lgb.Dataset(
        X_test, label=y_test,
        categorical_feature=CATEGORICAL_FEATURES,
        reference=train_ds,
    )

    model = lgb.train(
        LGBM_PARAMS,
        train_ds,
        valid_sets=[valid_ds],
        num_boost_round=NUM_BOOST_ROUND,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(100)],
        init_model=init_model,
    )

    y_pred = model.predict(X_test)
    metrics = {
        "mae":  round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "r2":   round(float(r2_score(y_test, y_pred)), 4),
    }
    return model, metrics


# ---------------------------------------------------------------------------
# Model versioning
# ---------------------------------------------------------------------------

def _next_version() -> int:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
        return max(int(k) for k in registry.keys()) + 1 if registry else 1
    return 1


def _save_version(
    model:        lgb.Booster,
    feature_cols: list[str],
    metrics:      dict,
    mode:         str,    # "initial" | "incremental" | "full_retrain"
    promote:      bool,
) -> int:
    """
    Save versioned model file, update registry, optionally promote to live.
    Returns the version number.
    """
    version      = _next_version()
    version_path = MODEL_DIR / f"metro_capacity_model_v{version}.txt"
    model.save_model(str(version_path))

    # save feature metadata (always reflects latest promoted model)
    meta = {"feature_cols": feature_cols}
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    # update registry
    registry: dict = {}
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)

    registry[str(version)] = {
        "path":      str(version_path),
        "timestamp": datetime.now().isoformat(),
        "mode":      mode,
        "metrics":   metrics,
        "promoted":  promote,
    }
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    if promote:
        shutil.copy2(version_path, LIVE_MODEL_PATH)
        logger.info("Model v%d promoted → %s", version, LIVE_MODEL_PATH)

    _print_metrics(metrics, version, mode, promote)
    return version


def _print_metrics(metrics: dict, version: int, mode: str, promoted: bool) -> None:
    tag = "✅ PROMOTED" if promoted else "⏸ NOT PROMOTED"
    print(f"\n{'='*55}")
    print(f"📊 MODEL v{version}  [{mode}]  {tag}")
    print(f"{'='*55}")
    print(f"MAE  : {metrics['mae']:.2f} people")
    print(f"RMSE : {metrics['rmse']:.2f} people")
    print(f"R²   : {metrics['r2']:.4f}")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def initial_train(json_file: str = "cairo_metro_7min_data.json") -> lgb.Booster:
    """
    Full training from a static JSON dump.
    Always promotes the result as the live model.
    """
    print(f"\n{'='*55}")
    print(f"🚇 INITIAL TRAINING from {json_file}")
    print(f"{'='*55}")

    with open(json_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw["data"])
    print(f"Loaded {len(df):,} records | {df['station_id'].nunique()} stations")

    df, feature_cols = engineer_features(df)
    X = df[feature_cols].copy()
    y = df["people_count"].copy()

    split_idx = int(len(df) * TRAIN_RATIO)
    model, metrics = _run_lgbm(
        X.iloc[:split_idx], y.iloc[:split_idx],
        X.iloc[split_idx:], y.iloc[split_idx:],
    )

    _save_version(model, feature_cols, metrics, mode="initial", promote=True)
    return model


def incremental_retrain(
    buffer:      SensorBuffer,
    current_mae: Optional[float] = None,
) -> Optional[lgb.Booster]:
    """
    Warm-start retrain on the live buffer data.

    Decision logic
    --------------
    1. Drift check  — skip if current_mae is below the threshold.
    2. Warm-start   — continues boosting from the live model.
    3. Promote only — if new MAE is strictly lower than the current live model.

    Returns promoted Booster, or None if retrain was skipped / not promoted.
    """
    if current_mae is not None and current_mae <= MAE_DRIFT_THRESHOLD:
        logger.info(
            "Skipping retrain: MAE %.2f <= threshold %.2f", current_mae, MAE_DRIFT_THRESHOLD
        )
        return None

    df = buffer.get_window()
    if df.empty or len(df) < 500:
        logger.warning("Buffer too small (%d rows). Skipping retrain.", len(df))
        return None

    print(f"\n🔄 Incremental retrain on {len(df):,} buffered rows …")
    df, feature_cols = engineer_features(df)

    X = df[feature_cols].copy()
    y = df["people_count"].copy()
    split_idx = int(len(df) * TRAIN_RATIO)

    if split_idx < 100:
        logger.warning("Not enough rows after split. Skipping.")
        return None

    # warm-start from live model if available
    init_model = None
    if LIVE_MODEL_PATH.exists():
        init_model = lgb.Booster(model_file=str(LIVE_MODEL_PATH))
        print("   Warm-starting from live model.")

    new_model, new_metrics = _run_lgbm(
        X.iloc[:split_idx], y.iloc[:split_idx],
        X.iloc[split_idx:], y.iloc[split_idx:],
        init_model=init_model,
    )

    # only promote if strictly better than current live model
    promote = True
    if init_model is not None:
        old_pred = init_model.predict(X.iloc[split_idx:])
        old_mae  = float(mean_absolute_error(y.iloc[split_idx:], old_pred))
        promote  = new_metrics["mae"] < old_mae
        if not promote:
            logger.warning(
                "New MAE %.2f >= old MAE %.2f — keeping current live model.",
                new_metrics["mae"], old_mae,
            )

    _save_version(new_model, feature_cols, new_metrics, mode="incremental", promote=promote)
    buffer.mark_retrained()
    return new_model if promote else None


def full_retrain_from_buffer(buffer: SensorBuffer) -> Optional[lgb.Booster]:
    """
    Full (non-warm-start) retrain from buffer.
    Use when drift is too severe for warm-starting to recover.
    Always promotes on success.
    """
    df = buffer.get_window()
    if df.empty or len(df) < 500:
        logger.warning("Buffer too small (%d rows). Skipping.", len(df))
        return None

    print(f"\n🔁 Full retrain from buffer ({len(df):,} rows) …")
    df, feature_cols = engineer_features(df)

    X = df[feature_cols].copy()
    y = df["people_count"].copy()
    split_idx = int(len(df) * TRAIN_RATIO)

    model, metrics = _run_lgbm(
        X.iloc[:split_idx], y.iloc[:split_idx],
        X.iloc[split_idx:], y.iloc[split_idx:],
        init_model=None,
    )

    _save_version(model, feature_cols, metrics, mode="full_retrain", promote=True)
    buffer.mark_retrained()
    return model


def plot_feature_importance(top_n: int = 20, save_path: str = "feature_importance.png") -> None:
    if not LIVE_MODEL_PATH.exists():
        print("No live model found.")
        return
    model = lgb.Booster(model_file=str(LIVE_MODEL_PATH))
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    feature_cols = meta["feature_cols"]
    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importance()}
    ).sort_values("importance", ascending=False)

    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(top)), top["importance"])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Features – Metro Capacity Prediction")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.show()
    print(f"✅ Saved → {save_path}")


def print_registry() -> None:
    if not REGISTRY_PATH.exists():
        print("No registry found.")
        return
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    print(f"\n{'='*65}")
    print(f"{'Ver':<5} {'Mode':<15} {'MAE':<8} {'R²':<8} {'Promoted':<10} Timestamp")
    print(f"{'-'*65}")
    for ver, entry in sorted(registry.items(), key=lambda x: int(x[0])):
        m = entry["metrics"]
        print(
            f"v{ver:<4} {entry['mode']:<15} {m['mae']:<8.2f} {m['r2']:<8.4f} "
            f"{'✅' if entry['promoted'] else '❌':<10} {entry['timestamp'][:19]}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("=" * 55)
    print("🚇 SEKKA – MODEL TRAINER")
    print("=" * 55)

    initial_train()
    plot_feature_importance()
    print_registry()

    print(f"\n✅ Initial training complete.")
    print(f"   Live model → {LIVE_MODEL_PATH}")
    print(f"   Run capacity_predictor.py to serve predictions.")
