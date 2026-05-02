# src/capacity_predictor.py
"""
CapacityPredictor – loads the live LightGBM model and serves predictions.
Uses real lag features from SensorBuffer.

All paths are resolved relative to the project root via config.py.
"""

import json
import logging
import pickle
from datetime import datetime
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Imports from the same src/ folder
from .config import (
    ALL_STATION_IDS,
    CATEGORICAL_FEATURES,
    INTERCHANGE_STATIONS,
    LIVE_MODEL_PATH,
    MAE_DRIFT_THRESHOLD,
    MAX_CAPACITY,
    META_PATH,
    RETRAIN_EVERY_HOURS,
    RETRAIN_EVERY_N_ROWS,
    SEAT_CAPACITY,
    STATION_LINE_MAP,
)
from .sensor_buffer import SensorBuffer

logger = logging.getLogger(__name__)


def _build_feature_row(
    station_id: int,
    hour: int,
    minute: int,
    day_of_week: int,
    lag_features: dict,  # real values from StationLagCache
    day_hour_avg: float = 30.0,
) -> dict:
    """Build a single flat feature dict for one station at one point in time."""
    is_morning_peak = int(7 <= hour <= 9)
    is_evening_peak = int(17 <= hour <= 19)

    return {
        "hour": hour,
        "minute": minute,
        "day_of_week": day_of_week,
        "is_weekend": int(day_of_week in (4, 5)),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * day_of_week / 7),
        "dow_cos": np.cos(2 * np.pi * day_of_week / 7),
        "is_morning_peak": is_morning_peak,
        "is_evening_peak": is_evening_peak,
        "is_peak": int(is_morning_peak or is_evening_peak),
        "is_friday_prayer": int(day_of_week == 4 and 12 <= hour <= 14),
        "is_interchange": int(station_id in INTERCHANGE_STATIONS),
        "station_id": station_id,
        "line_number": STATION_LINE_MAP.get(station_id, 1),
        "day_hour_avg": day_hour_avg,
        "station_hour_avg": lag_features.get("rolling_1h_mean", 30.0),
        **lag_features,  # people_count_lag_1/3/8, rolling_1h_mean, rolling_2h_mean
    }


class CapacityPredictor:
    """
    Loads the live LightGBM model and serves real‑time capacity predictions.

    Parameters
    ----------
    model_path : path to the live .txt model file (default: config.LIVE_MODEL_PATH)
    meta_path   : path to the .pkl metadata file   (default: config.META_PATH)
    buffer      : SensorBuffer instance (loads from disk if None)
    """

    def __init__(
        self,
        model_path=LIVE_MODEL_PATH,
        meta_path=META_PATH,
        buffer: Optional[SensorBuffer] = None,
    ) -> None:
        self.model_path = model_path
        self.meta_path = meta_path
        self.model: Optional[lgb.Booster] = None
        self.feature_cols: Optional[list[str]] = None
        self.buffer = buffer if buffer is not None else SensorBuffer.load()

        # rolling window of (actual, predicted) pairs for drift monitoring
        self._recent_actuals: list[float] = []
        self._recent_predicted: list[float] = []
        self._drift_window_size: int = 500

    def load(self) -> "CapacityPredictor":
        """Load model weights and metadata. Returns self for chaining."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No live model found at {self.model_path}. "
                "Run model_trainer.py first."
            )
        self.model = lgb.Booster(model_file=str(self.model_path))

        with open(self.meta_path, "rb") as f:
            meta = pickle.load(f)
        self.feature_cols = meta["feature_cols"]

        logger.info("✅ Model loaded from %s", self.model_path)
        logger.info("✅ Metadata loaded from %s", self.meta_path)
        logger.info("   Features: %d", len(self.feature_cols))
        return self

    def reload(self) -> None:
        """Hot‑reload the live model from disk (called after a successful retrain)."""
        self.model = lgb.Booster(model_file=str(self.model_path))
        with open(self.meta_path, "rb") as f:
            meta = pickle.load(f)
        self.feature_cols = meta["feature_cols"]
        logger.info("Model hot-reloaded from %s", self.model_path)

    def _ensure_loaded(self) -> None:
        if self.model is None or self.feature_cols is None:
            raise RuntimeError("Model not loaded. Call CapacityPredictor.load() first.")

    # ------------------------------------------------------------------
    # Ingest & drift monitoring
    # ------------------------------------------------------------------
    def ingest(self, payload: dict) -> dict:
        """Validate and store one sensor reading. Returns ingestion result dict."""
        result = self.buffer.ingest(payload)

        if result["status"] == "ok":
            # record actual for drift monitoring
            count = result["row"]["people_count"]
            sid = result["row"]["station_id"]
            ts = result["row"]["timestamp"]

            try:
                pred = self.predict_one(sid, ts.hour, ts.minute, ts.weekday())
                self._recent_actuals.append(count)
                self._recent_predicted.append(pred["predicted_count"])
                if len(self._recent_actuals) > self._drift_window_size:
                    self._recent_actuals.pop(0)
                    self._recent_predicted.pop(0)
            except Exception:
                pass

            self._check_retrain_triggers()

        return result

    def ingest_batch(self, payloads: list[dict]) -> dict:
        """Ingest multiple sensor payloads."""
        summary = self.buffer.ingest_batch(payloads)
        self._check_retrain_triggers()
        return summary

    @property
    def recent_mae(self) -> Optional[float]:
        """MAE over the last _drift_window_size predictions. None if too few samples."""
        if len(self._recent_actuals) < 50:
            return None
        return float(mean_absolute_error(self._recent_actuals, self._recent_predicted))

    def is_drifting(self) -> bool:
        mae = self.recent_mae
        return mae is not None and mae > MAE_DRIFT_THRESHOLD

    # ------------------------------------------------------------------
    # Retrain triggers (imported here to avoid circular imports)
    # ------------------------------------------------------------------
    def _check_retrain_triggers(self) -> None:
        from .model_trainer import full_retrain_from_buffer, incremental_retrain

        mae = self.recent_mae

        # severe drift → full retrain
        if mae is not None and mae > MAE_DRIFT_THRESHOLD * 1.5:
            logger.warning("Severe drift (MAE=%.2f). Triggering full retrain.", mae)
            new_model = full_retrain_from_buffer(self.buffer)
            if new_model:
                self.reload()
                self.buffer.save()
            return

        # volume or time or mild drift → incremental retrain
        if (
            self.buffer.should_retrain_volume(RETRAIN_EVERY_N_ROWS)
            or self.buffer.should_retrain_time(RETRAIN_EVERY_HOURS)
            or self.buffer.should_retrain_drift(mae or 0.0, MAE_DRIFT_THRESHOLD)
        ):
            logger.info("Retrain trigger fired. Running incremental retrain …")
            new_model = incremental_retrain(self.buffer, current_mae=mae)
            if new_model:
                self.reload()
            self.buffer.save()

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    def predict_batch(
        self,
        station_ids: list[int],
        hour: Optional[int] = None,
        minute: Optional[int] = None,
        day_of_week: Optional[int] = None,
    ) -> list[dict]:
        """
        Predict capacity for multiple stations using real lag features.
        Returns list of dicts with:
          timestamp, carriage_id, predicted_count, capacity_percent, seats_available, is_offline
        """
        self._ensure_loaded()
        if not station_ids:
            return []

        now = datetime.now()
        hour = hour if hour is not None else now.hour
        minute = minute if minute is not None else now.minute
        day_of_week = day_of_week if day_of_week is not None else now.weekday()
        ts_str = now.strftime("%Y-%m-%d %H:%M:%S")
        ts = pd.Timestamp(now)

        # check for offline stations
        offline = self.buffer.check_dropouts(station_ids, ts)
        if offline:
            logger.warning("Offline stations: %s — using interpolated counts.", offline)

        rows = []
        for sid in station_ids:
            lag_feats = self.buffer.lag_cache.get_lag_features(sid)

            if sid in offline:
                interp = self.buffer.interpolate_dropout(sid, ts)
                for k in lag_feats:
                    lag_feats[k] = interp

            day_hour_avg = self.buffer.lag_cache.station_hour_avg(sid, hour)
            rows.append(_build_feature_row(sid, hour, minute, day_of_week, lag_feats, day_hour_avg))

        X = pd.DataFrame(rows)

        # fill any one‑hot columns the model expects but are absent in this batch
        for col in self.feature_cols:
            if col not in X.columns and (col.startswith("type_") or col.startswith("zone_")):
                X[col] = 0

        X = X[self.feature_cols]
        for col in CATEGORICAL_FEATURES:
            X[col] = X[col].astype("category")

        counts = self.model.predict(X)

        results = []
        for sid, count in zip(station_ids, counts):
            count = max(0.0, float(count))
            results.append(
                {
                    "timestamp": ts_str,
                    "carriage_id": sid,
                    "predicted_count": int(round(count)),
                    "capacity_percent": round((count / MAX_CAPACITY) * 100, 1),
                    "seats_available": "YES" if count < SEAT_CAPACITY else "NO",
                    "is_offline": sid in offline,
                }
            )

        return results

    def predict_one(
        self,
        station_id: int,
        hour: Optional[int] = None,
        minute: Optional[int] = None,
        day_of_week: Optional[int] = None,
    ) -> dict:
        """Convenience wrapper for a single station."""
        return self.predict_batch([station_id], hour, minute, day_of_week)[0]

    # ------------------------------------------------------------------
    # Reporting & demo
    # ------------------------------------------------------------------
    def demo(self, station_ids: Optional[list[int]] = None) -> None:
        station_ids = station_ids or [119, 209, 313, 101, 215, 301, 335]
        results = self.predict_batch(station_ids)

        print("\n" + "=" * 75)
        print(" SEKKA CAPACITY PREDICTIONS")
        if self.recent_mae is not None:
            print(f"   Recent MAE: {self.recent_mae:.2f} people  {' DRIFTING' if self.is_drifting() else ' Healthy'}")
        print("=" * 75)
        print(f"\n{'ID':<6} {'Timestamp':<20} {'Capacity %':<12} {'Seats':<8} {'Count':<8} {'Offline?'}")
        print("-" * 75)
        for r in results:
            offline_tag = " offline" if r["is_offline"] else ""
            print(
                f"{r['carriage_id']:<6} {r['timestamp']:<20} "
                f"{r['capacity_percent']:<12}% {r['seats_available']:<8} "
                f"{r['predicted_count']:<8}{offline_tag}"
            )
        print("-" * 75)

    def demo_time_variations(self, station_id: int = 119) -> None:
        scenarios = [
            (8, 0, 0, "Morning Peak (Mon 8 AM)"),
            (13, 0, 4, "Friday Prayer (1 PM)"),
            (18, 0, 0, "Evening Peak (Mon 6 PM)"),
            (23, 0, 5, "Late Night (Sat 11 PM)"),
            (10, 0, 2, "Mid-morning (Wed 10 AM)"),
        ]
        print(f"\n{'='*60}")
        print(f" TIME VARIATIONS – STATION {station_id}")
        print(f"{'='*60}")
        print(f"\n{'Scenario':<30} {'Capacity %':<12} {'Seats':<8} Count")
        print("-" * 55)
        for h, m, d, label in scenarios:
            r = self.predict_one(station_id, h, m, d)
            print(f"{label:<30} {r['capacity_percent']:<12}% {r['seats_available']:<8} {r['predicted_count']}")

    def status(self) -> None:
        """Print predictor health summary."""
        print(f"\n{'='*50}")
        print("📡 PREDICTOR STATUS")
        print(f"{'='*50}")
        print(f"Buffer rows      : {self.buffer.get_window_size():,}")
        print(f"Rows ingested    : {self.buffer.rows_ingested:,}")
        print(f"Rows since retrain: {self.buffer.rows_since_retrain:,}")
        last = getattr(self.buffer, '_last_retrain_time', None)
        print(f"Last retrain     : {last.strftime('%Y-%m-%d %H:%M') if last else 'never'}")
        mae = self.recent_mae
        print(f"Recent MAE       : {f'{mae:.2f} people' if mae is not None else 'not enough data'}")
        print(f"Drifting         : {' YES' if self.is_drifting() else ' No'}")

    def export_json(
        self,
        station_ids: Optional[list[int]] = None,
        filename: str = "sekka_app_data.json",
    ) -> list[dict]:
        """Export current predictions to JSON for the Flutter app."""
        station_ids = station_ids or [119, 209, 313, 101, 215, 301, 335]
        results = self.predict_batch(station_ids)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\n📁 Exported {len(results)} predictions → '{filename}'")
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("=" * 70)
    print(" SEKKA – CAPACITY PREDICTOR (STANDALONE TEST)")
    print("=" * 70)

    predictor = CapacityPredictor().load()
    predictor.status()
    predictor.demo()
    predictor.demo_time_variations(station_id=119)
    predictor.export_json()

    print("\n✅ Done. Flutter app can read 'sekka_app_data.json'.")
