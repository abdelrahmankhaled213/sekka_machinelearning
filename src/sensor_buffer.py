"""
sensor_buffer.py
================
Responsible for:
  - Validating every incoming sensor payload before it touches the model
  - Maintaining a rolling window of recent readings (last N days)
  - Maintaining a per-station lag cache for real-time feature computation
  - Detecting and interpolating sensor dropouts
  - Persisting the buffer to disk so it survives restarts

Usage
-----
    from sensor_buffer import SensorBuffer

    buf = SensorBuffer.load()          # restore from disk (or create fresh)
    result = buf.ingest(payload)       # validate + store one reading
    recent_df = buf.get_window()       # full rolling window DataFrame
    lag = buf.get_lag_features(119)    # real lag values for station 119
    buf.save()                         # persist to disk
"""

import logging
import pickle
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Make relative imports work even when script is run directly
# ------------------------------------------------------------------
if __name__ == "__main__" and __package__ is None:
    # Add the project root to sys.path and set __package__
    import sys
    _current_dir = Path(__file__).parent
    sys.path.insert(0, str(_current_dir.parent))
    __package__ = "src"

# Now relative import works
from .config import (
    BUFFER_PATH,
    LAG_BUFFER_SIZE,
    ROLLING_WINDOW_DAYS,
    SENSOR_DROPOUT_THRESHOLD,
    SENSOR_SCHEMA,
    STATION_LINE_MAP,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    """Raised when a sensor payload fails schema checks."""


def validate_payload(payload: dict) -> dict:
    """
    Validate and coerce a raw sensor payload.

    Expected keys: station_id, people_count, timestamp (ISO-8601 str or datetime)
    Optional keys: station_type, zone, line_number  (filled with defaults if absent)

    Returns the cleaned payload dict.
    Raises ValidationError with a descriptive message on any failure.
    """
    cleaned = dict(payload)

    # --- required fields ---
    required = {"station_id", "people_count", "timestamp"}
    missing = required - set(cleaned.keys())
    if missing:
        raise ValidationError(f"Missing required fields: {missing}")

    # --- schema checks ---
    for field, rules in SENSOR_SCHEMA.items():
        raw = cleaned[field]
        if not isinstance(raw, rules["type"]):
            try:
                target_type = rules["type"][0] if isinstance(rules["type"], tuple) else rules["type"]
                cleaned[field] = target_type(raw)
            except (ValueError, TypeError):
                raise ValidationError(
                    f"Field '{field}' has wrong type: got {type(raw).__name__}"
                )
        val = cleaned[field]
        if val < rules["min"] or val > rules["max"]:
            raise ValidationError(
                f"Field '{field}' = {val} out of range [{rules['min']}, {rules['max']}]"
            )

    # --- timestamp ---
    ts = cleaned["timestamp"]
    if isinstance(ts, str):
        try:
            cleaned["timestamp"] = pd.to_datetime(ts)
        except Exception:
            raise ValidationError(f"Cannot parse timestamp: '{ts}'")
    elif not isinstance(ts, (datetime, pd.Timestamp)):
        raise ValidationError(f"Timestamp must be string or datetime, got {type(ts)}")

    cleaned["timestamp"] = pd.Timestamp(cleaned["timestamp"])

    # Guard against future timestamps (sensor clock drift)
    if cleaned["timestamp"] > pd.Timestamp.now() + pd.Timedelta(minutes=5):
        raise ValidationError(
            f"Timestamp {cleaned['timestamp']} is in the future — possible clock drift"
        )

    # --- optional fields with safe defaults ---
    sid = int(cleaned["station_id"])
    cleaned.setdefault("line_number", STATION_LINE_MAP.get(sid, 1))
    cleaned.setdefault("station_type", "regular")
    cleaned.setdefault("zone", "central")

    cleaned["people_count"] = float(cleaned["people_count"])
    cleaned["station_id"]   = sid

    return cleaned


# ---------------------------------------------------------------------------
# Per-station lag cache
# ---------------------------------------------------------------------------

def _make_lag_deque():
    """Module-level factory so pickle can serialize the defaultdict."""
    return deque(maxlen=LAG_BUFFER_SIZE)


class StationLagCache:
    """
    Circular buffer keeping the last LAG_BUFFER_SIZE readings per station.
    Provides real lag and rolling-mean values for inference-time features.
    """

    def __init__(self):
        # station_id → deque of (timestamp, people_count)
        self._cache: dict[int, deque] = defaultdict(_make_lag_deque)
        # consecutive missing-reading counter per station
        self._dropout_counter: dict[int, int] = defaultdict(int)

    def push(self, station_id: int, timestamp: pd.Timestamp, count: float) -> None:
        self._cache[station_id].append((timestamp, count))
        self._dropout_counter[station_id] = 0

    def mark_missing(self, station_id: int) -> None:
        self._dropout_counter[station_id] += 1
        if self._dropout_counter[station_id] >= SENSOR_DROPOUT_THRESHOLD:
            logger.warning(
                "Station %d has been silent for %d consecutive intervals — "
                "treating as offline.",
                station_id,
                self._dropout_counter[station_id],
            )

    def is_offline(self, station_id: int) -> bool:
        return self._dropout_counter[station_id] >= SENSOR_DROPOUT_THRESHOLD

    def get_lag_features(self, station_id: int) -> dict[str, float]:
        """
        Return real lag & rolling features for a station.
        Falls back to the buffer mean (or 30) when not enough history exists.
        """
        buf = self._cache[station_id]
        counts = [c for _, c in buf]

        fallback = float(np.mean(counts)) if counts else 30.0

        def _lag(n: int) -> float:
            return counts[-n] if len(counts) >= n else fallback

        def _rolling_mean(window: int) -> float:
            window_data = counts[-window:] if len(counts) >= window else counts
            return float(np.mean(window_data)) if window_data else fallback

        return {
            "people_count_lag_1": _lag(1),
            "people_count_lag_3": _lag(3),
            "people_count_lag_8": _lag(8),
            "rolling_1h_mean":    _rolling_mean(8),   # 8 × 7-min ≈ 1 h
            "rolling_2h_mean":    _rolling_mean(16),  # 16 × 7-min ≈ 2 h
        }

    def station_hour_avg(self, station_id: int, hour: int) -> float:
        """Approximate station-hour average from cached readings at the same hour."""
        buf = self._cache[station_id]
        same_hour = [c for ts, c in buf if ts.hour == hour]
        return float(np.mean(same_hour)) if same_hour else 30.0


# ---------------------------------------------------------------------------
# SensorBuffer — rolling window + lag cache
# ---------------------------------------------------------------------------

class SensorBuffer:
    """
    Central store for incoming sensor data.

    Attributes
    ----------
    window_df    : rolling DataFrame of the last ROLLING_WINDOW_DAYS days
    lag_cache    : StationLagCache for fast inference-time feature lookup
    rows_since_save : how many rows added since last disk save
    """

    def __init__(self):
        self.window_df: pd.DataFrame = pd.DataFrame()
        self.lag_cache: StationLagCache = StationLagCache()
        self.rows_ingested: int = 0          # total lifetime rows
        self.rows_since_retrain: int = 0     # reset after each retrain
        self._last_retrain_time: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, payload: dict) -> dict:
        """
        Validate a sensor payload and add it to the buffer.

        Returns
        -------
        {"status": "ok",    "row": cleaned_dict}   on success
        {"status": "error", "reason": str}          on validation failure
        """
        try:
            row = validate_payload(payload)
        except ValidationError as exc:
            logger.warning("Invalid sensor payload: %s | payload=%s", exc, payload)
            return {"status": "error", "reason": str(exc)}

        new_row = pd.DataFrame([row])
        self.window_df = (
            pd.concat([self.window_df, new_row], ignore_index=True)
            if not self.window_df.empty
            else new_row
        )

        self.lag_cache.push(row["station_id"], row["timestamp"], row["people_count"])
        self.rows_ingested += 1
        self.rows_since_retrain += 1

        self._prune()
        return {"status": "ok", "row": row}

    def ingest_batch(self, payloads: list[dict]) -> dict:
        """Ingest multiple payloads at once."""
        accepted, rejected, errors = 0, 0, []
        for p in payloads:
            result = self.ingest(p)
            if result["status"] == "ok":
                accepted += 1
            else:
                rejected += 1
                errors.append(result["reason"])
        return {"accepted": accepted, "rejected": rejected, "errors": errors}

    # ------------------------------------------------------------------
    # Dropout / interpolation
    # ------------------------------------------------------------------

    def check_dropouts(self, expected_station_ids: list[int], current_time: pd.Timestamp) -> list[int]:
        """Identify stations that haven't reported recently."""
        offline = []
        cutoff = current_time - pd.Timedelta(minutes=15)

        if self.window_df.empty:
            return offline

        recent = self.window_df[self.window_df["timestamp"] >= cutoff]
        reporting = set(recent["station_id"].unique())

        for sid in expected_station_ids:
            if sid not in reporting:
                self.lag_cache.mark_missing(sid)
                if self.lag_cache.is_offline(sid):
                    offline.append(sid)
                    logger.warning("Station %d classified as offline.", sid)

        return offline

    def interpolate_dropout(self, station_id: int, timestamp: pd.Timestamp) -> float:
        """Estimate count for offline station using recent same-hour data or global average."""
        if self.window_df.empty:
            return 30.0

        hour = timestamp.hour
        same = self.window_df[
            (self.window_df["station_id"] == station_id) &
            (self.window_df["timestamp"].dt.hour == hour)
        ]
        if not same.empty:
            return float(same["people_count"].mean())

        return float(self.window_df["people_count"].mean()) if not self.window_df.empty else 30.0

    # ------------------------------------------------------------------
    # Window access
    # ------------------------------------------------------------------

    def get_window(self) -> pd.DataFrame:
        return self.window_df.copy()

    def get_window_size(self) -> int:
        return len(self.window_df)

    # ------------------------------------------------------------------
    # Retrain trigger checks
    # ------------------------------------------------------------------

    def should_retrain_volume(self, threshold: int) -> bool:
        return self.rows_since_retrain >= threshold

    def should_retrain_time(self, every_hours: int) -> bool:
        if self._last_retrain_time is None:
            return True
        return (datetime.now() - self._last_retrain_time) >= timedelta(hours=every_hours)

    def should_retrain_drift(self, current_mae: float, threshold: float) -> bool:
        return current_mae > threshold

    def mark_retrained(self) -> None:
        self.rows_since_retrain = 0
        self._last_retrain_time = datetime.now()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path=BUFFER_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("SensorBuffer saved → %s (%d rows)", path, len(self.window_df))

    @classmethod
    def load(cls, path=BUFFER_PATH) -> "SensorBuffer":
        if path.exists():
            try:
                with open(path, "rb") as f:
                    buf = pickle.load(f)
                logger.info("SensorBuffer restored from %s (%d rows)", path, len(buf.window_df))
                return buf
            except Exception as e:
                logger.warning("Buffer file corrupted (%s) — starting fresh.", e)
                path.unlink(missing_ok=True)
        logger.info("No existing buffer found at %s — starting fresh.", path)
        return cls()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune(self) -> None:
        """Drop rows older than ROLLING_WINDOW_DAYS."""
        if self.window_df.empty:
            return
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=ROLLING_WINDOW_DAYS)
        self.window_df = self.window_df[self.window_df["timestamp"] >= cutoff].reset_index(drop=True)

    def __repr__(self) -> str:
        return (
            f"SensorBuffer(rows={len(self.window_df)}, "
            f"ingested={self.rows_ingested}, "
            f"since_retrain={self.rows_since_retrain})"
        )


# ---------------------------------------------------------------------------
# If run directly, create a test instance
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("=" * 60)
    print("SensorBuffer – standalone test")
    print("=" * 60)
    buf = SensorBuffer.load()
    print(buf)
    buf.save()
    print("✅ Done.")