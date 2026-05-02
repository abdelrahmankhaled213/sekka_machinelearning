# api/api.py - Sekka Metro Capacity API (full version, uses trained model)
"""
Sekka Metro Capacity API
Real-time capacity predictions for Cairo Metro (Lines 1, 2, 3)
Uses the full CapacityPredictor from src/ with proper feature engineering.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capacity_predictor import CapacityPredictor
from src.config import ALL_STATION_IDS, STATION_LINE_MAP

app = FastAPI(
    title="Sekka Metro Capacity API",
    description="Real‑time capacity predictions for Cairo Metro (Lines 1,2,3)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = CapacityPredictor().load()

@app.get("/")
def root():
    return {
        "app": "Sekka Metro Capacity API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "This information",
            "GET /health": "Health check",
            "GET /stations": "List all stations",
            "GET /predict/all": "Get capacity for all stations",
            "GET /predict/{station_id}": "Get capacity for one station",
            "GET /predict/line/{line_number}": "Get stations on a line (1,2,3)",
            "GET /docs": "Interactive documentation"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/stations")
def get_stations():
    """List all stations with their line numbers"""
    return [
        {"station_id": sid, "line": STATION_LINE_MAP.get(sid, 1)}
        for sid in ALL_STATION_IDS
    ]

@app.get("/predict/all")
def predict_all():
    """Get capacity predictions for all stations"""
    return predictor.predict_batch(ALL_STATION_IDS)

@app.get("/predict/{station_id}")
def predict_station(
    station_id: int,
    hour: Optional[int] = None,
    minute: Optional[int] = None,
    day: Optional[int] = None
):
    """Get capacity prediction for a specific station"""
    if station_id not in ALL_STATION_IDS:
        raise HTTPException(404, f"Station {station_id} not found (valid IDs: 101-135,201-220,301-345)")
    return predictor.predict_one(station_id, hour, minute, day)

@app.get("/predict/line/{line_number}")
def predict_line(line_number: int):
    """Get capacity predictions for all stations on a metro line (1,2,3)"""
    if line_number not in (1, 2, 3):
        raise HTTPException(400, "Line must be 1, 2, or 3")
    line_stations = [sid for sid in ALL_STATION_IDS if STATION_LINE_MAP.get(sid) == line_number]
    return predictor.predict_batch(line_stations)


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print(" SEKKA METRO CAPACITY API")
    print("=" * 60)
    print(" Server: http://127.0.0.1:8001")
    print(" Docs:   http://127.0.0.1:8001/docs")
    print(" Test:   http://127.0.0.1:8001/predict/119")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8001)
