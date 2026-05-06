from fastapi import FastAPI, HTTPException # تأكد من إضافة HTTPException هنا
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

# التأكد من أن السيرفر شايف الفولدرات الفرعية
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capacity_predictor import CapacityPredictor
from src.config import ALL_STATION_IDS, STATION_LINE_MAP

app = FastAPI(
    title="Sekka Metro Capacity API",
    description="Real‑time capacity predictions for Cairo Metro (Lines 1,2,3)",
    version="1.0.0"
)

# إضافة الـ Middleware
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

# ... (بقية الـ endpoints اللي كتبتها زي ما هي) ...

@app.get("/predict/line/{line_number}")
def predict_line(line_number: int):
    if line_number not in (1, 2, 3):
        raise HTTPException(400, "Line must be 1, 2, or 3")
    line_stations = [sid for sid in ALL_STATION_IDS if STATION_LINE_MAP.get(sid) == line_number]
    return predictor.predict_batch(line_stations)

# --- أهم جزء لـ Netlify ---
from mangum import Mangum
handler = Mangum(app) # الـ handler لازم يكون موجود عشان Netlify Functions تشغل FastAPI
# -------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)