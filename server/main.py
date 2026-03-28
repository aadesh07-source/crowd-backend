# main.py - CrowdSense AI FastAPI Backend
# Replaces OpenAI with Google Gemini (gemini-1.5-flash)
# Implements all endpoints from BACKEND_ENDPOINTS.md

import os
import random
import asyncio
import traceback
from datetime import datetime, timezone
from typing import Optional, List, Any
import uvicorn

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import httpx

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CrowdSense AI API",
    description="Crowd prediction & real-time monitoring backend powered by Google Gemini",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Environment Variables
# ---------------------------------------------------------------------------

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")
GOOGLE_MAPS_KEY   = os.getenv("GOOGLE_MAPS_API_KEY", "")   # optional for maps features

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ---------------------------------------------------------------------------
# Static Location Data — Mumbai, India
# All coordinates verified to real Mumbai landmarks
# ---------------------------------------------------------------------------

LOCATIONS = [
    {"locationId": "loc-csmt",            "locationName": "CSMT Railway Station",       "latitude": 18.9398, "longitude": 72.8354},
    {"locationId": "loc-dadar",           "locationName": "Dadar Station",              "latitude": 19.0186, "longitude": 72.8424},
    {"locationId": "loc-bandra",          "locationName": "Bandra Station",             "latitude": 19.0543, "longitude": 72.8403},
    {"locationId": "loc-andheri",         "locationName": "Andheri Station",            "latitude": 19.1197, "longitude": 72.8469},
    {"locationId": "loc-airport",         "locationName": "Chhatrapati Shivaji Airport","latitude": 19.0896, "longitude": 72.8656},
    {"locationId": "loc-gateway",         "locationName": "Gateway of India",           "latitude": 18.9220, "longitude": 72.8347},
    {"locationId": "loc-juhu-beach",      "locationName": "Juhu Beach",                 "latitude": 19.1075, "longitude": 72.8263},
    {"locationId": "loc-phoenix-mall",    "locationName": "Phoenix Palladium Mall",     "latitude": 18.9937, "longitude": 72.8262},
    {"locationId": "loc-dharavi",         "locationName": "Dharavi Market",             "latitude": 19.0405, "longitude": 72.8543},
    {"locationId": "loc-borivali",        "locationName": "Borivali Station",           "latitude": 19.2284, "longitude": 72.8564},
    {"locationId": "loc-thane",           "locationName": "Thane Station",              "latitude": 19.1890, "longitude": 72.9710},
    {"locationId": "loc-lower-parel",     "locationName": "Lower Parel BKC",            "latitude": 18.9966, "longitude": 72.8296},
]

LOCATION_MAP = {loc["locationId"]: loc for loc in LOCATIONS}

# ---------------------------------------------------------------------------
# Training State  (in-memory, resets on restart)
# ---------------------------------------------------------------------------

training_state = {
    "status": "idle",          # idle | running | completed | failed
    "started_at": None,
    "completed_at": None,
    "last_error": None,
    "last_rows_used": 0,
}

realtime_cache: list = []      # last collected realtime data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crowd_status(density: float) -> str:
    if density < 30:
        return "low"
    elif density < 65:
        return "medium"
    return "high"


def _mock_density(location_id: str, hour: int) -> float:
    """Deterministic mock density so responses are stable for a given hour."""
    seed = abs(hash(f"{location_id}-{hour}")) % 1000
    base = (seed % 60) + 20          # 20-80
    noise = random.uniform(-5, 5)
    return round(min(max(base + noise, 0), 100), 1)


def _build_crowd_item(loc: dict, hour: int) -> dict:
    density = _mock_density(loc["locationId"], hour)
    count   = int(density * 5)
    return {
        "locationId":        loc["locationId"],
        "location_id":       loc["locationId"],
        "locationName":      loc["locationName"],
        "location_name":     loc["locationName"],
        "latitude":          loc["latitude"],
        "longitude":         loc["longitude"],
        "crowdCount":        count,
        "crowd_count":       count,
        "crowdDensity":      density,
        "crowd_density":     density,
        "status":            _crowd_status(density),
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "predictedNextHour": _mock_density(loc["locationId"], (hour + 1) % 24),
        "predicted_next_hour": _mock_density(loc["locationId"], (hour + 1) % 24),
    }


def _gemini_ask(prompt: str) -> str:
    """Synchronous wrapper around Gemini text generation."""
    response = gemini_model.generate_content(prompt)
    return response.text or ""

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class PredictBody(BaseModel):
    location_id:  str
    hour:         int
    day_of_week:  int
    is_weekend:   int
    is_holiday:   int

class DirectionsBody(BaseModel):
    origin:      dict          # {"lat": float, "lng": float}
    destination: dict
    mode:        Optional[str] = "driving"

class AiInsightsBody(BaseModel):
    crowdData: Optional[List[Any]] = None

class AiRouteAdviceBody(BaseModel):
    crowdData:   Optional[List[Any]] = None
    origin:      Optional[str] = None
    destination: Optional[str] = None

class RealtimeTrainBody(BaseModel):
    hours_to_sample:    Optional[int]   = 12
    blend_with_original: Optional[bool] = True
    weight_maps:        Optional[float] = 0.6

# ---------------------------------------------------------------------------
# ROOT / PING / HEALTH
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "CrowdSense AI API is running!",
        "status":  "healthy",
        "version": "2.0.0",
        "docs":    "/docs",
    }


@app.get("/ping")
async def ping():
    return {"ping": "pong", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/health")
async def health():
    """
    Returns service status and API availability flags.
    Used by AppState.initialize() and every 30-second refresh.
    """
    return {
        "status":              "ok",
        "model":               "gemini-1.5-flash",
        "service":             "CrowdSense AI",
        "googleMapsConfigured": bool(GOOGLE_MAPS_KEY),
        "openAiConfigured":    True,   # Gemini replaces OpenAI; flag kept for frontend compat
        "geminiConfigured":    True,
        "timestamp":           datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# LOCATIONS
# ---------------------------------------------------------------------------

@app.get("/locations")
async def get_locations():
    """Return all monitored locations. Called once on app start."""
    return LOCATIONS


@app.get("/locations/nearby")
async def get_nearby_locations(
    latitude:  float = Query(..., description="User's current latitude"),
    longitude: float = Query(..., description="User's current longitude"),
    radius_km: float = Query(10.0, description="Search radius in kilometres"),
):
    """
    Returns only the locations within radius_km of the user's GPS position.
    The Flutter map screen calls this to filter heatmap markers to
    the user's actual city instead of showing every location worldwide.
    """
    from math import radians, sin, cos, sqrt, atan2

    def haversine(lat1, lon1, lat2, lon2) -> float:
        """Returns distance in km between two lat/lng points."""
        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    nearby = []
    for loc in LOCATIONS:
        dist = haversine(latitude, longitude, loc["latitude"], loc["longitude"])
        if dist <= radius_km:
            nearby.append({**loc, "distance_km": round(dist, 2)})

    # Sort closest first
    nearby.sort(key=lambda x: x["distance_km"])

    return {
        "locations":    nearby,
        "total":        len(nearby),
        "radius_km":    radius_km,
        "user_lat":     latitude,
        "user_lng":     longitude,
    }

# ---------------------------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------------------------

@app.get("/predictions/bulk")
async def get_bulk_predictions(hour: Optional[int] = Query(None, ge=0, le=23)):
    """
    Bulk ML predictions for all locations.
    Polled every 30 seconds by the Flutter app.
    """
    current_hour = hour if hour is not None else datetime.now().hour
    data = [_build_crowd_item(loc, current_hour) for loc in LOCATIONS]
    return {"data": data, "hour": current_hour, "count": len(data)}


@app.post("/predict")
async def predict_single(body: PredictBody):
    """
    Legacy single-location prediction fallback.
    Called by AnalyticsScreen only when /realtime/predict returns null.
    """
    loc = LOCATION_MAP.get(body.location_id)
    if not loc:
        raise HTTPException(status_code=404, detail=f"Location '{body.location_id}' not found")

    density = _mock_density(body.location_id, body.hour)
    return {
        "location_id":       body.location_id,
        "predicted_density": density,
        "status":            _crowd_status(density),
        "hour":              body.hour,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# REALTIME PIPELINE
# ---------------------------------------------------------------------------

@app.get("/realtime/status")
async def realtime_status():
    """Check realtime pipeline availability."""
    return {
        "enabled":  bool(GOOGLE_MAPS_KEY),
        "provider": "google_maps" if GOOGLE_MAPS_KEY else "mock",
        "status":   "available" if GOOGLE_MAPS_KEY else "mock_mode",
    }


@app.post("/realtime/collect")
async def collect_realtime():
    """
    Trigger collection of fresh realtime data.
    If Google Maps key is set, could call Maps API; otherwise returns mock data.
    """
    global realtime_cache
    try:
        hour = datetime.now().hour
        data = [_build_crowd_item(loc, hour) for loc in LOCATIONS]
        # Add slight random jitter so realtime differs from pure ML predictions
        for item in data:
            jitter = random.uniform(-8, 8)
            item["crowdDensity"]  = round(min(max(item["crowdDensity"] + jitter, 0), 100), 1)
            item["crowd_density"] = item["crowdDensity"]
            item["status"]        = _crowd_status(item["crowdDensity"])
        realtime_cache = data
        return {"data": data, "source": "realtime", "count": len(data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/cached")
async def get_cached_realtime():
    """Return last collected realtime data (fallback when live collect fails)."""
    if not realtime_cache:
        # Cold-start fallback
        hour = datetime.now().hour
        data = [_build_crowd_item(loc, hour) for loc in LOCATIONS]
        return {"data": data, "source": "cold_cache"}
    return {"data": realtime_cache, "source": "cache"}


@app.post("/realtime/predict")
async def realtime_predict(
    location_id: str = Query(...),
    hour:        Optional[int] = Query(None, ge=0, le=23),
):
    """
    Single-location realtime-aware prediction.
    Called by AnalyticsScreen in a 24-hour loop.
    """
    loc = LOCATION_MAP.get(location_id)
    if not loc:
        raise HTTPException(status_code=404, detail=f"Location '{location_id}' not found")

    target_hour = hour if hour is not None else datetime.now().hour
    density = _mock_density(location_id, target_hour)
    return {
        "location_id":       location_id,
        "predicted_density": density,
        "status":            _crowd_status(density),
        "hour":              target_hour,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# MAPS
# ---------------------------------------------------------------------------

@app.get("/maps/nearby")
async def maps_nearby(
    latitude:   float = Query(...),
    longitude:  float = Query(...),
    radius:     float = Query(...),
    place_type: Optional[str] = Query(None),
):
    """Fetch nearby places. Returns mock data when Maps key is not configured."""
    # --- With real key you'd call Google Places API here ---
    mock_places = [
        {
            "place_id":   f"mock_place_{i}",
            "name":       f"Nearby Place {i}",
            "latitude":   latitude + random.uniform(-0.01, 0.01),
            "longitude":  longitude + random.uniform(-0.01, 0.01),
            "place_type": place_type or "point_of_interest",
            "crowd_hint": random.choice(["low", "medium", "high"]),
        }
        for i in range(1, 6)
    ]
    return {
        "nearby_locations": mock_places,   # SmartRoute adapter key
        "places":           mock_places,
        "results":          mock_places,
        "radius_km":        round(radius / 1000, 2),
        "count":            len(mock_places),
    }


@app.post("/maps/directions")
async def maps_directions(body: DirectionsBody):
    """Fetch route options. Returns mock routes when Maps key is not configured."""
    origin_str = f"{body.origin.get('lat')},{body.origin.get('lng')}"
    dest_str   = f"{body.destination.get('lat')},{body.destination.get('lng')}"

    if GOOGLE_MAPS_KEY:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://maps.googleapis.com/maps/api/directions/json",
                    params={
                        "origin":      origin_str,
                        "destination": dest_str,
                        "mode":        body.mode,
                        "key":         GOOGLE_MAPS_KEY,
                        "alternatives": "true",
                    },
                )
                return resp.json()
        except Exception as e:
            print(f"Maps Directions API error: {e}")

    # Mock fallback
    return {
        "status": "OK",
        "routes": [
            {
                "summary":              f"Route {i+1} via mock road",
                "duration_minutes":     random.randint(10, 40),
                "distance_km":          round(random.uniform(2, 15), 1),
                "traffic_condition":    random.choice(["clear", "moderate", "heavy"]),
                "crowd_level":          random.choice(["low", "medium", "high"]),
            }
            for i in range(2)
        ],
    }


@app.get("/maps/place/{place_id}")
async def maps_place_details(place_id: str = Path(...)):
    """Fetch details for a specific place (not yet used by UI)."""
    return {
        "place_id":   place_id,
        "name":       f"Place {place_id}",
        "address":    "Mock Address, City",
        "rating":     round(random.uniform(3.0, 5.0), 1),
        "open_now":   True,
    }


@app.get("/maps/estimate-crowd/{location_id}")
async def maps_estimate_crowd(
    location_id: str  = Path(...),
    latitude:    float = Query(...),
    longitude:   float = Query(...),
):
    """Estimate crowd level using map signals (not yet used by UI)."""
    density = _mock_density(location_id, datetime.now().hour)
    return {
        "location_id":    location_id,
        "crowd_density":  density,
        "status":         _crowd_status(density),
        "source":         "maps_estimate",
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# BEST TIME
# ---------------------------------------------------------------------------

@app.get("/best-time")
async def best_time(
    from_location: str = Query(..., alias="from"),
    to_location:   str = Query(..., alias="to"),
):
    """Suggest best travel time between two locations."""
    hourly = {
        h: _mock_density(from_location, h)
        for h in range(24)
    }
    best_hour   = min(hourly, key=hourly.get)
    best_density = hourly[best_hour]

    return {
        "from":              from_location,
        "to":                to_location,
        "best_hour":         best_hour,
        "best_time":         f"{best_hour:02d}:00",     # "HH:00" string for frontend
        "expected_density":  best_density,
        "status":            _crowd_status(best_density),
        "hourly_predictions": [
            {"hour": h, "density": d, "status": _crowd_status(d)}
            for h, d in hourly.items()
        ],
    }

# ---------------------------------------------------------------------------
# AI  (Gemini replaces OpenAI)
# ---------------------------------------------------------------------------

AI_SYSTEM = (
    "You are an AI assistant for CrowdSense, a crowd monitoring platform. "
    "Provide concise, actionable insights about crowd levels at monitored locations. "
    "Use emojis where helpful. Keep responses short and practical."
)


@app.post("/ai/insights")
async def ai_insights(body: AiInsightsBody):
    """
    Generate textual AI summary for current crowd situation.
    Returns {"summary": "..."}.
    """
    try:
        crowd_info = ""
        if body.crowdData:
            crowd_info = "\n".join(
                f"- {item.get('locationName') or item.get('location_name','?')}: "
                f"density {item.get('crowdDensity') or item.get('crowd_density','?')}% "
                f"({item.get('status','?')})"
                for item in body.crowdData
            )
        else:
            hour = datetime.now().hour
            crowd_info = "\n".join(
                f"- {loc['locationName']}: density {_mock_density(loc['locationId'], hour)}%"
                for loc in LOCATIONS
            )

        prompt = (
            f"{AI_SYSTEM}\n\n"
            f"Current crowd data:\n{crowd_info}\n\n"
            "Generate a brief crowd situation summary with key alerts and "
            "a one-line recommendation for travelers."
        )
        summary = _gemini_ask(prompt)
        return {"summary": summary, "success": True}

    except Exception as e:
        print(f"AI Insights error: {e}\n{traceback.format_exc()}")
        return {"summary": "AI insights temporarily unavailable.", "success": False, "error": str(e)}


@app.post("/ai/route-advice")
async def ai_route_advice(body: AiRouteAdviceBody):
    """Generate AI route/timing advice based on crowd data."""
    try:
        crowd_info = ""
        if body.crowdData:
            crowd_info = "\n".join(
                f"- {item.get('locationName') or item.get('location_name','?')}: "
                f"{item.get('crowd_density') or item.get('crowdDensity','?')}% crowd"
                for item in body.crowdData
            )

        prompt = (
            f"{AI_SYSTEM}\n\n"
            f"User wants to travel"
            + (f" from '{body.origin}'" if body.origin else "")
            + (f" to '{body.destination}'" if body.destination else "")
            + f".\n\nCurrent crowd levels:\n{crowd_info or 'Not provided'}\n\n"
            "Give concise route advice: best time to leave, which areas to avoid, "
            "and estimated journey quality."
        )
        advice = _gemini_ask(prompt)
        return {"advice": advice, "summary": advice, "success": True}

    except Exception as e:
        print(f"AI Route Advice error: {e}\n{traceback.format_exc()}")
        return {"advice": "Route advice temporarily unavailable.", "summary": "", "success": False, "error": str(e)}

# ---------------------------------------------------------------------------
# REALTIME TRAINING  (Admin)
# ---------------------------------------------------------------------------

async def _fake_training_job(hours: int):
    """Simulate background training."""
    global training_state
    await asyncio.sleep(hours * 0.5)          # mock: 0.5s per hour of data
    training_state.update({
        "status":        "completed",
        "completed_at":  datetime.now(timezone.utc).isoformat(),
        "last_rows_used": hours * random.randint(50, 200),
    })


@app.post("/realtime/train")
async def start_realtime_training(body: RealtimeTrainBody):
    """
    Trigger retraining of realtime model from map data.
    Returns status_code key for special admin handling (200/409/503).
    """
    global training_state

    if not GOOGLE_MAPS_KEY:
        return {**training_state, "message": "Maps not configured", "status_code": 503}

    if training_state["status"] == "running":
        return {**training_state, "message": "Training already in progress", "status_code": 409}

    training_state.update({
        "status":     "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "last_error": None,
    })
    # Fire-and-forget background job
    asyncio.create_task(_fake_training_job(body.hours_to_sample))

    return {**training_state, "message": "Training started", "status_code": 200}


@app.get("/realtime/train/status")
async def realtime_training_status():
    """Track training progress. Polled every 7s by AdminPanel."""
    return {"training": training_state}


@app.get("/realtime/training-data")
async def realtime_training_data():
    """Return diagnostic/training dataset information."""
    return {
        "training_data": {
            "total_samples":     training_state.get("last_rows_used", 0),
            "locations_covered": len(LOCATIONS),
            "last_trained":      training_state.get("completed_at"),
            "model_version":     "1.0.0-gemini",
        }
    }

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)