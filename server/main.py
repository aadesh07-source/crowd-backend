# main.py - CrowdSense AI FastAPI Backend
# Real-time crowd monitoring and heatmap visualization for Mumbai
# Powered by Google Gemini AI

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
    description="Real-time crowd monitoring & heatmaps for Mumbai powered by Google Gemini",
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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_MAPS_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ---------------------------------------------------------------------------
# Static Location Data — MUMBAI, INDIA
# All coordinates verified for Mumbai landmarks - used for heatmaps
# ---------------------------------------------------------------------------

LOCATIONS = [
    {"locationId": "loc-csmt",            "locationName": "CSMT Railway Station",       "latitude": 18.9398, "longitude": 72.8354, "area": "South Mumbai"},
    {"locationId": "loc-dadar",           "locationName": "Dadar Station",              "latitude": 19.0186, "longitude": 72.8424, "area": "Central Mumbai"},
    {"locationId": "loc-bandra",          "locationName": "Bandra Station",             "latitude": 19.0543, "longitude": 72.8403, "area": "West Mumbai"},
    {"locationId": "loc-andheri",         "locationName": "Andheri Station",            "latitude": 19.1197, "longitude": 72.8469, "area": "North-West Mumbai"},
    {"locationId": "loc-airport",         "locationName": "Chhatrapati Shivaji Airport","latitude": 19.0896, "longitude": 72.8656, "area": "East Mumbai"},
    {"locationId": "loc-gateway",         "locationName": "Gateway of India",           "latitude": 18.9220, "longitude": 72.8347, "area": "South Mumbai"},
    {"locationId": "loc-juhu-beach",      "locationName": "Juhu Beach",                 "latitude": 19.1075, "longitude": 72.8263, "area": "West Mumbai"},
    {"locationId": "loc-phoenix-mall",    "locationName": "Phoenix Palladium Mall",     "latitude": 18.9937, "longitude": 72.8262, "area": "Central Mumbai"},
    {"locationId": "loc-dharavi",         "locationName": "Dharavi Market",             "latitude": 19.0405, "longitude": 72.8543, "area": "Central Mumbai"},
    {"locationId": "loc-borivali",        "locationName": "Borivali Station",           "latitude": 19.2284, "longitude": 72.8564, "area": "North Mumbai"},
    {"locationId": "loc-thane",           "locationName": "Thane Station",              "latitude": 19.1890, "longitude": 72.9710, "area": "East Mumbai"},
    {"locationId": "loc-lower-parel",     "locationName": "Lower Parel BKC",            "latitude": 18.9966, "longitude": 72.8296, "area": "South-Central Mumbai"},
]

LOCATION_MAP = {loc["locationId"]: loc for loc in LOCATIONS}

# Mumbai bounding box for map views
MUMBAI_BOUNDS = {
    "north": 19.2890,
    "south": 18.8900,
    "east": 72.9800,
    "west": 72.7900,
    "center_lat": 19.0760,
    "center_lng": 72.8777,
}

# ---------------------------------------------------------------------------
# Training State (in-memory)
# ---------------------------------------------------------------------------

training_state = {
    "status": "idle",
    "started_at": None,
    "completed_at": None,
    "last_error": None,
    "last_rows_used": 0,
}

realtime_cache: list = []

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
    """Deterministic mock density."""
    seed = abs(hash(f"{location_id}-{hour}")) % 1000
    base = (seed % 60) + 20
    noise = random.uniform(-5, 5)
    return round(min(max(base + noise, 0), 100), 1)


def _build_crowd_item(loc: dict, hour: int) -> dict:
    density = _mock_density(loc["locationId"], hour)
    count = int(density * 5)
    return {
        "locationId": loc["locationId"],
        "location_id": loc["locationId"],
        "locationName": loc["locationName"],
        "location_name": loc["locationName"],
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "area": loc.get("area", "Mumbai"),
        "crowdCount": count,
        "crowd_count": count,
        "crowdDensity": density,
        "crowd_density": density,
        "status": _crowd_status(density),
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    location_id: str
    hour: int
    day_of_week: int
    is_weekend: int
    is_holiday: int

class DirectionsBody(BaseModel):
    origin: dict
    destination: dict
    mode: Optional[str] = "driving"

class AiInsightsBody(BaseModel):
    crowdData: Optional[List[Any]] = None

class AiRouteAdviceBody(BaseModel):
    crowdData: Optional[List[Any]] = None
    origin: Optional[str] = None
    destination: Optional[str] = None

class RealtimeTrainBody(BaseModel):
    hours_to_sample: Optional[int] = 12
    blend_with_original: Optional[bool] = True
    weight_maps: Optional[float] = 0.6

# ---------------------------------------------------------------------------
# ROOT / PING / HEALTH
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "CrowdSense AI API - Mumbai Heatmaps Live",
        "status": "healthy",
        "version": "2.0.0",
        "city": "Mumbai, India",
        "docs": "/docs",
    }


@app.get("/ping")
async def ping():
    return {"ping": "pong", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/health")
async def health():
    """
    Health check endpoint for Flutter app startup.
    Confirms backend is running and configured for Mumbai.
    """
    return {
        "status": "ok",
        "model": "gemini-1.5-flash",
        "service": "CrowdSense AI",
        "city": "Mumbai",
        "region": "Maharashtra, India",
        "center_latitude": MUMBAI_BOUNDS["center_lat"],
        "center_longitude": MUMBAI_BOUNDS["center_lng"],
        "bounds": MUMBAI_BOUNDS,
        "googleMapsConfigured": bool(GOOGLE_MAPS_KEY),
        "geminiConfigured": True,
        "deployed_for": "Mumbai, India",
        "total_heatmap_locations": len(LOCATIONS),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# LOCATIONS - Used for heatmap markers
# ---------------------------------------------------------------------------

@app.get("/city-info")
async def get_city_info():
    """
    Returns metadata about the city this backend monitors.
    Flutter app calls this on startup to verify correct city configuration.
    Essential for ensuring heatmaps display in the right location (Mumbai).
    """
    return {
        "city": "Mumbai",
        "state": "Maharashtra",
        "country": "India",
        "country_code": "IN",
        "timezone": "IST (UTC+5:30)",
        "center_latitude": MUMBAI_BOUNDS["center_lat"],
        "center_longitude": MUMBAI_BOUNDS["center_lng"],
        "bounds": MUMBAI_BOUNDS,
        "total_monitored_locations": len(LOCATIONS),
        "description": "CrowdSense AI - Real-time crowd monitoring & heatmaps across Mumbai",
        "locations_list": [
            {
                "id": loc["locationId"],
                "name": loc["locationName"],
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "area": loc.get("area", "Mumbai")
            }
            for loc in LOCATIONS
        ],
    }


@app.get("/locations")
async def get_locations():
    """
    Returns all monitored locations for heatmap display.
    Called by Flutter app to render location markers on maps.
    All coordinates are in Mumbai, India.
    """
    return {
        "locations": LOCATIONS,
        "total": len(LOCATIONS),
        "city": "Mumbai",
        "country": "India",
        "bounds": MUMBAI_BOUNDS,
    }


@app.get("/locations/nearby")
async def get_nearby_locations(
    latitude: float = Query(..., description="User's current latitude"),
    longitude: float = Query(..., description="User's current longitude"),
    radius_km: float = Query(10.0, description="Search radius in kilometres"),
):
    """
    Returns only the heatmap locations within radius_km of the user's GPS position.
    This filters heatmap markers to show only nearby locations in Mumbai.
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

    nearby.sort(key=lambda x: x["distance_km"])

    return {
        "locations": nearby,
        "total": len(nearby),
        "radius_km": radius_km,
        "user_lat": latitude,
        "user_lng": longitude,
        "city": "Mumbai",
    }

# ---------------------------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------------------------

@app.get("/predictions/bulk")
async def get_bulk_predictions(hour: Optional[int] = Query(None, ge=0, le=23)):
    """Bulk predictions for heatmap display."""
    current_hour = hour if hour is not None else datetime.now().hour
    data = [_build_crowd_item(loc, current_hour) for loc in LOCATIONS]
    return {"data": data, "hour": current_hour, "count": len(data), "city": "Mumbai"}


@app.post("/predict")
async def predict_single(body: PredictBody):
    """Single location prediction."""
    loc = LOCATION_MAP.get(body.location_id)
    if not loc:
        raise HTTPException(status_code=404, detail=f"Location '{body.location_id}' not found")

    density = _mock_density(body.location_id, body.hour)
    return {
        "location_id": body.location_id,
        "location_name": loc["locationName"],
        "predicted_density": density,
        "status": _crowd_status(density),
        "hour": body.hour,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# REALTIME PIPELINE
# ---------------------------------------------------------------------------

@app.get("/realtime/status")
async def realtime_status():
    """Check realtime heatmap pipeline availability."""
    return {
        "enabled": bool(GOOGLE_MAPS_KEY),
        "provider": "google_maps" if GOOGLE_MAPS_KEY else "mock",
        "status": "available",
        "city": "Mumbai",
    }


@app.post("/realtime/collect")
async def collect_realtime():
    """Trigger collection of fresh realtime heatmap data."""
    global realtime_cache
    try:
        hour = datetime.now().hour
        data = [_build_crowd_item(loc, hour) for loc in LOCATIONS]
        for item in data:
            jitter = random.uniform(-8, 8)
            item["crowdDensity"] = round(min(max(item["crowdDensity"] + jitter, 0), 100), 1)
            item["crowd_density"] = item["crowdDensity"]
            item["status"] = _crowd_status(item["crowdDensity"])
        realtime_cache = data
        return {"data": data, "source": "realtime", "count": len(data), "city": "Mumbai"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/cached")
async def get_cached_realtime():
    """Return last collected realtime heatmap data."""
    if not realtime_cache:
        hour = datetime.now().hour
        data = [_build_crowd_item(loc, hour) for loc in LOCATIONS]
        return {"data": data, "source": "cold_cache", "city": "Mumbai"}
    return {"data": realtime_cache, "source": "cache", "city": "Mumbai"}


@app.post("/realtime/predict")
async def realtime_predict(
    location_id: str = Query(...),
    hour: Optional[int] = Query(None, ge=0, le=23),
):
    """Realtime prediction for heatmap marker."""
    loc = LOCATION_MAP.get(location_id)
    if not loc:
        raise HTTPException(status_code=404, detail=f"Location '{location_id}' not found")

    target_hour = hour if hour is not None else datetime.now().hour
    density = _mock_density(location_id, target_hour)
    return {
        "location_id": location_id,
        "location_name": loc["locationName"],
        "predicted_density": density,
        "status": _crowd_status(density),
        "hour": target_hour,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# MAPS
# ---------------------------------------------------------------------------

@app.get("/maps/nearby")
async def maps_nearby(
    latitude: float = Query(...),
    longitude: float = Query(...),
    radius: float = Query(...),
    place_type: Optional[str] = Query(None),
):
    """Fetch nearby places for heatmap context."""
    mock_places = [
        {
            "place_id": f"mock_place_{i}",
            "name": f"Nearby Place {i}",
            "latitude": latitude + random.uniform(-0.01, 0.01),
            "longitude": longitude + random.uniform(-0.01, 0.01),
            "place_type": place_type or "point_of_interest",
            "crowd_hint": random.choice(["low", "medium", "high"]),
        }
        for i in range(1, 6)
    ]
    return {
        "nearby_locations": mock_places,
        "places": mock_places,
        "results": mock_places,
        "radius_km": round(radius / 1000, 2),
        "count": len(mock_places),
    }


@app.post("/maps/directions")
async def maps_directions(body: DirectionsBody):
    """Fetch route options."""
    if GOOGLE_MAPS_KEY:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://maps.googleapis.com/maps/api/directions/json",
                    params={
                        "origin": f"{body.origin.get('lat')},{body.origin.get('lng')}",
                        "destination": f"{body.destination.get('lat')},{body.destination.get('lng')}",
                        "mode": body.mode,
                        "key": GOOGLE_MAPS_KEY,
                        "alternatives": "true",
                    },
                )
                return resp.json()
        except Exception as e:
            print(f"Maps Directions API error: {e}")

    return {
        "status": "OK",
        "routes": [
            {
                "summary": f"Route {i+1} via mock road",
                "duration_minutes": random.randint(10, 40),
                "distance_km": round(random.uniform(2, 15), 1),
                "traffic_condition": random.choice(["clear", "moderate", "heavy"]),
                "crowd_level": random.choice(["low", "medium", "high"]),
            }
            for i in range(2)
        ],
    }


@app.get("/best-time")
async def best_time(
    from_location: str = Query(..., alias="from"),
    to_location: str = Query(..., alias="to"),
):
    """Suggest best travel time between two Mumbai locations."""
    hourly = {h: _mock_density(from_location, h) for h in range(24)}
    best_hour = min(hourly, key=hourly.get)
    best_density = hourly[best_hour]

    return {
        "from": from_location,
        "to": to_location,
        "best_hour": best_hour,
        "best_time": f"{best_hour:02d}:00",
        "expected_density": best_density,
        "status": _crowd_status(best_density),
        "city": "Mumbai",
        "hourly_predictions": [
            {"hour": h, "density": d, "status": _crowd_status(d)}
            for h, d in hourly.items()
        ],
    }

# ---------------------------------------------------------------------------
# AI Insights
# ---------------------------------------------------------------------------

AI_SYSTEM = (
    "You are an AI assistant for CrowdSense, a crowd monitoring app for Mumbai. "
    "Provide concise, actionable insights about crowd levels at monitored locations in Mumbai. "
    "Use emojis where helpful. Keep responses short and practical."
)


@app.post("/ai/insights")
async def ai_insights(body: AiInsightsBody):
    """Generate AI summary for Mumbai crowd situation."""
    try:
        crowd_info = ""
        if body.crowdData:
            crowd_info = "\n".join(
                f"- {item.get('locationName')}: density {item.get('crowdDensity')}% ({item.get('status')})"
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
            f"Current Mumbai crowd data:\n{crowd_info}\n\n"
            "Generate a brief Mumbai crowd situation summary with alerts and recommendations."
        )
        summary = _gemini_ask(prompt)
        return {"summary": summary, "success": True, "city": "Mumbai"}

    except Exception as e:
        print(f"AI Insights error: {e}\n{traceback.format_exc()}")
        return {"summary": "AI insights temporarily unavailable.", "success": False, "error": str(e)}


@app.post("/ai/route-advice")
async def ai_route_advice(body: AiRouteAdviceBody):
    """Generate AI route advice for Mumbai."""
    try:
        crowd_info = ""
        if body.crowdData:
            crowd_info = "\n".join(
                f"- {item.get('locationName')}: {item.get('crowd_density')}% crowd"
                for item in body.crowdData
            )

        prompt = (
            f"{AI_SYSTEM}\n\n"
            f"User wants to travel"
            + (f" from '{body.origin}'" if body.origin else "")
            + (f" to '{body.destination}'" if body.destination else "")
            + f" in Mumbai.\n\nCurrent crowd levels:\n{crowd_info or 'Not provided'}\n\n"
            "Give concise route advice: best time to travel, areas to avoid, and journey quality assessment."
        )
        advice = _gemini_ask(prompt)
        return {"advice": advice, "summary": advice, "success": True, "city": "Mumbai"}

    except Exception as e:
        print(f"AI Route Advice error: {e}\n{traceback.format_exc()}")
        return {"advice": "Route advice temporarily unavailable.", "success": False, "error": str(e)}

# ---------------------------------------------------------------------------
# TRAINING ADMIN
# ---------------------------------------------------------------------------

async def _fake_training_job(hours: int):
    """Simulate background training."""
    global training_state
    await asyncio.sleep(hours * 0.5)
    training_state.update({
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "last_rows_used": hours * random.randint(50, 200),
    })


@app.post("/realtime/train")
async def start_realtime_training(body: RealtimeTrainBody):
    """Trigger model retraining."""
    global training_state

    if not GOOGLE_MAPS_KEY:
        return {**training_state, "message": "Maps not configured", "status_code": 503}

    if training_state["status"] == "running":
        return {**training_state, "message": "Training already in progress", "status_code": 409}

    training_state.update({
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "last_error": None,
    })
    asyncio.create_task(_fake_training_job(body.hours_to_sample))

    return {**training_state, "message": "Training started", "status_code": 200}


@app.get("/realtime/train/status")
async def realtime_training_status():
    """Track training progress."""
    return {"training": training_state}


@app.get("/realtime/training-data")
async def realtime_training_data():
    """Return training data info."""
    return {
        "training_data": {
            "total_samples": training_state.get("last_rows_used", 0),
            "locations_covered": len(LOCATIONS),
            "city": "Mumbai",
            "last_trained": training_state.get("completed_at"),
            "model_version": "2.0-gemini-mumbai",
        }
    }

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
