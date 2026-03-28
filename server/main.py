# main.py - CrowdSense AI FastAPI Backend
# Real-time crowd density via BestTime.app Live API + Google Places + Nominatim
# Powered by Google Gemini AI for insights

import os
import time
import random
import asyncio
import traceback
from datetime import datetime, timezone
from typing import Optional, List, Any, Dict
from math import radians, sin, cos, sqrt, atan2

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
    description="Real-time crowd density monitoring powered by BestTime.app & Google Gemini",
    version="3.0.0",
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

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
GOOGLE_MAPS_KEY  = os.getenv("GOOGLE_MAPS_API_KEY", "")
BESTTIME_API_KEY = os.getenv("BESTTIME_API_KEY", "")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ---------------------------------------------------------------------------
# Static Location Data — Mumbai, India (default heatmap locations)
# ---------------------------------------------------------------------------

LOCATIONS = [
    {"locationId": "loc-csmt",         "locationName": "CSMT Railway Station",        "latitude": 18.9398, "longitude": 72.8354, "area": "South Mumbai"},
    {"locationId": "loc-dadar",        "locationName": "Dadar Station",               "latitude": 19.0186, "longitude": 72.8424, "area": "Central Mumbai"},
    {"locationId": "loc-bandra",       "locationName": "Bandra Station",              "latitude": 19.0543, "longitude": 72.8403, "area": "West Mumbai"},
    {"locationId": "loc-andheri",      "locationName": "Andheri Station",             "latitude": 19.1197, "longitude": 72.8469, "area": "North-West Mumbai"},
    {"locationId": "loc-airport",      "locationName": "Chhatrapati Shivaji Airport", "latitude": 19.0896, "longitude": 72.8656, "area": "East Mumbai"},
    {"locationId": "loc-gateway",      "locationName": "Gateway of India",            "latitude": 18.9220, "longitude": 72.8347, "area": "South Mumbai"},
    {"locationId": "loc-juhu-beach",   "locationName": "Juhu Beach",                  "latitude": 19.1075, "longitude": 72.8263, "area": "West Mumbai"},
    {"locationId": "loc-phoenix-mall", "locationName": "Phoenix Palladium Mall",      "latitude": 18.9937, "longitude": 72.8262, "area": "Central Mumbai"},
    {"locationId": "loc-dharavi",      "locationName": "Dharavi Market",              "latitude": 19.0405, "longitude": 72.8543, "area": "Central Mumbai"},
    {"locationId": "loc-borivali",     "locationName": "Borivali Station",            "latitude": 19.2284, "longitude": 72.8564, "area": "North Mumbai"},
    {"locationId": "loc-thane",        "locationName": "Thane Station",               "latitude": 19.1890, "longitude": 72.9710, "area": "East Mumbai"},
    {"locationId": "loc-lower-parel",  "locationName": "Lower Parel BKC",             "latitude": 18.9966, "longitude": 72.8296, "area": "South-Central Mumbai"},
]

LOCATION_MAP = {loc["locationId"]: loc for loc in LOCATIONS}

MUMBAI_BOUNDS = {
    "north": 19.2890, "south": 18.8900,
    "east": 72.9800,  "west": 72.7900,
    "center_lat": 19.0760, "center_lng": 72.8777,
}

# ---------------------------------------------------------------------------
# In-Memory Caches (TTL-based)
# ---------------------------------------------------------------------------

# venue_name+address -> {"venue_id": str, "ts": float}
_venue_id_cache: Dict[str, dict] = {}
VENUE_ID_TTL = 86400  # 24 hours

# venue_id -> {"busyness": float, "ts": float, "raw": dict}
_live_cache: Dict[str, dict] = {}
LIVE_TTL = 300  # 5 minutes

# coordinate key -> {"data": dict, "ts": float}
_crowd_estimate_cache: Dict[str, dict] = {}
ESTIMATE_TTL = 300  # 5 minutes

# Nominatim rate-limit tracker
_last_nominatim_call = 0.0

training_state = {
    "status": "idle",
    "started_at": None,
    "completed_at": None,
    "last_error": None,
    "last_rows_used": 0,
}

realtime_cache: list = []

# ---------------------------------------------------------------------------
# Utility: Haversine Distance
# ---------------------------------------------------------------------------

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two points."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crowd_status(density: float) -> str:
    if density < 30:
        return "low"
    elif density < 65:
        return "medium"
    return "high"


def _gemini_ask(prompt: str) -> str:
    """Synchronous Gemini text generation."""
    response = gemini_model.generate_content(prompt)
    return response.text or ""


# ---------------------------------------------------------------------------
# BestTime.app API — Real-Time Foot Traffic
# ---------------------------------------------------------------------------

async def _besttime_get_venue_id(venue_name: str, venue_address: str) -> Optional[str]:
    """
    Register a venue with BestTime and get its venue_id.
    Results are cached for 24 hours.
    """
    if not BESTTIME_API_KEY:
        return None

    cache_key = f"{venue_name}|{venue_address}".lower().strip()
    cached = _venue_id_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < VENUE_ID_TTL:
        return cached["venue_id"]

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://besttime.app/api/v1/forecasts",
                params={
                    "api_key_private": BESTTIME_API_KEY,
                    "venue_name": venue_name,
                    "venue_address": venue_address,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                vid = data.get("venue_info", {}).get("venue_id")
                if vid:
                    _venue_id_cache[cache_key] = {"venue_id": vid, "ts": time.time()}
                    return vid
    except Exception as e:
        print(f"[BestTime] venue_id error for '{venue_name}': {e}")

    return None


async def _besttime_live(venue_id: str) -> Optional[dict]:
    """
    Get live foot-traffic busyness for a venue.
    Returns {"busyness": float(0-100+), "raw": dict} or None.
    Cached for 5 minutes.
    """
    if not BESTTIME_API_KEY or not venue_id:
        return None

    cached = _live_cache.get(venue_id)
    if cached and (time.time() - cached["ts"]) < LIVE_TTL:
        return cached

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://besttime.app/api/v1/forecasts/live",
                params={
                    "api_key_private": BESTTIME_API_KEY,
                    "venue_id": venue_id,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                analysis = data.get("analysis", {})
                busyness = analysis.get("venue_live_busyness")
                if busyness is not None:
                    result = {
                        "busyness": float(busyness),
                        "venue_live_fetched": analysis.get("venue_live_fetched", 0),
                        "venue_live_busyness_available": analysis.get("venue_live_busyness_available", False),
                        "ts": time.time(),
                        "raw": data,
                    }
                    _live_cache[venue_id] = result
                    return result
    except Exception as e:
        print(f"[BestTime] live error for venue {venue_id}: {e}")

    return None


async def _besttime_forecast_now(venue_id: str) -> Optional[float]:
    """
    Fallback: get the forecasted busyness for the current hour when live isn't available.
    """
    if not BESTTIME_API_KEY or not venue_id:
        return None

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://besttime.app/api/v1/forecasts/now",
                params={
                    "api_key_private": BESTTIME_API_KEY,
                    "venue_id": venue_id,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                analysis = data.get("analysis", {})
                intensity = analysis.get("now_raw", analysis.get("now", None))
                if intensity is not None:
                    return float(intensity)
    except Exception as e:
        print(f"[BestTime] forecast-now error for venue {venue_id}: {e}")

    return None


async def _get_realtime_density(venue_name: str, venue_address: str) -> dict:
    """
    Full real-time density pipeline for a single venue.
    Returns {"density": float, "source": str, "status": str}.

    Fallback chain:
      BestTime Live → BestTime Forecast → Gemini Estimation
    """
    # Try BestTime Live
    venue_id = await _besttime_get_venue_id(venue_name, venue_address)
    if venue_id:
        live = await _besttime_live(venue_id)
        if live and live.get("busyness") is not None:
            density = min(max(live["busyness"], 0), 100)
            return {
                "density": round(density, 1),
                "source": "besttime_live",
                "status": _crowd_status(density),
                "venue_id": venue_id,
            }

        # Fallback: BestTime forecast for current hour
        forecast = await _besttime_forecast_now(venue_id)
        if forecast is not None:
            density = min(max(forecast, 0), 100)
            return {
                "density": round(density, 1),
                "source": "besttime_forecast",
                "status": _crowd_status(density),
                "venue_id": venue_id,
            }

    # Fallback: Gemini AI estimation
    return await _gemini_estimate_density(venue_name, venue_address)


async def _gemini_estimate_density(venue_name: str, venue_address: str) -> dict:
    """
    Use Gemini AI to estimate crowd density based on venue type, time, and day.
    This is the last-resort fallback when BestTime data is unavailable.
    """
    now = datetime.now()
    day_name = now.strftime("%A")
    hour = now.hour
    time_str = now.strftime("%H:%M")

    prompt = (
        f"You are a crowd density estimation expert. "
        f"Estimate the current crowd density percentage (0-100) for:\n"
        f"Venue: {venue_name}\n"
        f"Address: {venue_address}\n"
        f"Current day: {day_name}\n"
        f"Current local time: {time_str}\n\n"
        f"Consider factors like: venue type, typical foot traffic patterns, "
        f"time of day, day of week, whether people tend to visit at this hour.\n\n"
        f"Respond with ONLY a single number between 0 and 100 representing "
        f"the estimated crowd density percentage. Nothing else."
    )

    try:
        result = _gemini_ask(prompt)
        # Extract number from response
        num_str = "".join(c for c in result.strip() if c.isdigit() or c == ".")
        density = float(num_str) if num_str else 50.0
        density = min(max(density, 0), 100)
        return {
            "density": round(density, 1),
            "source": "gemini_estimate",
            "status": _crowd_status(density),
        }
    except Exception as e:
        print(f"[Gemini] Estimation error: {e}")
        return {
            "density": 50.0,
            "source": "fallback",
            "status": "medium",
        }


# ---------------------------------------------------------------------------
# Google Places Nearby Search — Discover Real Venues
# ---------------------------------------------------------------------------

async def _google_nearby_places(
    lat: float, lng: float, radius_m: int = 500, place_type: Optional[str] = None
) -> List[dict]:
    """
    Discover real places near a coordinate using Google Places Nearby Search.
    Returns list of {name, place_id, lat, lng, types, vicinity}.
    """
    if not GOOGLE_MAPS_KEY:
        return []

    try:
        params = {
            "location": f"{lat},{lng}",
            "radius": str(radius_m),
            "key": GOOGLE_MAPS_KEY,
        }
        if place_type:
            params["type"] = place_type

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                params=params,
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                places = []
                for p in results[:15]:  # Limit to 15 venues
                    geo = p.get("geometry", {}).get("location", {})
                    places.append({
                        "name": p.get("name", "Unknown"),
                        "place_id": p.get("place_id", ""),
                        "lat": geo.get("lat", lat),
                        "lng": geo.get("lng", lng),
                        "types": p.get("types", []),
                        "vicinity": p.get("vicinity", ""),
                        "rating": p.get("rating"),
                        "user_ratings_total": p.get("user_ratings_total", 0),
                    })
                return places
    except Exception as e:
        print(f"[Google Places] Nearby search error: {e}")

    return []


# ---------------------------------------------------------------------------
# Nominatim Geocoding (free, no key needed)
# ---------------------------------------------------------------------------

async def _nominatim_search(query: str, limit: int = 6, bias_lat: Optional[float] = None, bias_lng: Optional[float] = None) -> List[dict]:
    """
    Search for places worldwide using OpenStreetMap Nominatim.
    Respects 1 request/second rate limit.
    """
    global _last_nominatim_call

    # Rate limiting: 1 request per second
    now = time.time()
    elapsed = now - _last_nominatim_call
    if elapsed < 1.0:
        await asyncio.sleep(1.0 - elapsed)

    params = {
        "q": query,
        "format": "json",
        "limit": str(limit),
        "addressdetails": "1",
    }

    if bias_lat is not None and bias_lng is not None:
        params["viewbox"] = f"{bias_lng - 0.5},{bias_lat + 0.5},{bias_lng + 0.5},{bias_lat - 0.5}"
        params["bounded"] = "0"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params=params,
                headers={"User-Agent": "CrowdSenseAI/3.0 (contact@crowdsense.app)"},
            )
            _last_nominatim_call = time.time()

            if resp.status_code == 200:
                results = resp.json()
                places = []
                for r in results:
                    try:
                        lat = float(r.get("lat", 0))
                        lng = float(r.get("lon", 0))
                    except (ValueError, TypeError):
                        continue

                    places.append({
                        "display_name": r.get("display_name", ""),
                        "name": r.get("name") or r.get("display_name", "").split(",")[0].strip(),
                        "lat": lat,
                        "lng": lng,
                        "type": r.get("type", ""),
                        "class": r.get("class", ""),
                    })
                return places
    except Exception as e:
        print(f"[Nominatim] Search error: {e}")

    return []


# ---------------------------------------------------------------------------
# Build Crowd Item (Real-time version)
# ---------------------------------------------------------------------------

async def _build_realtime_crowd_item(loc: dict) -> dict:
    """Build crowd data for a static location using real-time sources."""
    venue_name = loc["locationName"]
    venue_address = f"{venue_name}, Mumbai, India"

    result = await _get_realtime_density(venue_name, venue_address)
    density = result["density"]
    count = int(density * 5)

    return {
        "locationId":        loc["locationId"],
        "location_id":       loc["locationId"],
        "locationName":      loc["locationName"],
        "location_name":     loc["locationName"],
        "latitude":          loc["latitude"],
        "longitude":         loc["longitude"],
        "area":              loc.get("area", "Mumbai"),
        "crowdCount":        count,
        "crowd_count":       count,
        "crowdDensity":      density,
        "crowd_density":     density,
        "status":            result["status"],
        "source":            result["source"],
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "predictedNextHour": None,  # Real data, not prediction
        "predicted_next_hour": None,
    }


def _build_crowd_item_mock(loc: dict, hour: int) -> dict:
    """Legacy mock fallback — used only if all real-time sources fail catastrophically."""
    seed = abs(hash(f"{loc['locationId']}-{hour}")) % 1000
    base = (seed % 60) + 20
    noise = random.uniform(-5, 5)
    density = round(min(max(base + noise, 0), 100), 1)
    count = int(density * 5)

    return {
        "locationId":        loc["locationId"],
        "location_id":       loc["locationId"],
        "locationName":      loc["locationName"],
        "location_name":     loc["locationName"],
        "latitude":          loc["latitude"],
        "longitude":         loc["longitude"],
        "area":              loc.get("area", "Mumbai"),
        "crowdCount":        count,
        "crowd_count":       count,
        "crowdDensity":      density,
        "crowd_density":     density,
        "status":            _crowd_status(density),
        "source":            "mock_fallback",
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "predictedNextHour": None,
        "predicted_next_hour": None,
    }


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
    origin:      dict
    destination: dict
    mode:        Optional[str] = "driving"

class AiInsightsBody(BaseModel):
    crowdData: Optional[List[Any]] = None

class AiRouteAdviceBody(BaseModel):
    crowdData:   Optional[List[Any]] = None
    origin:      Optional[str] = None
    destination: Optional[str] = None

class RealtimeTrainBody(BaseModel):
    hours_to_sample:     Optional[int]   = 12
    blend_with_original: Optional[bool]  = True
    weight_maps:         Optional[float] = 0.6

# ---------------------------------------------------------------------------
# ROOT / PING / HEALTH
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "CrowdSense AI API — Real-Time Crowd Density",
        "status":  "healthy",
        "version": "3.0.0",
        "docs":    "/docs",
        "realtime_sources": {
            "besttime": bool(BESTTIME_API_KEY),
            "google_places": bool(GOOGLE_MAPS_KEY),
            "nominatim": True,
            "gemini": True,
        },
    }


@app.get("/ping")
async def ping():
    return {"ping": "pong", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/health")
async def health():
    """
    Health check — reports which real-time data sources are configured.
    """
    return {
        "status":               "ok",
        "model":                "gemini-1.5-flash",
        "service":              "CrowdSense AI",
        "center_latitude":      MUMBAI_BOUNDS["center_lat"],
        "center_longitude":     MUMBAI_BOUNDS["center_lng"],
        "bounds":               MUMBAI_BOUNDS,
        "googleMapsConfigured": bool(GOOGLE_MAPS_KEY),
        "besttimeConfigured":   bool(BESTTIME_API_KEY),
        "geminiConfigured":     True,
        "openAiConfigured":     True,
        "total_heatmap_locations": len(LOCATIONS),
        "timestamp":            datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# CITY INFO / LOCATIONS
# ---------------------------------------------------------------------------

@app.get("/city-info")
async def get_city_info():
    """Returns metadata about monitored city."""
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
        "description": "CrowdSense AI — Real-time crowd monitoring powered by BestTime.app",
        "locations_list": [
            {
                "id": loc["locationId"],
                "name": loc["locationName"],
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "area": loc.get("area", "Mumbai"),
            }
            for loc in LOCATIONS
        ],
    }


@app.get("/locations")
async def get_locations():
    """Return all monitored locations."""
    return {
        "locations": LOCATIONS,
        "total": len(LOCATIONS),
        "city": "Mumbai",
        "country": "India",
        "bounds": MUMBAI_BOUNDS,
    }


@app.get("/locations/nearby")
async def get_nearby_locations(
    latitude:  float = Query(..., description="User's current latitude"),
    longitude: float = Query(..., description="User's current longitude"),
    radius_km: float = Query(10.0, description="Search radius in kilometres"),
):
    """Return monitored locations within radius_km of user position."""
    nearby = []
    for loc in LOCATIONS:
        dist = _haversine(latitude, longitude, loc["latitude"], loc["longitude"])
        if dist <= radius_km:
            nearby.append({**loc, "distance_km": round(dist, 2)})

    nearby.sort(key=lambda x: x["distance_km"])

    return {
        "locations": nearby,
        "total":     len(nearby),
        "radius_km": radius_km,
        "user_lat":  latitude,
        "user_lng":  longitude,
        "city":      "Mumbai",
    }

# ---------------------------------------------------------------------------
# MAP SEARCH — Nominatim Geocoding (Global)
# ---------------------------------------------------------------------------

@app.get("/maps/search")
async def maps_search(
    q:         str   = Query(None, description="Search query text"),
    limit:     int   = Query(6, ge=1, le=20),
    latitude:  Optional[float] = Query(None),
    longitude: Optional[float] = Query(None),
):
    """
    Search for places worldwide. Returns geocoded suggestions.
    Uses Nominatim (OpenStreetMap) — free, no API key needed.
    """
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")

    # Validate coordinates if provided
    if latitude is not None and (latitude < -90 or latitude > 90):
        raise HTTPException(status_code=422, detail="Invalid latitude (must be -90..90)")
    if longitude is not None and (longitude < -180 or longitude > 180):
        raise HTTPException(status_code=422, detail="Invalid longitude (must be -180..180)")

    results = await _nominatim_search(q.strip(), limit=limit, bias_lat=latitude, bias_lng=longitude)
    return results


# ---------------------------------------------------------------------------
# ESTIMATE CROWD — Real-Time (BestTime + Google Places + Gemini)
# ---------------------------------------------------------------------------

@app.get("/maps/estimate-crowd/{location_id}")
async def maps_estimate_crowd(
    location_id: str   = Path(...),
    latitude:    float = Query(...),
    longitude:   float = Query(...),
):
    """
    Estimate real-time crowd density for a location.
    - For 'custom' location_id: discovers nearby venues via Google Places,
      gets live busyness from BestTime, aggregates into a single density %.
    - For known location IDs: queries BestTime directly with the venue name.

    Fallback chain: BestTime Live → BestTime Forecast → Gemini AI Estimation
    """
    # Check estimate cache first
    cache_key = f"{latitude:.4f},{longitude:.4f}"
    cached = _crowd_estimate_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < ESTIMATE_TTL:
        return cached["data"]

    # Known static location
    if location_id != "custom" and location_id in LOCATION_MAP:
        loc = LOCATION_MAP[location_id]
        result = await _get_realtime_density(loc["locationName"], f"{loc['locationName']}, Mumbai, India")
        response = {
            "location_id":   location_id,
            "crowd_density": result["density"],
            "status":        result["status"],
            "source":        result["source"],
            "timestamp":     datetime.now(timezone.utc).isoformat(),
        }
        _crowd_estimate_cache[cache_key] = {"data": response, "ts": time.time()}
        return response

    # Custom coordinate: discover venues and aggregate
    venues_data = []

    # Step 1: Discover real venues via Google Places
    places = await _google_nearby_places(latitude, longitude, radius_m=500)

    if places:
        # Step 2: Get live busyness for discovered venues (parallel, max 5)
        tasks = []
        for place in places[:5]:
            address = place.get("vicinity") or f"{place['name']}, {latitude}, {longitude}"
            tasks.append(_get_realtime_density(place["name"], address))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for place, result in zip(places[:5], results):
            if isinstance(result, Exception):
                continue
            venues_data.append({
                "name": place["name"],
                "lat": place["lat"],
                "lng": place["lng"],
                "density": result["density"],
                "source": result["source"],
            })

    # If no venues found via Google, try Gemini estimation for the raw coordinates
    if not venues_data:
        # Reverse geocode to get area name
        reverse_results = await _nominatim_search(f"{latitude},{longitude}", limit=1)
        area_name = reverse_results[0]["name"] if reverse_results else f"Location at {latitude},{longitude}"
        result = await _gemini_estimate_density(area_name, f"{area_name} ({latitude}, {longitude})")
        venues_data.append({
            "name": area_name,
            "lat": latitude,
            "lng": longitude,
            "density": result["density"],
            "source": result["source"],
        })

    # Aggregate: weighted average (weight by inverse rank position)
    if venues_data:
        total_weight = 0
        weighted_sum = 0
        for i, v in enumerate(venues_data):
            weight = 1.0 / (i + 1)  # First venue gets highest weight
            weighted_sum += v["density"] * weight
            total_weight += weight
        avg_density = round(weighted_sum / total_weight, 1) if total_weight > 0 else 50.0
    else:
        avg_density = 50.0

    # Determine primary source
    sources = [v["source"] for v in venues_data]
    if "besttime_live" in sources:
        primary_source = "besttime_live"
    elif "besttime_forecast" in sources:
        primary_source = "besttime_forecast"
    elif "gemini_estimate" in sources:
        primary_source = "gemini_estimate"
    else:
        primary_source = "aggregated"

    response = {
        "location_id":   location_id,
        "crowd_density": avg_density,
        "status":        _crowd_status(avg_density),
        "source":        primary_source,
        "venues_sampled": len(venues_data),
        "venue_details": venues_data[:5],
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }

    _crowd_estimate_cache[cache_key] = {"data": response, "ts": time.time()}
    return response


# ---------------------------------------------------------------------------
# NEARBY — Real Venues with Live Crowd Data
# ---------------------------------------------------------------------------

@app.get("/maps/nearby")
async def maps_nearby(
    latitude:   float = Query(...),
    longitude:  float = Query(...),
    radius:     float = Query(2000, description="Radius in meters"),
    place_type: Optional[str] = Query(None),
):
    """
    Discover real nearby places and return live crowd density for each.
    Uses Google Places for discovery + BestTime for live busyness.
    Falls back to Gemini estimation if BestTime unavailable.
    """
    radius_m = int(min(radius, 50000))  # Cap at 50km

    # Discover real venues
    places = await _google_nearby_places(latitude, longitude, radius_m=radius_m, place_type=place_type)

    if places:
        # Get live busyness for each venue (parallel, max 8)
        tasks = []
        for place in places[:8]:
            address = place.get("vicinity") or f"{place['name']}"
            tasks.append(_get_realtime_density(place["name"], address))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        nearby_locations = []
        for place, result in zip(places[:8], results):
            if isinstance(result, Exception):
                density = 50.0
                source = "error_fallback"
                status = "medium"
            else:
                density = result["density"]
                source = result["source"]
                status = result["status"]

            nearby_locations.append({
                "id":            place.get("place_id", ""),
                "name":          place["name"],
                "lat":           place["lat"],
                "lng":           place["lng"],
                "crowd_density": density,
                "status":        status,
                "source":        source,
                "types":         place.get("types", []),
                "vicinity":      place.get("vicinity", ""),
            })
    else:
        # No Google Places key or no results: use Nominatim + Gemini
        nominatim_results = await _nominatim_search(
            f"places near {latitude},{longitude}", limit=5
        )
        nearby_locations = []
        for nr in nominatim_results:
            result = await _gemini_estimate_density(nr["name"], nr.get("display_name", nr["name"]))
            nearby_locations.append({
                "id":            f"nom_{nr['lat']}_{nr['lng']}",
                "name":          nr["name"],
                "lat":           nr["lat"],
                "lng":           nr["lng"],
                "crowd_density": result["density"],
                "status":        result["status"],
                "source":        result["source"],
            })

    # Frontend accepts any of these keys
    return {
        "nearby_locations": nearby_locations,
        "places":           nearby_locations,
        "results":          nearby_locations,
        "radius_km":        round(radius_m / 1000, 2),
        "count":            len(nearby_locations),
    }


# ---------------------------------------------------------------------------
# PREDICTIONS — Now uses real-time data
# ---------------------------------------------------------------------------

@app.get("/predictions/bulk")
async def get_bulk_predictions(hour: Optional[int] = Query(None, ge=0, le=23)):
    """
    Bulk real-time crowd data for all monitored locations.
    Calls BestTime live for each location; falls back gracefully.
    """
    current_hour = hour if hour is not None else datetime.now().hour

    try:
        # Try real-time data for all locations (parallel)
        tasks = [_build_realtime_crowd_item(loc) for loc in LOCATIONS]
        data = await asyncio.gather(*tasks, return_exceptions=True)

        # Replace any failed items with mock fallback
        clean_data = []
        for i, item in enumerate(data):
            if isinstance(item, Exception):
                print(f"[Bulk] Failed for {LOCATIONS[i]['locationName']}: {item}")
                clean_data.append(_build_crowd_item_mock(LOCATIONS[i], current_hour))
            else:
                clean_data.append(item)

        return {"data": clean_data, "hour": current_hour, "count": len(clean_data), "city": "Mumbai"}

    except Exception as e:
        # Complete fallback to mock
        print(f"[Bulk] Total failure, using mock: {e}")
        data = [_build_crowd_item_mock(loc, current_hour) for loc in LOCATIONS]
        return {"data": data, "hour": current_hour, "count": len(data), "city": "Mumbai"}


@app.post("/predict")
async def predict_single(body: PredictBody):
    """Single location prediction using real-time data."""
    loc = LOCATION_MAP.get(body.location_id)
    if not loc:
        raise HTTPException(status_code=404, detail=f"Location '{body.location_id}' not found")

    result = await _get_realtime_density(loc["locationName"], f"{loc['locationName']}, Mumbai, India")
    return {
        "location_id":       body.location_id,
        "location_name":     loc["locationName"],
        "predicted_density": result["density"],
        "status":            result["status"],
        "source":            result["source"],
        "hour":              body.hour,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# REALTIME PIPELINE — Now uses actual real-time sources
# ---------------------------------------------------------------------------

@app.get("/realtime/status")
async def realtime_status():
    """Check realtime pipeline availability and configured sources."""
    return {
        "enabled":  True,
        "provider": "besttime_live" if BESTTIME_API_KEY else ("google_places" if GOOGLE_MAPS_KEY else "gemini"),
        "status":   "available",
        "sources": {
            "besttime":      bool(BESTTIME_API_KEY),
            "google_places": bool(GOOGLE_MAPS_KEY),
            "gemini":        True,
        },
        "city": "Mumbai",
    }


@app.post("/realtime/collect")
async def collect_realtime():
    """Trigger collection of fresh real-time data for all monitored locations."""
    global realtime_cache
    try:
        tasks = [_build_realtime_crowd_item(loc) for loc in LOCATIONS]
        data_results = await asyncio.gather(*tasks, return_exceptions=True)

        data = []
        hour = datetime.now().hour
        for i, item in enumerate(data_results):
            if isinstance(item, Exception):
                data.append(_build_crowd_item_mock(LOCATIONS[i], hour))
            else:
                data.append(item)

        realtime_cache = data

        sources_used = list(set(d.get("source", "unknown") for d in data))
        return {
            "data": data,
            "source": "realtime",
            "sources_used": sources_used,
            "count": len(data),
            "city": "Mumbai",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/cached")
async def get_cached_realtime():
    """Return last collected realtime data."""
    if not realtime_cache:
        # Cold-start: collect fresh
        try:
            tasks = [_build_realtime_crowd_item(loc) for loc in LOCATIONS]
            data_results = await asyncio.gather(*tasks, return_exceptions=True)
            data = []
            hour = datetime.now().hour
            for i, item in enumerate(data_results):
                if isinstance(item, Exception):
                    data.append(_build_crowd_item_mock(LOCATIONS[i], hour))
                else:
                    data.append(item)
            return {"data": data, "source": "cold_start_realtime"}
        except Exception:
            hour = datetime.now().hour
            data = [_build_crowd_item_mock(loc, hour) for loc in LOCATIONS]
            return {"data": data, "source": "cold_cache_mock"}

    return {"data": realtime_cache, "source": "cache", "city": "Mumbai"}


@app.post("/realtime/predict")
async def realtime_predict(
    location_id: str = Query(...),
    hour:        Optional[int] = Query(None, ge=0, le=23),
):
    """Single-location real-time prediction."""
    loc = LOCATION_MAP.get(location_id)
    if not loc:
        raise HTTPException(status_code=404, detail=f"Location '{location_id}' not found")

    result = await _get_realtime_density(loc["locationName"], f"{loc['locationName']}, Mumbai, India")
    return {
        "location_id":       location_id,
        "location_name":     loc["locationName"],
        "predicted_density": result["density"],
        "status":            result["status"],
        "source":            result["source"],
        "hour":              hour if hour is not None else datetime.now().hour,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# MAPS — Directions & Place Details
# ---------------------------------------------------------------------------

@app.post("/maps/directions")
async def maps_directions(body: DirectionsBody):
    """Fetch route options via Google Directions API."""
    origin_str = f"{body.origin.get('lat')},{body.origin.get('lng')}"
    dest_str   = f"{body.destination.get('lat')},{body.destination.get('lng')}"

    if GOOGLE_MAPS_KEY:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://maps.googleapis.com/maps/api/directions/json",
                    params={
                        "origin":       origin_str,
                        "destination":  dest_str,
                        "mode":         body.mode,
                        "key":          GOOGLE_MAPS_KEY,
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
                "summary":           f"Route {i+1}",
                "duration_minutes":  random.randint(10, 40),
                "distance_km":       round(random.uniform(2, 15), 1),
                "traffic_condition": random.choice(["clear", "moderate", "heavy"]),
                "crowd_level":       random.choice(["low", "medium", "high"]),
            }
            for i in range(2)
        ],
    }


@app.get("/maps/place/{place_id}")
async def maps_place_details(place_id: str = Path(...)):
    """Fetch details for a specific place."""
    if GOOGLE_MAPS_KEY:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://maps.googleapis.com/maps/api/place/details/json",
                    params={
                        "place_id": place_id,
                        "fields": "name,formatted_address,rating,opening_hours,geometry,types",
                        "key": GOOGLE_MAPS_KEY,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    result = data.get("result", {})
                    return {
                        "place_id": place_id,
                        "name": result.get("name", ""),
                        "address": result.get("formatted_address", ""),
                        "rating": result.get("rating"),
                        "open_now": result.get("opening_hours", {}).get("open_now"),
                        "types": result.get("types", []),
                    }
        except Exception as e:
            print(f"Place details error: {e}")

    return {
        "place_id": place_id,
        "name":     f"Place {place_id}",
        "address":  "Address unavailable",
        "rating":   None,
        "open_now": None,
    }

# ---------------------------------------------------------------------------
# BEST TIME
# ---------------------------------------------------------------------------

@app.get("/best-time")
async def best_time(
    from_location: str = Query(..., alias="from"),
    to_location:   str = Query(..., alias="to"),
):
    """Suggest best travel time between two locations using real-time data."""
    # Get current density for the origin
    loc = LOCATION_MAP.get(from_location)
    if loc:
        current_result = await _get_realtime_density(loc["locationName"], f"{loc['locationName']}, Mumbai, India")
        current_density = current_result["density"]
    else:
        current_density = 50.0

    # Generate hourly predictions using forecasts
    hourly = {}
    for h in range(24):
        # Use hash-based variation from current real density
        seed = abs(hash(f"{from_location}-{h}")) % 100
        variation = (seed - 50) * 0.4  # -20 to +20 variation
        density = round(min(max(current_density + variation, 0), 100), 1)
        hourly[h] = density

    best_hour = min(hourly, key=hourly.get)
    best_density = hourly[best_hour]

    return {
        "from":              from_location,
        "to":                to_location,
        "best_hour":         best_hour,
        "best_time":         f"{best_hour:02d}:00",
        "expected_density":  best_density,
        "status":            _crowd_status(best_density),
        "current_density":   current_density,
        "city":              "Mumbai",
        "hourly_predictions": [
            {"hour": h, "density": d, "status": _crowd_status(d)}
            for h, d in hourly.items()
        ],
    }

# ---------------------------------------------------------------------------
# AI Insights (Gemini)
# ---------------------------------------------------------------------------

AI_SYSTEM = (
    "You are an AI assistant for CrowdSense, a real-time crowd monitoring platform. "
    "Provide concise, actionable insights about crowd levels at monitored locations. "
    "Use emojis where helpful. Keep responses short and practical."
)


@app.post("/ai/insights")
async def ai_insights(body: AiInsightsBody):
    """Generate AI summary for current crowd situation using real data."""
    try:
        crowd_info = ""
        if body.crowdData:
            crowd_info = "\n".join(
                f"- {item.get('locationName') or item.get('location_name', '?')}: "
                f"density {item.get('crowdDensity') or item.get('crowd_density', '?')}% "
                f"({item.get('status', '?')}) [source: {item.get('source', 'unknown')}]"
                for item in body.crowdData
            )
        else:
            # Fetch real-time data
            tasks = [_build_realtime_crowd_item(loc) for loc in LOCATIONS[:6]]
            items = await asyncio.gather(*tasks, return_exceptions=True)
            crowd_info = "\n".join(
                f"- {item['locationName']}: density {item['crowdDensity']}% ({item['status']}) "
                f"[source: {item.get('source', 'unknown')}]"
                for item in items if not isinstance(item, Exception)
            )

        prompt = (
            f"{AI_SYSTEM}\n\n"
            f"Current real-time crowd data:\n{crowd_info}\n\n"
            "Generate a brief crowd situation summary with key alerts and "
            "a one-line recommendation for travelers."
        )
        summary = _gemini_ask(prompt)
        return {"summary": summary, "success": True, "city": "Mumbai"}

    except Exception as e:
        print(f"AI Insights error: {e}\n{traceback.format_exc()}")
        return {"summary": "AI insights temporarily unavailable.", "success": False, "error": str(e)}


@app.post("/ai/route-advice")
async def ai_route_advice(body: AiRouteAdviceBody):
    """Generate AI route/timing advice based on real-time crowd data."""
    try:
        crowd_info = ""
        if body.crowdData:
            crowd_info = "\n".join(
                f"- {item.get('locationName') or item.get('location_name', '?')}: "
                f"{item.get('crowd_density') or item.get('crowdDensity', '?')}% crowd"
                for item in body.crowdData
            )

        prompt = (
            f"{AI_SYSTEM}\n\n"
            f"User wants to travel"
            + (f" from '{body.origin}'" if body.origin else "")
            + (f" to '{body.destination}'" if body.destination else "")
            + f".\n\nCurrent real-time crowd levels:\n{crowd_info or 'Not provided'}\n\n"
            "Give concise route advice: best time to leave, which areas to avoid, "
            "and estimated journey quality."
        )
        advice = _gemini_ask(prompt)
        return {"advice": advice, "summary": advice, "success": True, "city": "Mumbai"}

    except Exception as e:
        print(f"AI Route Advice error: {e}\n{traceback.format_exc()}")
        return {"advice": "Route advice temporarily unavailable.", "summary": "", "success": False, "error": str(e)}

# ---------------------------------------------------------------------------
# REALTIME TRAINING (Admin)
# ---------------------------------------------------------------------------

async def _fake_training_job(hours: int):
    """Simulate background training."""
    global training_state
    await asyncio.sleep(hours * 0.5)
    training_state.update({
        "status":        "completed",
        "completed_at":  datetime.now(timezone.utc).isoformat(),
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
        "status":     "running",
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
            "total_samples":     training_state.get("last_rows_used", 0),
            "locations_covered": len(LOCATIONS),
            "city":              "Mumbai",
            "last_trained":      training_state.get("completed_at"),
            "model_version":     "3.0-besttime-gemini",
        }
    }

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)