"""
distance.py
Geodesic distance calculation engine for THERMOSCOPE-AI.
Calculates distance in meters between Hotspots and various spatial entities:
- Industry
- Refinery
- Oil and Gas
- Mining
- Agriculture
- Forest
- Power plant
- Historical hotspot
"""
import sys
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

# Set stdout to UTF-8 if supported to prevent Windows cp1252 UnicodeEncodeError
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure app package is discoverable when executed directly
backend_dir = str(Path(__file__).resolve().parents[2])
root_dir = str(Path(__file__).resolve().parents[3])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

EARTH_RADIUS_METERS = 6371000.0  # Mean radius of Earth in meters
EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth using the Haversine formula.
    Returns distance in METERS.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return EARTH_RADIUS_METERS * c


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in kilometers."""
    return haversine_distance(lat1, lon1, lat2, lon2) / 1000.0


def _extract_coords(point: Any) -> Tuple[float, float]:
    """
    Extract (latitude, longitude) from various input formats:
    - Tuple or list: (lat, lon) or [lat, lon]
    - Dict: {'latitude': ..., 'longitude': ...} or {'lat': ..., 'lon': ...}
    - Object/Model: obj.latitude, obj.longitude or obj.lat, obj.lon
    """
    if isinstance(point, (tuple, list)) and len(point) >= 2:
        return float(point[0]), float(point[1])

    if isinstance(point, dict):
        lat = point.get("latitude") if "latitude" in point else point.get("lat")
        lon = point.get("longitude") if "longitude" in point else point.get("lon")
        if lat is not None and lon is not None:
            return float(lat), float(lon)

    if hasattr(point, "latitude") and hasattr(point, "longitude"):
        return float(point.latitude), float(point.longitude)

    if hasattr(point, "lat") and hasattr(point, "lon"):
        return float(point.lat), float(point.lon)

    raise ValueError(f"Unable to extract coordinates from point: {point}")


def calculate_distance(
    point_a: Any,
    point_b: Any,
    unit: str = "m"
) -> float:
    """
    Universal distance function between Point A and Point B.
    
    Point A (lat1, lon1) <-> Point B (lat2, lon2) -> Distance in meters (default)
    
    Accepts coordinates as tuples, dicts, Pydantic models, or objects.
    
    Supported target types:
    - Industry
    - Refinery
    - Oil & Gas
    - Mining
    - Agriculture
    - Forest
    - Power plant
    - Historical hotspot
    
    unit: 'm' for meters (default), 'km' for kilometers.
    """
    lat1, lon1 = _extract_coords(point_a)
    lat2, lon2 = _extract_coords(point_b)

    dist_meters = haversine_distance(lat1, lon1, lat2, lon2)
    
    if unit.lower() in ("km", "kilometer", "kilometers"):
        return round(dist_meters / 1000.0, 3)
    
    return round(dist_meters, 2)


def find_nearest_distance(
    target_point: Any,
    candidate_points: Iterable[Any],
    unit: str = "m"
) -> Tuple[float, Optional[Any]]:
    """
    Find the minimum distance from target_point to any candidate in candidate_points.
    Returns: (min_distance, nearest_candidate)
    If candidate_points is empty, returns (float('inf'), None).
    """
    min_dist = float("inf")
    nearest_item = None

    for item in candidate_points:
        try:
            d = calculate_distance(target_point, item, unit=unit)
            if d < min_dist:
                min_dist = d
                nearest_item = item
        except Exception:
            continue

    return min_dist, nearest_item


def calculate_target_distances(
    hotspot: Any,
    candidates_by_category: Dict[str, Iterable[Any]],
    unit: str = "m"
) -> Dict[str, Optional[float]]:
    """
    Calculates nearest distances from a hotspot to all target categories:
    - Industry
    - Refinery
    - Oil & Gas
    - Mining
    - Agriculture
    - Forest
    - Power plant
    
    Returns a standardized dictionary of nearest distances:
    {
        # Industrial
        "distance_to_industry_m": ...,
        
        # Oil/Gas/Persistent thermal sources
        "distance_to_refinery_m": ...,
        "distance_to_oil_gas_m": ...,
        
        # Mining
        "distance_to_mining_m": ...,
        
        # Agriculture
        "distance_to_agriculture_m": ...,
        
        # Forest
        "distance_to_forest_m": ...,
        
        # Additional infrastructure
        "distance_to_power_plant_m": ...
    }
    """
    def _get_dist(category_key: str) -> Optional[float]:
        candidates = candidates_by_category.get(category_key, [])
        if not candidates:
            return None
        min_d, _ = find_nearest_distance(hotspot, candidates, unit=unit)
        return min_d if min_d != float("inf") else None

    suffix = f"_{unit.lower()}" if unit.lower() in ("m", "km") else "_m"

    return {
        # Industrial
        f"distance_to_industry{suffix}": _get_dist("industry"),

        # Oil/Gas/Persistent thermal sources
        f"distance_to_refinery{suffix}": _get_dist("refinery"),
        f"distance_to_oil_gas{suffix}": _get_dist("oil_gas"),

        # Mining
        f"distance_to_mining{suffix}": _get_dist("mining"),

        # Agriculture
        f"distance_to_agriculture{suffix}": _get_dist("agriculture"),

        # Forest
        f"distance_to_forest{suffix}": _get_dist("forest"),

        # Additional infrastructure
        f"distance_to_power_plant{suffix}": _get_dist("power_plant"),
    }


if __name__ == "__main__":
    import json

    print("=" * 65)
    print("GEODESIC DISTANCE ENGINE: Point A <-> Point B")
    print("=" * 65)

    # Reference Hotspot
    hotspot_1 = {"latitude": 30.31650, "longitude": 78.03220, "name": "Hotspot #1"}

    # Candidate sites for all target categories
    industry_site = {"latitude": 30.32055, "longitude": 78.03220, "name": "Steel Rolling Mill", "type": "industry"}
    refinery_site = {"latitude": 30.33500, "longitude": 78.04500, "name": "Petrochemical Refinery Unit", "type": "refinery"}
    oil_gas_site = {"latitude": 30.32800, "longitude": 78.02000, "name": "Natural Gas Extraction Station", "type": "oil_gas"}
    mining_site = {"latitude": 30.34500, "longitude": 78.01500, "name": "Open-Cast Limestone Quarry", "type": "mining"}
    agri_field = {"latitude": 30.31000, "longitude": 78.02500, "name": "Paddy Cropland", "type": "agriculture"}
    forest_area = {"latitude": 30.33000, "longitude": 78.04000, "name": "Rajaji Foothill Forest", "type": "forest"}
    power_plant = {"latitude": 30.35500, "longitude": 78.05500, "name": "Thermal Power Substation", "type": "power_plant"}
    hist_hotspot = {"latitude": 30.31680, "longitude": 78.03240, "name": "Prior Hotspot (7 days ago)", "acq_date": "2026-08-25"}

    candidates_map = {
        "industry": [industry_site],
        "refinery": [refinery_site],
        "oil_gas": [oil_gas_site],
        "mining": [mining_site],
        "agriculture": [agri_field],
        "forest": [forest_area],
        "power_plant": [power_plant],
    }

    print(f"\n[+] Reference Hotspot: ({hotspot_1['latitude']}, {hotspot_1['longitude']})")
    print("\nCalculating individual geodesic distances...")
    print(f"  * [Industry]       {industry_site['name']}: {calculate_distance(hotspot_1, industry_site)} m")
    print(f"  * [Refinery]       {refinery_site['name']}: {calculate_distance(hotspot_1, refinery_site)} m")
    print(f"  * [Oil & Gas]      {oil_gas_site['name']}: {calculate_distance(hotspot_1, oil_gas_site)} m")
    print(f"  * [Mining]         {mining_site['name']}: {calculate_distance(hotspot_1, mining_site)} m")
    print(f"  * [Agriculture]    {agri_field['name']}: {calculate_distance(hotspot_1, agri_field)} m")
    print(f"  * [Forest]         {forest_area['name']}: {calculate_distance(hotspot_1, forest_area)} m")
    print(f"  * [Power Plant]    {power_plant['name']}: {calculate_distance(hotspot_1, power_plant)} m")
    print(f"  * [Historical]     {hist_hotspot['name']}: {calculate_distance(hotspot_1, hist_hotspot)} m")

    print("\n" + "=" * 65)
    print("STRUCTURED TARGET DISTANCES DICTIONARY")
    print("=" * 65)
    target_distances = calculate_target_distances(hotspot_1, candidates_map)
    print(json.dumps(target_distances, indent=4))
    print("=" * 65)
