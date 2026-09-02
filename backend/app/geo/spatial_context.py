"""
spatial_context.py
Combines OSM Service + Distance Engine to perform geospatial context analysis.

Workflow:
1. Hotspot coordinates (Point A)
2. OSM Service queries nearby features:
   - Industries
   - Forests
   - Agriculture
   - Power plants / Infrastructure
3. Distance Engine calculates great-circle distance (meters) to each candidate.
4. Ranks candidates and identifies nearest feature per category.
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Set stdout to UTF-8 if supported to prevent Windows cp1252 UnicodeEncodeError
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure app package is discoverable
backend_dir = str(Path(__file__).resolve().parents[2])
root_dir = str(Path(__file__).resolve().parents[3])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.geo.distance import calculate_distance, _extract_coords
from app.services.osm_service import find_nearby_geographic_objects


def analyze_category_distances(
    hotspot: Any,
    candidates: List[Dict[str, Any]],
    category_name: str
) -> Dict[str, Any]:
    """
    Calculate distances from a hotspot to all candidates in a specific category,
    sort them by proximity, and determine the nearest feature.
    """
    if not candidates:
        return {
            "category": category_name,
            "nearest_distance_meters": None,
            "nearest": None,
            "candidates": []
        }

    evaluated_candidates = []
    for candidate in candidates:
        try:
            dist_m = calculate_distance(hotspot, candidate, unit="m")
            cand_copy = dict(candidate)
            cand_copy["distance_meters"] = dist_m
            evaluated_candidates.append(cand_copy)
        except Exception:
            continue

    # Sort ascending by distance: shortest distance first
    evaluated_candidates.sort(key=lambda x: x["distance_meters"])

    nearest_cand = evaluated_candidates[0] if evaluated_candidates else None
    nearest_dist = nearest_cand["distance_meters"] if nearest_cand else None

    return {
        "category": category_name,
        "nearest_distance_meters": nearest_dist,
        "nearest": nearest_cand,
        "candidates": evaluated_candidates
    }


def compute_geospatial_context(
    hotspot: Any,
    radius_meters: int = 15000,
    use_live_api: bool = True
) -> Dict[str, Any]:
    """
    Full pipeline combining OSM Service and Distance Engine.
    
    1. Extracts Hotspot Coordinates (lat, lon)
    2. Calls OSM Service to find nearby candidates
    3. Calculates distance in meters to each candidate
    4. Identifies nearest Industry, Forest, Agriculture, and Power Plant
    """
    lat, lon = _extract_coords(hotspot)
    point_dict = {"latitude": lat, "longitude": lon}

    # Step 1: OSM Service finds geographic objects
    raw_objects = find_nearby_geographic_objects(
        lat=lat,
        lon=lon,
        radius_meters=radius_meters,
        use_live_api=use_live_api
    )

    # Step 2: Distance Engine calculates distances for each category
    industry_analysis = analyze_category_distances(point_dict, raw_objects.get("industry", []), "industry")
    forest_analysis = analyze_category_distances(point_dict, raw_objects.get("forest", []), "forest")
    agriculture_analysis = analyze_category_distances(point_dict, raw_objects.get("agriculture", []), "agriculture")
    power_analysis = analyze_category_distances(point_dict, raw_objects.get("power_plant", []), "power_plant")

    # Step 3: Compile standardized summary metrics
    summary_distances = {
        "distance_to_industry_m": industry_analysis["nearest_distance_meters"],
        "distance_to_forest_m": forest_analysis["nearest_distance_meters"],
        "distance_to_agriculture_m": agriculture_analysis["nearest_distance_meters"],
        "distance_to_power_plant_m": power_analysis["nearest_distance_meters"],
    }

    return {
        "hotspot": point_dict,
        "summary_distances": summary_distances,
        "categories": {
            "industry": industry_analysis,
            "forest": forest_analysis,
            "agriculture": agriculture_analysis,
            "power_plant": power_analysis,
        }
    }


if __name__ == "__main__":
    print("=" * 68)
    print("GEOSPATIAL CONTEXT PIPELINE: OSM SERVICE + DISTANCE ENGINE")
    print("=" * 68)

    # Sample Hotspot
    test_hotspot = {
        "latitude": 30.3165,
        "longitude": 78.0322,
        "frp": 42.5,
        "brightness": 325.4,
        "confidence": "high",
        "acq_date": "2026-09-01"
    }

    print(f"\n[1] Selected Hotspot: ({test_hotspot['latitude']}, {test_hotspot['longitude']})")
    print(f"    FRP: {test_hotspot['frp']} MW | Confidence: {test_hotspot['confidence']}")
    print("    Running OSM query + Distance engine...")

    context = compute_geospatial_context(test_hotspot, radius_meters=15000, use_live_api=False)

    print("\n" + "=" * 68)
    print("DETAILED CANDIDATE DISTANCES & NEAREST SELECTION")
    print("=" * 68)

    for cat_name, cat_data in context["categories"].items():
        print(f"\n--- {cat_name.upper()} ---")
        candidates = cat_data["candidates"]
        if not candidates:
            print("  (No objects found within radius)")
            continue

        for cand in candidates:
            print(f"  {cand['name']} --> {cand['distance_meters']}m")

        nearest = cat_data["nearest"]
        print(f"  ==> Nearest {cat_name.title()}: {nearest['name']} ({cat_data['nearest_distance_meters']}m)")

    print("\n" + "=" * 68)
    print("FINAL GEOSPATIAL CONTEXT SUMMARY")
    print("=" * 68)
    for metric, dist in context["summary_distances"].items():
        val_str = f"{dist} meters" if dist is not None else "None"
        print(f"  * {metric}: {val_str}")
    print("=" * 68)
