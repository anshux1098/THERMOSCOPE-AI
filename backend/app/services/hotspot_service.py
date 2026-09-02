"""
hotspot_service.py
ORCHESTRATOR for THERMOSCOPE-AI (SIH26162).

Coordinates the pipeline between:
- firms_service.py      -> NASA FIRMS hotspot data
- osm_service.py        -> OpenStreetMap nearby geographic entities
- distance.py           -> Geodesic Haversine distance engine
- spatial_context.py    -> Distance calculation and candidate ranking
- schemas/hotspot.py    -> Hotspot data schema
- schemas/spatial_context.py -> SpatialContext schema
- schemas/analysis.py   -> HotspotAnalysis output schema
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Set stdout to UTF-8 if supported to prevent Windows cp1252 UnicodeEncodeError
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure backend package is discoverable
backend_dir = str(Path(__file__).resolve().parents[2])
root_dir = str(Path(__file__).resolve().parents[3])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.schemas.hotspot import Hotspot
from app.schemas.spatial_context import SpatialContext
from app.schemas.analysis import HotspotAnalysis
from app.geo.spatial_context import compute_geospatial_context
from app.services.firms_service import get_standardized_hotspots


def analyze_single_hotspot(
    hotspot: Union[Hotspot, Dict[str, Any]],
    radius_meters: int = 15000,
    use_live_api: bool = False
) -> HotspotAnalysis:
    """
    Orchestrates spatial analysis for a single Hotspot.
    
    Pipeline:
    1. Validates input as a standardized Hotspot schema object.
    2. Coordinates with spatial_context module to fetch OSM features and calculate distances.
    3. Safely extracts nearest distances (industry, forest, agriculture).
    4. Combines them into a standardized HotspotAnalysis response model.
    """
    # Step 1: Ensure input is a validated Hotspot schema object
    if isinstance(hotspot, Hotspot):
        hotspot_obj = hotspot
    elif isinstance(hotspot, dict):
        hotspot_obj = Hotspot.from_dict(hotspot)
    else:
        raise ValueError(f"Expected Hotspot instance or dict, got: {type(hotspot)}")

    # Step 2: Delegate geospatial context computation to spatial_context engine
    geo_context = compute_geospatial_context(
        hotspot=hotspot_obj,
        radius_meters=radius_meters,
        use_live_api=use_live_api
    )

    summary_distances = geo_context.get("summary_distances", {})

    # Step 3: Safely extract distance values, handling missing categories with None
    ind_dist = summary_distances.get("distance_to_industry_m")
    forest_dist = summary_distances.get("distance_to_forest_m")
    agri_dist = summary_distances.get("distance_to_agriculture_m")

    # Step 4: Construct the validated SpatialContext schema
    spatial_context = SpatialContext(
        nearest_industry_m=int(round(ind_dist)) if ind_dist is not None else None,
        nearest_forest_m=int(round(forest_dist)) if forest_dist is not None else None,
        nearest_agriculture_m=int(round(agri_dist)) if agri_dist is not None else None
    )

    # Step 5: Construct and return the combined HotspotAnalysis object
    return HotspotAnalysis(
        hotspot=hotspot_obj,
        spatial_context=spatial_context
    )


def batch_analyze_hotspots(
    hotspots: List[Union[Hotspot, Dict[str, Any]]],
    radius_meters: int = 15000,
    use_live_api: bool = False
) -> List[HotspotAnalysis]:
    """
    Orchestrates spatial analysis for a list of Hotspots.
    """
    results: List[HotspotAnalysis] = []
    for h in hotspots:
        try:
            analysis = analyze_single_hotspot(
                hotspot=h,
                radius_meters=radius_meters,
                use_live_api=use_live_api
            )
            results.append(analysis)
        except Exception as err:
            print(f"[Warning] Failed to analyze hotspot {h}: {err}")
            continue
    return results


def get_live_or_cached_hotspot_analyses(
    days: int = 3,
    limit: Optional[int] = 5,
    radius_meters: int = 15000,
    use_live_api: bool = False
) -> List[HotspotAnalysis]:
    """
    Full End-to-End Flow:
    1. Retrieves real NASA FIRMS hotspots via firms_service.
    2. Runs spatial analysis on the requested number of hotspots.
    3. Returns a list of standardized HotspotAnalysis objects.
    """
    raw_hotspots = get_standardized_hotspots(days=days)
    if limit is not None and limit > 0:
        raw_hotspots = raw_hotspots[:limit]
    return batch_analyze_hotspots(raw_hotspots, radius_meters=radius_meters, use_live_api=use_live_api)


if __name__ == "__main__":
    import json

    print("=" * 68)
    print("THERMOSCOPE-AI: HOTSPOT SERVICE ORCHESTRATOR TEST RUN")
    print("=" * 68)

    # Sample Input: Standardized Hotspot Object
    sample_hotspot = Hotspot(
        latitude=30.3165,
        longitude=78.0322,
        frp=42.5,
        brightness=325.4,
        confidence="high",
        acq_date="2026-09-01"
    )

    print("\n[Input Hotspot]:")
    print(json.dumps(sample_hotspot.model_dump(), indent=2))

    print("\nOrchestrating analysis pipeline...")
    result: HotspotAnalysis = analyze_single_hotspot(sample_hotspot, use_live_api=False)

    print("\n[Orchestrated HotspotAnalysis Output]:")
    print(json.dumps(result.model_dump(), indent=2))
    print("=" * 68)
