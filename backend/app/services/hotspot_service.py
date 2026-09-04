"""
hotspot_service.py
ORCHESTRATOR for THERMOSCOPE-AI (SIH26162) — Phase D.

Coordinates the full pipeline:
- firms_service.py      -> NASA FIRMS hotspot data
- osm_service.py        -> OpenStreetMap nearby geographic entities
- distance.py           -> Geodesic Haversine distance engine
- spatial_context.py    -> Distance calculation and candidate ranking
- intelligence/labeling_functions.py -> 14 weak-supervision LFs
- intelligence/label_aggregator.py   -> Vote aggregation
- intelligence/hybrid_engine.py     -> Phase C: LFs + XGBoost fusion
- ml/predict.py         -> XGBoost probability predictions
- ml/models/*.joblib    -> Trained model artifacts
- schemas/hotspot.py    -> Hotspot data schema
- schemas/spatial_context.py -> SpatialContext schema (7 distance fields)
- schemas/analysis.py   -> HotspotAnalysis output schema
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Set stdout to UTF-8 if supported
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
from app.intelligence.hybrid_engine import classify_hotspot, HybridEngine


# Module-level singleton
_engine: Optional[HybridEngine] = None


def _get_engine() -> HybridEngine:
    """Lazy-init the hybrid engine."""
    global _engine
    if _engine is None:
        _engine = HybridEngine()
    return _engine


def _build_feature_record(
    hotspot_obj: Hotspot,
    dist_industry_m: Optional[float],
    dist_refinery_m: Optional[float],
    dist_oil_gas_m: Optional[float],
    dist_mining_m: Optional[float],
    dist_agriculture_m: Optional[float],
    dist_forest_m: Optional[float],
    dist_power_plant_m: Optional[float],
) -> Dict[str, Any]:
    """Build the flat feature dict the hybrid engine expects.

    Distances are in METERS. The hybrid engine's get_distance_meters
    helper handles both meters and km-alias columns, but we feed it
    the canonical meter fields plus km aliases for robustness.
    """
    h = hotspot_obj.model_dump() if hasattr(hotspot_obj, "model_dump") else hotspot_obj.dict()

    def _m_to_km(v):
        if v is None:
            return None
        return float(v) / 1000.0

    return {
        # FIRMS thermal / metadata
        "frp": h.get("frp"),
        "bright_ti4": h.get("bright_ti4") or h.get("brightness"),
        "bright_ti5": h.get("bright_ti5"),
        "confidence": h.get("confidence"),
        "daynight": h.get("daynight"),
        "acq_date": h.get("acq_date"),
        "acq_time": h.get("acq_time"),
        "satellite": h.get("satellite"),
        # Canonical meter fields (LFs read these first)
        "distance_to_industry_m": dist_industry_m,
        "distance_to_refinery_m": dist_refinery_m,
        "distance_to_oil_gas_m": dist_oil_gas_m,
        "distance_to_mining_m": dist_mining_m,
        "distance_to_agriculture_m": dist_agriculture_m,
        "distance_to_forest_m": dist_forest_m,
        "distance_to_power_plant_m": dist_power_plant_m,
        # km aliases (LFs' fallback path: dist_* read as km and x1000)
        "dist_industry": _m_to_km(dist_industry_m),
        "dist_factory": _m_to_km(dist_industry_m),
        "dist_industrial_zone": _m_to_km(dist_industry_m),
        "dist_refinery": _m_to_km(dist_refinery_m),
        "dist_oil_gas": _m_to_km(dist_oil_gas_m),
        "dist_mining": _m_to_km(dist_mining_m),
        "dist_agriculture": _m_to_km(dist_agriculture_m),
        "dist_forest": _m_to_km(dist_forest_m),
        "dist_powerplant": _m_to_km(dist_power_plant_m),
        "dist_power_plant": _m_to_km(dist_power_plant_m),
        # Flags + counts (from CSV or spatial context)
        "has_industrial_2km": int(dist_industry_m is not None and dist_industry_m <= 2000),
        "has_factory_5km": int(dist_industry_m is not None and dist_industry_m <= 5000),
        "has_refinery_5km": int(dist_refinery_m is not None and dist_refinery_m <= 5000),
        "has_powerplant_5km": int(dist_power_plant_m is not None and dist_power_plant_m <= 5000),
        "has_forest_5km": int(dist_forest_m is not None and dist_forest_m <= 5000),
        "has_agriculture_5km": int(dist_agriculture_m is not None and dist_agriculture_m <= 5000),
        "count_ind_5km": 0,  # populated by spatial_context if available
        "count_ref_5km": 0,
        "count_forest_5km": 0,
        "count_agriculture_5km": 0,
    }


def analyze_single_hotspot(
    hotspot: Union[Hotspot, Dict[str, Any]],
    radius_meters: int = 15000,
    use_live_api: bool = False,
    run_classification: bool = True,
) -> HotspotAnalysis:
    """
    Orchestrates spatial analysis + hybrid intelligence classification for a single Hotspot.

    Pipeline (Phase D):
    1. Validate input as a standardized Hotspot schema object.
    2. Compute geospatial context: fetch OSM features, calculate distances to 7 categories.
    3. Build the flat feature record (canonical meter fields + km aliases).
    4. Run the Hybrid Intelligence Engine (14 LFs + XGBoost ML, 5-case fusion).
    5. Combine everything into a standardized HotspotAnalysis response.

    Args:
        hotspot: Hotspot Pydantic model or dict with FIRMS fields.
        radius_meters: OSM search radius (default 15 km).
        use_live_api: If True, query Overpass live; else use cached OSM data.
        run_classification: If True, run hybrid engine; else skip and return spatial-only.

    Returns:
        HotspotAnalysis with hotspot, spatial_context, and (optional) classification.
    """
    # Step 1: Validate input
    if isinstance(hotspot, Hotspot):
        hotspot_obj = hotspot
    elif isinstance(hotspot, dict):
        hotspot_obj = Hotspot.from_dict(hotspot)
    else:
        raise ValueError(f"Expected Hotspot instance or dict, got: {type(hotspot)}")

    # Step 2: Compute geospatial context (OSM lookup + distance calculation)
    geo_context = compute_geospatial_context(
        hotspot=hotspot_obj,
        radius_meters=radius_meters,
        use_live_api=use_live_api
    )
    summary_distances = geo_context.get("summary_distances", {})

    # Step 3: Extract 7 distances (in meters) from spatial context
    ind_m = summary_distances.get("distance_to_industry_m")
    ref_m = summary_distances.get("distance_to_refinery_m")
    oil_gas_m = summary_distances.get("distance_to_oil_gas_m")
    mining_m = summary_distances.get("distance_to_mining_m")
    agri_m = summary_distances.get("distance_to_agriculture_m")
    forest_m = summary_distances.get("distance_to_forest_m")
    power_m = summary_distances.get("distance_to_power_plant_m")

    # Build SpatialContext schema
    spatial_ctx = SpatialContext(
        nearest_industry_m=int(round(ind_m)) if ind_m is not None else None,
        nearest_refinery_m=int(round(ref_m)) if ref_m is not None else None,
        nearest_oil_gas_m=int(round(oil_gas_m)) if oil_gas_m is not None else None,
        nearest_mining_m=int(round(mining_m)) if mining_m is not None else None,
        nearest_agriculture_m=int(round(agri_m)) if agri_m is not None else None,
        nearest_forest_m=int(round(forest_m)) if forest_m is not None else None,
        nearest_power_plant_m=int(round(power_m)) if power_m is not None else None,
    )

    # Step 4: Run hybrid classification (Phase C)
    classification_decision = None
    if run_classification:
        feature_record = _build_feature_record(
            hotspot_obj, ind_m, ref_m, oil_gas_m, mining_m, agri_m, forest_m, power_m
        )
        engine = _get_engine()
        classification_decision = engine.classify(feature_record)

    # Step 5: Compose final analysis
    return HotspotAnalysis(
        hotspot=hotspot_obj,
        spatial_context=spatial_ctx,
        classification=classification_decision,
    )


def get_hotspots_with_classification(
    days: int = 3,
    radius_meters: int = 15000,
    use_live_api: bool = False,
    limit: Optional[int] = None,
) -> List[HotspotAnalysis]:
    """
    Fetch recent FIRMS hotspots and run full analysis pipeline on each.

    Args:
        days: Number of days back to fetch from FIRMS.
        radius_meters: OSM search radius.
        use_live_api: If True, query Overpass live.
        limit: Max number of hotspots to analyze (None = all).

    Returns:
        List of HotspotAnalysis (one per hotspot).
    """
    hotspots = get_standardized_hotspots(days=days)
    if limit is not None:
        hotspots = hotspots[:limit]

    results = []
    for i, hs in enumerate(hotspots):
        try:
            analysis = analyze_single_hotspot(
                hotspot=hs,
                radius_meters=radius_meters,
                use_live_api=use_live_api,
                run_classification=True,
            )
            results.append(analysis)
        except Exception as e:
            print(f"  [warn] hotspot #{i} failed: {e}")
            continue

    return results


if __name__ == "__main__":
    import json

    print("=" * 70)
    print("THERMOSCOPE-AI: Hotspot Service — Phase D End-to-End Demo")
    print("=" * 70)

    # Demo: run on a synthetic hotspot near industry (Gujarat, India)
    print("\n[Demo] Analyzing a hotspot near industrial area (Gujarat)...")

    demo_hotspot = {
        "latitude": 21.1051,
        "longitude": 72.6438,
        "frp": 5.9,
        "bright_ti4": 330.8,
        "bright_ti5": 300.0,
        "confidence": "n",
        "daynight": "D",
        "acq_date": "2026-09-01",
        "satellite": "N",
    }

    analysis = analyze_single_hotspot(
        hotspot=demo_hotspot,
        radius_meters=15000,
        use_live_api=False,
        run_classification=True,
    )

    print(f"\n  Hotspot: ({analysis.hotspot.latitude}, {analysis.hotspot.longitude})")
    print(f"  FRP: {analysis.hotspot.frp} MW | Brightness: {analysis.hotspot.brightness} K")
    print(f"  Confidence: {analysis.hotspot.confidence}")

    print(f"\n  Spatial Context:")
    sc = analysis.spatial_context
    print(f"    Industry:   {sc.nearest_industry_m}m" if sc.nearest_industry_m else "    Industry:   N/A")
    print(f"    Refinery:   {sc.nearest_refinery_m}m" if sc.nearest_refinery_m else "    Refinery:   N/A")
    print(f"    Oil/Gas:    {sc.nearest_oil_gas_m}m" if sc.nearest_oil_gas_m else "    Oil/Gas:    N/A")
    print(f"    Mining:     {sc.nearest_mining_m}m" if sc.nearest_mining_m else "    Mining:     N/A")
    print(f"    Forest:     {sc.nearest_forest_m}m" if sc.nearest_forest_m else "    Forest:     N/A")
    print(f"    Agriculture:{sc.nearest_agriculture_m}m" if sc.nearest_agriculture_m else "    Agriculture:N/A")
    print(f"    Power plant:{sc.nearest_power_plant_m}m" if sc.nearest_power_plant_m else "    Power plant:N/A")

    if analysis.classification:
        cls = analysis.classification
        print(f"\n  Classification:")
        print(f"    Final label:      {cls['final_label']}")
        print(f"    Confidence:       {cls['hybrid_confidence']:.3f} ({cls['confidence_level']})")
        print(f"    Decision source:  {cls['decision_source']}")
        print(f"    Agreement:        {cls['agreement']}, Conflict: {cls['conflict']}")
        print(f"    Requires review:  {cls['requires_human_review']}")
        if cls.get("review_reason"):
            print(f"    Review reason:    {cls['review_reason']}")
        print(f"\n    Explanation:")
        for line in cls.get("explanation", []):
            print(f"      - {line}")

    print("\n" + "=" * 70)
    print("Phase D orchestration complete.")
    print("=" * 70)
