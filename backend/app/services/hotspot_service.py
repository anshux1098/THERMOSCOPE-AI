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
from app.schemas.analysis_response import (
    AnalysisResponse,
    ClassificationSummary,
    HotspotEcho,
    Recommendation,
)
from app.core.constants import CLASS_LABEL_DISPLAY, CLASS_COLORS
from app.geo.spatial_context import compute_geospatial_context
from app.geo.spatial_features import CATEGORIES, build_candidates_by_category, compute_spatial_features
from app.services.firms_service import get_standardized_hotspots
from app.services.recommendation_service import generate_recommendations, risk_level_for
from app.intelligence.hybrid_engine import (
    HIGH_CONFIDENCE,
    MODERATE_CONFIDENCE,
    classify_hotspot,
    HybridEngine,
)


# Module-level singleton
_engine: Optional[HybridEngine] = None


def _get_engine() -> HybridEngine:
    """Lazy-init the hybrid engine."""
    global _engine
    if _engine is None:
        _engine = HybridEngine()
    return _engine


# ---------------------------------------------------------------------------
# Shared feature contract (Phase B P1.3)
# ---------------------------------------------------------------------------
def _spatial_features_from_geo_context(geo_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the shared spatial feature block from compute_geospatial_context output.

    Batch producer (scripts/build_real_dataset.py), the live hotspot service
    (here) and run_pipeline.cpp all funnel through compute_spatial_features so
    their feature math is identical. This mirrors build_real_dataset.
    """
    candidates_by_category = {}
    categories = geo_context.get("categories") or geo_context.get("osm_objects") or {}
    if isinstance(categories, dict):
        for c in CATEGORIES:
            bucket = categories.get(c) or {}
            candidates_by_category[c] = (bucket.get("candidates") or []) if isinstance(bucket, dict) else list(bucket)
    elif isinstance(categories, list):
        # Legacy: flat list of OSM objects grouped generically.
        for c in CATEGORIES:
            candidates_by_category[c] = [
                s for s in categories if s.get("category") == c or s.get("site_type") == c
            ]

    lat = float(geo_context.get("latitude") or 0.0)
    lon = float(geo_context.get("longitude") or 0.0)
    data_sources = geo_context.get("data_sources") or {}
    radius = geo_context.get("radius_meters") or 15000
    return compute_spatial_features(lat, lon, candidates_by_category, data_sources=data_sources, radius_m=radius)


def _build_feature_record(
    spot_dict: Dict[str, Any],
    feats: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the flat feature dict the hybrid engine expects.

    All spatial values come from the SHARED canonical feature contract
    (compute_spatial_features) so this matches the batch producer exactly
    (Phase B P1.3 batch==live parity).
    """
    return {
        # FIRMS thermal / metadata
        "frp": spot_dict.get("frp"),
        "bright_ti4": spot_dict.get("bright_ti4") or spot_dict.get("brightness"),
        "bright_ti5": spot_dict.get("bright_ti5"),
        "confidence": spot_dict.get("confidence"),
        "daynight": spot_dict.get("daynight"),
        "acq_date": spot_dict.get("acq_date"),
        "acq_time": spot_dict.get("acq_time"),
        "satellite": spot_dict.get("satellite"),
        "firms_type": spot_dict.get("firms_type"),
        # Canonical meter fields (LFs read these first)
        "distance_to_industry_m": feats["distance_to_industry_m"],
        "distance_to_refinery_m": feats["distance_to_refinery_m"],
        "distance_to_oil_gas_m": feats["distance_to_oil_gas_m"],
        "distance_to_mining_m": feats["distance_to_mining_m"],
        "distance_to_agriculture_m": feats["distance_to_agriculture_m"],
        "distance_to_forest_m": feats["distance_to_forest_m"],
        "distance_to_power_plant_m": feats["distance_to_power_plant_m"],
        # km aliases (LFs' fallback path)
        "dist_industry": feats["dist_industry"],
        "dist_factory": feats["dist_factory"],
        "dist_industrial_zone": feats["dist_industrial_zone"],
        "dist_refinery": feats["dist_refinery"],
        "dist_oil_gas": feats["dist_oil_gas"],
        "dist_mining": feats["dist_mining"],
        "dist_agriculture": feats["dist_agriculture"],
        "dist_forest": feats["dist_forest"],
        "dist_powerplant": feats["dist_powerplant"],
        # Flags + REAL neighbour counts (Phase B P1.5)
        "has_industrial_2km": feats["has_industrial_2km"],
        "has_factory_5km": feats["has_factory_5km"],
        "has_refinery_5km": feats["has_refinery_5km"],
        "has_powerplant_5km": feats["has_powerplant_5km"],
        "has_forest_5km": feats["has_forest_5km"],
        "has_agriculture_5km": feats["has_agriculture_5km"],
        "industrial_sites_within_2km": feats["industrial_sites_within_2km"],
        "industrial_sites_within_5km": feats["industrial_sites_within_5km"],
        "refinery_sites_within_3km": feats["refinery_sites_within_3km"],
        "refinery_sites_within_5km": feats["refinery_sites_within_5km"],
        "forest_sites_within_5km": feats["forest_sites_within_5km"],
        "agriculture_sites_within_5km": feats["agriculture_sites_within_5km"],
        "count_forest_5km": feats["count_forest_5km"],
        "count_agriculture_5km": feats["count_agriculture_5km"],
        # Legacy ML-schema count aliases (DEPRECATED, kept byte-compatible)
        "count_ind_5km": feats["count_ind_5km"],
        "count_ref_5km": feats["count_ref_5km"],
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
        spot_dict = hotspot_obj.model_dump() if hasattr(hotspot_obj, "model_dump") else hotspot_obj.dict()
        feats = _spatial_features_from_geo_context(geo_context)
        feature_record = _build_feature_record(spot_dict, feats)
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


# ---------------------------------------------------------------------------
# Presentation-ready output (frontend contract)
# ---------------------------------------------------------------------------

def _to_presentation_response(
    analysis: HotspotAnalysis,
) -> AnalysisResponse:
    """
    Project an internal HotspotAnalysis + hybrid decision into the frontend
    AnalysisResponse contract.

    Pure translation: reuses the existing geo context and hybrid decision.
    No reclassification, no new thresholds, no synthetic data.

    CONFIDENCE SEMANTICS (honest): the final `confidence_score` refers ONLY to
    the final class decision. When the engine abstains (`kind == 'unclassified'`)
    the score is None and `confidence_level` is 'insufficient_evidence'. The raw
    ML probability is exposed separately as `ml_confidence` alongside
    `ml_top_prediction` — never as confidence in the final class.
    """

    hotspot = analysis.hotspot
    sc = analysis.spatial_context
    decision = analysis.classification or {}

    final_label = str(decision.get("final_label", "unclassified"))
    decided = final_label != "unclassified"
    engine_confidence = float(decision.get("hybrid_confidence", 0.0))
    engine_tier = str(decision.get("confidence_level", "low"))
    decision_source = str(decision.get("decision_source", "uncertain"))
    requires_review = bool(decision.get("requires_human_review", False))
    review_reason = decision.get("review_reason")

    # classification_status is the engine's final decision status. When a legacy
    # / unit decision dict omits it (e.g. explicit fixture decisions), derive it
    # from the decision + confidence rather than blind-defaulting to uncertain.
    engine_status = decision.get("classification_status")
    if engine_status in ("confirmed", "probable", "uncertain"):
        classification_status = str(engine_status)
    elif not decided:
        classification_status = "uncertain"
    elif engine_confidence >= HIGH_CONFIDENCE:
        classification_status = "confirmed"
    elif engine_confidence >= MODERATE_CONFIDENCE:
        classification_status = "probable"
    else:
        classification_status = "uncertain"

    rule_engine = decision.get("rule_engine") or {}
    rule_prediction = rule_engine.get("prediction")
    rule_vote_strength = rule_engine.get("active_votes")

    ml_engine = decision.get("ml_engine") or {}
    ml_top_prediction = ml_engine.get("prediction")
    ml_confidence = ml_engine.get("confidence")

    display_name = CLASS_LABEL_DISPLAY.get(final_label, final_label)
    color = CLASS_COLORS.get(final_label, "#808080")
    risk = risk_level_for(final_label, engine_confidence)

    if decided:
        confidence_score = round(engine_confidence, 4)
        confidence_level = engine_tier
        if requires_review:
            decision_status = "requires_review"
        else:
            decision_status = classification_status
    else:
        # Abstention: there is no class decision and no final-class confidence.
        confidence_score = None
        confidence_level = "insufficient_evidence"
        decision_status = "uncertain"

    why_this_class: List[str] = list(decision.get("explanation", []))
    if not why_this_class:
        why_this_class = [
            f"Classification '{final_label}' with hybrid confidence "
            f"{engine_confidence:.2f} (source: {decision_source})."
        ]

    recommendations = generate_recommendations(decision)

    echo = HotspotEcho(
        latitude=float(hotspot.latitude),
        longitude=float(hotspot.longitude),
        frp=float(hotspot.frp),
        brightness=float(hotspot.brightness),
        confidence=hotspot.confidence,
    )

    summary = ClassificationSummary(
        kind=final_label,
        display_name=display_name,
        color=color,
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        risk_level=risk,
        decision_source=decision_source,
        decision_status=decision_status,
        classification_status=classification_status,
        rule_prediction=rule_prediction,
        rule_vote_strength=rule_vote_strength,
        ml_top_prediction=ml_top_prediction,
        ml_confidence=ml_confidence,
        requires_human_review=requires_review,
        review_reason=review_reason,
    )

    return AnalysisResponse(
        status="success",
        analysis_type="presentation",
        hotspot=echo,
        spatial_context=SpatialContext(
            nearest_industry_m=sc.nearest_industry_m,
            nearest_refinery_m=sc.nearest_refinery_m,
            nearest_oil_gas_m=sc.nearest_oil_gas_m,
            nearest_mining_m=sc.nearest_mining_m,
            nearest_agriculture_m=sc.nearest_agriculture_m,
            nearest_forest_m=sc.nearest_forest_m,
            nearest_power_plant_m=sc.nearest_power_plant_m,
        ),
        classification=summary,
        why_this_class=why_this_class,
        recommendations=recommendations,
    )


def analyze_hotspot_presentation(
    hotspot: Union[Hotspot, Dict[str, Any]],
    radius_meters: int = 15000,
    use_live_api: bool = False,
) -> AnalysisResponse:
    """
    Production orchestrator entry point for the frontend.

    Runs the full pipeline once (via analyze_single_hotspot) and returns the
    presentation-ready AnalysisResponse contract:

        status, hotspot, spatial_context,
        classification{ kind, display_name, confidence_score, risk_level,
                        decision_source, requires_human_review },
        why_this_class[], recommendations[]

    FastAPI stays thin: `result = analyze_hotspot_presentation(request)`.
    """
    analysis = analyze_single_hotspot(
        hotspot=hotspot,
        radius_meters=radius_meters,
        use_live_api=use_live_api,
        run_classification=True,
    )
    return _to_presentation_response(analysis)


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
