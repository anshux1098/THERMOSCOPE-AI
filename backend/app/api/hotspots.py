"""
api/hotspots.py — HTTP adapter over the end-to-end hotspot analysis service.

Thin layer only: all classification logic stays in
app/services/hotspot_service.py -> app/intelligence/*. No thresholds, label
functions, or model parameters live here.
"""
from fastapi import APIRouter

from app.schemas.hotspot import Hotspot
from app.schemas.analysis import HotspotAnalysis
from app.schemas.analysis_response import AnalysisResponse
from app.services.hotspot_service import analyze_single_hotspot, analyze_hotspot_presentation

router = APIRouter(prefix="/hotspots", tags=["hotspots"])


@router.post(
    "/analyze",
    response_model=HotspotAnalysis,
    summary="Analyze a single thermal hotspot",
)
def analyze_hotspot(
    hotspot: Hotspot,
    radius_meters: int = 15000,
    use_live_api: bool = False,
) -> HotspotAnalysis:
    """
    Run the full production pipeline for one NASA FIRMS hotspot:

    1. OSM spatial context (cached or live)
    2. 14 labeling functions + weak-supervision aggregation
    3. XGBoost prediction
    4. Hybrid Rules x ML fusion

    Returns the MVP outputs: classification / class, hybrid confidence,
    decision source, agreement, conflict, human-review requirement, review
    reason, and explanation bullets (list under `classification.explanation`).
    """
    return analyze_single_hotspot(
        hotspot,
        radius_meters=radius_meters,
        use_live_api=use_live_api,
        run_classification=True,
    )


@router.post(
    "/analyze/presentation",
    response_model=AnalysisResponse,
    summary="Analyze a hotspot (frontend-ready presentation contract)",
)
def analyze_hotspot_presentation_endpoint(
    hotspot: Hotspot,
    radius_meters: int = 15000,
    use_live_api: bool = False,
) -> AnalysisResponse:
    """
    Same full production pipeline as /analyze, but returns the frontend-ready
    AnalysisResponse contract:

        classification{ kind, display_name, color, confidence_score,
                        confidence_level, risk_level, decision_source,
                        requires_human_review, review_reason }
        why_this_class[]   (human-readable evidence)
        recommendations[]  (priority + action)

    Thin adapter only: orchestrates via
    app/services/hotspot_service.analyze_hotspot_presentation.
    """
    return analyze_hotspot_presentation(
        hotspot,
        radius_meters=radius_meters,
        use_live_api=use_live_api,
    )