"""
analysis_response.py
Presentation-ready final output contract for THERMOSCOPE-AI (SIH26162).

This is the FRONTEND contract. It is a thin, presentation-layer projection of
the internal `HotspotAnalysis` + hybrid classification decision. All
classification logic lives downstream in app/services/hotspot_service.py and
app/intelligence/* — nothing here recomputes features, thresholds, or labels.

Fields map to the MVP response requirements:
    class                 -> classification.kind (canonical label)
    confidence_score      -> classification.confidence_score
    why_this_class        -> `explanation` (human-readable evidence bullets)
    recommendations       -> `recommendations` (priority + action)
    requires_human_review -> classification.requires_human_review
"""
from typing import List

from pydantic import BaseModel, Field

from app.schemas.hotspot import Hotspot
from app.schemas.spatial_context import SpatialContext


class HotspotEcho(BaseModel):
    """A clean projection of the input hotspot for the response envelope."""

    latitude: float = Field(..., description="Latitude in decimal degrees")
    longitude: float = Field(..., description="Longitude in decimal degrees")
    frp: float = Field(..., description="Fire Radiative Power in MW")
    brightness: float = Field(..., description="Brightness temperature in Kelvin")
    confidence: str = Field(..., description="Detection confidence: high/nominal/low")


class ClassificationSummary(BaseModel):
    """The final, frontend-facing classification envelope — honest semantics.

    Semantics contract (enforced in hotspot_service._to_presentation_response):

    - ``confidence_score`` is confidence in the FINAL CLASS DECISION only.
      It is ``None`` when the system abstained (``kind == "unclassified"``).
    - ``ml_confidence`` is the raw XGBoost probability of ``ml_top_prediction``.
      When ML itself abstains, ``ml_top_prediction == "unclassified"`` and the
      probability refers to that abstention output — it is NOT confidence in
      any real class.
    - ``confidence_level`` is ``"insufficient_evidence"`` when the final
      decision abstained; otherwise it mirrors the hybrid engine tier
      (low/medium/high).
    - ``classification_status`` ∈ {confirmed, probable, uncertain} is the
      final decision status from the hybrid engine — distinct from model
      probability (``ml_confidence``) and from ``decision_status``, which also
      reflects the operator-review requirement.
    - ``decision_status`` ∈ {confirmed, probable, uncertain, requires_review}
      and is derived from the final class + confidence + review requirement.
    """

    kind: str = Field(..., description="Canonical classification label (e.g. industrial_fire)")
    display_name: str = Field(..., description="Human-readable class display name")
    color: str = Field(..., description="Hex colour for map/UI visualisation")
    confidence_score: float | None = Field(
        ...,
        description="Confidence in the FINAL class decision in [0,1]; None when the "
        "system abstained (kind == 'unclassified').",
    )
    confidence_level: str = Field(
        ...,
        description="low / medium / high tier, or 'insufficient_evidence' on abstention",
    )
    risk_level: str = Field(..., description="Overall risk tier (critical/high/medium/low)")
    decision_source: str = Field(
        ...,
        description="Source of the decision (hybrid_agreement/ml_assisted/rule_dominant/ml_dominant/conflict/uncertain)",
    )
    decision_status: str = Field(
        ...,
        description="confirmed / probable / uncertain / requires_review (status + review requirement)",
    )
    classification_status: str = Field(
        ...,
        description="Final decision status: confirmed / probable / uncertain (engine classification_status)",
    )
    rule_prediction: str | None = Field(
        None, description="Rule-engine consensus label ('unclassified' when rules abstain)"
    )
    rule_vote_strength: int | None = Field(
        None, description="Number of active labeling functions behind the rule consensus"
    )
    ml_top_prediction: str | None = Field(
        ...,
        description="Raw XGBoost top class (may be 'unclassified' when ML abstains)",
    )
    ml_confidence: float | None = Field(
        ...,
        description="Raw XGBoost probability of ml_top_prediction; never implies "
        "confidence in the final class when the engine abstains.",
    )
    requires_human_review: bool = Field(
        ..., description="True when an operator should verify this classification"
    )
    review_reason: str | None = Field(
        None, description="Human-readable reason for review, when required"
    )


class Recommendation(BaseModel):
    """A single actionable recommendation with a priority tier."""

    priority: str = Field(..., description="critical / high / medium / low")
    action: str = Field(..., description="Human-readable recommended action")


class AnalysisResponse(BaseModel):
    """Frontend-ready output for a single analysed hotspot."""

    status: str = Field("success", description="success / error")
    analysis_type: str = Field(
        "presentation", description="Fixed marker: this is the frontend contract"
    )
    hotspot: HotspotEcho
    spatial_context: SpatialContext
    classification: ClassificationSummary
    why_this_class: List[str] = Field(
        ..., description="Human-readable evidence bullets explaining the decision"
    )
    recommendations: List[Recommendation] = Field(
        ..., description="Prioritised recommended actions"
    )