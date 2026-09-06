"""
recommendation_service.py
Deterministic recommendation engine for THERMOSCOPE-AI (SIH26162).

Generates presentation-layer recommendations from an existing hybrid
classification decision. CRITICAL ARCHITECTURE RULE: this module MUST NOT
reclassify, change confidence, alter thresholds, or modify any label. It only
*translates* an already-final decision (final_label, confidence, risk level,
review requirement) into human-readable recommended actions for the response.

Risk levels are derived READ-ONLY from the final label + hybrid confidence.
"""
from typing import Any, Dict, List

from app.schemas.analysis_response import Recommendation


def risk_level_for(final_label: str, confidence: float) -> str:
    """
    Derive an overall risk tier from the FINAL classification.

    This is a presentation classifier over already-final output — it never
    feeds back into the hybrid engine. Pure function over (label, confidence).
    """
    high_risk_labels = {"forest_natural_fire", "industrial_fire", "industrial_process_heat"}
    medium_risk_labels = {"gas_flare", "mining_activity"}

    if final_label in high_risk_labels:
        base = "high"
    elif final_label in medium_risk_labels:
        base = "medium"
    elif final_label == "agricultural_burn":
        base = "medium"
    else:  # unclassified and any unknown
        base = "low"

    # Raise tier on low confidence / unresolved cases (read-only provenance).
    if confidence < 0.60 or final_label == "unclassified":
        # unclassified is inherently a review situation.
        return "critical" if final_label == "unclassified" else base
    return base


def _fire_class_recommendations(final_label: str, confidence: float) -> List[Recommendation]:
    if final_label in ("industrial_fire", "industrial_process_heat"):
        return [
            Recommendation(
                priority="critical",
                action="Alert relevant industrial safety and fire authorities; inspect proximate factories/refineries.",
            ),
            Recommendation(
                priority="high",
                action="Verify the event against recent high-resolution satellite observations.",
            ),
            Recommendation(
                priority="medium",
                action="Monitor neighbouring industrial infrastructure for heat persistence over the next detection cycle.",
            ),
        ]
    if final_label == "forest_natural_fire":
        return [
            Recommendation(
                priority="critical",
                action="Notify forest and disaster-management authorities for immediate response.",
            ),
            Recommendation(
                priority="high",
                action="Issue an early-warning advisory for nearby populated areas and park infrastructure.",
            ),
            Recommendation(
                priority="medium",
                action="Track the fire perimeter across subsequent FIRMS detections to estimate spread.",
            ),
        ]
    if final_label == "gas_flare":
        return [
            Recommendation(
                priority="high",
                action="Contact the responsible oil/gas operator to confirm the flaring source and status.",
            ),
            Recommendation(
                priority="medium",
                action="Compare against persistent flare signatures to distinguish routine flaring from an incident.",
            ),
            Recommendation(
                priority="low",
                action="Log as routine energy-sector activity unless the signature intensifies over time.",
            ),
        ]
    if final_label == "mining_activity":
        return [
            Recommendation(
                priority="medium",
                action="Inform mining-oversight authorities of a thermal signature at an extraction site.",
            ),
            Recommendation(
                priority="medium",
                action="Check for deliberate blasting or waste-burn activity before escalation.",
            ),
        ]
    if final_label == "agricultural_burn":
        return [
            Recommendation(
                priority="medium",
                action="Flag to state agriculture / pollution-control authorities for stubble-burn advisories.",
            ),
            Recommendation(
                priority="low",
                action="Monitor for seasonal clustering that indicates a regional burning campaign.",
            ),
        ]
    return []


def _unclassified_recommendations(confidence: float) -> List[Recommendation]:
    return [
        Recommendation(
            priority="critical",
            action="Route to a human analyst for visual verification; evidence is insufficient for automation.",
        ),
        Recommendation(
            priority="high",
            action="Pull the geohistory and adjacent FIRMS detections to determine the thermal source.",
        ),
        Recommendation(
            priority="medium",
            action="Enrich with fresher OSM/land-use context and re-run the analysis.",
        ),
    ]


def generate_recommendations(decision: Dict[str, Any]) -> List[Recommendation]:
    """
    Map a final hybrid decision into prioritized actions.

    Policy (Phase F) keys on `classification_status`:
      - final_label == 'unclassified'      -> analyst-verification action set.
      - classification_status == "probable"  -> act, but verify before escalation.
      - classification_status == "confirmed" -> class-specific actions (no
        forced verification beyond `requires_human_review`).
    A generic operator-review reminder is appended ONLY when the engine
    explicitly requires human review AND the label is decided (the unclassified
    set already contains verification actions — no duplication).

    Args:
        decision: the hybrid engine output dict (final_label, hybrid_confidence,
                  confidence_level, decision_source, classification_status,
                  requires_human_review, ...).

    Returns:
        List[Recommendation], ordered most-important first.
    """
    final_label = str(decision.get("final_label", "unclassified"))
    confidence = float(decision.get("hybrid_confidence", 0.0))
    status = str(decision.get("classification_status", ""))
    if status not in ("confirmed", "probable", "uncertain"):
        status = "uncertain" if final_label == "unclassified" else "confirmed"

    if final_label == "unclassified":
        return _unclassified_recommendations(confidence)

    recs: List[Recommendation] = []
    if status == "probable":
        recs.append(
            Recommendation(
                priority="high",
                action="Probable (single-source/contested): verify the thermal "
                       "source with fresh imagery or operator checks before "
                       "high-stakes escalation.",
            )
        )

    recs.extend(_fire_class_recommendations(final_label, confidence))

    # Surface a review reminder when the engine requires operator review — but
    # not on abstention, where the unclassified action set already covers it.
    if decision.get("requires_human_review"):
        recs.append(
            Recommendation(
                priority="high",
                action="Confirm the classification with a human operator before downstream action.",
            )
        )

    return recs