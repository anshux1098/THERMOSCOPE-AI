"""
hybrid_engine.py
Phase C — Hybrid Intelligence Fusion Engine for THERMOSCOPE-AI (SIH26162).

Combines rule-based weak supervision (14 labeling functions) with
machine learning (XGBoost) to produce calibrated classifications
with human-review flags and explainable evidence.

Fusion logic — deterministic decision matrix (ordered evaluation).
Each row yields a classification_status:
  A) Rules + ML agree                  -> CONFIRMED (hybrid_agreement)
  B) Rules abstain + strong ML         -> PROBABLE  (ml_assisted)
  C) Strong rules + weak ML            -> CONFIRMED (rule_dominant)
  D) Rules + ML conflict               -> UNCERTAIN (strong/close-tie conflict)
                                          or PROBABLE (contested strong single source)
  E) Both abstain / below threshold    -> UNCERTAIN

Semantics: `classification_status` (confirmed/probable/uncertain) is the final
decision status. `hybrid_confidence` is the confidence in that final decision
(0.0 for a non-decision). `raw_ml_confidence` is the raw XGBoost probability of
the ML top class and is NOT decision confidence. `rule_vote_strength` is the
active labeling-function count. These are never conflated.
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make app package discoverable
backend_dir = str(Path(__file__).resolve().parents[2])
root_dir = str(Path(__file__).resolve().parents[3])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.intelligence.label_aggregator import (
    apply_labeling_functions,
    aggregate_votes,
    compute_vote_summary,
    UNCLASSIFIED,
)
from app.ml.predict import predict_proba


# Confidence tier thresholds
HIGH_CONFIDENCE = 0.80
MODERATE_CONFIDENCE = 0.60
REVIEW_CONFIDENCE = 0.60

# ML vote-strength thresholds
ML_HIGH = 0.80
ML_MODERATE = 0.60
ML_SIGNAL_MIN = 0.45  # minimum ML probability for a real-class signal

# Rule strength thresholds
STRONG_RULE_VOTES = 2  # >= 2 active LFs = strong domain consensus

# --- Decision policy (audit-justified; see reports/HYBRID_DECISION_AUDIT.md) ---
# Uncontested (rules abstain) strong ML at/above this threshold -> PROBABLE.
# Justification: every real-class ML prediction on the canonical snapshot sits
# >= 0.60, and the existing high-confidence tier is 0.80. The Phase A audit
# found 0 eligible rows, so this is a forward-compatibility policy constant —
# it is NOT a lever to inflate the classification rate.
ML_PROBABLE_THRESHOLD = 0.80

# Decision confidence assigned to single-evidence-source PROBABLE outcomes.
# Deliberately moderate: model probability (routinely ~0.99+) is model output,
# not calibrated validated accuracy, so it is mapped to a bounded DECISION
# confidence rather than forwarded wholesale.
MODERATE_DECISION_CONFIDENCE = 0.70

# Case B (ml_assisted) outcomes below this ML probability require operator
# review even though they are auto-classified single-source results.
ML_ASSISTED_REVIEW_MIN_PROB = 0.90


def _confidence_tier(score: float) -> str:
    """Map numerical confidence to categorical tier."""
    if score >= HIGH_CONFIDENCE:
        return "high"
    elif score >= MODERATE_CONFIDENCE:
        return "medium"
    return "low"


def _add_feature_explanations(features: Dict[str, Any], explanation: List[str]) -> None:
    """Generate human-readable evidence statements from features."""
    # Thermal signature
    frp = features.get("frp")
    if frp is not None:
        try:
            frp_val = float(frp)
            if frp_val >= 15.0:
                explanation.append(f"High-intensity thermal signature (FRP: {frp_val:.1f} MW).")
            elif frp_val >= 5.0:
                explanation.append(f"Moderate thermal signature (FRP: {frp_val:.1f} MW).")
            else:
                explanation.append(f"Low-intensity thermal source (FRP: {frp_val:.1f} MW).")
        except (ValueError, TypeError):
            pass

    # Industrial proximity
    dist_factory = features.get("dist_factory") or features.get("dist_industry")
    if dist_factory is not None:
        try:
            d_km = float(dist_factory)
            if d_km <= 1.0:
                explanation.append(f"Industrial infrastructure within {d_km:.2f} km.")
            elif d_km <= 5.0:
                explanation.append(f"Industrial infrastructure within {d_km:.1f} km.")
        except (ValueError, TypeError):
            pass

    # Refinery proximity
    dist_ref = features.get("dist_refinery")
    if dist_ref is not None:
        try:
            d_km = float(dist_ref)
            if d_km <= 2.0:
                explanation.append(f"Petroleum/chemical refinery within {d_km:.1f} km.")
        except (ValueError, TypeError):
            pass

    # Mining proximity
    dist_min = features.get("dist_mining")
    if dist_min is not None:
        try:
            d_km = float(dist_min)
            if d_km <= 2.0:
                explanation.append(f"Mining/extraction site within {d_km:.1f} km.")
        except (ValueError, TypeError):
            pass

    # Forest proximity
    dist_forest = features.get("dist_forest")
    if dist_forest is not None:
        try:
            d_km = float(dist_forest)
            if d_km <= 5.0:
                explanation.append(f"Forest cover within {d_km:.1f} km.")
        except (ValueError, TypeError):
            pass

    # Agriculture proximity
    dist_agri = features.get("dist_agriculture")
    if dist_agri is not None:
        try:
            d_km = float(dist_agri)
            if d_km <= 5.0:
                explanation.append(f"Agricultural land within {d_km:.1f} km.")
        except (ValueError, TypeError):
            pass


def classify_hotspot(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run hybrid intelligence classification on a single hotspot.

    Args:
        record: Dict with hotspot features (frp, bright_ti4, dist_*, etc.)

    Returns:
        Dict with final_label, hybrid_confidence, decision_source,
        agreement, conflict, requires_human_review, explanation,
        rule_engine, ml_engine
    """
    # Step 1: Run rule-based labeling functions
    rule_votes_raw = apply_labeling_functions(record)
    vote_summary = compute_vote_summary(rule_votes_raw)

    rule_pred = str(vote_summary["consensus"])
    active_votes = int(vote_summary["active_votes_count"])
    vote_breakdown = {str(k): int(v) for k, v in vote_summary["vote_breakdown"].items()}
    active_lfs = {str(k): str(v) for k, v in vote_summary["active_lfs"].items()}

    # Step 2: Run ML classifier
    try:
        ml_result = predict_proba(record)
        ml_pred = str(ml_result.get("label", UNCLASSIFIED))
        ml_prob = float(ml_result.get("probability", 0.0))
        all_probs = {str(k): float(v) for k, v in ml_result.get("all_probabilities", {}).items()}
    except Exception as e:
        # ML inference failed — fall back to rules only
        ml_pred = UNCLASSIFIED
        ml_prob = 0.0
        all_probs = {}

    # Step 3: Fusion logic
    rule_has_signal = (active_votes > 0) and (rule_pred != UNCLASSIFIED)
    ml_has_signal = (ml_pred != UNCLASSIFIED) and (ml_prob >= ML_SIGNAL_MIN)

    explanation: List[str] = []
    requires_human_review = False
    review_reason: Optional[str] = None

    # Build feature context
    _add_feature_explanations(record, explanation)

    final_label = UNCLASSIFIED
    decision_source = "uncertain"
    classification_status = "uncertain"
    agreement = False
    conflict = False
    hybrid_confidence = 0.0

    if rule_has_signal and ml_has_signal and (rule_pred == ml_pred):
        # CASE A: Rules + ML agree -> CONFIRMED
        final_label = ml_pred
        decision_source = "hybrid_agreement"
        classification_status = "confirmed"
        agreement = True
        boost = min(0.12, 0.04 * active_votes)
        hybrid_confidence = min(0.99, ml_prob + boost)
        explanation.append(
            f"{active_votes} labeling function(s) voted '{rule_pred}' "
            f"({', '.join(active_lfs.keys())})."
        )
        explanation.append(
            f"XGBoost predicted '{ml_pred}' with {ml_prob * 100:.1f}% probability."
        )
        explanation.append(
            "Decision [hybrid_agreement]: independent rule evidence and ML agree "
            f"on '{final_label}' — status confirmed."
        )

    elif (not rule_has_signal) and ml_has_signal and (ml_prob >= ML_PROBABLE_THRESHOLD):
        # CASE B: Rules abstain + strong uncontested ML -> PROBABLE (ml_assisted)
        final_label = ml_pred
        decision_source = "ml_assisted"
        classification_status = "probable"
        hybrid_confidence = MODERATE_DECISION_CONFIDENCE
        explanation.append("Rule engine abstained (no independent spatial/heuristic evidence).")
        explanation.append(
            f"XGBoost classified the event as '{ml_pred}' at {ml_prob * 100:.1f}% "
            "model probability (not validated accuracy)."
        )
        explanation.append(
            "Decision [ml_assisted]: strong but single-source ML evidence; "
            "no independent spatial confirmation — status probable."
        )
        if ml_prob < ML_ASSISTED_REVIEW_MIN_PROB:
            requires_human_review = True
            review_reason = (
                "ML-only classification below the no-review threshold "
                f"({ML_ASSISTED_REVIEW_MIN_PROB * 100:.0f}% ML probability)."
            )
            explanation.append(
                "Single evidence source below the no-review threshold — "
                "operator confirmation recommended."
            )

    elif rule_has_signal and (active_votes >= STRONG_RULE_VOTES) and (ml_prob < ML_MODERATE or ml_pred == UNCLASSIFIED):
        # CASE C: Strong rules + weak/absent ML -> CONFIRMED via rule consensus
        final_label = rule_pred
        decision_source = "rule_dominant"
        classification_status = "confirmed"
        agreement = (rule_pred == ml_pred)
        conflict = (ml_pred != UNCLASSIFIED and rule_pred != ml_pred)
        hybrid_confidence = min(0.85, 0.55 + (0.10 * active_votes))
        explanation.append(
            f"Strong domain consensus: {active_votes} labeling function(s) voted "
            f"'{rule_pred}' ({', '.join(active_lfs.keys())})."
        )
        if ml_pred == UNCLASSIFIED:
            explanation.append(
                f"ML model found no confident class (most probable class "
                f"'unclassified' at {ml_prob * 100:.1f}%)."
            )
        else:
            explanation.append(
                f"ML model exhibited low confidence ({ml_prob * 100:.1f}% for '{ml_pred}')."
            )
        explanation.append(
            "Decision [rule_dominant]: independent multi-rule consensus is "
            "sufficient — status confirmed."
        )

    elif rule_has_signal and ml_has_signal and (rule_pred != ml_pred):
        # CASE D: Rules + ML conflict — policy by evidence strength
        agreement = False
        conflict = True
        if active_votes >= STRONG_RULE_VOTES and ml_prob >= ML_HIGH:
            # D1: strong rules AND strong ML actively disagree -> UNCERTAIN + review
            final_label = UNCLASSIFIED
            decision_source = "conflict"
            classification_status = "uncertain"
            hybrid_confidence = 0.0
            requires_human_review = True
            review_reason = (
                f"Strong rule consensus ('{rule_pred}') and strong ML evidence "
                f"('{ml_pred}', {ml_prob * 100:.1f}%) actively disagree."
            )
            explanation.append(
                f"Strong disagreement: {active_votes} labeling function(s) voted "
                f"'{rule_pred}' while ML predicted '{ml_pred}' at "
                f"{ml_prob * 100:.1f}%."
            )
            explanation.append(
                "Decision [conflict]: contradictory strong evidence — status "
                "uncertain, operator review required."
            )
        elif ml_prob >= ML_HIGH and active_votes < STRONG_RULE_VOTES:
            # D2: strong ML vs weak contradicting rule -> PROBABLE (ml_dominant) + review
            final_label = ml_pred
            decision_source = "ml_dominant"
            classification_status = "probable"
            hybrid_confidence = MODERATE_DECISION_CONFIDENCE
            requires_human_review = True
            review_reason = (
                f"Strong ML evidence ('{ml_pred}', {ml_prob * 100:.1f}%) "
                f"conflicts with a weak rule ('{rule_pred}')."
            )
            explanation.append(
                f"Contradiction: a weak rule ('{rule_pred}') conflicts with strong "
                f"ML evidence ('{ml_pred}', {ml_prob * 100:.1f}%)."
            )
            explanation.append(
                "Decision [ml_dominant]: strong ML favored, but the conflict is "
                "documented — status probable, operator review required."
            )
        elif active_votes >= STRONG_RULE_VOTES and ml_prob < ML_HIGH:
            # D3: strong rules vs weaker contradicting ML -> PROBABLE (rule_dominant) + review
            final_label = rule_pred
            decision_source = "rule_dominant"
            classification_status = "probable"
            hybrid_confidence = MODERATE_DECISION_CONFIDENCE
            requires_human_review = True
            review_reason = (
                f"Strong rule consensus ('{rule_pred}') conflicts with ML evidence "
                f"('{ml_pred}', {ml_prob * 100:.1f}%)."
            )
            explanation.append(
                f"Contradiction: strong rule consensus ({active_votes} votes for "
                f"'{rule_pred}') conflicts with ML ('{ml_pred}', "
                f"{ml_prob * 100:.1f}%)."
            )
            explanation.append(
                "Decision [rule_dominant]: multi-rule consensus favored, but the "
                "conflict is documented — status probable, operator review required."
            )
        else:
            # D4: close tie -> UNCERTAIN + review
            final_label = UNCLASSIFIED
            decision_source = "conflict"
            classification_status = "uncertain"
            hybrid_confidence = 0.0
            requires_human_review = True
            review_reason = (
                f"No clear winner: rule consensus ('{rule_pred}') and ML "
                f"('{ml_pred}', {ml_prob * 100:.1f}%) conflict at close strength."
            )
            explanation.append(
                f"Close conflict: rules voted '{rule_pred}', ML predicted "
                f"'{ml_pred}' at {ml_prob * 100:.1f}%."
            )
            explanation.append(
                "Decision [conflict]: no dominant evidence — status uncertain, "
                "operator review required."
            )

    else:
        # CASE E: Both abstain (or the only signal is below acceptance thresholds).
        # There is NO class decision, so hybrid_confidence is 0.0. Any ML
        # probability attached to the 'unclassified' output is the model's own
        # abstention signal (see ml_engine) — it is never final-class confidence.
        final_label = UNCLASSIFIED
        decision_source = "uncertain"
        classification_status = "uncertain"
        agreement = False  # reciprocal abstention is not positive agreement
        conflict = False
        hybrid_confidence = 0.0
        requires_human_review = True
        if rule_has_signal:
            review_reason = (
                f"Rule evidence below the strong-consensus threshold "
                f"({active_votes}/{STRONG_RULE_VOTES} votes) and ML offered no "
                "confident class."
            )
            explanation.append(
                f"Only {active_votes} labeling function(s) fired — below the "
                f"strong-consensus threshold ({STRONG_RULE_VOTES})."
            )
        else:
            review_reason = "Both rule engine and ML model have insufficient evidence."
            explanation.append("All 14 labeling functions abstained due to lack of spatial context.")
        if ml_pred == UNCLASSIFIED and ml_prob > 0:
            explanation.append(
                f"XGBoost also abstained: its most probable class was "
                f"'unclassified' ({ml_prob * 100:.1f}%), meaning no specific "
                f"class met the evidence threshold."
            )
        elif ml_prob > 0:
            explanation.append(
                f"XGBoost's most probable class was '{ml_pred}' at only "
                f"{ml_prob * 100:.1f}%, below the evidence threshold."
            )
        explanation.append(
            "Decision [uncertain]: classified as unknown requiring ground review."
        )

    # Calibrate confidence tier
    confidence_level = _confidence_tier(hybrid_confidence)

    # Flag for review if confidence below threshold
    if hybrid_confidence < REVIEW_CONFIDENCE:
        requires_human_review = True
        if review_reason is None:
            review_reason = f"Hybrid confidence ({hybrid_confidence:.2f}) below verification threshold ({REVIEW_CONFIDENCE:.2f})."

    return {
        "final_label": final_label,
        "classification_status": classification_status,
        "hybrid_confidence": round(float(hybrid_confidence), 4),
        "decision_confidence": round(float(hybrid_confidence), 4),
        "raw_ml_confidence": round(float(ml_prob), 4),
        "ml_probability": round(float(ml_prob), 4),
        "confidence_level": confidence_level,
        "decision_confidence_level": confidence_level,
        "decision_source": decision_source,
        "agreement": agreement,
        "conflict": conflict,
        "requires_human_review": requires_human_review,
        "review_reason": review_reason,
        "rule_engine": {
            "prediction": rule_pred,
            "votes": vote_breakdown,
            "active_votes": active_votes,
            "active_lfs": active_lfs,
        },
        "ml_engine": {
            "prediction": ml_pred,
            "confidence": round(float(ml_prob), 4),
            "probabilities": {k: round(float(v), 4) for k, v in all_probs.items()},
        },
        "rule_consensus": rule_pred,
        "rule_vote_strength": active_votes,
        "explanation": explanation,
    }


# Module-level singleton
_engine_instance: Optional["HybridEngine"] = None


class HybridEngine:
    """Class wrapper for the hybrid classification logic."""

    def __init__(self):
        pass

    def classify(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return classify_hotspot(record)


def get_hybrid_engine() -> HybridEngine:
    """Get singleton instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = HybridEngine()
    return _engine_instance


if __name__ == "__main__":
    import json

    print("=" * 65)
    print("THERMOSCOPE-AI: Hybrid Intelligence Engine — Phase C Demo")
    print("=" * 65)

    # Test 1: Hotspot with industrial context
    test_industrial = {
        "frp": 12.0,
        "bright_ti4": 335.0,
        "bright_ti5": 305.0,
        "dist_industry": 0.5,
        "dist_factory": 0.5,
        "dist_refinery": 15.0,
        "dist_oil_gas": 25.0,
        "dist_mining": 999.0,
        "dist_forest": 45.0,
        "dist_agriculture": 10.0,
        "dist_powerplant": 8.0,
        "has_industrial_2km": 1,
        "count_ind_5km": 4,
        "confidence": "n",
        "daynight": "D",
    }

    print("\n[Test 1] Hotspot near industrial area:")
    result = classify_hotspot(test_industrial)
    print(f"  Final label: {result['final_label']}")
    print(f"  Confidence:  {result['hybrid_confidence']}")
    print(f"  Source:      {result['decision_source']}")
    print(f"  Agreement:   {result['agreement']}, Conflict: {result['conflict']}")
    print(f"  Review:      {result['requires_human_review']}")
    print(f"  Rule votes:  {result['rule_engine']['active_votes']}")
    print(f"  ML prob:     {result['ml_engine']['confidence']}")
    print("\n  Explanation:")
    for line in result["explanation"]:
        print(f"    - {line}")

    # Test 2: Hotspot with no context
    test_isolated = {
        "frp": 4.0,
        "bright_ti4": 320.0,
        "bright_ti5": 295.0,
        "dist_industry": 999.0,
        "dist_factory": 999.0,
        "dist_refinery": 999.0,
        "dist_oil_gas": 999.0,
        "dist_mining": 999.0,
        "dist_forest": 999.0,
        "dist_agriculture": 999.0,
        "dist_powerplant": 999.0,
        "has_industrial_2km": 0,
        "count_ind_5km": 0,
        "confidence": "n",
        "daynight": "D",
    }

    print("\n[Test 2] Isolated hotspot (no spatial context):")
    result = classify_hotspot(test_isolated)
    print(f"  Final label: {result['final_label']}")
    print(f"  Confidence:  {result['hybrid_confidence']}")
    print(f"  Source:      {result['decision_source']}")
    print(f"  Review:      {result['requires_human_review']}")
    if result["review_reason"]:
        print(f"  Review reason: {result['review_reason']}")

    print("\n" + "=" * 65)
    print("Phase C hybrid engine operational.")
    print("=" * 65)
