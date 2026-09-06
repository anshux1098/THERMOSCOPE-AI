"""
tests/test_hybrid_policy.py — deterministic decision-matrix tests.

Drives the hybrid engine through each fusion case by monkeypatching the rule
side (`apply_labeling_functions` / `compute_vote_summary`) and ML side
(`predict_proba`) with controlled inputs, so no trained artifacts or live OSM
data are required and the outcomes are fully deterministic.

Decision matrix under test (ordered evaluation):
    A  rules + ML agree                       -> CONFIRMED
    B  rules abstain + ML >= ML_PROBABLE_THRESHOLD
                                              -> PROBABLE (ml_assisted)
    C  strong rules + weak/absent ML          -> CONFIRMED (rule_dominant)
    D1 strong rules + strong ML (disagree)    -> UNCERTAIN (conflict)
    D2 strong ML vs weak rule contest         -> PROBABLE (ml_dominant) + review
    D3 strong rules vs weaker ML contest      -> PROBABLE (rule_dominant) + review
    D4 close tie / weak conflict              -> UNCERTAIN (conflict)
    E  both abstain / below thresholds        -> UNCERTAIN
"""
import sys
from collections import Counter
from pathlib import Path

import pytest

backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
root_dir = str(Path(__file__).resolve().parents[1])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.core.constants import CLASS_LABELS
from app.intelligence import hybrid_engine
from app.intelligence.hybrid_engine import (
    HIGH_CONFIDENCE,
    ML_ASSISTED_REVIEW_MIN_PROB,
    ML_PROBABLE_THRESHOLD,
    MODERATE_CONFIDENCE,
    MODERATE_DECISION_CONFIDENCE,
    STRONG_RULE_VOTES,
)


def _run_case(
    monkeypatch: pytest.MonkeyPatch,
    rule_votes,
    ml_label: str,
    ml_prob: float,
):
    """Run classify_hotspot with fully controlled rule + ML inputs."""
    vote_dict = {f"lf_{i}": label for i, label in enumerate(rule_votes)}
    summary = {
        "consensus": rule_votes[0] if rule_votes else "unclassified",
        "total_lfs": 14,
        "active_votes_count": len(rule_votes),
        "abstain_count": 14 - len(rule_votes),
        "vote_breakdown": Counter(rule_votes),
        "active_lfs": vote_dict,
    }
    monkeypatch.setattr(hybrid_engine, "apply_labeling_functions", lambda record: vote_dict)
    monkeypatch.setattr(hybrid_engine, "compute_vote_summary", lambda votes: summary)

    preds = {c: 0.001 for c in CLASS_LABELS}
    preds[ml_label] = ml_prob
    monkeypatch.setattr(
        hybrid_engine,
        "predict_proba",
        lambda record: {
            "label": ml_label,
            "probability": ml_prob,
            "all_probabilities": preds,
        },
    )
    return hybrid_engine.classify_hotspot({})


def _assert_semantic_aliases(decision) -> None:
    """The semantic alias fields must agree with their canonical keys."""
    assert decision["classification_status"] in ("confirmed", "probable", "uncertain")
    assert decision["decision_confidence"] == decision["hybrid_confidence"]
    assert decision["decision_confidence_level"] == decision["confidence_level"]
    assert decision["ml_probability"] == decision["raw_ml_confidence"]
    assert decision["rule_consensus"] == decision["rule_engine"]["prediction"]
    assert decision["rule_vote_strength"] == decision["rule_engine"]["active_votes"]
    for line in decision["explanation"]:
        assert "confidence is low" not in line.lower(), f"contradiction: {line}"


def test_case_a_agreement_is_confirmed(monkeypatch) -> None:
    d = _run_case(monkeypatch, ["industrial_fire", "industrial_fire"], "industrial_fire", 0.90)
    assert d["final_label"] == "industrial_fire"
    assert d["classification_status"] == "confirmed"
    assert d["decision_source"] == "hybrid_agreement"
    assert d["agreement"] is True
    assert d["conflict"] is False
    assert d["requires_human_review"] is False
    assert d["hybrid_confidence"] == pytest.approx(min(0.99, 0.90 + 0.08))
    _assert_semantic_aliases(d)


def test_case_b_rules_abstain_strong_ml_is_probable(monkeypatch) -> None:
    d = _run_case(monkeypatch, [], "gas_flare", 0.85)
    assert d["final_label"] == "gas_flare"
    assert d["classification_status"] == "probable"
    assert d["decision_source"] == "ml_assisted"
    assert d["hybrid_confidence"] == pytest.approx(MODERATE_DECISION_CONFIDENCE)
    assert d["confidence_level"] == "medium"
    # Below the no-review threshold (0.85 < 0.90) -> operator review required.
    assert d["requires_human_review"] is True
    assert d["review_reason"] and "no-review" in d["review_reason"]
    _assert_semantic_aliases(d)


def test_case_b_strong_ml_above_no_review_threshold(monkeypatch) -> None:
    d = _run_case(monkeypatch, [], "gas_flare", 0.95)
    assert d["classification_status"] == "probable"
    assert d["decision_source"] == "ml_assisted"
    assert d["requires_human_review"] is False
    _assert_semantic_aliases(d)


def test_case_b_ml_below_probable_threshold_stays_uncertain(monkeypatch) -> None:
    """Rules abstain + real-class ML below the acceptance threshold -> UNCERTAIN."""
    d = _run_case(monkeypatch, [], "gas_flare", ML_PROBABLE_THRESHOLD - 0.01)
    assert d["final_label"] == "unclassified"
    assert d["classification_status"] == "uncertain"
    assert d["decision_source"] == "uncertain"
    assert d["hybrid_confidence"] == 0.0
    assert d["requires_human_review"] is True
    _assert_semantic_aliases(d)


def test_case_c_strong_rules_weak_ml_is_confirmed(monkeypatch) -> None:
    d = _run_case(monkeypatch, ["industrial_fire", "industrial_fire"], "unclassified", 0.97)
    assert d["final_label"] == "industrial_fire"
    assert d["classification_status"] == "confirmed"
    assert d["decision_source"] == "rule_dominant"
    assert d["agreement"] is False
    assert d["conflict"] is False
    assert d["hybrid_confidence"] == pytest.approx(min(0.85, 0.55 + 0.10 * 2))
    assert d["requires_human_review"] is False
    _assert_semantic_aliases(d)


def test_case_d1_strong_strong_disagreement_is_uncertain(monkeypatch) -> None:
    d = _run_case(monkeypatch, ["industrial_fire", "industrial_fire"], "forest_natural_fire", 0.88)
    assert d["final_label"] == "unclassified"
    assert d["classification_status"] == "uncertain"
    assert d["decision_source"] == "conflict"
    assert d["conflict"] is True
    assert d["hybrid_confidence"] == 0.0
    assert d["requires_human_review"] is True
    assert "disagree" in d["review_reason"]
    _assert_semantic_aliases(d)


def test_case_d2_strong_ml_vs_weak_rule_is_probable_review(monkeypatch) -> None:
    d = _run_case(monkeypatch, ["industrial_fire"], "agricultural_burn", 0.85)
    assert d["final_label"] == "agricultural_burn"
    assert d["classification_status"] == "probable"
    assert d["decision_source"] == "ml_dominant"
    assert d["conflict"] is True
    assert d["requires_human_review"] is True
    assert d["hybrid_confidence"] == pytest.approx(MODERATE_DECISION_CONFIDENCE)
    _assert_semantic_aliases(d)


def test_case_d3_strong_rules_vs_weaker_ml_is_probable_review(monkeypatch) -> None:
    d = _run_case(monkeypatch, ["industrial_fire", "industrial_fire"], "gas_flare", 0.65)
    assert d["final_label"] == "industrial_fire"
    assert d["classification_status"] == "probable"
    assert d["decision_source"] == "rule_dominant"
    assert d["conflict"] is True
    assert d["requires_human_review"] is True
    _assert_semantic_aliases(d)


def test_case_d4_close_tie_is_uncertain(monkeypatch) -> None:
    d = _run_case(monkeypatch, ["industrial_fire"], "gas_flare", 0.50)
    assert d["final_label"] == "unclassified"
    assert d["classification_status"] == "uncertain"
    assert d["decision_source"] == "conflict"
    assert d["conflict"] is True
    assert d["hybrid_confidence"] == 0.0
    assert d["requires_human_review"] is True
    _assert_semantic_aliases(d)


def test_case_e_both_abstain_is_uncertain_honest(monkeypatch) -> None:
    d = _run_case(monkeypatch, [], "unclassified", 0.9983)
    assert d["final_label"] == "unclassified"
    assert d["classification_status"] == "uncertain"
    assert d["decision_source"] == "uncertain"
    assert d["agreement"] is False  # reciprocal abstention is not agreement
    assert d["conflict"] is False
    assert d["hybrid_confidence"] == 0.0
    assert d["raw_ml_confidence"] == pytest.approx(0.9983)
    assert d["requires_human_review"] is True
    _assert_semantic_aliases(d)


def test_threshold_canary_values() -> None:
    """Thresholds are centralized and consistent with the decision policy."""
    from app.intelligence import hybrid_engine as he
    assert he.ML_PROBABLE_THRESHOLD == he.HIGH_CONFIDENCE == 0.80
    assert he.STRONG_RULE_VOTES == 2
    assert 0.0 < he.MODERATE_DECISION_CONFIDENCE < he.HIGH_CONFIDENCE
    assert he.HIGH_CONFIDENCE < he.ML_ASSISTED_REVIEW_MIN_PROB <= 1.0
    assert he.ML_SIGNAL_MIN <= he.ML_PROBABLE_THRESHOLD