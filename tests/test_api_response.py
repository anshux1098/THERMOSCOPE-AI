"""
tests/test_api_response.py — frontend-ready presentation contract tests.

Verifies the /api/v1/hotspots/analyze/presentation endpoint returns the
AnalysisResponse shape (status / hotspot / classification{kind, confidence,
risk_level, decision_source, decision_status, ml internals} / why_this_class /
recommendations) without regressing the internal /analyze contract.

CONFIDENCE SEMANTICS CONTRACT (enforced here):
- ``confidence_score`` is confidence in the FINAL class decision only; it is
  ``null`` when the engine abstains (kind == 'unclassified').
- ``ml_confidence`` / ``ml_top_prediction`` report the raw XGBoost output and
  are NEVER presented as final-class confidence when the engine abstains.
- ``confidence_level`` is 'insufficient_evidence' on abstention.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
root_dir = str(Path(__file__).resolve().parents[1])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.main import app
from app.schemas.analysis import HotspotAnalysis
from app.schemas.analysis_response import AnalysisResponse, Recommendation
from app.schemas.hotspot import Hotspot
from app.schemas.spatial_context import SpatialContext
from app.services.hotspot_service import _to_presentation_response
from app.services.recommendation_service import risk_level_for, generate_recommendations

ALL_CLASSES = (
    "industrial_fire",
    "gas_flare",
    "mining_activity",
    "agricultural_burn",
    "forest_natural_fire",
    "industrial_process_heat",
    "unclassified",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


PAYLOAD = {
    "latitude": 21.1051,
    "longitude": 72.6438,
    "frp": 5.9,
    "brightness": 330.8,
    "confidence": "nominal",
    "acq_date": "2026-09-01",
}


def test_presentation_endpoint_returns_frontend_contract(client: TestClient) -> None:
    resp = client.post("/api/v1/hotspots/analyze/presentation", json=PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "success"
    assert body["analysis_type"] == "presentation"

    # hotspot echo
    assert body["hotspot"]["latitude"] == PAYLOAD["latitude"]
    assert body["hotspot"]["confidence"] in ("high", "nominal", "low")

    # classification envelope
    cls = body["classification"]
    for key in (
        "kind",
        "display_name",
        "color",
        "confidence_score",
        "confidence_level",
        "risk_level",
        "decision_source",
        "decision_status",
        "classification_status",
        "rule_prediction",
        "rule_vote_strength",
        "ml_top_prediction",
        "ml_confidence",
        "requires_human_review",
        "review_reason",
    ):
        assert key in cls, f"missing classification field: {key}"
    assert cls["kind"] in ALL_CLASSES
    assert cls["risk_level"] in ("critical", "high", "medium", "low")
    assert cls["classification_status"] in ("confirmed", "probable", "uncertain")
    assert cls["decision_status"] in (
        "confirmed",
        "probable",
        "uncertain",
        "requires_review",
    )
    # confidence_score is confidence in the FINAL class decision: null on abstention
    if cls["confidence_score"] is None:
        assert cls["kind"] == "unclassified"
        assert cls["confidence_level"] == "insufficient_evidence"
    else:
        assert isinstance(cls["confidence_score"], float)
        assert 0.0 <= cls["confidence_score"] <= 1.0
        assert cls["confidence_level"] in ("low", "medium", "high")

    # why_this_class
    assert isinstance(body["why_this_class"], list) and len(body["why_this_class"]) >= 1

    # recommendations
    assert isinstance(body["recommendations"], list) and len(body["recommendations"]) >= 1
    for rec in body["recommendations"]:
        assert rec["priority"] in ("critical", "high", "medium", "low")
        assert isinstance(rec["action"], str) and rec["action"]
    actions = [rec["action"] for rec in body["recommendations"]]
    assert len(actions) == len(set(actions)), "recommendations must not be duplicated"


def test_presentation_keeps_internal_analyze_contract(client: TestClient) -> None:
    """The internal /analyze endpoint must be unaffected by the new endpoint."""
    resp = client.post("/api/v1/hotspots/analyze", json=PAYLOAD)
    assert resp.status_code == 200
    cls = resp.json()["classification"]
    for key in (
        "final_label",
        "hybrid_confidence",
        "decision_source",
        "agreement",
        "conflict",
        "requires_human_review",
        "explanation",
    ):
        assert key in cls


def test_unclassified_presentation_is_honest(client: TestClient) -> None:
    """
    CRITICAL: an abstention must never masquerade as high-confidence.

    This payload verifiably produces decision_source == 'uncertain' with ML
    predicting 'unclassified' at ~99.8%. The high ML number is the probability
    of the ABSTENTION output, so the final contract must express NO class
    confidence — never a 0.998 confidence_score / 'high' tier.
    """
    resp = client.post("/api/v1/hotspots/analyze/presentation", json=PAYLOAD)
    assert resp.status_code == 200
    cls = resp.json()["classification"]

    assert cls["kind"] == "unclassified"
    assert cls["decision_source"] == "uncertain"
    assert cls["confidence_score"] is None
    assert cls["confidence_level"] == "insufficient_evidence"
    assert cls["decision_status"] == "uncertain"
    assert cls["requires_human_review"] is True
    assert isinstance(cls["review_reason"], str) and cls["review_reason"]

    # ML internals are exposed SEPARATELY — as the probability of 'unclassified'.
    assert cls["ml_top_prediction"] == "unclassified"
    assert isinstance(cls["ml_confidence"], float)
    assert cls["ml_confidence"] > 0.9
    assert cls["confidence_score"] is not None or cls["ml_confidence"] != cls["confidence_score"]

    # Explanation must not claim a low-confidence phrasing for a 99.8% ML number.
    for line in resp.json()["why_this_class"]:
        assert "confidence is low" not in line.lower(), f"contradiction: {line}"

    # Unclassified carries its own analyst-verification recommendation; the extra
    # generic review reminder must NOT be appended (dedupe).
    actions = [rec["action"] for rec in resp.json()["recommendations"]]
    assert not any("Confirm the classification" in a for a in actions)


def _project(
    label: str,
    conf: float,
    tier: str,
    source: str,
    review: bool,
    review_reason: str | None,
    ml_pred: str,
    ml_conf: float,
) -> AnalysisResponse:
    """Build the presentation projection over an EXPLICIT hybrid decision."""
    analysis = HotspotAnalysis(
        hotspot=Hotspot(
            latitude=21.1051,
            longitude=72.6438,
            frp=5.9,
            brightness=330.8,
            confidence="nominal",
            acq_date="2026-09-01",
        ),
        spatial_context=SpatialContext(
            nearest_industry_m=500,
            nearest_refinery_m=10000,
            nearest_oil_gas_m=20000,
            nearest_mining_m=30000,
            nearest_agriculture_m=7000,
            nearest_forest_m=15000,
            nearest_power_plant_m=12000,
        ),
        classification={
            "final_label": label,
            "hybrid_confidence": conf,
            "confidence_level": tier,
            "decision_source": source,
            "requires_human_review": review,
            "review_reason": review_reason,
            "ml_engine": {"prediction": ml_pred, "confidence": ml_conf},
            "explanation": [f"Provenance: {source} for {label}."],
        },
    )
    return _to_presentation_response(analysis)


def test_projection_class_matrix_semantics() -> None:
    """
    Class matrix over ALL seven labels (fast, deterministic).

    Real API-path evidence exists only for industrial_fire (batch row 159),
    agricultural_burn (1022), gas_flare (1141) and unclassified (0); the other
    three classes are covered here at the presentation-projection layer and are
    reported as a dataset coverage limitation, not fabricated end-to-end rows.
    """
    cases = {
        "industrial_fire": dict(
            conf=0.92, tier="high", source="hybrid_agreement", review=False,
            review_reason=None, ml_pred="industrial_fire", ml_conf=0.88,
            exp_status="confirmed",
        ),
        "agricultural_burn": dict(
            conf=0.85, tier="high", source="rule_dominant", review=False,
            review_reason=None, ml_pred="agricultural_burn", ml_conf=0.52,
            exp_status="confirmed",
        ),
        "gas_flare": dict(
            conf=0.50, tier="medium", source="ml_dominant", review=True,
            review_reason="Conflict between rule consensus and ML.",
            ml_pred="gas_flare", ml_conf=0.50, exp_status="requires_review",
        ),
        "forest_natural_fire": dict(
            conf=0.71, tier="medium", source="hybrid_agreement", review=False,
            review_reason=None, ml_pred="industrial_fire", ml_conf=0.60,
            exp_status="probable",
        ),
        "mining_activity": dict(
            conf=0.64, tier="medium", source="rule_dominant", review=False,
            review_reason=None, ml_pred="unclassified", ml_conf=0.44,
            exp_status="probable",
        ),
        "industrial_process_heat": dict(
            conf=0.83, tier="high", source="hybrid_agreement", review=False,
            review_reason=None, ml_pred="industrial_process_heat", ml_conf=0.74,
            exp_status="confirmed",
        ),
        "unclassified": dict(
            conf=0.0, tier="low", source="uncertain", review=True,
            review_reason="Both rule engine and ML model have insufficient evidence.",
            ml_pred="unclassified", ml_conf=0.9983, exp_status="uncertain",
        ),
    }

    for label, cfg in cases.items():
        cfg = dict(cfg)
        exp_status = cfg.pop("exp_status")
        resp = _project(label, **cfg)
        c = resp.classification
        assert c.kind == label
        assert c.decision_status == exp_status, f"{label}: {c.decision_status}"
        assert c.decision_source == cfg["source"]
        assert c.ml_top_prediction == cfg["ml_pred"]
        assert c.ml_confidence == cfg["ml_conf"]
        assert c.display_name
        assert c.review_reason == cfg["review_reason"]

        # classification_status must stay coherent with decision_status:
        # non-review decisions map 1:1; requires_review keeps the underlying status.
        if c.decision_status in ("confirmed", "probable"):
            assert c.classification_status == c.decision_status, (
                f"{label}: classification_status={c.classification_status}"
            )
        elif c.decision_status == "requires_review":
            assert c.classification_status in ("confirmed", "probable", "uncertain")
        else:
            assert c.classification_status == "uncertain"

        if label == "unclassified":
            assert c.confidence_score is None
            assert c.confidence_level == "insufficient_evidence"
            assert c.requires_human_review is True
        else:
            assert isinstance(c.confidence_score, float)
            assert 0.0 <= c.confidence_score <= 1.0
            assert c.confidence_level in ("low", "medium", "high")
            assert c.confidence_score == round(cfg["conf"], 4)

        # Recommendations: non-empty, priorities valid, actions never duplicated.
        actions = [r.action for r in resp.recommendations]
        assert actions, f"{label}: no recommendations"
        assert len(actions) == len(set(actions)), f"{label}: duplicate recommendations"
        assert all(r.priority in ("critical", "high", "medium", "low") for r in resp.recommendations)

    # Review-reminder dedupe: unclassified already carries analyst-verification,
    # so the generic operator-confirmation reminder must not reappear.
    unc_cfg = {k: v for k, v in cases["unclassified"].items() if k != "exp_status"}
    unc_recs = _project("unclassified", **unc_cfg).recommendations
    assert not any("Confirm the classification" in r.action for r in unc_recs)
    # Decided + review (e.g. gas_flare close-tie) DOES carry exactly one such reminder.
    gas_cfg = {k: v for k, v in cases["gas_flare"].items() if k != "exp_status"}
    gas_confirmed = [r.action for r in _project("gas_flare", **gas_cfg).recommendations]
    assert sum("Confirm the classification" in a for a in gas_confirmed) == 1


def test_recommendation_service_ranges() -> None:
    """Risk mapping + recommendation generation are total over all labels."""
    for label in ALL_CLASSES:
        risk = risk_level_for(label, 0.9)
        assert risk in ("critical", "high", "medium", "low")
        decision = {
            "final_label": label,
            "hybrid_confidence": 0.9,
            "confidence_level": "high",
            "decision_source": "hybrid_agreement",
            "requires_human_review": False,
            "review_reason": None,
        }
        recs = generate_recommendations(decision)
        assert isinstance(recs, list) and len(recs) >= 1
        assert all(isinstance(r, Recommendation) for r in recs)


def test_unclassified_raises_risk_and_review(client: TestClient) -> None:
    """Unclassified must be flagged for review with critical risk."""
    assert risk_level_for("unclassified", 0.5) == "critical"
    decision = {
        "final_label": "unclassified",
        "hybrid_confidence": 0.5,
        "confidence_level": "low",
        "decision_source": "uncertain",
        "requires_human_review": True,
        "review_reason": "insufficient evidence",
    }
    recs = generate_recommendations(decision)
    priorities = [r.priority for r in recs]
    assert "critical" in priorities
    assert any("human" in r.action.lower() for r in recs)