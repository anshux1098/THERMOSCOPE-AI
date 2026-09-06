"""
tests/test_api.py — FastAPI layer tests.

The API is a thin adapter over the existing services, so these tests verify the
HTTP contract and the presence of the MVP output fields, not classification logic.
"""
import json
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


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_analyze_returns_mvp_outputs(client: TestClient) -> None:
    """End-to-end through the API must surface the MVP intelligence outputs."""
    payload = {
        "latitude": 21.1051,
        "longitude": 72.6438,
        "frp": 5.9,
        "brightness": 330.8,
        "confidence": "nominal",
        "acq_date": "2026-09-01",
    }
    resp = client.post("/api/v1/hotspots/analyze", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    assert body["hotspot"]["latitude"] == payload["latitude"]
    assert "nearest_industry_m" in body["spatial_context"]

    cls = body["classification"]
    assert cls is not None
    for key in (
        "final_label",
        "hybrid_confidence",
        "decision_source",
        "agreement",
        "conflict",
        "requires_human_review",
        "review_reason",
        "explanation",
    ):
        assert key in cls, f"missing MVP output field: {key}"

    assert cls["decision_source"] in (
        "rules_only",
        "ml_only",
        "rules_and_ml_agree",
        "hybrid_agreement",
        "conflict",
        "rule_dominant",
        "ml_dominant",
        "ml_assisted",
        "uncertain",
    )
    assert cls["final_label"] in (
        "industrial_fire",
        "gas_flare",
        "mining_activity",
        "agricultural_burn",
        "forest_fire",
        "process_heat",
        "unclassified",
    )
    assert cls["classification_status"] in ("confirmed", "probable", "uncertain")
    assert cls["decision_confidence"] == cls["hybrid_confidence"]
    assert cls["ml_probability"] == cls["raw_ml_confidence"]
    assert cls["rule_consensus"] == cls["rule_engine"]["prediction"]
    assert cls["rule_vote_strength"] == cls["rule_engine"]["active_votes"]
    assert isinstance(cls["explanation"], list) and len(cls["explanation"]) >= 1


def test_analyze_validates_input(client: TestClient) -> None:
    """Malformed hotspots must be rejected by the schema, not the engine."""
    resp = client.post(
        "/api/v1/hotspots/analyze",
        json={"latitude": "not-a-number", "longitude": 0, "frp": 0, "brightness": 0, "confidence": "x", "acq_date": "x"},
    )
    assert resp.status_code == 422