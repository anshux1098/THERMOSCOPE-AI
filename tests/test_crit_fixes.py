"""
tests/test_crit_fixes.py — Phase D release-gate regression tests for:

  CRIT-1  Live API loses bright_ti5, causing batch/live ML feature mismatch.
          Fixed: optional bright_ti5 in HotspotBase/from_dict flows end-to-end
          into the ML feature record.
  CRIT-2  Live spatial evidence discovery gate differed from the batch feature
          gate (1.2x vs 1.5x nominal radius), silently dropping out-of-nominal
          candidates such as the 17.969 km power plant at idx 1124.
          Fixed: shared SPATIAL_BBOX_GATE_MULTIPLIER == 1.5 in both paths.
  MED-1   Stored derived pipeline snapshot must be reproducible by the current
          engine (snapshot-fidelity guard).

Run:
    pytest tests/test_crit_fixes.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
root_dir = str(Path(__file__).resolve().parents[1])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.schemas.hotspot import Hotspot
from app.geo.spatial_features import SPATIAL_BBOX_GATE_MULTIPLIER
from app.services.hotspot_service import analyze_single_hotspot


def _build_feature_record_from(hotspot, radius_meters=15000):
    """Live-style full feature record: schema -> spatial context -> shared contract."""
    from app.geo.spatial_context import compute_geospatial_context
    from app.services.hotspot_service import _build_feature_record, _spatial_features_from_geo_context
    ctx = compute_geospatial_context(hotspot=hotspot, radius_meters=radius_meters, use_live_api=False)
    geo = ctx["geo_context"] if isinstance(ctx, dict) and "geo_context" in ctx else ctx
    return _build_feature_record(hotspot.model_dump(), _spatial_features_from_geo_context(geo))


# ---------------------------------------------------------------------------
# CRIT-1  bright_ti5 feature parity
# ---------------------------------------------------------------------------
class TestCrit1BrightTi5Parity:
    def test_schema_accepts_bright_ti5(self):
        hs = Hotspot.from_dict({
            "latitude": 26.5, "longitude": 90.5, "frp": 5.79,
            "brightness": 328.03, "bright_ti5": 289.42,
            "confidence": "n", "acq_date": "2026-09-03",
        })
        assert hs.bright_ti5 == 289.42

    def test_parsing_preserves_bright_ti5(self):
        hs = Hotspot.from_dict({
            "latitude": 26.5, "longitude": 90.5, "frp": 5.79,
            "brightness": 328.03, "bright_ti5": 288.83,
            "confidence": "n", "acq_date": "2026-09-03",
        })
        assert "bright_ti5" in hs.model_dump()
        assert hs.model_dump()["bright_ti5"] == 288.83

    def test_bright_ti5_reaches_feature_record(self):
        # The feature record passed to the engine/ML must carry the supplied
        # value, never silently replaced with 0.0.
        hs = Hotspot.from_dict({
            "latitude": 26.5, "longitude": 90.5, "frp": 5.79,
            "brightness": 328.03, "bright_ti5": 290.0,
            "confidence": "n", "acq_date": "2026-09-03",
        })
        spot = hs.model_dump()
        from app.services.hotspot_service import _build_feature_record
        from app.geo.spatial_features import compute_spatial_features
        feats = compute_spatial_features(26.5, 90.5, {})
        record = _build_feature_record(spot, feats)
        assert record["bright_ti5"] == 290.0

    def test_backward_compat_without_bright_ti5(self):
        hs = Hotspot.from_dict({
            "latitude": 26.5, "longitude": 90.5, "frp": 5.79,
            "brightness": 328.03, "confidence": "n", "acq_date": "2026-09-03",
        })
        assert hs.bright_ti5 is None
        spot = hs.model_dump()
        from app.services.hotspot_service import _build_feature_record
        from app.geo.spatial_features import compute_spatial_features
        feats = compute_spatial_features(26.5, 90.5, {})
        record = _build_feature_record(spot, feats)
        assert record["bright_ti5"] is None  # legacy fallback behavior preserved

    def test_daynight_passthrough_reaches_feature_record(self):
        # CRIT-1 completion: daynight must survive schema + from_dict and reach
        # the feature record so night-gated rules (lf_refinery_flare, etc.) match batch.
        hs = Hotspot.from_dict({
            "latitude": 26.5, "longitude": 90.5, "frp": 5.79, "brightness": 328.03,
            "bright_ti5": 288.83, "confidence": "n", "acq_date": "2026-09-03",
            "daynight": "N", "satellite": "N20", "acq_time": 2002,
        })
        assert hs.daynight == "N"
        assert hs.satellite == "N20"
        assert hs.acq_time == "2002"
        from app.geo.spatial_features import compute_spatial_features
        from app.services.hotspot_service import _build_feature_record
        feats = compute_spatial_features(26.5, 90.5, {})
        record = _build_feature_record(hs.model_dump(), feats)
        assert record["daynight"] == "N"
        assert record["satellite"] == "N20"
        assert record["acq_time"] == "2002"

    def test_daynight_optional_default_none(self):
        hs = Hotspot.from_dict({
            "latitude": 26.5, "longitude": 90.5, "frp": 5.79,
            "brightness": 328.03, "confidence": "n", "acq_date": "2026-09-03",
        })
        assert hs.daynight is None and hs.satellite is None and hs.acq_time is None


# ---------------------------------------------------------------------------
# CRIT-2  Spatial evidence parity (shared gate multiplier)
# ---------------------------------------------------------------------------
class TestCrit2SpatialEvidenceParity:
    def test_shared_bbox_gate_multiplier_is_used_by_both_paths(self):
        import inspect
        from app.services.osm_service import find_nearby_geographic_objects
        from app.geo.spatial_features import compute_spatial_features

        osvc_src = inspect.getsource(find_nearby_geographic_objects)
        sfeat_src = inspect.getsource(compute_spatial_features)
        # Both discovery and feature generation must reference the SAME
        # multiplier constant — never independent 1.2 / 1.5 literals.
        assert "SPATIAL_BBOX_GATE_MULTIPLIER" in osvc_src
        assert "SPATIAL_BBOX_GATE_MULTIPLIER" in sfeat_src
        assert SPATIAL_BBOX_GATE_MULTIPLIER == 1.5

    def test_live_discovers_entity_batch_features_admit(self):
        # A power plant lat ~0.16 deg from the hotspot in longitude (approx
        # 16.5-18 km): inside batch's 1.5x=~22.5 km effective gate at 15 km
        # nominal, but would have been outside live's old 1.2x gate. Now both
        # share the same multiplier, so live must discover it.
        from app.services.osm_service import find_nearby_geographic_objects
        lat, lon = 26.52291, 90.5325   # idx 1124 hotspot
        ctx = find_nearby_geographic_objects(lat, lon, radius_meters=15000, use_live_api=False)
        assert ctx["data_sources"]["power_plant"] == "cache"
        assert len(ctx["power_plant"]) >= 1


# ---------------------------------------------------------------------------
# Regression hotspots — idx 1022 (bright_ti5) and idx 1124 (spatial + ti5)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def v2_df():
    return pd.read_csv("data/processed/hotspots/classified_hotspots_v2.csv")


def _payload_from_v2(v2_df, idx, include_ti5=True):
    r = v2_df.loc[idx]
    payload = {
        "latitude": float(r["latitude"]), "longitude": float(r["longitude"]),
        "frp": float(r["frp"]), "brightness": float(r["bright_ti4"]),
        "confidence": str(r["confidence"]), "acq_date": str(r["acq_date"]),
    }
    if include_ti5 and r.get("bright_ti5") is not None:
        payload["bright_ti5"] = float(r["bright_ti5"])
    for field in ("daynight", "satellite", "acq_time"):
        if field in r and pd.notna(r.get(field)):
            payload[field] = str(r[field])
    return payload


class TestRegressionHotspots:
    def test_idx1022_batch_parity(self, v2_df):
        # Batch: (agricultural_burn, confirmed, hybrid_agreement) ml=0.9843
        # rules=agri 3. Matches regenerated enriched snapshot after the
        # sample-weighted retrain + widened agri/forest thresholds.
        payload = _payload_from_v2(v2_df, 1022)
        c = analyze_single_hotspot(payload, radius_meters=15000,
                                   use_live_api=False, run_classification=True).classification
        assert c["final_label"] == "agricultural_burn"
        assert c["classification_status"] == "confirmed"
        assert c["decision_source"] == "hybrid_agreement"
        assert round(float(c["ml_probability"]), 4) == 0.9843
        assert c["rule_engine"]["prediction"] == "agricultural_burn"
        assert int(c["rule_engine"]["active_votes"]) == 3

    def test_idx1124_batch_parity(self, v2_df):
        # Batch (full FIRMS metadata incl. daynight='N'): industrial entities
        # + night -> rules industrial_fire (4 votes) AND ML industrial_fire
        # 0.9986 -> confirmed/hybrid_agreement. Matches regenerated enriched
        # snapshot after the sample-weighted retrain + widened industrial/
        # power-plant thresholds.
        payload = _payload_from_v2(v2_df, 1124)
        c = analyze_single_hotspot(payload, radius_meters=15000,
                                   use_live_api=False, run_classification=True).classification
        assert c["final_label"] == "industrial_fire"
        assert c["classification_status"] == "confirmed"
        assert c["decision_source"] == "hybrid_agreement"
        assert round(float(c["ml_probability"]), 4) == 0.9986
        assert c["rule_engine"]["prediction"] == "industrial_fire"
        assert int(c["rule_engine"]["active_votes"]) == 4

    def test_night_gated_gas_flare_rule_parity(self, v2_df):
        # CRIT-1 completion: daynight passthrough. A refinery-adjacent DETECTION
        # whose batch record is night ('N') must fire lf_refinery_flare live.
        # Without the passthrough this class downgraded confirmed->probable/unclassified.
        from app.intelligence.labeling_functions import is_night, lf_refinery_flare
        payload = _payload_from_v2(v2_df, 1124)
        assert payload.get("daynight") == "N"
        record = _build_feature_record_from(hotspot=Hotspot.from_dict(payload), radius_meters=15000)
        assert is_night(record) is True
        assert lf_refinery_flare(record) is not None


# ---------------------------------------------------------------------------
# MED-1  Snapshot fidelity guard (lightweight, deterministic)
# ---------------------------------------------------------------------------
class TestSnapshotFidelity:
    def test_enriched_snapshot_matches_current_engine_distribution(self):
        # Deterministic on a fixed stratified subset so it is fast and
        # reproducible, yet catches silent drift between the stored derived
        # artifact and current pipeline semantics.
        import random
        from scripts import run_pipeline
        from app.intelligence.hybrid_engine import classify_hotspot

        enr = pd.read_csv("data/processed/hotspots/classified_hotspots_v2_enriched.csv")
        v2 = pd.read_csv("data/processed/hotspots/classified_hotspots_v2.csv")

        random.seed(42)
        idx = random.sample(range(len(v2)), min(60, len(v2)))
        mismatches = 0
        for i in idx:
            rec = run_pipeline._build_feature_row(v2.iloc[i])
            d = classify_hotspot(rec)
            stored = enr.loc[enr["input_index"] == i]
            assert len(stored) == 1
            stored_label = stored["final_label"].iloc[0]
            if d["final_label"] != stored_label:
                mismatches += 1
        assert mismatches == 0, f"stored snapshot diverged from engine on {mismatches}/{len(idx)} sampled labels"