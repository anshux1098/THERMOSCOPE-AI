"""
tests/test_phase_b_spatial_fixes.py

Regression tests for the Phase B spatial/data fixes:

  P0.1  Batch producer merges the existing forest/agriculture OSM cache, so
        dist_forest / dist_agriculture are REAL (not constant sentinel).
  P0.2  firms_type tri-state: missing/-1 is UNKNOWN evidence, never silently
        positive (old `firms_type is None` shortcut) or silently negative.
  P1.1  (covered below) batch == live single shared feature contract.
  P1.4  One canonical "no nearby entity" sentinel for batch AND live, always
        numeric (never None), and never confused with SPATIAL_EVIDENCE_INFLUENCE_M.
  P1.5  Real neighbourhood counts (industrial_sites_within_2km/5km, etc.),
        with legacy bucket aliases count_ind_5km / count_ref_5km kept.

Run:
    pytest tests/test_phase_b_spatial_fixes.py -v
"""
import sys
import os
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Make app package importable
backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
root_dir = str(Path(__file__).resolve().parents[1])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.geo.spatial_features import (
    SENTINEL_DISTANCE_M,
    SENTINEL_DISTANCE_KM,
    SPATIAL_EVIDENCE_INFLUENCE_M,
    CATEGORIES,
    build_candidates_by_category,
    compute_spatial_features,
)
from app.intelligence.labeling_functions import (
    ABSTAIN,
    INDUSTRIAL_FIRE,
    FOREST_NATURAL_FIRE,
    AGRICULTURAL_BURN,
    get_firms_type,                 # noqa: F401  (kept for tri-state construction)
    get_firms_type_state,
    FIRMS_TYPE_KNOWN_STATE,
    FIRMS_TYPE_UNKNOWN_STATE,
    lf_forest_vegetation_fire,
    lf_agriculture_vegetation_fire,
    lf_industrial_zone_cluster,
)


def _mk_site(lat, lon, category=None, tags=None, site_type=None, distance_meters=None):
    s = {"lat": lat, "lon": lon, "category": category, "tags": tags or {}, "site_type": site_type}
    if distance_meters is not None:
        s["distance_meters"] = distance_meters
    return s


# ---------------------------------------------------------------------------
# P1.4  Sentinel contract
# ---------------------------------------------------------------------------
class TestSentinelContract:
    def test_canonical_sentinel_values(self):
        assert SENTINEL_DISTANCE_M == 999000.0
        assert SENTINEL_DISTANCE_KM == 999.0
        assert SENTINEL_DISTANCE_M == SENTINEL_DISTANCE_KM * 1000.0
        # Named "negligible influence" rule threshold, distinct from "no entity".
        assert SPATIAL_EVIDENCE_INFLUENCE_M == 45000.0
        assert SENTINEL_DISTANCE_M > SPATIAL_EVIDENCE_INFLUENCE_M

    def test_feature_distances_never_none(self):
        feats = compute_spatial_features(20.0, 75.0, {})
        for c in CATEGORIES:
            v = feats[f"distance_to_{c}_m"]
            assert v == SENTINEL_DISTANCE_M
            assert v is not None
            assert isinstance(v, float) or isinstance(v, int)
        # Empty candidates are "no nearby entity", never "influence threshold".
        assert feats["distance_to_mining_m"] == SENTINEL_DISTANCE_M
        assert feats["dist_mining"] == SENTINEL_DISTANCE_KM

    def test_isolated_row_reports_none_source(self):
        feats = compute_spatial_features(20.0, 75.0, {})
        for c in CATEGORIES:
            assert feats[f"src_{c}"] == "none"


# ---------------------------------------------------------------------------
# P1.5  Real counts vs legacy bucket aliases
# ---------------------------------------------------------------------------
class TestRealCounts:
    def _candidates(self):
        # Valid lat/lon required; distance_meters pre-computed value is reused
        # by compute_spatial_features to avoid re-calc.
        return {
            "industry": [
                _mk_site(0.0004, 0.0, tags={"landuse": "industrial"}, distance_meters=1000.0),
                _mk_site(0.0080, 0.0, tags={"landuse": "industrial"}, distance_meters=1800.0),
                _mk_site(0.0189, 0.0, tags={"landuse": "industrial"}, distance_meters=4200.0),
                _mk_site(0.0405, 0.0, tags={"landuse": "industrial"}, distance_meters=9000.0),
            ],
            "refinery": [
                _mk_site(0.0126, 0.0, tags={"industrial": "refinery"}, distance_meters=2800.0),
                _mk_site(0.0234, 0.0, tags={"industrial": "refinery"}, distance_meters=5200.0),
            ],
            "forest": [
                _mk_site(0.0054, 0.0, category="forest", distance_meters=1200.0),
                _mk_site(0.0194, 0.0, category="forest", distance_meters=4300.0),
                _mk_site(0.0365, 0.0, category="forest", distance_meters=8100.0),
            ],
            "agriculture": [
                _mk_site(0.0032, 0.0, category="agriculture", distance_meters=700.0),
            ],
        }

    def test_real_counts_reflect_actual_neighbourhood(self):
        feats = compute_spatial_features(0.0, 0.0, self._candidates())
        # 3 industrial sites <= 5 km, 2 <= 2 km.
        assert feats["industrial_sites_within_2km"] == 2
        assert feats["industrial_sites_within_5km"] == 3
        # 1 refinery <= 3 km, 1 <= 5 km (the 5200 m site is excluded).
        assert feats["refinery_sites_within_3km"] == 1
        assert feats["refinery_sites_within_5km"] == 1
        # 2 forest <= 5 km, 8100 m excluded; 1 agriculture <= 5 km.
        assert feats["forest_sites_within_5km"] == 2
        assert feats["count_forest_5km"] == 2
        assert feats["agriculture_sites_within_5km"] == 1
        assert feats["count_agriculture_5km"] == 1

    def test_legacy_bucket_aliases_are_byte_compatible(self):
        feats = compute_spatial_features(0.0, 0.0, self._candidates())
        # count_ind_5km: 3 if <=2km else 1 if <=5km else 0 -> nearest is 1000 m -> 3.
        assert feats["count_ind_5km"] == 3
        # count_ref_5km: 2 if <=3km else 1 if <=5km else 0 -> nearest 2800 m -> 2.
        assert feats["count_ref_5km"] == 2

    def test_bucket_alias_uses_nearest_not_any_site(self):
        # A lone industrial site at 2200 m must produce count_ind_5km == 1
        # (the old bucket code), never >= 2, even though it IS within 5 km.
        cands = {"industry": [_mk_site(0.0099, 0.0, tags={"landuse": "industrial"}, distance_meters=2200.0)]}
        feats = compute_spatial_features(0.0, 0.0, cands)
        assert feats["industrial_sites_within_5km"] == 1
        assert feats["count_ind_5km"] == 1

    def test_flags_match_real_distances(self):
        feats = compute_spatial_features(0.0, 0.0, self._candidates())
        assert feats["has_industrial_2km"] == 1
        assert feats["has_factory_5km"] == 1
        assert feats["has_refinery_5km"] == 1
        assert feats["has_forest_5km"] == 1
        assert feats["has_agriculture_5km"] == 1


# ---------------------------------------------------------------------------
# P0.2  firms_type tri-state
# ---------------------------------------------------------------------------
class TestFirmsTypeTriState:
    def test_missing_is_unknown(self):
        assert get_firms_type_state({}) == FIRMS_TYPE_UNKNOWN_STATE
        assert get_firms_type_state({"firms_type": None}) == FIRMS_TYPE_UNKNOWN_STATE
        assert get_firms_type_state({"firms_type_mode": None}) == FIRMS_TYPE_UNKNOWN_STATE

    def test_virs_neg1_is_unknown_not_negative(self):
        # VIIRS 'unknown' code must NOT be treated as evidence of any kind.
        assert get_firms_type_state({"firms_type": -1}) == FIRMS_TYPE_UNKNOWN_STATE
        assert get_firms_type_state({"firms_type_mode": -1}) == FIRMS_TYPE_UNKNOWN_STATE

    def test_real_codes_are_known(self):
        assert get_firms_type_state({"firms_type": 0}) == FIRMS_TYPE_KNOWN_STATE
        assert get_firms_type_state({"firms_type": 2}) == FIRMS_TYPE_KNOWN_STATE
        assert get_firms_type_state({"firms_type": 3}) == FIRMS_TYPE_KNOWN_STATE

    @pytest.mark.parametrize("missing_firms", [{}, {"firms_type": None}, {"firms_type": -1}])
    def test_unknown_type_fires_on_circumstantial_forest_evidence(self, missing_firms):
        # Forest proximity + isolation + thermal signal is documented
        # circumstantial evidence: the LF must VOTE even when the type is
        # unknown, but it never claims "type == vegetation".
        record = {
            "distance_to_forest_m": 900.0,
            "dist_industry": None,
            "distance_to_industry_m": SENTINEL_DISTANCE_M,
            "frp": 8.0,
            **missing_firms,
        }
        assert lf_forest_vegetation_fire(record) == FOREST_NATURAL_FIRE

    def test_known_non_vegetation_type_abstains_from_forest(self):
        # Volcano (2), static land source (3), offshore (4) are NEGATIVE evidence
        # for a vegetation claim even with forest proximity.
        for code in (2, 3, 4):
            record = {
                "distance_to_forest_m": 500.0,
                "distance_to_industry_m": SENTINEL_DISTANCE_M,
                "frp": 12.0,
                "firms_type": code,
            }
            assert lf_forest_vegetation_fire(record) == ABSTAIN

    def test_known_non_vegetation_type_abstains_from_agriculture(self):
        record = {
            "distance_to_agriculture_m": 400.0,
            "distance_to_industry_m": SENTINEL_DISTANCE_M,
            "frp": 3.0,
            "firms_type": 3,
        }
        assert lf_agriculture_vegetation_fire(record) == ABSTAIN

    def test_vegetation_type_within_agri_evidence_fires(self):
        record = {
            "distance_to_agriculture_m": 300.0,
            "distance_to_industry_m": SENTINEL_DISTANCE_M,
            "frp": 2.0,
            "firms_type": 0,
        }
        assert lf_agriculture_vegetation_fire(record) == AGRICULTURAL_BURN


# ---------------------------------------------------------------------------
# P1.5  LF density gate uses the REAL neighbourhood count
# ---------------------------------------------------------------------------
class TestIndustrialDensityGate:
    def test_real_count_authoritative(self):
        # Even though the legacy bucket says "code 3" (near industrial), the
        # real count of ONLY ONE site within 5 km is NOT dense -> abstain.
        record = {
            "distance_to_industry_m": 800.0,
            "frp": 8.0,
            "industrial_sites_within_5km": 1,
            "count_ind_5km": 3,
        }
        assert lf_industrial_zone_cluster(record) == ABSTAIN

    def test_real_count_triggers(self):
        record = {
            "distance_to_industry_m": 1200.0,
            "frp": 9.0,
            "industrial_sites_within_5km": 2,
            "count_ind_5km": 3,
        }
        assert lf_industrial_zone_cluster(record) == INDUSTRIAL_FIRE

    def test_legacy_fallback_when_real_count_column_absent(self):
        # Pre-Phase-B records (no real-count column) must still work via the
        # legacy bucket path, keeping old saved checks reproducible.
        record = {
            "distance_to_industry_m": 900.0,
            "frp": 7.0,
            "has_industrial_2km": 1,
            "count_ind_5km": 3,
        }
        assert lf_industrial_zone_cluster(record) == INDUSTRIAL_FIRE

    def test_low_frp_still_abstains(self):
        record = {
            "distance_to_industry_m": 900.0,
            "frp": 2.0,
            "industrial_sites_within_5km": 5,
        }
        assert lf_industrial_zone_cluster(record) == ABSTAIN


# ---------------------------------------------------------------------------
# P0.1  Batch producer emits REAL forest/agri distances
# ---------------------------------------------------------------------------
class TestBatchProducesRealForestAgri:
    def _firms_csv(self, tmp_dir):
        path = os.path.join(tmp_dir, "firms_test.csv")
        data = {
            "latitude":   [20.0000, 24.0000],
            "longitude":  [75.0000, 80.0000],
            "bright_ti4": [320.0, 330.0],
            "bright_ti5": [295.0, 300.0],
            "frp":        [10.0, 45.0],
            "confidence": ["h", "n"],
            "daynight":   ["D", "N"],
            "acq_date":   ["2026-09-01", "2026-09-01"],
            "acq_time":   [652, 652],
            "satellite":  ["N", "N"],
            "instrument": ["VIIRS", "VIIRS"],
            "source_dataset": ["VIIRS_SNPP_NRT"] * 2,
        }
        pd.DataFrame(data).to_csv(path, index=False)
        return path

    def _industrial_cache(self, tmp_dir):
        path = os.path.join(tmp_dir, "osm_industrial.json")
        payload = {
            "schema_version": "v2_with_forest_agri_mining",
            "sites": [
                {"id": 1, "osm_type": "node", "lat": 0.0, "lon": 0.0,
                 "tags": {"landuse": "industrial"}, "site_type": "industrial_zone",
                 "category": "industry", "name": "Far", "state": "dummy"},
            ],
        }
        json.dump(payload, open(path, "w"))
        return path

    def _forest_agri_cache(self, tmp_dir):
        path = os.path.join(tmp_dir, "osm_forest_agriculture.json")
        # Bare-list legacy format (matches data/raw/osm/osm_forest_agriculture.json):
        # ~180 m from hotspot #1 (20.0, 75.0) and ~12 km from it.
        json.dump(
            [
                {"id": 101, "name": "Test Reserve Forest", "category": "forest",
                 "lat": 20.0012, "lon": 75.0012, "state": "test"},
                {"id": 102, "name": "Test Farmland", "category": "agriculture",
                 "lat": 20.0500, "lon": 75.0500, "state": "test"},
            ],
            open(path, "w"),
        )
        return path

    def test_forest_agri_distances_are_real_not_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            from scripts.build_real_dataset import build_real_dataset
            firms = self._firms_csv(tmp)
            ind = self._industrial_cache(tmp)
            fa = self._forest_agri_cache(tmp)
            out = os.path.join(tmp, "classified_hotspots_v2.csv")
            build_real_dataset(
                firms_path=firms, output_path=out,
                osm_cache_path=ind, osm_forest_agri_cache_path=fa,
                use_live_api=False, verbose=False,
            )
            df = pd.read_csv(out)

            assert (df["distance_to_forest_m"] < 999000.0).any()
            # Hotspot #1 sits ~180 m from the forest reserve.
            assert df.loc[df["latitude"] == 20.0, "distance_to_forest_m"].iloc[0] < 1000.0
            assert (df["distance_to_agriculture_m"] < 15000.0).any()
            # The far hotspot stays sentinel.
            assert df.loc[df["latitude"] == 24.0, "distance_to_forest_m"].iloc[0] == 999000.0
            # Real counts present and populated.
            assert "count_forest_5km" in df.columns
            assert df.loc[df["latitude"] == 20.0, "count_forest_5km"].iloc[0] == 1
            # Legacy schema aliases intact.
            assert "dist_forest" in df.columns
            assert df["dist_forest"].nunique() > 1


# ---------------------------------------------------------------------------
# P1.3  Batch == live parity (single shared contract)
# ---------------------------------------------------------------------------
class TestBatchLiveParity:
    def test_live_service_uses_identical_spatial_features(self):
        from app.services.hotspot_service import _build_feature_record

        candidates = {
            "forest": [_mk_site(0.0041, 0.0, category="forest", distance_meters=900.0)],
            "industry": [_mk_site(0.0144, 0.0, tags={"landuse": "industrial"}, distance_meters=3200.0)],
        }
        feats = compute_spatial_features(0.0, 0.0, candidates)
        spot = {"frp": 6.0, "bright_ti4": 320.0, "bright_ti5": 299.0, "confidence": "n",
                "daynight": "D", "acq_date": "2026-09-01", "acq_time": 652, "satellite": "N"}
        record = _build_feature_record(spot, feats)

        assert record["distance_to_forest_m"] == feats["distance_to_forest_m"]
        assert record["distance_to_industry_m"] == feats["distance_to_industry_m"]
        assert record["count_forest_5km"] == feats["count_forest_5km"]
        assert record["industrial_sites_within_2km"] == feats["industrial_sites_within_2km"]
        assert record["has_forest_5km"] == feats["has_forest_5km"]
        assert record["dist_forest"] == feats["dist_forest"]

    def test_run_pipeline_build_feature_row_matches_shared_contract(self):
        from scripts import run_pipeline

        candidates = {
            "forest": [_mk_site(0.0041, 0.0, category="forest", distance_meters=900.0)],
            "industry": [_mk_site(0.0144, 0.0, tags={"landuse": "industrial"}, distance_meters=3200.0)],
        }
        feats = compute_spatial_features(0.0, 0.0, candidates)
        row = pd.Series({
            "frp": 6.0, "bright_ti4": 320.0, "bright_ti5": 299.0,
            "confidence": "n", "daynight": "D", "acq_date": "2026-09-01",
            "acq_time": 652, "satellite": "N",
            "dist_industry": feats["dist_industry"],
            "dist_factory": feats["dist_factory"],
            "dist_industrial_zone": feats["dist_industrial_zone"],
            "dist_refinery": feats["dist_refinery"],
            "dist_oil_gas": feats["dist_oil_gas"],
            "dist_mining": feats["dist_mining"],
            "dist_agriculture": feats["dist_agriculture"],
            "dist_forest": feats["dist_forest"],
            "dist_powerplant": feats["dist_powerplant"],
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
            "count_ind_5km": feats["count_ind_5km"],
            "count_ref_5km": feats["count_ref_5km"],
            "firms_type": None,
        })
        record = run_pipeline._build_feature_row(row)

        # Sentinel round-trips EXACTLY (999.0 km -> 999000.0 m).
        assert run_pipeline._km(999.0) == SENTINEL_DISTANCE_M == 999000.0
        # Real distances match within the km-alias rounding tolerance (<= 1 m).
        assert abs(record["distance_to_forest_m"] - feats["distance_to_forest_m"]) <= 1.0
        assert abs(record["distance_to_industry_m"] - feats["distance_to_industry_m"]) <= 1.0
        assert record["count_forest_5km"] == feats["count_forest_5km"]
        assert record["industrial_sites_within_5km"] == feats["industrial_sites_within_5km"]

    def test_km_sentinel_never_becomes_false_proximity(self):
        from scripts.run_pipeline import _km
        assert _km(999.0) == SENTINEL_DISTANCE_M        # km sentinel -> m sentinel
        assert _km("999.0") == SENTINEL_DISTANCE_M
        assert _km(999.5) > SPATIAL_EVIDENCE_INFLUENCE_M  # never folds into 45 km
        assert _km(None) is None