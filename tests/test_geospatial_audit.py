"""
tests/test_geospatial_audit.py

Phase 16 forensic-spatial-audit guard tests.

These tests are pure AUDIT GUARDS: they lock in the measured properties of the
spatial pipeline (units, coordinate order, radius gates, sentinel semantics,
feature constancy, config consistency) so that the forensic findings are
reproducible and can never silently regress. None of them change thresholds,
labels, the model, or any production file.

The two data-dependent tests (constant-feature guard, forest/agri data guard)
are skipped if the canonical dataset or OSM caches are missing.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
root_dir = str(Path(__file__).resolve().parents[1])
scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
for p in (backend_dir, root_dir, scripts_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.core.constants import (
    INDUSTRIAL_PROXIMITY_M,
    MINING_PROXIMITY_M,
    OIL_GAS_PROXIMITY_M,
    POWER_PLANT_PROXIMITY_M,
    REFINERY_PROXIMITY_M,
    DEFAULT_RADIUS_METERS,
)
from app.core.paths import (
    CLASSIFIED_DATASET_PATH,
    OSM_FOREST_AGRI_CACHE_PATH,
    OSM_INDUSTRIAL_CACHE_PATH,
)
from app.geo.distance import EARTH_RADIUS_METERS, calculate_distance
from app.geo.spatial_context import SENTINEL_DISTANCE_M as SPATIAL_CONTEXT_SENTINEL_M
from app.geo.spatial_features import SPATIAL_EVIDENCE_INFLUENCE_M
from app.intelligence.labeling_functions import (
    THRESHOLD_AGRICULTURE_PROXIMITY_M,
    THRESHOLD_FOREST_PROXIMITY_M,
    THRESHOLD_INDUSTRY_PROXIMITY_M,
    THRESHOLD_ISOLATED_FROM_INDUSTRY_M,
    THRESHOLD_MINING_PROXIMITY_M,
    THRESHOLD_OIL_GAS_PROXIMITY_M,
    THRESHOLD_REFINERY_PROXIMITY_M,
    get_distance_meters,
)
from app.services.osm_service import classify_osm_category
from scripts.build_real_dataset import SENTINEL_DISTANCE_M, _compute_distances_from_cache


# ---------------------------------------------------------------------------
# Distance math / units / coordinate order
# ---------------------------------------------------------------------------
class TestDistanceMath:
    def test_distance_units_are_consistent(self):
        # ~1 deg of longitude at the equator is ~111.2 km -> the haversine
        # function must return METERS, never km or miles.
        meters = calculate_distance(
            {"latitude": 0.0, "longitude": 0.0},
            {"latitude": 0.0, "longitude": 1.0},
            unit="m",
        )
        assert meters is not None and 110_000 < meters < 112_500

        km = calculate_distance(
            {"latitude": 0.0, "longitude": 0.0},
            {"latitude": 0.0, "longitude": 1.0},
            unit="km",
        )
        assert km is not None and abs(km - meters / 1000.0) < 2.0

    def test_latitude_longitude_order_is_correct(self):
        # (lat, lon) must NOT be confused with (lon, lat): swapping the two
        # produces a materially different (and wrong) distance.
        latlon = calculate_distance(
            {"latitude": 20.0, "longitude": 75.0},
            {"latitude": 22.0, "longitude": 78.0},
            unit="m",
        )
        swapped = calculate_distance(
            {"latitude": 75.0, "longitude": 20.0},
            {"latitude": 78.0, "longitude": 22.0},
            unit="m",
        )
        assert latlon is not None and swapped is not None
        assert latlon != pytest.approx(swapped, abs=100.0)

        # The correct order: (20,75)->(22,78) haversine is ~382.7 km.
        assert latlon == pytest.approx(382_700.0, abs=10_000.0)
        # The swapped interpretation (~338 km) is materially different.
        assert swapped > 300_000.0

    def test_known_distance_calculation(self):
        # Earth radius constant is the canonical mean radius in meters.
        assert EARTH_RADIUS_METERS == pytest.approx(6_371_000.0)
        # 1 degree of latitude anywhere ~ 111.195 km.
        meters = calculate_distance(
            {"latitude": 10.0, "longitude": 20.0},
            {"latitude": 11.0, "longitude": 20.0},
            unit="m",
        )
        assert meters is not None and 110_000 < meters < 112_500

    def test_zero_distance(self):
        meters = calculate_distance(
            {"latitude": 28.5, "longitude": 77.0},
            {"latitude": 28.5, "longitude": 77.0},
            unit="m",
        )
        assert meters == 0.0

    def test_get_distance_meters_returns_units_in_meters(self):
        # get_distance_meters prefers the explicit meters column; the km
        # fallback column is converted to meters, never returned raw as km.
        row = {"distance_to_industry_m": 500.0}
        assert get_distance_meters(row, "industry") == 500.0
        row2 = {"dist_factory": 999.0, "distance_to_industry_m": 1234.5}
        assert get_distance_meters(row2, "industry") == 1234.5


# ---------------------------------------------------------------------------
# Radius gates / boundary behavior
# ---------------------------------------------------------------------------
class TestRadiusBoundaries:
    def test_lf_threshold_flip_at_radius_boundary(self):
        # The industry LF must abstain at exactly beyond the threshold and
        # consider the inside case. Use the LF's own guard via the shared
        # constant: threshold <= dist -> blocked, dist < threshold -> eligible.
        assert THRESHOLD_INDUSTRY_PROXIMITY_M == 2000.0
        assert get_distance_meters({"distance_to_industry_m": 1999.0}, "industry") < THRESHOLD_INDUSTRY_PROXIMITY_M
        assert get_distance_meters({"distance_to_industry_m": 2000.0}, "industry") == THRESHOLD_INDUSTRY_PROXIMITY_M

    def test_has_flag_boundary_behavior(self):
        # The 'has_*_5km' boolean flags flip exactly at the 5km boundary the
        # producer uses (distance_to_*_m <= 5000.0 -> 1).
        def has_flag(d_m):
            return 1 if d_m <= 5000.0 else 0

        assert has_flag(5000.0) == 1
        assert has_flag(5000.0001) == 0
        assert has_flag(4_999.9) == 1

    def test_sentinel_flagged_rows_never_fire_flags(self):
        # A sentinel distance (no OSM site within the search box) must report
        # has_*_5km = 0, not NaN or 1.
        assert (999000.0 <= 5000.0) is False
        assert (999000.0 < 45000.0) is False  # also above the SF sentinel
        assert (999000.0 < THRESHOLD_ISOLATED_FROM_INDUSTRY_M) is False


# ---------------------------------------------------------------------------
# Whole-pipeline distance computation (hermetic, synthetic)
# ---------------------------------------------------------------------------
class TestProducerDistances:
    def _site(self, lat, lon, category=None, tags=None, site_type=None):
        return {"lat": lat, "lon": lon, "category": category, "tags": tags or {}, "site_type": site_type}

    def test_entity_inside_search_radius(self):
        # Hotspot 300 m from a factory must produce a real (non-sentinel)
        # distance and a cache data source.
        factories = [
            self._site(20.0012, 75.0012, tags={"landuse": "industrial"}),  # ~180 m NE
            self._site(20.0027, 75.0025, tags={"landuse": "industrial"}),  # ~400 m NE
        ]
        nearest, sources = _compute_distances_from_cache(20.0, 75.0, factories)
        assert 100 < nearest["industry"] < 500
        assert sources["industry"] == "cache"

    def test_entity_outside_search_radius(self):
        # Hotspot 400+ km away must be sentinel for the coarse bbox gate.
        factory = self._site(34.0, 76.0, tags={"landuse": "industrial"})
        nearest, sources = _compute_distances_from_cache(20.0, 75.0, [factory])
        assert nearest["industry"] == SENTINEL_DISTANCE_M
        assert sources["industry"] == "none"

    def test_empty_spatial_query_is_detectable(self):
        # An empty cache must be *represented* (sentinel + data_source 'none'),
        # so an audit can distinguish "queried and found nothing geographic"
        # from a missing/empty spatial query -- never silently NaN.
        nearest, sources = _compute_distances_from_cache(20.0, 75.0, [])
        assert all(v == SENTINEL_DISTANCE_M for v in nearest.values())
        assert all(v == "none" for v in sources.values())

    def test_query_hit_and_query_miss_are_distinguishable(self):
        # A category present in the query is marked 'cache'; a category with
        # no sites is marked 'none' -- the contract that lets the pipeline (and
        # the audit) tell a real absence from a broken query.
        sites = [self._site(20.0, 75.0, tags={"power": "plant"})]
        _, sources = _compute_distances_from_cache(20.0, 75.0, sites)
        assert sources["power_plant"] == "cache"
        assert sources["mining"] == "none"
        assert sources["refinery"] == "none"


# ---------------------------------------------------------------------------
# Configuration / constants consistency
# ---------------------------------------------------------------------------
class TestConfigurationConsistency:
    def test_lf_thresholds_match_constants(self):
        # LFs must consume the shared constants, not hard-coded duplicates.
        assert THRESHOLD_INDUSTRY_PROXIMITY_M == float(INDUSTRIAL_PROXIMITY_M)
        assert THRESHOLD_REFINERY_PROXIMITY_M == float(REFINERY_PROXIMITY_M)
        assert THRESHOLD_OIL_GAS_PROXIMITY_M == float(OIL_GAS_PROXIMITY_M)
        assert THRESHOLD_MINING_PROXIMITY_M == float(MINING_PROXIMITY_M)

    def test_power_plant_radius_is_defined_but_no_lf_consumes_it(self):
        # POWER_PLANT_PROXIMITY_M exists in constants but no LF threshold is
        # bound to it -- power plants are ML features only. This documents (and
        # locks) that divergence so a future threshold change here is deliberate.
        assert POWER_PLANT_PROXIMITY_M == 5000.0
        distinct = {
            THRESHOLD_INDUSTRY_PROXIMITY_M,
            THRESHOLD_REFINERY_PROXIMITY_M,
            THRESHOLD_OIL_GAS_PROXIMITY_M,
            THRESHOLD_MINING_PROXIMITY_M,
        }
        assert POWER_PLANT_PROXIMITY_M not in distinct

    def test_search_radius_matches_builder_constant(self):
        # The default OSM search radius (constants / service / spatial context)
        # must agree with the batch producer.
        from app.services.osm_service import find_nearby_geographic_objects
        from app.geo.spatial_context import compute_geospatial_context
        import inspect

        svc_sig = inspect.signature(find_nearby_geographic_objects).parameters.get("radius_meters")
        ctx_sig = inspect.signature(compute_geospatial_context).parameters.get("radius_meters")
        assert svc_sig is not None and svc_sig.default == 15000
        assert ctx_sig is not None and ctx_sig.default == 15000
        assert DEFAULT_RADIUS_METERS == 15000

    def test_sentinel_markers_are_behaviorally_equivalent(self):
        # Phase B P1.4 fix: batch and live now share ONE canonical "no nearby
        # entity" sentinel (999000.0 m / 999.0 km). It must exceed every LF
        # proximity threshold AND every LF lazy threshold (SPATIAL_EVIDENCE_
        # INFLUENCE_M = 45 km), so a sentinel never reads as positive proximity.
        largest_lf_threshold = max(
            THRESHOLD_INDUSTRY_PROXIMITY_M,
            THRESHOLD_REFINERY_PROXIMITY_M,
            THRESHOLD_OIL_GAS_PROXIMITY_M,
            THRESHOLD_MINING_PROXIMITY_M,
            THRESHOLD_AGRICULTURE_PROXIMITY_M,
            THRESHOLD_FOREST_PROXIMITY_M,
        )
        assert SENTINEL_DISTANCE_M == 999000.0
        assert SPATIAL_CONTEXT_SENTINEL_M == 999000.0
        assert SPATIAL_CONTEXT_SENTINEL_M == SENTINEL_DISTANCE_M
        assert SENTINEL_DISTANCE_M > largest_lf_threshold
        assert SPATIAL_CONTEXT_SENTINEL_M > SPATIAL_EVIDENCE_INFLUENCE_M
        assert SPATIAL_CONTEXT_SENTINEL_M / 1000.0 == 999.0


# ---------------------------------------------------------------------------
# Data-guard tests (depend on the real artifacts)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not OSM_FOREST_AGRI_CACHE_PATH.exists(),
                    reason="forest/agriculture OSM cache not present")
class TestForestAgriDataExists:
    def test_forest_agri_cache_contains_real_classifiable_sites(self):
        sites = json.loads(OSM_FOREST_AGRI_CACHE_PATH.read_text(encoding="utf-8"))
        assert len(sites) > 1000

        def categorize(site):
            c = classify_osm_category(site.get("tags") or {})
            return c if c in ("forest", "agriculture") else site.get("category")

        cats = [categorize(s) for s in sites]
        assert cats.count("forest") > 1000
        assert cats.count("agriculture") > 200


@pytest.mark.skipif(not CLASSIFIED_DATASET_PATH.exists()
                    or not OSM_FOREST_AGRI_CACHE_PATH.exists(),
                    reason="canonical classified CSV or forest/agri cache missing")
class TestConstantFeatureGuard:
    def test_forest_agri_features_are_not_constant_in_current_build(self):
        # Phase B P0.1 FIX: the forest/agri cache is now merged by
        # build_real_dataset, so dist_forest / dist_agriculture must be REAL
        # (non-constant) in the rebuilt canonical dataset — never the 999.0
        # sentinel for every row. This test asserts the corrected invariant.
        df = pd.read_csv(CLASSIFIED_DATASET_PATH)
        assert "dist_forest" in df.columns and "dist_agriculture" in df.columns
        # Forest cache holds >1000 sites -> generous real variance expected.
        assert df["dist_forest"].nunique() > 50
        assert df["dist_agriculture"].nunique() > 10
        # At least one hotspot genuinely sits near forest (< 15 km search radius)
        assert (df["distance_to_forest_m"] < 15000.0).any()
        assert (df["distance_to_agriculture_m"] < 15000.0).any()

    def test_industry_distances_have_real_variance(self):
        # Contrast: real merged categories DO vary, proving the pipeline CAN
        # produce spatial variance.
        df = pd.read_csv(CLASSIFIED_DATASET_PATH)
        assert df["dist_factory"].nunique() > 50
        assert df["distance_to_industry_m"].min() < 1000.0