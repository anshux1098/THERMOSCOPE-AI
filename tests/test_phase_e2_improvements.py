"""
tests/test_phase_e2_improvements.py

Phase E2 — Targeted Evidence-Based Labeling Improvements (SIH26162).

Regression tests locked to the Phase E1 audit findings:

  E2-A  Forest FRP gate recalibrated 5.0 -> FRP_FOREST_MODERATE_MW (2.0).
        lf_strong_forest_fire remains an intentionally rare high-intensity
        rule bound to FRP_VERY_HIGH_MW (15.0) - no hard-coded duplicate.
  E2-B  Agriculture comparative evidence policy is already the
        `dist_agri < dist_ind` branch in the agriculture LFs. Measured:
        ZERO rows are recoverable (all 33 blocked rows have industry
        substantially CLOSER than agriculture), so the isolation rule is
        preserved. Literal thresholds were centralized into named constants
        (behavior unchanged).
  E2-C  New lf_power_plant_process_heat: power-plant proximity is supporting
        evidence for industrial_process_heat ONLY when the plant is the
        nearest industrial-adjacent entity AND a supporting thermal signature
        exists (FRP >= FRP_PROCESS_HEAT_MIN_MW, brightness >=
        BRIGHTNESS_MODERATE_K). Never a proximity-only / auto-fire label.
  E2-D  Conservative industry calibration: low-FRP AND low-brightness rows
        near industry MUST keep abstaining (guarded policy). Measured: every
        guarded combination recovers ZERO rows among the unclassified
        industrial cohort, so no global FRP lowering was applied.

Run:
    pytest tests/test_phase_e2_improvements.py -v
"""
import sys
from pathlib import Path

backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
root_dir = str(Path(__file__).resolve().parents[1])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest

from app.core.constants import POWER_PLANT_PROXIMITY_M
from app.geo.spatial_features import SPATIAL_EVIDENCE_INFLUENCE_M, SENTINEL_DISTANCE_M
from app.intelligence.labeling_functions import (
    ABSTAIN,
    AGRICULTURAL_BURN,
    FOREST_NATURAL_FIRE,
    INDUSTRIAL_FIRE,
    INDUSTRIAL_PROCESS_HEAT,
    BRIGHTNESS_MODERATE_K,
    BRIGHTNESS_STUBBLE_CEILING_K,
    FRP_FOREST_MODERATE_MW,
    FRP_MODERATE_MW,
    FRP_PROCESS_HEAT_MIN_MW,
    FRP_VERY_HIGH_MW,
    lf_agriculture_burn_context,
    lf_agriculture_vegetation_fire,
    lf_factory_proximity_thermal,
    lf_forest_vegetation_fire,
    lf_industry_high_frp,
    lf_industrial_zone_cluster,
    lf_power_plant_process_heat,
    lf_strong_forest_fire,
)


# ---------------------------------------------------------------------------
# E2-A  Forest calibration
# ---------------------------------------------------------------------------
class TestE2AForestCalibration:
    def test_forest_frp_gate_uses_named_constant_not_literal(self):
        # E2-A locks the calibrated 2 MW gate through a shared constant
        # (controls "no duplicate magic numbers" / "no silent threshold changes").
        assert FRP_FOREST_MODERATE_MW == 2.0
        assert FRP_FOREST_MODERATE_MW < FRP_MODERATE_MW

    def test_forest_fires_above_calibrated_gate(self):
        record = {
            "distance_to_forest_m": 900.0,
            "distance_to_industry_m": SENTINEL_DISTANCE_M,
            "frp": FRP_FOREST_MODERATE_MW,  # 2.0 MW: was ABSTAIN pre-E2
        }
        assert lf_forest_vegetation_fire(record) == FOREST_NATURAL_FIRE

    def test_forest_abstains_below_calibrated_gate(self):
        # 1.0 MW stays below the noise floor -> ABSTAIN, never force-labeled.
        record = {
            "distance_to_forest_m": 900.0,
            "distance_to_industry_m": SENTINEL_DISTANCE_M,
            "frp": 1.0,
        }
        assert lf_forest_vegetation_fire(record) == ABSTAIN

    def test_forest_still_requires_industry_isolation_at_low_frp(self):
        # The mandatory isolation gate is NOT weakened by the calibration.
        record = {
            "distance_to_forest_m": 800.0,
            "distance_to_industry_m": 400.0,   # industry much closer
            "frp": 4.0,
        }
        assert lf_forest_vegetation_fire(record) == ABSTAIN

    def test_strong_forest_fire_uses_shared_very_high_constant(self):
        # LF 11 is the intentionally rare high-intensity cap rule.
        low = {
            "distance_to_forest_m": 500.0,
            "distance_to_industry_m": SENTINEL_DISTANCE_M,
            "distance_to_agriculture_m": SENTINEL_DISTANCE_M,
            "frp": 10.0,
        }
        assert lf_strong_forest_fire(low) == ABSTAIN

        high = {
            "distance_to_forest_m": 500.0,
            "distance_to_industry_m": SENTINEL_DISTANCE_M,
            "distance_to_agriculture_m": SENTINEL_DISTANCE_M,
            "frp": FRP_VERY_HIGH_MW,  # >= 15 MW unambiguous wildfire signal
        }
        assert lf_strong_forest_fire(high) == FOREST_NATURAL_FIRE


# ---------------------------------------------------------------------------
# E2-B  Agriculture comparative evidence policy
# ---------------------------------------------------------------------------
class TestE2BAgricultureComparativeEvidence:
    def test_agri_closer_than_industry_fires(self):
        # Comparative evidence: agriculture strictly closer than industry.
        record = {
            "distance_to_agriculture_m": 1200.0,
            "distance_to_industry_m": 4000.0,
            "frp": 3.0,
        }
        assert lf_agriculture_vegetation_fire(record) == AGRICULTURAL_BURN

    def test_industry_closer_than_agri_abstains(self):
        # Measured population: ALL 33 blocked agri-close rows have industry
        # 2.9x-22.9x closer. They stay unclassified (no forced agri label).
        record = {
            "distance_to_agriculture_m": 4000.0,
            "distance_to_industry_m": 650.0,  # industry ~6x closer
            "frp": 2.6,
        }
        assert lf_agriculture_vegetation_fire(record) == ABSTAIN
        assert lf_agriculture_burn_context(record) == ABSTAIN

    def test_agri_context_uses_named_constants(self):
        # E2-B centralized the stubble-band + isolation literals without
        # changing behavior (5.0 -> FRP_MODERATE_MW, 345.0 -> ceiling,
        # 1500.0 -> THRESHOLD_ISOLATED_FROM_INDUSTRY_M).
        assert FRP_MODERATE_MW == 5.0
        assert BRIGHTNESS_STUBBLE_CEILING_K == 345.0


# ---------------------------------------------------------------------------
# E2-C  Power-plant supporting evidence
# ---------------------------------------------------------------------------
class TestE2CPowerPlantEvidence:
    def test_pp_nearest_with_supporting_thermal_fires(self):
        # Documented E2 candidate: day detection, FRP 3-5 MW, brightness
        # 330-341 K, power plant the nearest industrial-adjacent entity.
        record = {
            "distance_to_power_plant_m": 2000.0,
            "distance_to_industry_m": 4000.0,
            "distance_to_refinery_m": SENTINEL_DISTANCE_M,
            "distance_to_mining_m": SENTINEL_DISTANCE_M,
            "frp": 4.0,
            "bright_ti4": 335.0,
        }
        assert lf_power_plant_process_heat(record) == INDUSTRIAL_PROCESS_HEAT

    def test_pp_too_far_abstains(self):
        record = {
            "distance_to_power_plant_m": POWER_PLANT_PROXIMITY_M + 1000.0,
            "distance_to_industry_m": SENTINEL_DISTANCE_M,
            "frp": 6.0,
            "bright_ti4": 340.0,
        }
        assert lf_power_plant_process_heat(record) == ABSTAIN

    def test_industry_closer_than_pp_abstains(self):
        # Identity protection: when industry is nearer, the industrial fire /
        # process-heat semantics own the record. Power plant must be strictly
        # the nearest industrial-adjacent evidence.
        record = {
            "distance_to_power_plant_m": 1500.0,
            "distance_to_industry_m": 300.0,
            "distance_to_refinery_m": SENTINEL_DISTANCE_M,
            "distance_to_mining_m": SENTINEL_DISTANCE_M,
            "frp": 6.0,
            "bright_ti4": 335.0,
        }
        assert lf_power_plant_process_heat(record) == ABSTAIN

    def test_no_supporting_thermal_abstains(self):
        # Proximity alone is NEVER an auto-fire label (supporting evidence
        # requires FRP + brightness).
        record = {
            "distance_to_power_plant_m": 800.0,
            "distance_to_industry_m": 3000.0,
            "distance_to_refinery_m": SENTINEL_DISTANCE_M,
            "distance_to_mining_m": SENTINEL_DISTANCE_M,
            "frp": 1.0,
            "bright_ti4": 300.0,
        }
        assert lf_power_plant_process_heat(record) == ABSTAIN

    def test_known_non_static_type_abstains(self):
        # firms_type 0 (vegetation) / 2 (volcano) / 4 (offshore) are negative
        # evidence for a process-heat claim.
        for code in (0, 2, 4):
            record = {
                "distance_to_power_plant_m": 1000.0,
                "distance_to_industry_m": 4000.0,
                "firms_type": code,
                "frp": 6.0,
                "bright_ti4": 335.0,
            }
            assert lf_power_plant_process_heat(record) == ABSTAIN


# ---------------------------------------------------------------------------
# E2-D  Conservative industry guarded policy
# ---------------------------------------------------------------------------
class TestE2DIndustryGuardedPolicy:
    def test_low_frp_and_low_brightness_near_industry_abstain(self):
        # Measured null result: the entire unclassified industrial cohort is
        # night, FRP < 5, brightness < 310 K. NO guarded combination with the
        # existing gates (proximity + density + confidence + brightness)
        # recovers a single row, so no global FRP lowering was applied.
        record = {
            "distance_to_industry_m": 800.0,
            "frp": 2.2,
            "bright_ti4": 304.0,
            "confidence": "n",
            "industrial_sites_within_2km": 5,
            "industrial_sites_within_5km": 8,
            "daynight": "N",
        }
        assert lf_industry_high_frp(record) == ABSTAIN
        assert lf_factory_proximity_thermal(record) == ABSTAIN
        assert lf_industrial_zone_cluster(record) == ABSTAIN

    def test_industry_lfs_still_fire_with_full_signal(self):
        # Guarded policy keeps precision: moderate FRP + elevated brightness +
        # density still fire (unchanged from Phase B).
        record = {
            "distance_to_industry_m": 900.0,
            "frp": 8.0,
            "bright_ti4": 340.0,
            "confidence": "h",
            "industrial_sites_within_5km": 3,
        }
        assert lf_industry_high_frp(record) == INDUSTRIAL_FIRE
        assert lf_factory_proximity_thermal(record) == INDUSTRIAL_FIRE
        assert lf_industrial_zone_cluster(record) == INDUSTRIAL_FIRE