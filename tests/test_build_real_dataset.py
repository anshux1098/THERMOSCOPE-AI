"""
tests/test_build_real_dataset.py

Unit tests for the feature-encoding correctness of build_real_dataset.py.

Specifically catches Bug 1 (confidence numeric conversion) and Bug 2 (daynight int conversion)
that were introduced in the first version of build_real_dataset.py.

These tests verify that:
  - confidence is written as raw FIRMS strings ('h', 'n', 'l')
    so that get_confidence() in labeling_functions returns 'high'/'nominal'/'low'.
  - daynight is written as raw FIRMS strings ('D', 'N')
    so that is_night() in labeling_functions returns False/True correctly.

Run:
    pytest tests/test_build_real_dataset.py -v
    # or as part of the pre-training guard:
    python scripts/check_data_integrity.py
"""
import sys
import os
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

from app.intelligence.labeling_functions import get_confidence, is_night


# ---------------------------------------------------------------------------
# Bug 1 & 2: Confidence and Daynight Raw Pass-Through
# ---------------------------------------------------------------------------

class TestConfidenceEncoding:
    """
    Verifies that the confidence column in the output CSV preserves raw FIRMS
    strings ('h', 'n', 'l') so that get_confidence() works correctly.
    
    Bug was: _conf_to_numeric() converted 'h'->80.0, 'n'->50.0, 'l'->30.0.
    get_confidence() then received floats and returned 'nominal' for all values
    in the 30-70 range, breaking high/low detection.
    """

    def test_high_confidence_string(self):
        """'h' raw string -> get_confidence() returns 'high'"""
        row = {"confidence": "h"}
        assert get_confidence(row) == "high", (
            "confidence='h' should map to 'high' in get_confidence(). "
            "Check that build_real_dataset.py passes raw 'h' not numeric 80."
        )

    def test_nominal_confidence_string(self):
        """'n' raw string -> get_confidence() returns 'nominal'"""
        row = {"confidence": "n"}
        assert get_confidence(row) == "nominal", (
            "confidence='n' should map to 'nominal' in get_confidence(). "
            "Check that build_real_dataset.py passes raw 'n' not numeric 50."
        )

    def test_low_confidence_string(self):
        """'l' raw string -> get_confidence() returns 'low'"""
        row = {"confidence": "l"}
        assert get_confidence(row) == "low", (
            "confidence='l' should map to 'low' in get_confidence(). "
            "Check that build_real_dataset.py passes raw 'l' not numeric 30."
        )

    def test_all_three_confidence_values_reachable(self):
        """All 3 confidence outputs ('high', 'nominal', 'low') must be reachable."""
        results = {
            get_confidence({"confidence": "h"}),
            get_confidence({"confidence": "n"}),
            get_confidence({"confidence": "l"}),
        }
        assert results == {"high", "nominal", "low"}, (
            f"Expected {{'high', 'nominal', 'low'}}, got {results}. "
            "All 3 confidence levels must be reachable via raw FIRMS strings."
        )

    def test_numeric_80_is_NOT_what_we_want(self):
        """
        Regression test: documents why passing numeric was wrong.
        The old bug encoded 'l' as 30.0. get_confidence() checks:
          val < 0.3 -> 'low'  (but 30.0 > 0.3, so this check FAILS)
          val > 0.7 -> 'high' (but 30.0 > 0.7, so it returned 'HIGH' for 'l'!)
        So 'l' hotspots got mislabeled as 'high' confidence.
        """
        # 80.0 (old encoding of 'h') -> returns 'high' (accidentally correct)
        assert get_confidence({"confidence": 80.0}) == "high"
        # 30.0 (old encoding of 'l') -> 30.0 > 0.7 is True -> returns 'HIGH'! BUG.
        result_for_old_l = get_confidence({"confidence": 30.0})
        assert result_for_old_l == "high", (
            "Confirmed the old bug: 30.0 (old encoding of 'l') maps to 'high' "
            "because get_confidence checks val > 0.7 and 30.0 > 0.7 is True. "
            "The fix is to pass raw 'l' string instead."
        )


class TestDayNightEncoding:
    """
    Verifies that the daynight column in the output CSV preserves raw FIRMS
    strings ('D', 'N') so that is_night() works correctly.
    
    Bug was: _daynight_to_int() converted 'D'->1, 'N'->0.
    is_night() then received integers. is_night() checks:
        if isinstance(val, (int, float)): return val >= 0.5
    So 1 -> True (day detected as night!) and 0 -> False (night detected as day!).
    Both day and night were INVERTED.
    """

    def test_D_is_not_night(self):
        """'D' raw string -> is_night() returns False (daytime)"""
        row = {"daynight": "D"}
        result = is_night(row)
        assert result is False, (
            "daynight='D' should return is_night()=False (it's daytime). "
            "Check that build_real_dataset.py passes raw 'D' not integer 1. "
            f"Got: {result!r}"
        )

    def test_N_is_night(self):
        """'N' raw string -> is_night() returns True (nighttime)"""
        row = {"daynight": "N"}
        result = is_night(row)
        assert result is True, (
            "daynight='N' should return is_night()=True (it's nighttime). "
            "Check that build_real_dataset.py passes raw 'N' not integer 0. "
            f"Got: {result!r}"
        )

    def test_integer_1_inversion_bug(self):
        """
        Regression: integer 1 (old 'D' encoding) was INCORRECTLY treated as night.
        Documents the exact bug: is_night({daynight: 1}) == True (wrong!).
        """
        row_with_int_1 = {"daynight": 1}  # old encoding of 'D'
        result = is_night(row_with_int_1)
        # 1 >= 0.5 is True -> is_night says True for what was Day -> BUG
        assert result is True, (
            "This confirms the old bug: integer 1 (old encoding of 'D') "
            "was incorrectly treated as night by is_night(). "
            "The fix is to pass raw 'D' string."
        )

    def test_integer_0_inversion_bug(self):
        """
        Regression: integer 0 (old 'N' encoding) was INCORRECTLY treated as daytime.
        Documents the exact bug: is_night({daynight: 0}) == False (wrong!).
        """
        row_with_int_0 = {"daynight": 0}  # old encoding of 'N'
        result = is_night(row_with_int_0)
        # 0 >= 0.5 is False -> is_night says False for what was Night -> BUG
        assert result is False, (
            "This confirms the old bug: integer 0 (old encoding of 'N') "
            "was incorrectly treated as daytime by is_night(). "
            "The fix is to pass raw 'N' string."
        )

    def test_both_daynight_values_distinct(self):
        """D and N must produce different is_night() results."""
        result_d = is_night({"daynight": "D"})
        result_n = is_night({"daynight": "N"})
        assert result_d != result_n, (
            f"D -> {result_d}, N -> {result_n}: they should differ. "
            "If they are the same, the encoding is broken."
        )


# ---------------------------------------------------------------------------
# Integration: build_real_dataset output columns are correct types
# ---------------------------------------------------------------------------

class TestBuildRealDatasetOutputFormat:
    """
    Smoke-tests that build_real_dataset.py writes confidence/daynight as
    strings (not ints/floats) in the output CSV.
    
    We do this without calling Overpass by mocking the OSM cache as empty,
    so all distances will be sentinel and all labeling functions will ABSTAIN.
    """

    def _make_minimal_firms_csv(self, tmp_dir: str) -> str:
        """Create a tiny FIRMS CSV with known confidence/daynight values."""
        path = os.path.join(tmp_dir, "firms_test.csv")
        data = {
            "latitude":   [20.0, 20.1, 20.2],
            "longitude":  [78.0, 78.1, 78.2],
            "bright_ti4": [320.0, 330.0, 315.0],
            "bright_ti5": [295.0, 300.0, 290.0],
            "frp":        [10.0, 45.0, 5.0],
            "confidence": ["h", "n", "l"],
            "daynight":   ["D", "N", "D"],
            "acq_date":   ["2026-09-01", "2026-09-01", "2026-09-01"],
            "acq_time":   [652, 652, 652],
            "satellite":  ["N", "N", "N"],
            "instrument": ["VIIRS", "VIIRS", "VIIRS"],
            "source_dataset": ["VIIRS_SNPP_NRT"] * 3,
        }
        pd.DataFrame(data).to_csv(path, index=False)
        return path

    def _make_minimal_osm_cache(self, tmp_dir: str) -> str:
        """
        Create a minimal OSM cache with a single dummy industrial site
        placed far from the test coordinates so no hotspot gets a real hit.
        This avoids FileNotFoundError while keeping all LF distances at sentinel.
        """
        import json
        path = os.path.join(tmp_dir, "osm_sites.json")
        dummy_site = {
            "id": 1, "osm_type": "node",
            "lat": 0.0, "lon": 0.0,  # Far from test coords (20.x, 78.x)
            "tags": {"landuse": "industrial"}, "site_type": "industrial_zone",
            "category": "industry", "name": "Dummy Far Site", "state": "dummy",
        }
        payload = {
            "schema_version": "v2_with_forest_agri_mining",
            "sites": [dummy_site],
        }
        with open(path, "w") as f:
            json.dump(payload, f)
        return path

    def test_confidence_written_as_string(self):
        """confidence column must contain strings 'h'/'n'/'l', never floats."""
        with tempfile.TemporaryDirectory() as tmp:
            from scripts.build_real_dataset import build_real_dataset
            firms = self._make_minimal_firms_csv(tmp)
            osm   = self._make_minimal_osm_cache(tmp)
            out   = os.path.join(tmp, "classified_hotspots_v2.csv")
            build_real_dataset(
                firms_path=firms,
                output_path=out,
                osm_cache_path=osm,
                use_live_api=False,
                verbose=False,
            )
            df = pd.read_csv(out, dtype={"confidence": str, "daynight": str})
            conf_values = set(df["confidence"].str.strip().str.lower().unique())
            assert conf_values.issubset({"h", "n", "l"}), (
                f"confidence column has unexpected values: {conf_values}. "
                "Expected only raw strings 'h', 'n', 'l'."
            )

    def test_daynight_written_as_string(self):
        """daynight column must contain strings 'D'/'N', never integers."""
        with tempfile.TemporaryDirectory() as tmp:
            from scripts.build_real_dataset import build_real_dataset
            firms = self._make_minimal_firms_csv(tmp)
            osm   = self._make_minimal_osm_cache(tmp)
            out   = os.path.join(tmp, "classified_hotspots_v2.csv")
            build_real_dataset(
                firms_path=firms,
                output_path=out,
                osm_cache_path=osm,
                use_live_api=False,
                verbose=False,
            )
            df = pd.read_csv(out, dtype={"confidence": str, "daynight": str})
            dn_values = set(df["daynight"].str.strip().str.upper().unique())
            assert dn_values.issubset({"D", "N"}), (
                f"daynight column has unexpected values: {dn_values}. "
                "Expected only raw strings 'D' or 'N'."
            )

    def test_no_synthetic_demo_marker(self):
        """Real dataset must never have _is_synthetic_demo column."""
        with tempfile.TemporaryDirectory() as tmp:
            from scripts.build_real_dataset import build_real_dataset
            firms = self._make_minimal_firms_csv(tmp)
            osm   = self._make_minimal_osm_cache(tmp)
            out   = os.path.join(tmp, "classified_hotspots_v2.csv")
            build_real_dataset(
                firms_path=firms,
                output_path=out,
                osm_cache_path=osm,
                use_live_api=False,
                verbose=False,
            )
            df = pd.read_csv(out)
            assert "_is_synthetic_demo" not in df.columns, (
                "Real dataset must not contain _is_synthetic_demo column."
            )
