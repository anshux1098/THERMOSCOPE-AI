"""
tests/test_pipeline_cli.py

Phase E run_pipeline CLI behavior tests (resume / force / limit / dry-run).

The classifier is mocked so the tests verify pipeline mechanics
(row selection, skip logic, merge, dedupe) deterministically and
without requiring trained ML artifacts.
"""
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

from run_pipeline import RESUME_KEY, process_hotspots


def _canned_result() -> dict:
    return {
        "final_label": "industrial_fire",
        "hybrid_confidence": 0.82,
        "raw_ml_confidence": 0.71,
        "confidence_level": "high",
        "decision_source": "rules_and_ml_agree",
        "agreement": True,
        "conflict": False,
        "requires_human_review": False,
        "review_reason": None,
        "rule_engine": {"prediction": "industrial_fire", "active_votes": 5},
        "ml_engine": {"prediction": "industrial_fire", "confidence": 0.71},
        "explanation": ["High-intensity thermal signature (FRP: 80.0 MW)."],
    }


@pytest.fixture
def mock_classifier(monkeypatch):
    import run_pipeline
    monkeypatch.setattr(run_pipeline, "classify_hotspot",
                        lambda features: _canned_result())


@pytest.fixture
def input_csv(tmp_path) -> Path:
    path = tmp_path / "classified_hotspots_v2.csv"
    rows = 20
    df = pd.DataFrame(
        {
            "latitude": [20.0 + i for i in range(rows)],
            "longitude": [75.0 + i for i in range(rows)],
            "frp": [40.0] * rows,
            "dist_refinery": [100.0] * rows,
            "dist_factory": [100.0] * rows,
            "dist_industrial_zone": [100.0] * rows,
            "dist_oil_gas": [100.0] * rows,
            "dist_mining": [100.0] * rows,
            "dist_forest": [100.0] * rows,
            "dist_agriculture": [100.0] * rows,
            "dist_powerplant": [100.0] * rows,
        }
    )
    df.to_csv(path, index=False)
    return path


def _read_output(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _assert_unique_input_index(df: pd.DataFrame):
    assert df[RESUME_KEY].is_unique
    assert df[RESUME_KEY].notna().all()


class TestResumeMode:
    def test_default_resume_skips_existing(self, mock_classifier, input_csv, tmp_path):
        out = tmp_path / "out.csv"
        stats1 = process_hotspots(str(input_csv), str(out), verbose=False)
        assert stats1["processed"] == 20
        assert stats1["skipped"] == 0

        stats2 = process_hotspots(str(input_csv), str(out), verbose=False)
        assert stats2["processed"] == 0
        assert stats2["skipped"] == 20
        assert stats2["dry_run"] is False

        df = _read_output(out)
        assert len(df) == 20
        _assert_unique_input_index(df)

    def test_resume_keeps_all_columns(self, mock_classifier, input_csv, tmp_path):
        out = tmp_path / "out.csv"
        process_hotspots(str(input_csv), str(out), limit=3, verbose=False)
        process_hotspots(str(input_csv), str(out), limit=3, verbose=False)
        process_hotspots(str(input_csv), str(out), limit=3, verbose=False)
        df = _read_output(out)
        # Never collapse to a one-column output: schema must persist across resumes.
        assert "final_label" in df.columns
        assert "latitude" in df.columns
        assert len(df) == 3
        _assert_unique_input_index(df)


class TestLimitMode:
    def test_limit_processes_n_rows(self, mock_classifier, input_csv, tmp_path):
        out = tmp_path / "out.csv"
        stats = process_hotspots(str(input_csv), str(out), limit=7, verbose=False)
        assert stats["processed"] == 7
        df = _read_output(out)
        assert sorted(df[RESUME_KEY].tolist()) == list(range(7))
        _assert_unique_input_index(df)


class TestForceMode:
    def test_force_reprocesses_selected_and_never_duplicates(
        self, mock_classifier, input_csv, tmp_path
    ):
        out = tmp_path / "out.csv"
        process_hotspots(str(input_csv), str(out), limit=5, verbose=False)
        assert len(_read_output(out)) == 5

        # Force full reprocess -> identical 20 rows, replaced not duplicated.
        stats = process_hotspots(str(input_csv), str(out), force=True, verbose=False)
        assert stats["processed"] == 20
        df = _read_output(out)
        assert len(df) == 20
        _assert_unique_input_index(df)

    def test_force_limit_replaces_without_duplicates(
        self, mock_classifier, input_csv, tmp_path
    ):
        out = tmp_path / "out.csv"
        process_hotspots(str(input_csv), str(out), limit=20, verbose=False)
        # Manually corrupt a value to confirm force actually replaces it.
        df = _read_output(out)
        df.loc[0, "final_label"] = "stale"
        df.to_csv(out, index=False)

        stats = process_hotspots(str(input_csv), str(out), force=True, limit=1,
                                 verbose=False)
        assert stats["processed"] == 1
        df = _read_output(out)
        assert len(df) == 20                # no growth
        _assert_unique_input_index(df)
        assert (df["final_label"] == "stale").sum() == 0  # replaced


class TestDryRun:
    def test_dry_run_writes_nothing(self, mock_classifier, input_csv, tmp_path):
        out = tmp_path / "out.csv"
        stats = process_hotspots(str(input_csv), str(out), limit=5, dry_run=True,
                                 verbose=False)
        assert stats["dry_run"] is True
        assert stats["processed"] == 0
        assert not out.exists()