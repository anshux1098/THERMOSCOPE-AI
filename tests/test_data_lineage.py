"""
tests/test_data_lineage.py

Data lineage / canonical path tests for THERMOSCOPE-AI.

Verifies:
  - A single canonical classified/training/enriched path exists.
  - Consuming stages reference the canonical path (no divergent constants).
  - build_real_dataset writes to the canonical path.
  - Data contract validation raises descriptive errors on missing / synthetic /
    stale data.
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

from app.core import paths as canonical_paths
from app.core.lineage import (
    validate_classified_dataset,
    validate_training_dataset,
    detect_conflicting_classified_copies,
    warn_if_stale_classified_copy,
)


# ---------------------------------------------------------------------------
# Canonical path tests
# ---------------------------------------------------------------------------
class TestCanonicalDatasetPaths:
    def test_canonical_paths_are_absolute(self):
        assert isinstance(canonical_paths.CLASSIFIED_DATASET_PATH, Path)
        assert canonical_paths.CLASSIFIED_DATASET_PATH.is_absolute()
        assert canonical_paths.TRAINING_DATASET_PATH.is_absolute()
        assert canonical_paths.ENRICHED_DATASET_PATH.is_absolute()

    def test_canonical_paths_live_under_processed_hotspots(self):
        assert canonical_paths.CLASSIFIED_DATASET_PATH.parent == canonical_paths.HOTSPOTS_DIR
        assert canonical_paths.TRAINING_DATASET_PATH.parent == canonical_paths.HOTSPOTS_DIR
        assert canonical_paths.ENRICHED_DATASET_PATH.parent == canonical_paths.HOTSPOTS_DIR

    def test_canonical_classified_name(self):
        assert canonical_paths.CLASSIFIED_DATASET_PATH.name == "classified_hotspots_v2.csv"
        assert canonical_paths.TRAINING_DATASET_PATH.name == "training_dataset.csv"
        assert canonical_paths.ENRICHED_DATASET_PATH.name == "classified_hotspots_v2_enriched.csv"


class TestNoConflictingDatasetPaths:
    """The pipeline must have exactly one canonical classified path source."""

    def test_dataset_builder_uses_canonical_path(self):
        import app.ml.dataset_builder as db
        assert Path(db.DEFAULT_INPUT_CSV).resolve() == canonical_paths.CLASSIFIED_DATASET_PATH.resolve()
        assert Path(db.DEFAULT_OUTPUT_CSV).resolve() == canonical_paths.TRAINING_DATASET_PATH.resolve()

    def test_run_pipeline_uses_canonical_classified_path(self):
        sys.path.insert(0, str(Path(root_dir) / "scripts"))
        import run_pipeline  # scripts/run_pipeline.py
        assert Path(run_pipeline.DEFAULT_INPUT_CSV).resolve() == canonical_paths.CLASSIFIED_DATASET_PATH.resolve()
        assert Path(run_pipeline.DEFAULT_OUTPUT_CSV).resolve() == canonical_paths.ENRICHED_DATASET_PATH.resolve()

    def test_build_real_dataset_writes_canonical_classified_path(self):
        # Source-level check: the builder's OUTPUT_CSV constant must equal the
        # canonical path (import the module without running the pipeline).
        import importlib.util as ilu
        builder_path = Path(root_dir) / "scripts" / "build_real_dataset.py"
        spec = ilu.spec_from_file_location("build_real_dataset", builder_path)
        # Loading the module executes only module-level code (safe: no pipeline run).
        mod = ilu.module_from_spec(spec)
        sys.modules["build_real_dataset"] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        assert Path(mod.OUTPUT_CSV).resolve() == canonical_paths.CLASSIFIED_DATASET_PATH.resolve()
        sys.modules.pop("build_real_dataset", None)

    def test_no_hardcoded_legacy_path_in_consumers(self):
        """Consumers must not reference the legacy data/classified location."""
        import app.ml.dataset_builder as db
        src = Path(db.__file__).read_text(encoding="utf-8")
        assert "data/classified" not in src

        import run_pipeline
        src = Path(run_pipeline.__file__).read_text(encoding="utf-8")
        assert "data/classified" not in src


# ---------------------------------------------------------------------------
# Data contract tests
# ---------------------------------------------------------------------------
class TestDataContractValidation:
    def _make_real_csv(self, path: Path, rows=5, extra_cols=None):
        df = pd.DataFrame(
            {
                "latitude": [20.0 + i for i in range(rows)],
                "longitude": [75.0 + i for i in range(rows)],
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
        for c in extra_cols or []:
            df[c] = 1
        df.to_csv(path, index=False)
        return df

    def test_validate_classified_dataset_ok(self, tmp_path):
        path = tmp_path / "classified_hotspots_v2.csv"
        self._make_real_csv(path)
        df = validate_classified_dataset(path)
        assert len(df) == 5

    def test_validate_classified_dataset_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="canonical|not found"):
            validate_classified_dataset(tmp_path / "does_not_exist.csv")

    def test_validate_classified_dataset_synthetic(self, tmp_path):
        path = tmp_path / "classified_hotspots_v2.csv"
        self._make_real_csv(path, extra_cols=["_is_synthetic_demo"])
        with pytest.raises(ValueError, match="synthetic"):
            validate_classified_dataset(path)

    def test_validate_classified_dataset_missing_columns(self, tmp_path):
        path = tmp_path / "classified_hotspots_v2.csv"
        self._make_real_csv(path)
        # Drop a required column by rewriting.
        df = pd.read_csv(path).drop(columns=["dist_forest"])
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            validate_classified_dataset(path)

    def test_validate_classified_dataset_empty(self, tmp_path):
        path = tmp_path / "classified_hotspots_v2.csv"
        pd.DataFrame(columns=["latitude", "longitude"]).to_csv(path, index=False)
        with pytest.raises(ValueError, match="empty"):
            validate_classified_dataset(path)

    def test_validate_training_dataset_synthetic(self, tmp_path):
        path = tmp_path / "training_dataset.csv"
        pd.DataFrame({"label": ["x"], "_is_synthetic_demo": [True]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="synthetic"):
            validate_training_dataset(path)


# ---------------------------------------------------------------------------
# Stale / conflicting copy detection
# ---------------------------------------------------------------------------
class TestConflictingCopies:
    def test_detect_conflicting_classified_copies(self, tmp_path, monkeypatch):
        monkeypatch.setattr(canonical_paths, "LEGACY_CLASSIFIED_DATASET_PATH",
                            tmp_path / "legacy" / "classified_hotspots_v2.csv")
        monkeypatch.setattr(canonical_paths, "CLASSIFIED_DATASET_PATH",
                            tmp_path / "canonical" / "classified_hotspots_v2.csv")
        tmp_path.joinpath("legacy").mkdir()
        (tmp_path / "legacy" / "classified_hotspots_v2.csv").write_text("a,b\n1,2\n")
        conflicts = detect_conflicting_classified_copies()
        assert len(conflicts) == 1
        assert conflicts[0].name == "classified_hotspots_v2.csv"

    def test_warn_if_stale_classified_copy_no_crash_when_absent(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(canonical_paths, "LEGACY_CLASSIFIED_DATASET_PATH",
                            tmp_path / "missing.csv")
        warn_if_stale_classified_copy()  # should not raise