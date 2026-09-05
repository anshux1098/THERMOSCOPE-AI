"""
paths.py
Centralized, canonical dataset / model path management for THERMOSCOPE-AI (SIH26162).

Single source of truth for every file path the pipeline touches, so that no
stage silently reads a stale or duplicate dataset and no stage diverges from
the canonical data flow:

    RAW FIRMS + RAW OSM
    -> build_real_dataset.py   -> CLASSIFIED_DATASET_PATH
    -> dataset_builder.py      -> TRAINING_DATASET_PATH
    -> train.py                -> MODEL_DIR / MODEL_PATH
    -> hybrid_engine/predict   -> (in-memory model)
    -> run_pipeline.py         -> ENRICHED_DATASET_PATH

All paths are ABSOLUTE and derived from the repository root, so they work
regardless of the current working directory.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root (absolute, CWD-independent)
# ---------------------------------------------------------------------------
# backend/app/core/paths.py -> parents: core -> app -> backend -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Top-level data dirs
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
HOTSPOTS_DIR = PROCESSED_DATA_DIR / "hotspots"
CLASSIFIED_DATA_DIR = DATA_DIR / "classified"  # LEGACY; superseded by HOTSPOTS_DIR

# ---------------------------------------------------------------------------
# Raw input datasets / caches
# ---------------------------------------------------------------------------
FIRMS_DATASET_PATH = RAW_DATA_DIR / "firms_recent.csv"
OSM_CACHE_DIR = RAW_DATA_DIR / "osm"
OSM_INDUSTRIAL_CACHE_PATH = OSM_CACHE_DIR / "osm_industrial_sites.json"
OSM_FOREST_AGRI_CACHE_PATH = OSM_CACHE_DIR / "osm_forest_agriculture.json"

# ---------------------------------------------------------------------------
# Canonical derived datasets (data/processed/hotspots)
# ---------------------------------------------------------------------------
CLASSIFIED_DATASET_PATH = HOTSPOTS_DIR / "classified_hotspots_v2.csv"
TRAINING_DATASET_PATH = HOTSPOTS_DIR / "training_dataset.csv"
ENRICHED_DATASET_PATH = HOTSPOTS_DIR / "classified_hotspots_v2_enriched.csv"

# ---------------------------------------------------------------------------
# Model artifacts
# ---------------------------------------------------------------------------
MODEL_DIR = PROJECT_ROOT / "backend" / "app" / "ml" / "models"
MODEL_PATH = MODEL_DIR / "hotspot_classifier.joblib"
FEATURE_COLS_PATH = MODEL_DIR / "feature_columns.joblib"
LABEL_CLASSES_PATH = MODEL_DIR / "label_classes.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
METRICS_PATH = MODEL_DIR / "training_metrics.json"

# ---------------------------------------------------------------------------
# Evaluation / reporting artifacts
# ---------------------------------------------------------------------------
PLOT_DIR = HOTSPOTS_DIR
CM_PLOT_PATH = HOTSPOTS_DIR / "confusion_matrix.png"
FI_PLOT_PATH = HOTSPOTS_DIR / "feature_importance.png"
REPORT_PATH = HOTSPOTS_DIR / "classification_report.txt"
EVAL_METRICS_PATH = HOTSPOTS_DIR / "evaluation_metrics.json"

# ---------------------------------------------------------------------------
# Legacy / conflicting copies (detected for lineage protection)
# ---------------------------------------------------------------------------
LEGACY_CLASSIFIED_DATASET_PATH = CLASSIFIED_DATA_DIR / "classified_hotspots_v2.csv"


def ensure_dir(path: Path) -> Path:
    """Create the parent directory of a file path if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve(path: Path) -> str:
    """Return an absolute, normalized string path."""
    return path.resolve().as_posix()
