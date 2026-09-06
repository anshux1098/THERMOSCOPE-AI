"""
feature_schema.py
Single authoritative ML feature contract for THERMOSCOPE-AI (SIH26162).

This module is the ONE source of truth for the feature ordering shared by:

    - production inference   (app.ml.predict           -> FEATURE_COLUMNS)
    - training data building (app.ml.dataset_builder   -> re-exports the contract)
    - training / evaluation  (train.py, evaluate.py, experiment_runner.py)
    - the deployed model     (models/feature_columns.joblib)

Architecture note (production/runtime must not depend on training/research):
`predict.py` consumes THIS module directly. It never imports the training-only
`dataset_builder` module, so production inference has no dependency on the
training pipeline. Dataset-builder defaults (DEFAULT_INPUT_CSV / DEFAULT_OUTPUT_CSV)
remain in dataset_builder.py.

CONTRACT RULES:
- FEATURE_COLUMNS ordering must NEVER change without retraining the model and
  regenerating models/feature_columns.joblib.
- predictions are served in the order of feature_columns.joblib; the deployed
  order is verified against FEATURE_COLUMNS at load time (see predict._load_artifacts).
"""
from typing import List

# Thermal signature columns (from NASA FIRMS)
THERMAL_COLUMNS: List[str] = [
    "frp",
    "bright_ti4",
    "bright_ti5",
]

# Distance columns (km in source CSV)
DISTANCE_COLUMNS: List[str] = [
    "dist_refinery",
    "dist_factory",
    "dist_industrial_zone",
    "dist_oil_gas",
    "dist_mining",
    "dist_forest",
    "dist_agriculture",
    "dist_powerplant",
]

FLAG_COLUMNS: List[str] = [
    "has_refinery_5km",
    "has_powerplant_5km",
    "has_factory_5km",
    "has_industrial_2km",
]

COUNT_COLUMNS: List[str] = [
    "count_ind_5km",
    "count_ref_5km",
]

# The canonical 17-feature schema (3 + 8 + 4 + 2).
FEATURE_COLUMNS: List[str] = (
    THERMAL_COLUMNS + DISTANCE_COLUMNS + FLAG_COLUMNS + COUNT_COLUMNS
)

# Target column used for weak-supervision training labels.
LABEL_COLUMN = "label"

# 999 sentinel means "no feature of that type found within search radius"
MISSING_SENTINEL = 999.0

EXPECTED_FEATURE_COUNT = 17


def validate_feature_columns(columns: List[str]) -> bool:
    """Return True iff a column list exactly matches the canonical contract.

    Used to verify deployed model feature ordering (feature_columns.joblib)
    against the authoritative schema before serving inference.
    """
    return list(columns) == FEATURE_COLUMNS