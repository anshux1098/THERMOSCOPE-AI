"""
dataset_builder.py
Phase A — Build the ML training dataset for THERMOSCOPE-AI (SIH26162).

Pipeline:
1. Load the processed classified_hotspots.csv (642 rows with FIRMS + OSM features).
2. For each row, run ALL labeling functions in app.intelligence.labeling_functions.
3. Aggregate votes via majority voting (aggregate_votes).
4. Use the consensus as the weak-supervision label.
5. Select the ML feature columns and write training_dataset.csv.

This is the only piece of Phase A that was still missing — the labeling functions,
constants, and consensus aggregation already exist in app.intelligence.

Notes for the team:
- Distances in classified_hotspots.csv are in KILOMETERS (with 999 as the
  'not found' sentinel). We pass rows directly to the LFs; the LFs'
  get_distance_meters() helper handles unit conversion (km -> m).
- Mining distance is almost always 999 (sentinel), so the mining class
  will be underrepresented. Synthetic minority examples are recommended
  post-Phase B if class balance is poor (see Phase B report).
- The output CSV is the ONLY input to app.ml.train.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Make app package importable when run as a script
backend_dir = str(Path(__file__).resolve().parents[2])
root_dir = str(Path(__file__).resolve().parents[3])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd

from app.intelligence.labeling_functions import (
    apply_labeling_functions,
    aggregate_votes,
    ALL_LABELING_FUNCTIONS,
)
from app.core.constants import CLASS_LABELS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_INPUT_CSV = "data/processed/hotspots/classified_hotspots_v2.csv"
DEFAULT_OUTPUT_CSV = "data/processed/hotspots/training_dataset.csv"

# ---------------------------------------------------------------------------
# ML feature columns (the schema XGBoost will see)
# ---------------------------------------------------------------------------
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

FEATURE_COLUMNS: List[str] = (
    THERMAL_COLUMNS + DISTANCE_COLUMNS + FLAG_COLUMNS + COUNT_COLUMNS
)
LABEL_COLUMN = "label"

# 999 sentinel means "no feature of that type found within search radius"
MISSING_SENTINEL = 999.0


def _safe_get(row: pd.Series, col: str, default: float = 0.0) -> float:
    """Read a feature from a row, treating 999 / NaN as 'missing' -> large value.

    XGBoost handles trees, so this just needs to be consistent.
    We keep 999 as-is (a 'very far' value) since the LFs use the same convention
    and changing it would desync the rule and ML paths.
    """
    val = row.get(col, default)
    try:
        if pd.isna(val):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def build_feature_row(row: pd.Series) -> Dict[str, float]:
    """Extract the ML feature dict for one hotspot row."""
    features: Dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        features[col] = _safe_get(row, col)
    return features


def build_labels(df: pd.DataFrame) -> List[str]:
    """Run all LFs on each row and aggregate votes into a consensus label."""
    labels: List[str] = []
    for _, row in df.iterrows():
        votes = apply_labeling_functions(row)
        labels.append(aggregate_votes(votes))
    return labels


def build_dataset(
    input_csv: str = DEFAULT_INPUT_CSV,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the training dataset.

    Returns the final DataFrame (also written to output_csv).
    Columns:
        - All FEATURE_COLUMNS (numeric, ready for XGBoost)
        - LABEL_COLUMN (weak-supervision consensus)
        - latitude, longitude (kept for traceability / debugging)
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}\n"
            "Run the existing hotspot pipeline first to generate it."
        )

    if verbose:
        print(f"[dataset_builder] Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    if verbose:
        print(f"  -> {len(df)} rows, {len(df.columns)} columns")

    # INTEGRITY GUARD: Reject synthetic demo CSVs immediately.
    # If build_demo_dataset.py accidentally generated this file, it will have
    # a _is_synthetic_demo column. Training on synthetic data is a silent failure.
    if "_is_synthetic_demo" in df.columns:
        raise ValueError(
            "\n" + "="*70 + "\n"
            "INTEGRITY ERROR: The input CSV was generated by build_demo_dataset.py"
            " (detected '_is_synthetic_demo' marker column).\n"
            "Training on synthetic data is FORBIDDEN.\n"
            "Run:  python scripts/build_real_dataset.py\n"
            "Then re-run dataset_builder.\n"
            + "="*70
        )

    # 1. Generate weak-supervision labels via the existing LF pipeline
    if verbose:
        print("[dataset_builder] Running 14 labeling functions on each row...")
    labels = build_labels(df)
    df[LABEL_COLUMN] = labels

    # 2. Build the numeric feature matrix
    if verbose:
        print("[dataset_builder] Extracting ML feature columns...")
    feature_rows = [build_feature_row(row) for _, row in df.iterrows()]
    feature_df = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)

    # 3. Compose final dataset (features + label + a couple of geo cols for debugging)
    final_df = pd.concat(
        [
            feature_df,
            df[[LABEL_COLUMN, "latitude", "longitude"]].reset_index(drop=True),
        ],
        axis=1,
    )

    # 4. Sanity check: only keep rows where the label is one of the canonical classes
    valid_mask = final_df[LABEL_COLUMN].isin(CLASS_LABELS)
    dropped = (~valid_mask).sum()
    if dropped > 0 and verbose:
        print(f"[dataset_builder] Warning: dropped {dropped} rows with invalid labels")
    final_df = final_df[valid_mask].reset_index(drop=True)

    # 5. Write output
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    if verbose:
        print(f"[dataset_builder] Wrote {len(final_df)} rows to {output_csv}")
        print("\n[dataset_builder] Label distribution:")
        print(final_df[LABEL_COLUMN].value_counts().to_string())
        print(f"\n[dataset_builder] Feature columns ({len(FEATURE_COLUMNS)}):")
        for c in FEATURE_COLUMNS:
            print(f"  - {c}")

    return final_df


if __name__ == "__main__":
    build_dataset()
