"""
lineage.py
Data lineage + data-contract validation helpers for THERMOSCOPE-AI.

Used by build_real_dataset.py, dataset_builder.py, train.py and run_pipeline.py
to:
  * Detect conflicting / stale duplicate copies of derived datasets.
  * Validate that a dataset exists at the canonical path with an expected schema.
  * Emit a readable DATASET LINEAGE block at each producing/consuming stage.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from app.core.paths import (
    CLASSIFIED_DATASET_PATH,
    LEGACY_CLASSIFIED_DATASET_PATH,
)


# ---------------------------------------------------------------------------
# Lineage logging
# ---------------------------------------------------------------------------
def log_lineage(
    stage: str,
    input_path: Optional[str],
    input_rows: Optional[int],
    output_path: Optional[str],
    output_rows: Optional[int],
    rows_removed: Optional[int] = None,
    reason_for_removal: str = "",
    extra: Optional[List[str]] = None,
) -> None:
    """Print a standard DATASET LINEAGE block for a producing stage."""
    line = "=" * 52
    print("\n" + line)
    print("DATASET LINEAGE")
    print(line)
    print(f"Stage:            {stage}")
    if input_path is not None:
        print(f"Input path:       {input_path}")
    if input_rows is not None:
        print(f"Input rows:       {input_rows}")
    if output_path is not None:
        print(f"Output path:      {output_path}")
    if output_rows is not None:
        print(f"Output rows:      {output_rows}")
    if rows_removed is not None:
        print(f"Rows removed:     {rows_removed}")
    if reason_for_removal:
        print(f"Reason for removal: {reason_for_removal}")
    if extra:
        for e in extra:
            print(f"  {e}")
    print(f"Timestamp:        {datetime.now().isoformat(timespec='seconds')}")
    print(line + "\n")


# ---------------------------------------------------------------------------
# Stale / conflicting copy detection
# ---------------------------------------------------------------------------
def detect_conflicting_classified_copies() -> List[Path]:
    """
    Return a list of non-canonical copies of the classified dataset that exist
    on disk (currently the legacy data/classified copy).
    """
    conflicts: List[Path] = []
    if LEGACY_CLASSIFIED_DATASET_PATH.exists():
        conflicts.append(LEGACY_CLASSIFIED_DATASET_PATH)
    return conflicts


def warn_if_stale_classified_copy() -> None:
    """Warn loudly if a stale duplicate classified dataset exists anywhere."""
    conflicts = detect_conflicting_classified_copies()
    if not conflicts:
        return
    line = "=" * 68
    print("\n" + line)
    print("WARNING: Conflicting copy of the classified dataset detected.")
    print(line)
    print(f"Canonical path (used by the pipeline):")
    print(f"  {CLASSIFIED_DATASET_PATH}")
    print(f"Conflicting non-canonical copy found:")
    for p in conflicts:
        print(f"  {p}")
    print(
        "\nThe pipeline ALWAYS reads the canonical path above. The conflicting "
        "copy is NOT consumed, but its presence can cause confusion and stale "
        "data lineage."
    )
    print(
        "To remove the obsolete copy, run:\n"
        f"  Remove-Item -LiteralPath '{conflicts[0]}'"
    )
    print(line + "\n")


# ---------------------------------------------------------------------------
# Data contract validation
# ---------------------------------------------------------------------------
def validate_classified_dataset(
    path: Path = CLASSIFIED_DATASET_PATH,
    required_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Assert that the classified dataset exists at the canonical path, is not
    empty, is not synthetic, and has the expected schema. Returns the loaded
    DataFrame.

    Raises FileNotFoundError / ValueError with descriptive messages.
    """
    from app.core.paths import resolve

    if required_columns is None:
        required_columns = [
            "latitude",
            "longitude",
            "dist_refinery",
            "dist_factory",
            "dist_industrial_zone",
            "dist_oil_gas",
            "dist_mining",
            "dist_forest",
            "dist_agriculture",
            "dist_powerplant",
        ]

    if not path.exists():
        raise FileNotFoundError(
            "ERROR: Canonical classified dataset not found.\n"
            f"Expected: {resolve(path)}\n"
            "Run `python scripts/build_real_dataset.py` to generate it."
        )

    df = pd.read_csv(path)

    if "_is_synthetic_demo" in df.columns:
        raise ValueError(
            "\n" + "=" * 68 + "\n"
            "DATA CONTRACT ERROR: The canonical classified dataset contains the\n"
            "'_is_synthetic_demo' marker — it is synthetic and must NEVER be used.\n"
            "Run `python scripts/build_real_dataset.py` to regenerate it.\n"
            + "=" * 68
        )

    if df.empty:
        raise ValueError(
            f"DATA CONTRACT ERROR: Classified dataset is empty: {resolve(path)}"
        )

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            "DATA CONTRACT ERROR: Classified dataset is missing required columns: "
            f"{missing}\nPath: {resolve(path)}"
        )

    return df


def validate_training_dataset(path: Path) -> pd.DataFrame:
    """Validate the training dataset contract and return the loaded DataFrame."""
    from app.core.paths import resolve

    if not path.exists():
        raise FileNotFoundError(
            "ERROR: Training dataset not found.\n"
            f"Expected: {resolve(path)}\n"
            "Run `python -m backend.app.ml.dataset_builder` first."
        )
    df = pd.read_csv(path)
    if "_is_synthetic_demo" in df.columns:
        raise ValueError(
            "DATA CONTRACT ERROR: Training dataset is synthetic (contains "
            "'_is_synthetic_demo'). Regenerate with build_real_dataset.py + dataset_builder."
        )
    if df.empty:
        raise ValueError(f"DATA CONTRACT ERROR: Training dataset is empty: {resolve(path)}")
    return df
