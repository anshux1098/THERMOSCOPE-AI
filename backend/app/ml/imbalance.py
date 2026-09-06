"""
imbalance.py
Phase E4 — Controlled, reusable training-time class-imbalance mitigation.

This module contains the ONLY sanctioned Phase E4 intervention tooling:

  - 'none'           : uniform weights (Phase E3 control reproduction)
  - 'inverse'        : w_c = N_train / n_c                      (raw inverse frequency)
  - 'sqrt_inverse'   : w_c = sqrt(N_train / n_c)                (damped inverse frequency)
  - 'capped_inverse' : w_c = min(N_train / n_c, cap)            (bounded inverse frequency)

Every strategy:
  * is computed from **training labels only** — test data never enters (§11 leakage protection);
  * never duplicates rows and never synthesizes feature vectors — duplication is NOT new
    independent evidence (§6);
  * produces finite, strictly-positive weights by construction;
  * optionally mean-normalizes the sample weights (uniform scaling is invariance-neutral
    for an XGBoost tree, so normalization only keeps strategies comparable);
  * emits a complete per-class audit (class, frequency, raw weight, cap applied, final weight).

`bounded_oversample` is the isolated, seed-reproducible oversampling helper used by experiment
E4-C. It duplicates *training* rows only and never touches test/canonical data. Duplication is
explicitly flagged as non-evidence: 100 copies of one sample remain one independent example.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

WEIGHT_STRATEGIES: Tuple[str, ...] = (
    "none",
    "inverse",
    "sqrt_inverse",
    "capped_inverse",
)

OVERSAMPLE_SEED = 42


def validate_strategy(strategy: str) -> None:
    if strategy not in WEIGHT_STRATEGIES:
        raise ValueError(
            f"Unknown weight strategy '{strategy}'. Valid: {WEIGHT_STRATEGIES}"
        )


def class_frequencies(y: pd.Series) -> pd.Series:
    """Per-class training frequencies (sorted by label, matching LabelEncoder order)."""
    return y.value_counts().sort_index()


def raw_class_weights(
    strategy: str, freqs: pd.Series, cap: Optional[float] = None
) -> pd.Series:
    """
    Per-class raw weight from the strategy formula, BEFORE normalization/cap bookkeeping.

    'capped_inverse' applies an upper bound on the raw inverse-frequency weight; that bound
    is where a cap is introduced.
    """
    validate_strategy(strategy)
    n_total = int(freqs.sum())

    if strategy == "none":
        w = pd.Series(1.0, index=freqs.index, dtype=float)
    elif strategy == "inverse":
        w = n_total / freqs.astype(float)
    elif strategy == "sqrt_inverse":
        w = np.sqrt(n_total / freqs.astype(float))
    elif strategy == "capped_inverse":
        if cap is None:
            raise ValueError("Strategy 'capped_inverse' requires an explicit cap (float > 1).")
        if cap <= 1:
            raise ValueError("Cap must be > 1 (a cap <= 1 would flatten every class to 1.0).")
        raw = n_total / freqs.astype(float)
        w = raw.clip(upper=float(cap))
    return w


def compute_sample_weights(
    y_train: pd.Series,
    strategy: str = "none",
    cap: Optional[float] = None,
    normalize: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute a per-row sample weight vector for the training labels.

    Args:
        y_train: Training labels (class-name strings; index is the training row index).
        strategy: One of WEIGHT_STRATEGIES.
        cap: Upper bound for 'capped_inverse' (predefined before any result inspection).
        normalize: If True, divide all weights by their mean so the mean weight == 1.

    Returns:
        (weight vector aligned to y_train.index, per-class audit DataFrame with columns
         ['class', 'frequency', 'raw_weight', 'cap_applied', 'final_weight',
          'weighted_rows']).
    """
    validate_strategy(strategy)
    if len(y_train) == 0:
        raise ValueError("Cannot compute weights on an empty training set.")

    y = y_train.astype(str)
    freqs = class_frequencies(y)

    raw = raw_class_weights(strategy, freqs, cap=cap)
    weight_by_class = raw.to_dict()
    weights = np.array([weight_by_class[c] for c in y.values], dtype=np.float64)

    norm_factor = 1.0
    if normalize:
        mean_w = float(weights.mean())
        if not np.isfinite(mean_w) or mean_w <= 0:
            raise ValueError("Normalization failed: non-positive/non-finite mean weight.")
        norm_factor = mean_w
        weights = weights / mean_w

    final_weight_by_class = {c: w / norm_factor for c, w in weight_by_class.items()}

    # Hard validation: no NaN/inf, strictly positive (per-row and per-class).
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Sample weights contain NaN/inf or zero/negative values — aborting.")
    if any(not np.isfinite(w) or w <= 0 for w in final_weight_by_class.values()):
        raise ValueError("Final class weights contain NaN/inf or zero/negative values — aborting.")

    cap_applied = pd.Series(False, index=freqs.index)
    if cap is not None:
        raw_unbounded = int(freqs.sum()) / freqs.astype(float)
        cap_applied = raw_unbounded > float(cap)

    audit = pd.DataFrame(
        {
            "class": freqs.index.tolist(),
            "frequency": freqs.values,
            "raw_weight": raw.values,
            "cap_applied": cap_applied.values,
            "final_weight": [final_weight_by_class[c] for c in freqs.index],
        }
    )
    audit["weighted_rows"] = audit["frequency"] * audit["final_weight"]
    audit = audit.reset_index(drop=True)

    return weights, audit


def bounded_oversample(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    floor: int,
    random_state: int = OVERSAMPLE_SEED,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Deterministic, bounded, TRAINING-ONLY oversampling.

    Every class present in the training split is raised to at least `floor` rows by
    drawing random rows of that class WITH replacement (seed `random_state`). Original
    rows are always kept. Test data and the canonical dataset are untouched.

    Scientific caveat enforced by Phase E4: duplication increases a class' abstraction
    in the loss but does NOT add independent evidence. Results for tier B/C classes must
    be read as memorization probes, never as evidence of learned generalization.
    """
    if floor < 1:
        raise ValueError("Oversample floor must be >= 1.")
    rng = np.random.RandomState(random_state)

    frames_x, frames_y = [], []
    for cls in sorted(y_train.unique()):
        mask = (y_train.values == cls)
        cls_x = X_train.loc[mask]
        cls_y = y_train.loc[mask]
        need = max(int(floor) - int(len(cls_y)), 0)
        if need > 0:
            pos = np.where(mask)[0]
            idx = rng.choice(pos, size=need, replace=True)
            frames_x.append(X_train.iloc[idx])
            frames_y.append(y_train.iloc[idx])
        frames_x.append(cls_x)
        frames_y.append(cls_y)

    X_dup = pd.concat(frames_x, ignore_index=True)
    y_dup = pd.concat(frames_y, ignore_index=True)
    return X_dup, y_dup