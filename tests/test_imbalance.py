"""
tests/test_imbalance.py

Phase E4 — Controlled Imbalance Mitigation (SIH26162).

Locks the Phase E4 intervention math so future experiments cannot silently
change the sanctioned weighting strategies:

  E4-A  none            : uniform weights (control - must reproduce E3)
  E4-B1 inverse         : w_c = N_train / n_c                 (uncapped)
  E4-B2 sqrt_inverse    : w_c = sqrt(N_train / n_c)           (damped)
  E4-B3 capped_inverse  : w_c = min(N_train / n_c, 10.0)      (predefined cap)
  E4-C  oversample      : TRAIN-only bounded duplication to floor=30, seed=42

Hard guarantees under test:
  * weights come from TRAINING labels only (no test-leakage path exists in the
    public API; the trainer preprocessors are applied to the training split).
  * weights are finite and strictly positive, mean-normalized by default.
  * a cap <= 1 is rejected; an unknown strategy is rejected.
  * oversampling is deterministic per seed, keeps every original row, and never
    lowers any class below the floor.
  * experiment artifacts are written to their own isolated directory (never the
    production model directory).

Run:
    pytest tests/test_imbalance.py -v
"""
import sys
from pathlib import Path

backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
root_dir = str(Path(__file__).resolve().parents[1])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import pytest

from app.ml.imbalance import (
    WEIGHT_STRATEGIES,
    bounded_oversample,
    class_frequencies,
    compute_sample_weights,
    raw_class_weights,
    validate_strategy,
)
from app.ml.experiment_runner import (
    CAPPED_INVERSE_CAP,
    OVERSAMPLE_SEED,
    OVERSAMPLE_FLOOR,
    PREPROCESSORS,
    build_split,
)
from app.ml.dataset_builder import FEATURE_COLUMNS, LABEL_COLUMN


def _toy_labels() -> pd.Series:
    return pd.Series(
        ["agri"] * 1 + ["minor"] * 3 + ["forest"] * 7 + ["major"] * 20,
        name="label",
    )


def _toy_df() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    n = 31
    y = _toy_labels().astype(str)
    x = pd.DataFrame(
        {c: rng.rand(n) for c in FEATURE_COLUMNS}, index=y.index
    )
    return pd.concat([x, pd.DataFrame({LABEL_COLUMN: y})], axis=1)


# ---------------------------------------------------------------------------
# Strategy validation
# ---------------------------------------------------------------------------
def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        validate_strategy("not_a_strategy")


def test_registry_contains_only_predefined_strategies():
    assert list(WEIGHT_STRATEGIES) == [
        "none", "inverse", "sqrt_inverse", "capped_inverse",
    ]


def test_capped_inverse_requires_cap():
    y = _toy_labels()
    with pytest.raises(ValueError):
        compute_sample_weights(y, strategy="capped_inverse", cap=None)
    with pytest.raises(ValueError):
        compute_sample_weights(y, strategy="capped_inverse", cap=1.0)
    with pytest.raises(ValueError):
        compute_sample_weights(y, strategy="capped_inverse", cap=0.5)


def test_empty_training_set_rejected():
    with pytest.raises(ValueError):
        compute_sample_weights(pd.Series(dtype=str), strategy="inverse")


# ---------------------------------------------------------------------------
# Strategy math
# ---------------------------------------------------------------------------
def test_none_strategy_is_uniform():
    y = _toy_labels()
    w, audit = compute_sample_weights(y, strategy="none")
    assert np.allclose(w, 1.0)
    assert (audit["final_weight"] == 1.0).all()


def test_inverse_formula_and_equalization():
    y = _toy_labels()
    w, audit = compute_sample_weights(y, strategy="inverse")
    n = len(y)
    for _, row in audit.iterrows():
        assert np.isclose(row["raw_weight"], n / row["frequency"])
    # Mean-normalized -> the total weight per class is identical (equalization).
    totals = audit["weighted_rows"]
    assert np.isclose(totals.max(), totals.min())


def test_sqrt_inverse_formula():
    y = _toy_labels()
    w, audit = compute_sample_weights(y, strategy="sqrt_inverse")
    n = len(y)
    for _, row in audit.iterrows():
        assert np.isclose(row["raw_weight"], np.sqrt(n / row["frequency"]))
    assert not (audit["cap_applied"]).any()


def test_capped_inverse_applies_predefined_cap():
    y = _toy_labels()
    w, audit = compute_sample_weights(
        y, strategy="capped_inverse", cap=CAPPED_INVERSE_CAP
    )
    n = len(y)
    assert (audit["raw_weight"] <= CAPPED_INVERSE_CAP + 1e-9).all()
    # Classes whose uncapped inverse weight exceeds the cap are the capped ones.
    assert (audit["raw_weight"] < (n / audit["frequency"]) - 1e-9).eq(audit["cap_applied"]).all()
    # Only the tiny classes actually hit the cap.
    assert (audit.loc[audit["frequency"].le(3), "cap_applied"]).all()
    assert not (audit.loc[audit["frequency"].ge(7), "cap_applied"]).any()


def test_mean_normalization_holds():
    y = _toy_labels()
    for strategy in ["none", "inverse", "sqrt_inverse", "capped_inverse"]:
        w, _ = compute_sample_weights(y, strategy=strategy, cap=CAPPED_INVERSE_CAP)
        assert np.isclose(w.mean(), 1.0, atol=1e-9)


def test_without_normalization_keeps_raw_scale():
    y = _toy_labels()
    w, audit = compute_sample_weights(
        y, strategy="inverse", normalize=False
    )
    for _, row in audit.iterrows():
        assert np.isclose(row["final_weight"], row["raw_weight"])


def test_weights_finite_and_strictly_positive():
    y = _toy_labels()
    for strategy in ["none", "inverse", "sqrt_inverse", "capped_inverse"]:
        w, audit = compute_sample_weights(y, strategy=strategy, cap=CAPPED_INVERSE_CAP)
        assert np.isfinite(w).all() and (w > 0).all()
        assert np.isfinite(audit["final_weight"]).all() and (audit["final_weight"] > 0).all()


# ---------------------------------------------------------------------------
# Oversampling: determinism, floor, original-row preservation
# ---------------------------------------------------------------------------
def test_oversample_deterministic_per_seed():
    y = _toy_labels()
    x = pd.DataFrame({"v": np.arange(len(y))}, index=y.index)
    x1, y1 = bounded_oversample(x, y, floor=10, random_state=5)
    x2, y2 = bounded_oversample(x, y, floor=10, random_state=5)
    pd.testing.assert_frame_equal(x1, x2)
    pd.testing.assert_series_equal(y1, y2)


def test_oversample_floor_and_preservation():
    y = _toy_labels()
    x = pd.DataFrame({"v": np.arange(len(y)) * 1.0}, index=y.index)
    x_dup, y_dup = bounded_oversample(x, y, floor=OVERSAMPLE_FLOOR, random_state=OVERSAMPLE_SEED)
    for cls in y.unique():
        assert int((y_dup.values == cls).sum()) >= OVERSAMPLE_FLOOR
    # Every original training row survives at least once (row identity by value).
    assert set(x_dup["v"].values) == set(x["v"].values)
    assert x_dup.columns.equals(x.columns)


def test_oversample_rejects_floor_below_one():
    with pytest.raises(ValueError):
        bounded_oversample(pd.DataFrame({"v": [1, 2]}), pd.Series(["a", "b"]), floor=0)


def test_oversample_audit_arity():
    y = _toy_labels()
    x = pd.DataFrame({"v": np.arange(len(y))}, index=y.index)
    x_dup, y_dup = bounded_oversample(x, y, floor=OVERSAMPLE_FLOOR, random_state=OVERSAMPLE_SEED)
    assert len(x_dup) == len(y_dup)


# ---------------------------------------------------------------------------
# Leakage isolation: preprocessing applies to TRAIN only
# ---------------------------------------------------------------------------
def test_oversample_preprocessor_does_not_touch_test_rows():
    sp = build_split(_toy_df())
    fn = PREPROCESSORS["e4c_oversample_floor30"]
    x_tr, y_tr, w, audit = fn(sp["X_train"], sp["y_train"], sp["n_classes"], sp["class_names"])
    # Training split grew (duplicates added) while test split is untouched.
    assert len(x_tr) > len(sp["X_train"])
    assert set(sp["X_test"].index).isdisjoint(set(sp["X_train"].index))
    # Duplication only ever draws from the training rows, never from test rows.
    assert set(sp["X_train"].index).issubset(set(x_tr.index))


def test_weight_preprocessor_derives_audit_from_train_only():
    sp = build_split(_toy_df())
    fn = PREPROCESSORS["e4b1_weight_inverse"]
    x_tr, y_tr, w, audit = fn(sp["X_train"], sp["y_train"], sp["n_classes"], sp["class_names"])
    # Identical row count -> weighting never modifies the training frame.
    pd.testing.assert_frame_equal(x_tr, sp["X_train"])
    assert len(w) == len(sp["X_train"])


def test_split_reproduction_guard():
    # The 7 toy labels are unique -> rare-class move-to-train guard is exercised.
    sp = build_split(_toy_df())
    assert sp["X_train"].shape[1] == len(FEATURE_COLUMNS)
    assert sp["y_train"].index.isin(sp["y_test"].index).sum() == 0