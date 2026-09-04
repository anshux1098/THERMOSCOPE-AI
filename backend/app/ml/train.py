"""
train.py
Phase B — Train an XGBoost classifier on the weakly supervised training dataset.

Inputs:
    data/processed/hotspots/training_dataset.csv
        (produced by app.ml.dataset_builder.build_dataset)

Outputs:
    backend/app/ml/models/hotspot_classifier.joblib    - trained XGBoost model
    backend/app/ml/models/feature_columns.joblib       - ordered list of feature names
    backend/app/ml/models/label_classes.joblib         - ordered list of class names (for predict)

Usage:
    python -m app.ml.train
    # or, from inside the project:
    from app.ml.train import train_model
    model, metrics = train_model()

Notes:
- Uses stratified 80/20 train/test split to keep class proportions.
- Reports accuracy + macro F1 + per-class precision/recall.
- Adds stratified 5-fold cross-validation (mean +/- std) for small-dataset variance.
- Classes with fewer than 15 training samples are flagged as coverage gaps.
- Saves the model + feature column order together. predict.py relies on the
  same column order, so do not change FEATURE_COLUMNS after training without
  re-running this script.
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Make app package importable when run as a script
backend_dir = str(Path(__file__).resolve().parents[2])
root_dir = str(Path(__file__).resolve().parents[3])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from app.ml.dataset_builder import FEATURE_COLUMNS, LABEL_COLUMN, DEFAULT_OUTPUT_CSV

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "hotspot_classifier.joblib"
FEATURE_COLS_PATH = MODEL_DIR / "feature_columns.joblib"
LABEL_CLASSES_PATH = MODEL_DIR / "label_classes.joblib"
METRICS_PATH = MODEL_DIR / "training_metrics.json"

# Classes with fewer than this many training samples are flagged as coverage gaps
COVERAGE_GAP_THRESHOLD = 15


def _ensure_model_dir() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _safe_predict(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    Safe wrapper for XGBClassifier.predict() that handles both the case where
    predict() returns a 1D array (correct) or a 2D probability array (multi:softprob
    with certain XGBoost versions). Always returns a 1D integer class index array.
    """
    preds = model.predict(X)
    if preds.ndim == 2:
        preds = np.argmax(preds, axis=1)
    return preds.astype(int)


def train_model(
    dataset_csv: str = DEFAULT_OUTPUT_CSV,
    random_state: int = 42,
    test_size: float = 0.2,
    n_cv_folds: int = 5,
) -> Tuple[XGBClassifier, Dict]:
    """
    Train XGBoost on the training dataset and persist artifacts.

    Returns:
        (model, metrics_dict)

    Raises:
        FileNotFoundError: if dataset_csv doesn't exist
        ValueError: if dataset is empty or has only one class
    """
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(
            f"Training dataset not found: {dataset_csv}\n"
            "Run `python -m app.ml.dataset_builder` first."
        )

    print(f"[train] Loading {dataset_csv}...")
    df = pd.read_csv(dataset_csv)
    print(f"  -> {len(df)} rows, {len(df.columns)} columns")

    if df.empty:
        raise ValueError("Training dataset is empty.")
    if df[LABEL_COLUMN].nunique() < 2:
        raise ValueError(
            f"Need at least 2 classes to train. Found: "
            f"{df[LABEL_COLUMN].unique().tolist()}"
        )

    X = df[FEATURE_COLUMNS].astype(np.float32)
    y_raw = df[LABEL_COLUMN].astype(str)

    # XGBoost 3.x requires integer labels via LabelEncoder
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw), index=y_raw.index)
    class_names = le.classes_.tolist()
    n_classes = len(class_names)

    # Stratified split preserves class proportions in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[train] Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"[train] Classes: {class_names}")
    print(f"[train] Class distribution (train):")
    print(pd.Series(y_train).value_counts().sort_index().to_string())
    for idx, name in enumerate(class_names):
        n_train = int((y_train == idx).sum())
        print(f"    {idx} -> {name} ({n_train} train)")

    # Flag classes with very few samples -- honest coverage-gap report
    coverage_gaps = []
    for idx, name in enumerate(class_names):
        n_train = int((y_train == idx).sum())
        if n_train < COVERAGE_GAP_THRESHOLD:
            coverage_gaps.append((name, n_train))
    if coverage_gaps:
        print(f"\n[train] WARNING: COVERAGE GAPS (< {COVERAGE_GAP_THRESHOLD} train samples):")
        for name, n in coverage_gaps:
            print(f"    {name}: {n} samples -- expect 0% recall for this class")
        print("  This is real data sparsity, not a bug. Do NOT synthesize data to fix.")

    # XGBoost with safe defaults; small dataset so we keep it simple
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )

    print("\n[train] Fitting XGBoost...")
    model.fit(X_train, y_train)

    # -----------------------------------------------------------------------
    # Evaluation -- single 80/20 split
    # -----------------------------------------------------------------------
    y_pred_train = _safe_predict(model, X_train)
    y_pred_test = _safe_predict(model, X_test)

    # Convert integer predictions back to label names for the report
    y_test_names = le.inverse_transform(np.array(y_test, dtype=int))
    y_pred_test_names = le.inverse_transform(y_pred_test)
    y_train_names = le.inverse_transform(np.array(y_train, dtype=int))
    y_pred_train_names = le.inverse_transform(y_pred_train)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    macro_f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)

    print(f"\n[train] Train accuracy: {train_acc:.4f}")
    print(f"[train] Test accuracy:  {test_acc:.4f}")
    print(f"[train] Test macro F1:  {macro_f1:.4f}")
    print("\n[train] Test classification report:")
    print(classification_report(
        y_test_names, y_pred_test_names, labels=class_names, zero_division=0
    ))
    for c in class_names:
        supp = (y_test_names == c).sum()
        if supp == 0:
            print(f"  {c}: NOT EVALUATED (0 test samples) — accuracy figure does not reflect this class")


    cm = confusion_matrix(
        y_test_names, y_pred_test_names, labels=sorted(class_names)
    )
    print("[train] Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=sorted(class_names), columns=sorted(class_names)))

    # -----------------------------------------------------------------------
    # Stratified k-fold cross-validation
    # (reduces variance from a single split on a small dataset)
    # -----------------------------------------------------------------------
    cv_scores_acc = []
    cv_scores_f1 = []
    print(f"\n[train] Running stratified {n_cv_folds}-fold cross-validation...")

    skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_f_train, X_f_val = X.iloc[train_idx], X.iloc[val_idx]
        y_f_train, y_f_val = y.iloc[train_idx], y.iloc[val_idx]

        # Skip folds where a class would have 0 train samples (degenerate dataset)
        if y_f_train.nunique() < 2:
            print(f"  Fold {fold_i}: SKIPPED (only 1 class in fold train set)")
            continue

        fold_model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", random_state=random_state,
            n_jobs=-1, tree_method="hist",
        )
        fold_model.fit(X_f_train, y_f_train)
        y_fold_pred = _safe_predict(fold_model, X_f_val)
        fold_acc = accuracy_score(y_f_val, y_fold_pred)
        fold_f1 = f1_score(y_f_val, y_fold_pred, average="macro", zero_division=0)
        cv_scores_acc.append(fold_acc)
        cv_scores_f1.append(fold_f1)
        print(f"  Fold {fold_i}: acc={fold_acc:.4f}  macro_f1={fold_f1:.4f}")

    cv_acc_mean = float(np.mean(cv_scores_acc)) if cv_scores_acc else float("nan")
    cv_acc_std  = float(np.std(cv_scores_acc))  if cv_scores_acc else float("nan")
    cv_f1_mean  = float(np.mean(cv_scores_f1))  if cv_scores_f1  else float("nan")
    cv_f1_std   = float(np.std(cv_scores_f1))   if cv_scores_f1  else float("nan")

    print(f"\n[train] CV accuracy  : {cv_acc_mean:.4f} +/- {cv_acc_std:.4f}")
    print(f"[train] CV macro F1  : {cv_f1_mean:.4f} +/- {cv_f1_std:.4f}")

    if coverage_gaps:
        print("\n[train] Limitation note: some classes had 0% recall.")
        print("  This is an OSM data coverage gap for India, not a code bug.")
        print("  See REPRODUCIBILITY.md for the honest baseline.")

    # -----------------------------------------------------------------------
    # Feature importances
    # -----------------------------------------------------------------------
    importances = model.feature_importances_
    feat_imp = sorted(
        zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True
    )
    print("\n[train] Feature importances (top 10):")
    for feat, imp in feat_imp[:10]:
        print(f"  {feat:30s}: {imp:.4f}")

    # -----------------------------------------------------------------------
    # Persist model artifacts
    # -----------------------------------------------------------------------
    _ensure_model_dir()
    joblib.dump(model, MODEL_PATH)
    joblib.dump(FEATURE_COLUMNS, FEATURE_COLS_PATH)
    joblib.dump(class_names, LABEL_CLASSES_PATH)
    print(f"\n[train] Saved model -> {MODEL_PATH}")

    metrics: Dict = {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(macro_f1),
        "cv_accuracy_mean": cv_acc_mean,
        "cv_accuracy_std": cv_acc_std,
        "cv_f1_mean": cv_f1_mean,
        "cv_f1_std": cv_f1_std,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classes": class_names,
        "coverage_gaps": [{"class": n, "n_train": c} for n, c in coverage_gaps],
        "feature_importances": {f: float(i) for f, i in feat_imp},
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] Saved metrics -> {METRICS_PATH}")

    return model, metrics


if __name__ == "__main__":
    train_model()
