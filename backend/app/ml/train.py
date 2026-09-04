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
from sklearn.model_selection import train_test_split
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


def _ensure_model_dir() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_model(
    dataset_csv: str = DEFAULT_OUTPUT_CSV,
    random_state: int = 42,
    test_size: float = 0.2,
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

    # Stratified split preserves class proportions in train and test.
    # If any class has < 2 samples (e.g. agricultural_burn with 1 sample),
    # stratified split fails — fall back to random split and note it.
    stratify = y
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 2].index.tolist()
    if rare_classes:
        rare_names = [class_names[i] for i in rare_classes]
        print(f"[train] WARNING: {len(rare_classes)} class(es) with < 2 samples: {rare_names}")
        print(f"[train] Falling back to non-stratified split (stratify disabled)")
        stratify = None

    # Stratified split preserves class proportions in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    print(f"[train] Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"[train] Classes: {class_names}")
    print(f"[train] Class distribution (train):")
    print(pd.Series(y_train).value_counts().sort_index().to_string())
    for idx, name in enumerate(class_names):
        print(f"    {idx} -> {name} ({int((y_train == idx).sum())} train)")

    # XGBoost with safe defaults; small dataset so we keep it simple
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(class_names),
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )

    print("\n[train] Fitting XGBoost...")
    model.fit(X_train, y_train)

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Convert integer predictions back to label names for the report
    y_test_names = le.inverse_transform(y_test)
    y_pred_test_names = le.inverse_transform(y_pred_test)
    y_train_names = le.inverse_transform(y_train)
    y_pred_train_names = le.inverse_transform(y_pred_train)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    macro_f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)

    print(f"\n[train] Train accuracy: {train_acc:.4f}")
    print(f"[train] Test accuracy:  {test_acc:.4f}")
    print(f"[train] Test macro F1:  {macro_f1:.4f}")
    print("\n[train] Test classification report:")
    print(classification_report(
        y_test_names, y_pred_test_names, zero_division=0
    ))

    cm = confusion_matrix(
        y_test_names, y_pred_test_names, labels=sorted(class_names)
    )
    print("[train] Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=sorted(class_names), columns=sorted(class_names)))

    metrics: Dict = {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(macro_f1),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "classes": sorted(class_names),
        "confusion_matrix": cm.tolist(),
    }

    # -----------------------------------------------------------------------
    # Persist artifacts (model + LabelEncoder + column order)
    # -----------------------------------------------------------------------
    _ensure_model_dir()
    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(FEATURE_COLUMNS), FEATURE_COLS_PATH)
    joblib.dump(sorted(class_names), LABEL_CLASSES_PATH)
    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[train] Saved model -> {MODEL_PATH}")
    print(f"[train] Saved feature columns -> {FEATURE_COLS_PATH}")
    print(f"[train] Saved label classes -> {LABEL_CLASSES_PATH}")
    print(f"[train] Saved label encoder -> {MODEL_DIR / 'label_encoder.joblib'}")
    print(f"[train] Saved metrics -> {METRICS_PATH}")

    return model, metrics


if __name__ == "__main__":
    train_model()
