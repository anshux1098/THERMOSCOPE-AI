"""
evaluate.py
Phase B — Generate evaluation artifacts (plots + report) for the trained model.

Outputs (in data/processed/hotspots/):
    confusion_matrix.png       - heatmap of true vs predicted classes
    feature_importance.png     - top 15 features by gain
    classification_report.txt  - full sklearn classification report

Re-runs the same stratified 80/20 split as train.py (same random_state=42)
so the numbers are directly comparable to what train.py printed.

Usage:
    python -m app.ml.evaluate
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
import matplotlib

matplotlib.use("Agg")  # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from app.ml.dataset_builder import FEATURE_COLUMNS, LABEL_COLUMN, DEFAULT_OUTPUT_CSV
from app.ml.train import MODEL_PATH, FEATURE_COLS_PATH, LABEL_CLASSES_PATH

# Output paths
PLOT_DIR = Path("data/processed/hotspots")
CM_PLOT_PATH = PLOT_DIR / "confusion_matrix.png"
FI_PLOT_PATH = PLOT_DIR / "feature_importance.png"
REPORT_PATH = PLOT_DIR / "classification_report.txt"


def _ensure_plot_dir() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _load_model_and_data() -> Tuple:
    """Load the trained model and re-create the test split used at training time."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run `python -m app.ml.train` first."
        )
    if not os.path.exists(DEFAULT_OUTPUT_CSV):
        raise FileNotFoundError(
            f"Training dataset not found: {DEFAULT_OUTPUT_CSV}\n"
            "Run `python -m app.ml.dataset_builder` first."
        )

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLS_PATH)
    label_classes = joblib.load(LABEL_CLASSES_PATH)

    df = pd.read_csv(DEFAULT_OUTPUT_CSV)
    X = df[feature_columns].astype(np.float32)
    y = df[LABEL_COLUMN].astype(str)

    # Same split as train.py: random_state=42, test_size=0.2, stratify=y
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return model, feature_columns, X_test, y_test, label_classes


def _decode_predictions(y_pred_int) -> np.ndarray:
    """Convert XGBoost integer predictions back to class-name strings using
    the label_classes ordering (alphabetically sorted at training time)."""
    label_classes_local = joblib.load(LABEL_CLASSES_PATH)
    return np.array([label_classes_local[int(i)] for i in y_pred_int])


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    classes: List[str],
    out_path: Path = CM_PLOT_PATH,
) -> None:
    """Save a heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(max(8, len(classes) * 1.2), max(6, len(classes) * 1.0)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=False,
    )
    plt.title("Confusion Matrix — THERMOSCOPE-AI Hotspot Classifier")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[evaluate] Saved confusion matrix -> {out_path}")


def plot_feature_importance(
    model,
    feature_columns: List[str],
    out_path: Path = FI_PLOT_PATH,
    top_n: int = 15,
) -> None:
    """Save a bar chart of the top-N most important features by XGBoost gain."""
    importances = model.feature_importances_
    fi = pd.DataFrame(
        {"feature": feature_columns, "importance": importances}
    ).sort_values("importance", ascending=False)

    top = fi.head(top_n).iloc[::-1]  # reverse for horizontal bar plot
    plt.figure(figsize=(9, max(4, top_n * 0.35)))
    sns.barplot(
        data=top,
        x="importance",
        y="feature",
        palette="viridis",
        hue="feature",
        legend=False,
    )
    plt.title(f"Top {top_n} Feature Importance (XGBoost gain)")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[evaluate] Saved feature importance -> {out_path}")

    # Also print the table to stdout
    print(f"\n[evaluate] Top {top_n} features:")
    print(fi.head(top_n).to_string(index=False))


def evaluate(
    output_dir: str = str(PLOT_DIR),
    random_state: int = 42,
) -> Dict:
    """
    Re-evaluate the trained model on the held-out 20% test split and
    persist confusion-matrix + feature-importance plots + a text report.

    Returns the metrics dict (same shape as train.train_model).
    """
    _ensure_plot_dir()
    model, feature_columns, X_test, y_test, label_classes = _load_model_and_data()

    print(f"[evaluate] Test set: {len(X_test)} rows")
    print(f"[evaluate] Classes: {label_classes}")

    y_pred = model.predict(X_test)
    y_pred_names = _decode_predictions(y_pred)
    report = classification_report(
        y_test, y_pred_names, labels=label_classes, zero_division=0
    )
    print("\n[evaluate] Classification report:")
    print(report)

    # Persist text report
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("THERMOSCOPE-AI — Hotspot Classifier Evaluation\n")
        f.write(f"Test set: {len(X_test)} rows | Features: {len(feature_columns)}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)

    # Plots
    plot_confusion_matrix(y_test, y_pred_names, label_classes, out_dir / "confusion_matrix.png")
    plot_feature_importance(model, feature_columns, out_dir / "feature_importance.png")

    # Persist metrics JSON
    from sklearn.metrics import accuracy_score, f1_score

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred_names)),
        "test_macro_f1": float(
            f1_score(y_test, y_pred_names, average="macro", zero_division=0)
        ),
        "n_test": int(len(X_test)),
        "n_features": int(len(feature_columns)),
        "classes": label_classes,
    }
    with open(out_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    evaluate()
