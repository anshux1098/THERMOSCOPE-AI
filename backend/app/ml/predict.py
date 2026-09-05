"""
predict.py
Phase B — Load a trained XGBoost model and run inference on new hotspots.

Usage:
    from app.ml.predict import predict_proba, predict_label
    result = predict_proba({"frp": 80, "dist_refinery": 0.5, ...})
    # -> {"label": "industrial_fire", "probability": 0.71, "all_probabilities": {...}}

    label = predict_label({"frp": 80, "dist_refinery": 0.5, ...})

The feature dict must contain ALL columns from FEATURE_COLUMNS (defined in
dataset_builder). Missing columns default to 0.0. Extra columns are ignored.

The model, feature column order, and label classes are loaded once on first
call and cached in module-level globals. This keeps inference latency low
when predict() is called many times in a row.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Make app package importable when run as a script
backend_dir = str(Path(__file__).resolve().parents[2])
root_dir = str(Path(__file__).resolve().parents[3])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from app.ml.dataset_builder import FEATURE_COLUMNS
from app.core.paths import (
    MODEL_PATH,
    FEATURE_COLS_PATH,
    LABEL_CLASSES_PATH,
    LABEL_ENCODER_PATH,
)

import threading

# Module-level cache (loaded lazily)
_model: Optional[XGBClassifier] = None
_feature_columns: Optional[List[str]] = None
_label_classes: Optional[List[str]] = None
_label_encoder: Optional[LabelEncoder] = None
_lock = threading.Lock()


def _load_artifacts() -> None:
    """Load model + feature columns + label encoder (once, cached)."""
    global _model, _feature_columns, _label_classes, _label_encoder
    if _model is not None:
        return
        
    with _lock:
        if _model is not None:
            return
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}\n"
                "Run `python -m app.ml.train` first."
            )
        _model = joblib.load(MODEL_PATH)
        _feature_columns = joblib.load(FEATURE_COLS_PATH)
        _label_classes = joblib.load(LABEL_CLASSES_PATH)
        if LABEL_ENCODER_PATH.exists():
            _label_encoder = joblib.load(LABEL_ENCODER_PATH)


def _features_to_array(features: Dict[str, float]) -> np.ndarray:
    """
    Build a (1, n_features) numpy array from a feature dict, using the
    trained column order. Missing columns default to 0.0.

    Casts all values to float32 (matches training-time dtype).
    """
    _load_artifacts()
    assert _feature_columns is not None
    row = []
    for col in _feature_columns:
        try:
            v = float(features.get(col, 0.0))
            if np.isnan(v) or np.isinf(v):
                v = 0.0
        except (TypeError, ValueError):
            v = 0.0
        row.append(v)
    return np.array([row], dtype=np.float32)


def predict_proba(features: Dict[str, float]) -> Dict:
    """
    Run the trained model on a single feature dict.

    Returns:
        {
            "label": str,             # predicted class (e.g. "industrial_fire")
            "probability": float,     # confidence in that class (0..1)
            "all_probabilities": {class_name: prob, ...}  # full distribution
        }
    """
    _load_artifacts()
    assert _model is not None and _label_classes is not None

    X = _features_to_array(features)
    proba = _model.predict_proba(X)[0]  # shape (n_classes,)

    # XGBoost returns probabilities in the order it learned them. With our
    # LabelEncoder, that's the sorted-class order. We rely on
    # _label_classes (sorted list saved at training time) to map indices to
    # names. This is robust to XGBoost's internal ordering changes.
    if _label_classes is not None and len(_label_classes) == len(proba):
        all_probs = {str(c): float(p) for c, p in zip(_label_classes, proba)}
        best_idx = int(np.argmax(proba))
        best_class = str(_label_classes[best_idx])
        best_prob = float(proba[best_idx])
    else:
        # Fallback: use whatever classes_ the model exposes
        classes = list(_model.classes_)
        all_probs = {str(c): float(p) for c, p in zip(classes, proba)}
        best_idx = int(np.argmax(proba))
        best_class = str(classes[best_idx])
        best_prob = float(proba[best_idx])

    return {
        "label": best_class,
        "probability": best_prob,
        "all_probabilities": all_probs,
    }


def predict_label(features: Dict[str, float]) -> str:
    """Convenience: return just the predicted label."""
    return predict_proba(features)["label"]


def batch_predict(features_list: List[Dict[str, float]]) -> List[Dict]:
    """Run inference on a list of feature dicts in a single call (faster)."""
    _load_artifacts()
    assert _model is not None and _feature_columns is not None

    rows = []
    for features in features_list:
        row = []
        for col in _feature_columns:
            try:
                v = float(features.get(col, 0.0))
                if np.isnan(v) or np.isinf(v):
                    v = 0.0
            except (TypeError, ValueError):
                v = 0.0
            row.append(v)
        rows.append(row)

    X = np.array(rows, dtype=np.float32)
    proba = _model.predict_proba(X)

    if _label_classes is not None and len(_label_classes) == proba.shape[1]:
        classes = list(_label_classes)
    else:
        classes = [str(c) for c in _model.classes_]

    out: List[Dict] = []
    for i in range(len(rows)):
        p = proba[i]
        best_idx = int(np.argmax(p))
        out.append(
            {
                "label": classes[best_idx],
                "probability": float(p[best_idx]),
                "all_probabilities": {c: float(x) for c, x in zip(classes, p)},
            }
        )
    return out


if __name__ == "__main__":
    # Quick smoke test: load model and run a couple of synthetic predictions
    sample = {
        "frp": 80.0,
        "bright_ti4": 340.0,
        "bright_ti5": 305.0,
        "confidence_val": 1.0,
        "is_night": 1.0,
        "is_viiirs": 1.0,
        "is_modis": 0.0,
        "is_offshore": 0.0,
        "dist_refinery": 0.5,
        "dist_factory": 0.3,
        "dist_industrial_zone": 0.2,
        "dist_oil_gas": 5.0,
        "dist_mining": 999.0,
        "dist_volcano": 999.0,
        "dist_powerplant": 12.0,
        "has_refinery_5km": 1.0,
        "has_powerplant_5km": 0.0,
        "has_factory_5km": 1.0,
        "has_volcano_10km": 0.0,
        "has_industrial_2km": 1.0,
        "count_ind_5km": 3.0,
        "count_ref_5km": 1.0,
    }
    import json
    print("[predict] Smoke test on synthetic industrial-fire-style input:")
    print(json.dumps(predict_proba(sample), indent=2))
