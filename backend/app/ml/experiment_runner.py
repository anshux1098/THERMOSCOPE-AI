"""
experiment_runner.py
Phase E4 — Controlled imbalance-mitigation experiment harness.

Runs the predefined E4 experiment matrix on the SAME weak-supervision training
dataset, preserving the E3 baseline as the control:

    E4-A  control                 : no intervention (must reproduce E3 metrics)
    E4-B1 inverse                 : w_c = N_train / n_c          (uncapped)
    E4-B2 sqrt_inverse            : w_c = sqrt(N_train / n_c)
    E4-B3 capped_inverse          : min(N_train / n_c, cap=10.0) (predefined cap)
    E4-C  oversample_floor30      : bounded train-only duplication to floor=30, seed 42

Isolation & integrity guarantees:
  * NEVER writes to the production model directory (backend/app/ml/models/).
  * Every candidate is written to backend/app/ml/experiments/e4/<name>/ plus a manifest.
  * The train/test split reproduces train.py EXACTLY: same random_state=42, same
    non-stratified fallback, same rare-class move-to-train guard, same 253 test rows.
  * Weights are computed from TRAINING labels only (test data never enters weighting).
  * CV replicates train.py's StratifiedKFold(5, shuffle=True, random_state=42) with the
    exact fold-skip rules; skipped folds are reported, never silently averaged.
  * The hybrid pass calls the FROZEN hybrid engine (classify_hotspot) with the candidate
    model swapped in-memory, over the canonically-built feature dicts — production model
    files and the frozen rules/hybrid logic are untouched.

Usage:
    python -m app.ml.experiment_runner --experiments e4a_control,e4b1_weight_inverse
    python -m app.ml.experiment_runner --all
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Make app package importable
backend_dir = str(Path(__file__).resolve().parents[2])
root_dir = str(Path(__file__).resolve().parents[3])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier

from app.ml.dataset_builder import FEATURE_COLUMNS, LABEL_COLUMN, DEFAULT_OUTPUT_CSV
from app.ml.imbalance import (
    compute_sample_weights,
    bounded_oversample,
    OVERSAMPLE_SEED,
)
from app.core.paths import (
    CLASSIFIED_DATASET_PATH,
    ENRICHED_DATASET_PATH,
)
from app.core.lineage import validate_training_dataset

# ---------------------------------------------------------------------------
# Experiment registry (predefined BEFORE any result inspection — no cherry picking)
# ---------------------------------------------------------------------------
EXPERIMENT_DIR = Path(backend_dir) / "app" / "ml" / "experiments" / "e4"
REPORT_DIR = Path(root_dir) / "reports" / "e4"

# Cap chosen BEFORE looking at any outcome: bounds the 1-sample class influence
# (raw inverse weight for agricultural_burn would be ~1015) to 10x majority.
CAPPED_INVERSE_CAP = 10.0
# Oversampling floor chosen BEFORE any outcome: raise every class present in the
# train split to >= 30 rows (comparable to the natural forest support of 36).
OVERSAMPLE_FLOOR = 30

# Frozen XGBoost architecture (identical to train.py — E3 and E4 share it).
def make_model(n_classes: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_predict(model: XGBClassifier, X) -> np.ndarray:
    preds = model.predict(X)
    if getattr(preds, "ndim", 0) == 2:
        preds = np.argmax(preds, axis=1)
    return preds.astype(int)


# ---------------------------------------------------------------------------
# Split — exact replica of train.py's deterministic split (audit + reproduction)
# ---------------------------------------------------------------------------
def build_split(df: pd.DataFrame) -> Dict[str, Any]:
    X = df[FEATURE_COLUMNS].astype(np.float32)
    y_raw = df[LABEL_COLUMN].astype(str)

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw), index=y_raw.index)
    class_names = le.classes_.tolist()
    n_classes = len(class_names)

    stratify = y
    class_counts = y.value_counts()
    if (class_counts < 2).any():
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    missing = set(y.unique()) - set(y_train.unique())
    for cls in missing:
        mask_test = y_test.eq(cls)
        X_train = pd.concat([X_train, X_test.loc[mask_test]])
        y_train = pd.concat([y_train, y_test[mask_test]])
        X_test = X_test.loc[~mask_test]
        y_test = y_test.loc[~mask_test]

    return {
        "X": X, "y": y,
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "class_names": class_names, "n_classes": n_classes,
    }


def split_support_table(sp: Dict[str, Any], class_names: List[str]) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(class_names):
        rows.append({
            "class": c,
            "total": int((sp["y"] == i).sum()),
            "train": int((sp["y_train"] == i).sum()),
            "test": int((sp["y_test"] == i).sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Preprocessing strategies (applied to TRAIN only — leakage-safe by construction)
# ---------------------------------------------------------------------------
def _prep_weights(strategy: str, cap: Optional[float] = None) -> Callable:
    def fn(x_tr: pd.DataFrame, y_tr: pd.Series, n_classes: int, class_names: List[str]):
        y_names = pd.Series([class_names[int(i)] for i in y_tr], index=y_tr.index)
        w, audit = compute_sample_weights(y_names, strategy=strategy, cap=cap, normalize=True)
        return x_tr, y_tr, w, audit
    return fn


def _prep_oversample(floor: int = OVERSAMPLE_FLOOR, seed: int = OVERSAMPLE_SEED) -> Callable:
    def fn(x_tr: pd.DataFrame, y_tr: pd.Series, n_classes: int, class_names: List[str]):
        y_names = pd.Series([class_names[int(i)] for i in y_tr], index=y_tr.index)
        x_dup, y_dup = bounded_oversample(
            x_tr, y_names, floor=floor, random_state=seed
        )
        audit = pd.DataFrame(
            {
                "class": y_names.value_counts().sort_index().index,
                "original_frequency": y_names.value_counts().sort_index().values,
                "final_frequency": [
                    int((y_dup.values == c).sum()) for c in y_names.value_counts().sort_index().index
                ],
            }
        )
        audit["duplicated_rows"] = audit["final_frequency"] - audit["original_frequency"]
        # XGBoost requires integer labels: map names back to class indices.
        idx = {c: i for i, c in enumerate(class_names)}
        y_dup_int = pd.Series([idx[v] for v in y_dup], index=y_dup.index)
        return x_dup, y_dup_int, None, audit
    return fn


def _prep_control(x_tr: pd.DataFrame, y_tr: pd.Series, n_classes: int, class_names: List[str]):
    audit = pd.DataFrame(
        {
            "class": [class_names[int(i)] for i in sorted(set(int(v) for v in y_tr))],
            "frequency": [int((y_tr == i).sum()) for i in sorted(set(int(v) for v in y_tr))],
            "raw_weight": 1.0, "cap_applied": False, "final_weight": 1.0,
            "weighted_rows": [int((y_tr == i).sum()) for i in sorted(set(int(v) for v in y_tr))],
        }
    )
    return x_tr, y_tr, None, audit.copy()


PREPROCESSORS: Dict[str, Callable] = {
    "e4a_control": _prep_control,
    "e4b1_weight_inverse": _prep_weights("inverse"),
    "e4b2_weight_sqrt_inverse": _prep_weights("sqrt_inverse"),
    "e4b3_weight_capped_inverse": _prep_weights("capped_inverse", cap=CAPPED_INVERSE_CAP),
    "e4c_oversample_floor30": _prep_oversample(floor=OVERSAMPLE_FLOOR),
}


def _audit_from_audit_or_identity(w_audit, sp, class_names) -> pd.DataFrame:
    if w_audit is not None and len(w_audit) > 0:
        return w_audit
    supp = split_support_table(sp, class_names)
    return supp[["class", "train"]].rename(columns={"train": "frequency"}).assign(
        raw_weight=1.0, cap_applied=False, final_weight=1.0,
        weighted_rows=lambda d: d["frequency"],
    )


# ---------------------------------------------------------------------------
# Evaluation (single held-out split — identical test rows for every experiment)
# ---------------------------------------------------------------------------
def evaluate_model(model, x_test, y_test, class_names: List[str]) -> Dict[str, Any]:
    pred_int = _safe_predict(model, x_test)
    proba = model.predict_proba(x_test)

    y_true = [class_names[i] for i in np.asarray(y_test, dtype=int)]
    y_pred = [class_names[i] for i in pred_int]

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "class": class_names,
            "precision": p, "recall": r, "f1": f,
            "support_true": [int((np.asarray(y_test, dtype=int) == i).sum()) for i in range(len(class_names))],
            "support_pred": [int((pred_int == i).sum()) for i in range(len(class_names))],
        }
    )

    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=class_names),
        index=[f"true:{c}" for c in class_names],
        columns=[f"pred:{c}" for c in class_names],
    )

    # Multi-class Brier score (mean squared error vs one-hot).
    one_hot = np.zeros_like(proba)
    for r_i, t in enumerate(np.asarray(y_test, dtype=int)):
        one_hot[r_i, t] = 1.0
    brier = float(np.mean((proba - one_hot) ** 2))

    # Macro-F1 over ALL canonical classes (strict; zero for unrepresentable classes)
    macro_f1_all = float(np.mean(f))
    # Macro-F1 exactly as train.py stores it: f1_score(average='macro') with default
    # labels = union(y_true, y_pred) present on the held-out split.
    macro_f1_union = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_p_all = float(np.mean(p))
    macro_r_all = float(np.mean(r))

    metrics = {
        "test_n": int(len(x_test)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": macro_p_all,
        "macro_recall": macro_r_all,
        "macro_f1": macro_f1_all,                       # over all 7 canonical classes
        "macro_f1_test_classes": macro_f1_union,        # E3/train.py-comparable macro
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_score": brier,
    }

    prob_audit = _probability_audit(proba, pred_int, y_test, class_names, y_true, y_pred)

    return {"metrics": metrics, "per_class": per_class, "confusion": cm,
            "probability": prob_audit}


def _probability_audit(proba, pred_int, y_test, class_names, y_true, y_pred) -> Dict[str, Any]:
    top = proba.max(axis=1)
    correct = np.asarray(pred_int, dtype=int) == np.asarray(y_test, dtype=int)

    by_pred = []
    for i, c in enumerate(class_names):
        m = pred_int == i
        if m.any():
            by_pred.append({
                "predicted_class": c, "predicted_count": int(m.sum()),
                "prob_min": float(top[m].min()), "prob_q25": float(np.percentile(top[m], 25)),
                "prob_median": float(np.median(top[m])), "prob_mean": float(top[m].mean()),
                "prob_q75": float(np.percentile(top[m], 75)), "prob_max": float(top[m].max()),
            })
    by_pred = pd.DataFrame(by_pred)

    return {
        "top_prob": {
            "min": float(top.min()), "q25": float(np.percentile(top, 25)),
            "median": float(np.median(top)), "mean": float(top.mean()),
            "q75": float(np.percentile(top, 75)), "max": float(top.max()),
        },
        "mean_top_prob_correct": float(top[correct].mean()),
        "mean_top_prob_incorrect": float(top[~correct].mean()) if (~correct).any() else None,
        "n_correct": int(correct.sum()),
        "n_incorrect": int((~correct).sum()),
        "high_conf_wrong_ge0.9": int(((~correct) & (top >= 0.90)).sum()),
        "high_conf_wrong_ge0.8": int(((~correct) & (top >= 0.80)).sum()),
        "high_conf_pred_ge0.9": int((top >= 0.90).sum()),
        "high_conf_pred_ge0.8": int((top >= 0.80).sum()),
        "by_predicted_class": by_pred,
    }


# ---------------------------------------------------------------------------
# Cross-validation audit — exact replica of train.py's StratifiedKFold protocol
# ---------------------------------------------------------------------------
def run_cv(x, y, n_classes, class_names, preprocess) -> Dict[str, Any]:
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    valid_acc, valid_f1 = [], []
    for fold_i, (tr, va) in enumerate(skf.split(x, y), 1):
        x_tr, x_va = x.iloc[tr], x.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        reason = None
        if y_tr.nunique() < 2:
            reason = "fold train set has only 1 class"
        else:
            lo, hi = int(y_tr.min()), int(y_tr.max())
            if y_tr.nunique() != (hi - lo + 1) or lo != 0:
                reason = "fold train set misses a label class feeding the [0,n_classes) range"

        if reason is not None:
            folds.append({"fold": fold_i, "status": "skipped", "reason": reason})
            continue

        x_tr_p, y_tr_p, w, _ = preprocess(x_tr, y_tr, n_classes, class_names)
        m = make_model(n_classes)
        m.fit(x_tr_p, y_tr_p, sample_weight=w)
        p = _safe_predict(m, x_va)
        acc = float(accuracy_score(y_va, p))
        f1 = float(f1_score(y_va, p, average="macro", zero_division=0))
        valid_acc.append(acc)
        valid_f1.append(f1)
        folds.append({
            "fold": fold_i, "status": "valid",
            "accuracy": acc, "macro_f1": f1,
            "train_support": {class_names[int(c)]: int(k) for c, k in y_tr.value_counts().items()},
            "train_n": int(len(y_tr)),
        })

    return {
        "requested_folds": 5,
        "valid_folds": len(valid_acc),
        "skipped_folds": 5 - len(valid_acc),
        "fold_details": folds,
        "cv_accuracy_mean": float(np.mean(valid_acc)) if valid_acc else None,
        "cv_accuracy_std": float(np.std(valid_acc)) if valid_acc else None,
        "cv_macro_f1_mean": float(np.mean(valid_f1)) if valid_f1 else None,
        "cv_macro_f1_std": float(np.std(valid_f1)) if valid_f1 else None,
    }


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------
def run_experiment(name: str, source_csv: str = DEFAULT_OUTPUT_CSV) -> Dict[str, Any]:
    t0 = time.time()
    out_dir = EXPERIMENT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\n[E4] EXPERIMENT {name}\n{'='*70}")

    df = validate_training_dataset(Path(source_csv))
    sp = build_split(df)
    class_names = sp["class_names"]
    n_classes = sp["n_classes"]

    support = split_support_table(sp, class_names)

    preprocess = PREPROCESSORS[name]
    x_tr, y_tr = sp["X_train"], sp["y_train"]
    x_tr_p, y_tr_p, w, w_audit = preprocess(x_tr, y_tr, n_classes, class_names)

    model = make_model(n_classes)
    model.fit(x_tr_p, y_tr_p, sample_weight=w)

    eval_result = evaluate_model(model, sp["X_test"], sp["y_test"], class_names)
    cv = run_cv(sp["X"], sp["y"], n_classes, class_names, preprocess)

    # Probability audit to CSV (expand by_predicted_class)
    prob_by_pred = eval_result["probability"].pop("by_predicted_class")

    metrics = eval_result["metrics"] | {
        "cv": cv,
        "n_train_raw": int(len(sp["y_train"])),
        "n_train_effective": int(len(y_tr_p)),
        "n_test": int(len(sp["y_test"])),
    }

    # Persist isolated artifacts
    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(class_names, out_dir / "label_classes.joblib")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    eval_result["per_class"].to_csv(out_dir / "per_class.csv", index=False)
    eval_result["confusion"].to_csv(out_dir / "confusion_matrix.csv")
    prob_by_pred.to_csv(out_dir / "probability_audit.csv", index=False)
    support.to_csv(out_dir / "support.csv", index=False)
    with open(out_dir / "probability_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_result["probability"], f, indent=2, default=str)
    if w_audit is not None and len(w_audit) > 0:
        w_audit.to_csv(out_dir / "weights_audit.csv", index=False)
    else:
        _audit_from_audit_or_identity(w_audit, sp, class_names).to_csv(
            out_dir / "weights_audit.csv", index=False
        )

    manifest = {
        "experiment_name": name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "random_state": 42,
        "dataset": {
            "training_dataset": str(Path(source_csv)),
            "rows": int(len(df)),
            "sha256": _sha256(Path(source_csv)),
        },
        "split": {
            "n_train_raw": int(len(sp["y_train"])),
            "n_test": int(len(sp["y_test"])),
            "test_support": support[["class", "test"]].set_index("class")["test"].to_dict(),
            "train_support": support[["class", "train"]].set_index("class")["train"].to_dict(),
        },
        "model": {
            "estimator": "XGBClassifier",
            "params": {
                "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                "objective": "multi:softprob", "random_state": 42,
                "tree_method": "hist", "n_jobs": -1,
            },
            "n_classes": n_classes,
            "classes": class_names,
            "feature_schema": list(FEATURE_COLUMNS),
        },
        "intervention": preprocess.__name__ if hasattr(preprocess, "__name__") else name,
        "weighting": {"strategy": name, "cap": CAPPED_INVERSE_CAP if "capped" in name else None},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[E4 {name}] test_acc={metrics['accuracy']:.4f} "
          f"macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f}")
    print(f"[E4 {name}] balanced_acc={metrics['balanced_accuracy']:.4f} "
          f"brier={metrics['brier_score']:.4f}")
    print(f"[E4 {name}] CV: {metrics['cv']['valid_folds']} valid / "
          f"{metrics['cv']['skipped_folds']} skipped | "
          f"cv_f1={metrics['cv']['cv_macro_f1_mean']:.4f} ± {metrics['cv']['cv_macro_f1_std']:.4f}")
    print(f"[E4 {name}] artifacts -> {out_dir}")

    return {
        "name": name,
        "manifest": manifest,
        "metrics": metrics,
        "per_class": eval_result["per_class"],
        "confusion": eval_result["confusion"],
        "probability": eval_result["probability"],
        "support": support,
        "weights_audit": w_audit,
        "model_path": str(out_dir / "model.joblib"),
    }


# ---------------------------------------------------------------------------
# Frozen hybrid pass for a candidate (swaps model in-memory; never touches files)
# ---------------------------------------------------------------------------
def run_hybrid_candidate(result: Dict[str, Any]) -> pd.DataFrame:
    """Recompute per-row hybrid decisions with the candidate model through the FROZEN
    hybrid engine, over the same canonical feature dicts the production pipeline uses."""
    import app.ml.predict as predict_mod
    from app.intelligence.hybrid_engine import classify_hotspot
    from scripts.run_pipeline import _build_feature_row

    model = joblib.load(result["model_path"])
    class_names = result["manifest"]["model"]["classes"]
    canonical = pd.read_csv(CLASSIFIED_DATASET_PATH)

    orig = (predict_mod._model, predict_mod._feature_columns, predict_mod._label_classes)
    try:
        predict_mod._model = model
        predict_mod._feature_columns = list(FEATURE_COLUMNS)
        predict_mod._label_classes = list(class_names)

        rows = []
        for i in range(len(canonical)):
            rec = classify_hotspot(_build_feature_row(canonical.iloc[i]))
            rows.append({
                "input_index": i,
                "final_label": rec["final_label"],
                "decision_source": rec["decision_source"],
                "hybrid_confidence": rec["hybrid_confidence"],
                "agreement": bool(rec["agreement"]),
                "conflict": bool(rec["conflict"]),
                "requires_human_review": bool(rec["requires_human_review"]),
                "review_reason": rec.get("review_reason"),
                "rule_prediction": rec["rule_engine"]["prediction"],
                "rule_active_votes": rec["rule_engine"]["active_votes"],
                "ml_prediction": rec["ml_engine"]["prediction"],
                "ml_top_probability": rec["ml_engine"]["confidence"],
            })
    finally:
        predict_mod._model, predict_mod._feature_columns, predict_mod._label_classes = orig

    return pd.DataFrame(rows)


def hybrid_summary(hyb: pd.DataFrame) -> Dict[str, Any]:
    return {
        "final_labeled": int((hyb["final_label"] != "unclassified").sum()),
        "final_unclassified": int((hyb["final_label"] == "unclassified").sum()),
        "decision_source": hyb["decision_source"].value_counts().to_dict(),
        "review_flags": int(hyb["requires_human_review"].sum()),
        "conflict_flags": int(hyb["conflict"].sum()),
        "agreements": int(hyb["agreement"].sum()),
        "final_class_counts": hyb["final_label"].value_counts().to_dict(),
        "ml_prediction_counts": hyb["ml_prediction"].value_counts().to_dict(),
    }


def audit_transitions(e3: pd.DataFrame, cand: pd.DataFrame) -> Dict[str, Any]:
    m = e3.merge(cand, on="input_index", suffixes=("_e3", "_cand"))
    tm = pd.crosstab(m["final_label_e3"], m["final_label_cand"], dropna=False)
    losses = m[(m["final_label_e3"] != "unclassified") & (m["final_label_cand"] == "unclassified")]
    changed = m[m["final_label_e3"] != m["final_label_cand"]]
    new_conflicts = m[(m["conflict_e3"] == False) & (m["conflict_cand"] == True)]  # noqa: E712
    return {
        "transition_matrix": tm,
        "labeled_to_unclassified_losses": int(len(losses)),
        "lost_rows": losses["input_index"].tolist(),
        "changed_rows": changed["input_index"].tolist(),
        "n_changed": int(len(changed)),
        "new_conflicts_count": int(len(new_conflicts)),
        "new_conflict_rows": new_conflicts["input_index"].tolist(),
        "rows": m,
    }


def load_experiment(name: str) -> Dict[str, Any]:
    """Reload a previously persisted experiment for reporting/aggregation."""
    d = EXPERIMENT_DIR / name
    if not (d / "manifest.json").exists():
        raise FileNotFoundError(f"Experiment {name} not found. Run it first.")
    return {
        "name": name,
        "manifest": json.loads((d / "manifest.json").read_text()),
        "metrics": json.loads((d / "metrics.json").read_text()),
        "per_class": pd.read_csv(d / "per_class.csv"),
        "confusion": pd.read_csv(d / "confusion_matrix.csv", index_col=0),
        "support": pd.read_csv(d / "support.csv"),
        "weights_audit": pd.read_csv(d / "weights_audit.csv"),
        "model_path": str(d / "model.joblib"),
    }


def run_hybrid_phase(names: List[str]) -> None:
    """STEP O — run the FROZEN hybrid engine with each candidate model over the same
    1,268 canonical rows, and persist per-row hybrid results + E3-side comparison."""
    e3 = pd.read_csv(ENRICHED_DATASET_PATH)
    e3_cols = {"input_index", "final_label", "decision_source", "agreement", "conflict",
               "requires_human_review", "rule_prediction", "rule_active_votes",
               "ml_prediction", "ml_top_probability"}
    if not e3_cols.issubset(set(e3.columns)):
        raise ValueError(f"E3 enriched missing required columns. Have: {sorted(e3.columns)}")

    for name in names:
        result = load_experiment(name)
        print(f"[E4 hybrid] {name} — running frozen hybrid engine over 1,268 canonical rows...")
        hyb = run_hybrid_candidate(result)
        pure_rule = e3[["input_index", "rule_prediction", "rule_active_votes"]].copy()
        hyb = hyb.merge(pure_rule, on="input_index", suffixes=("", "_e3"))
        # Immutability cross-check: the frozen rules must reproduce the E3 rule layer exactly.
        rule_ok = (hyb["rule_prediction"] == hyb["rule_prediction_e3"]).all() and \
                  (hyb["rule_active_votes"] == hyb["rule_active_votes_e3"]).all()
        print(f"[E4 hybrid] {name} — rule-layer immutability (rules==E3): {rule_ok}")
        if not rule_ok:
            print("[E4 hybrid] WARNING: rule layer drifted — frozen LFs should be identical.")

        hyb = hyb.drop(columns=["rule_prediction_e3", "rule_active_votes_e3"])
        hyb.to_csv(EXPERIMENT_DIR / name / "hybrid_results.csv", index=False)

        summary = hybrid_summary(hyb)
        with open(EXPERIMENT_DIR / name / "hybrid_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        audit = audit_transitions(e3, hyb)
        audit["transition_matrix"].to_csv(EXPERIMENT_DIR / name / "transition_matrix.csv")
        audit["rows"].to_csv(EXPERIMENT_DIR / name / "transition_rows.csv", index=False)
        with open(EXPERIMENT_DIR / name / "transition_audit.json", "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in audit.items() if k not in ("transition_matrix", "rows")},
                f, indent=2,
            )
        print(f"[E4 hybrid] {name} — labeled->unclassified losses: "
              f"{audit['labeled_to_unclassified_losses']} | changed rows: "
              f"{audit['n_changed']} | new conflicts: {audit['new_conflicts_count']}")
    print("[E4 hybrid] DONE — hybrid results + transition audits written per experiment.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default=",".join(PREPROCESSORS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--hybrid", action="store_true", help="Run frozen hybrid + audits for all experiments")
    args = parser.parse_args()

    names = list(PREPROCESSORS) if args.all else [n.strip() for n in args.experiments.split(",") if n.strip()]
    for n in names:
        run_experiment(n)
    if args.hybrid:
        run_hybrid_phase(names)
    print("\n[DONE] All requested experiments persisted under", EXPERIMENT_DIR)