# THERMOSCOPE-AI

**AI-Based Detection and Classification of Industrial Fires and Persistent Thermal Sources**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-ready-green)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-multi--class-orange)](https://xgboost.readthedocs.io/)
[![SIH 2026](https://img.shields.io/badge/SIH-2026-red)](https://www.sih.gov.in/)
[![Test Accuracy](https://img.shields.io/badge/Test_Accuracy-94.33%25-brightgreen)](data/processed/hotspots/classification_report.txt)
[![Macro F1](https://img.shields.io/badge/Macro_F1-0.936-yellowgreen)](backend/app/ml/models/training_metrics.json)

> Smart India Hackathon 2026 — Problem Statement **SIH26162** (NTRO / Disaster Management)

## 📑 Table of Contents

- [Executive Summary](#-executive-summary)
- [Classification Taxonomy](#-7-canonical-classification-taxonomy)
- [Architecture & Data Flow](#%EF%B8%8F-project-architecture--data-flow)
- [Project Structure](#-complete-project-structure--file-index)
- [Module Guide](#-what-every-module--file-does)
- [Model Performance](#-model-performance--evaluation-results)
- [Quickstart](#-quickstart--installation)
- [Execution Commands](#-execution--verification-commands)
- [Hybrid Intelligence (Phase C)](#-hybrid-intelligence-fusion-phase-c)
- [Data Integrity](#%EF%B8%8F-data-integrity--reproducibility-guarantees)
- [Known Limitations](#-known-limitations--honest-baseline)
- [Team & Problem Statement](#-problem-statement--team)

## 📌 Executive Summary

Satellite thermal sensors (NASA VIIRS and MODIS) detect high-temperature infrared anomalies across the Indian subcontinent daily. But raw telemetry is just hot pixels — latitude, longitude, brightness temperature, Fire Radiative Power — with no distinction between an active industrial plant fire, an oil/gas flare, a surface coal mine, a forest wildfire, or agricultural stubble burning.

**THERMOSCOPE-AI** closes that gap with a multi-stage geospatial intelligence + machine learning pipeline:

1. **Near-real-time ingestion** — active thermal detections across India via the NASA FIRMS API (`VIIRS_SNPP_NRT`, `VIIRS_NOAA20_NRT`, `MODIS_NRT`), cached locally with 24 h freshness.
2. **Granular spatial enrichment** — **24,022** OpenStreetMap sites indexed (20,231 industrial: factories, industrial zones, power plants, refineries, oil/gas, mines + 3,791 forest/agriculture) with geodesic Haversine distance vectors per hotspot.
3. **Weak supervision engine** — **13** domain-expert Labeling Functions (LFs) with majority-vote consensus generate high-confidence labels with zero manual annotation.
4. **ML classifier** — multi-class XGBoost on 17 engineered features with stratified 5-fold cross-validation: **94.33% test accuracy, 0.936 macro F1** (over classes with test support).
5. **Real-time inference & explainability** — per-class probability distributions (`predict_proba` / `batch_predict`), XGBoost gain rankings, GIS-ready Pydantic schemas, and a hybrid rules×ML fusion layer with human-review flagging.

## 🎯 7 Canonical Classification Taxonomy

Defined in `backend/app/core/constants.py` (`CLASS_LABELS`, with display names and GIS colors):

| # | Canonical Label | Display Label | Signature |
|---|---|---|---|
| 1 | `industrial_fire` | Industrial Fire 🔴 `#FF4500` | High-intensity event (FRP ≥ 35 MW, elevated brightness) inside/adjacent to factories or industrial zones |
| 2 | `gas_flare` | Gas Flare / Persistent Thermal Source 🟠 `#FF8C00` | Persistent source at oil/gas, chemical or refinery sites (often nocturnal) |
| 3 | `mining_activity` | Mining Activity 🟤 `#8B4513` | Surface coal/mineral extraction or quarry operations + thermal signal |
| 4 | `agricultural_burn` | Agricultural Burn 🟡 `#DAA520` | Seasonal crop-residue / stubble burning on farmland, moderate FRP |
| 5 | `forest_natural_fire` | Forest / Natural Fire 🟢 `#228B22` | Vegetation/wildfire in designated forests, isolated from industry |
| 6 | `industrial_process_heat` | Industrial Process Heat 🔺 `#DC143C` | Steady low-to-moderate emission (kilns, smelters, boilers; FRP ≤ 15 MW), no spreading fire |
| 7 | `unclassified` | Unknown / Requires Verification ⚪ `#808080` | Insufficient evidence or ambiguous signature — abstain, don't guess |

## 🏗️ Project Architecture & Data Flow

```text
┌───────────────────────────────┐
│   NASA FIRMS Satellite API    │
│  (VIIRS SNPP, NOAA-20, MODIS) │
└───────────────┬───────────────┘
                │ Raw hotspots (lat/lon, FRP, brightness)
                ▼
┌───────────────────────────────┐
│  OpenStreetMap (OSM) Cache    │
│  20,231 industrial +          │
│  3,791 forest/agriculture     │
└───────────────┬───────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────┐
│   Geodesic Distance Engine (geo/distance.py)           │
│   Nearest distance to all 7 spatial categories         │
│   (meters + km aliases, dual naming convention)        │
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│   Weak Supervision (intelligence/labeling_functions.py)│
│   13 domain-expert LFs → majority-vote consensus       │
│   (label_aggregator.py)                                │
└────────────────────────────┬───────────────────────────┘
                             │ training_dataset.csv
                             ▼
┌────────────────────────────────────────────────────────┐
│   XGBoost Multi-Class Classifier (ml/train.py)         │
│   Stratified 80/20 split + 5-fold CV                   │
└────────────────────────────┬───────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│  Evaluation Artifacts   │     │   Live Inference        │
│  - confusion_matrix.png │     │   - predict_proba()     │
│  - feature_importance   │     │   - batch_predict()     │
│  - metrics JSON/report  │     │   - hybrid_engine       │
└─────────────────────────┘     └─────────────────────────┘
```

## 📁 Complete Project Structure & File Index

```text
THERMOSCOPE-AI/
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py          # Pydantic settings: FIRMS key, DB, CORS, India bbox, cache
│       │   └── constants.py       # 7-class taxonomy, display names, GIS colors, thresholds
│       ├── geo/
│       │   ├── __init__.py
│       │   ├── distance.py        # Haversine engine (meters + km), batch helpers
│       │   └── spatial_context.py # Multi-category distance aggregation → schemas
│       ├── intelligence/
│       │   ├── __init__.py
│       │   ├── labeling_functions.py  # 13 weak-supervision rules + 9-scenario test harness
│       │   ├── label_aggregator.py    # Majority-vote consensus, vote summaries
│       │   └── hybrid_engine.py       # Phase C: rules × ML fusion + human-review flag
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── dataset_builder.py # 17-feature matrix + weak labels → training_dataset.csv
│       │   ├── train.py           # XGBoost + 5-fold CV, rare-class + coverage-gap guards
│       │   ├── evaluate.py        # Confusion matrix, feature importance, metrics JSON
│       │   ├── predict.py         # Cached singleton: predict_proba() / batch_predict()
│       │   └── models/
│       │       ├── hotspot_classifier.joblib
│       │       ├── feature_columns.joblib
│       │       ├── label_classes.joblib
│       │       ├── label_encoder.joblib
│       │       └── training_metrics.json
│       ├── schemas/
│       │   ├── __init__.py
│       │   ├── hotspot.py         # FIRMS detection input records
│       │   ├── spatial_context.py # Multi-distance geospatial context
│       │   └── analysis.py        # HotspotAnalysis: label, confidence, explanation, risk
│       └── services/
│           ├── __init__.py
│           ├── firms_service.py   # FIRMS API ingestion, 24 h cache
│           ├── osm_service.py     # Overpass client + local cache (26 states), tag taxonomy
│           └── hotspot_service.py # Orchestrator: ingest → enrich → hybrid classify → analyze
├── data/
│   ├── classified/
│   │   └── classified_hotspots_v2.csv      # Build output incl. hybrid labels (archived copy)
│   ├── processed/hotspots/
│   │   ├── classified_hotspots_v2.csv      # Canonical 642-row feature dataset (42 cols, both unit conventions)
│   │   ├── classified_hotspots_v2_enriched.csv  # Hybrid-engine outputs + risk scores
│   │   ├── training_dataset.csv            # Numeric 17-feature matrix + consensus labels (968 rows)
│   │   ├── classification_report.txt       # Precision/recall/F1 per class
│   │   ├── evaluation_metrics.json         # Accuracy, macro F1, test-split metadata
│   │   ├── confusion_matrix.png            # Actual vs predicted heatmap
│   │   └── feature_importance.png          # Top-15 XGBoost gain chart
│   └── raw/
│       ├── firms_recent.csv                # Cached FIRMS detections for India
│       ├── firms_recent_india.csv          # India-filtered FIRMS snapshot
│       └── osm/
│           ├── osm_industrial_sites.json   # 20,231 industrial/refinery/power/mining nodes
│           └── osm_forest_agriculture.json # 3,791 forest + agriculture sites (27 states)
├── scripts/
│   ├── __init__.py
│   ├── build_synth_v2.py      # FIRMS × OSM enrichment → v2 CSV (canonical producer)
│   ├── build_real_dataset.py  # Real-only nationwide builder (no demo fallback, 45 km sentinel)
│   ├── build_demo_dataset.py  # Synthetic demo generator (FORBIDDEN for training — integrity-guarded)
│   ├── add_synthetic_mining.py# Known Indian mining-cluster proxies for the OSM cache
│   ├── check_data_integrity.py# Pre-training guard: no synthetic markers / demo fallbacks
│   ├── fetch_firms_api.py     # Live FIRMS API fetcher
│   ├── fetch_osm_daily.py     # Daily Overpass fetcher for forest/agriculture landuse
│   ├── fetch_mining.py        # Overpass queries for coal/iron-ore/bauxite mines
│   └── fetch_state.py         # Per-state bounding-box query utility
├── tests/
│   ├── __init__.py
│   └── test_build_real_dataset.py  # Unit tests (confidence, day/night, integrity)
├── check_states_v2.py         # Diagnostic: hotspot counts per state
├── gate2_check.py             # Phase D gate verification script
├── .env.example               # Config template (copy to .env, add your FIRMS key)
├── .gitignore
├── REPRODUCIBILITY.md         # Academic reproducibility + baseline integrity notes
├── requirements.txt
└── README.md
```

## 🔬 What Every Module & File Does

**`backend/app/core/`** — `config.py` reads `.env` via Pydantic `BaseSettings` (FIRMS key, DB URL, CORS, India bbox, cache hours; comma-separated bbox strings handled). `constants.py` is the single source of truth: 7-class taxonomy, display names, GIS colors, sensor types, proximity thresholds.

**`backend/app/geo/`** — `distance.py` implements the spherical Haversine formula (R = 6,371 km) with point-to-point, batch, and nearest-candidate helpers in meters or km. `spatial_context.py` aggregates distances across all 7 infrastructure categories into structured dicts/models.

**`backend/app/intelligence/`** — `labeling_functions.py` holds **13** explainable detectors (industrial fire ×3, gas flare ×2, mining ×2, agriculture ×2, forest ×2, process heat ×2) plus safe helpers (`get_distance_meters` accepts meter fields and km aliases, `is_missing` treats 999/NaN/Inf as missing). Zero eager guessing — insufficient evidence returns abstain (`None`). Includes a 9-scenario self-test harness (run: `python -m app.intelligence.labeling_functions` → 9/9 PASS). `label_aggregator.py` applies majority voting; total abstention or ties fall back to `unclassified`. `hybrid_engine.py` fuses LF consensus with XGBoost probabilities into `final_label` + `hybrid_confidence` + `decision_source`, flagging disagreements for human review.

**`backend/app/ml/`** — `dataset_builder.py` runs the LFs over the v2 CSV, extracts 17 numeric features, writes `training_dataset.csv`. `train.py` fits XGBoost (200 trees, depth 6) on a stratified 80/20 split + 5-fold CV, with automatic fallback to a non-stratified split when a class has < 2 samples and explicit coverage-gap warnings (< 15 train samples). `evaluate.py` regenerates the confusion matrix, feature-importance chart, text report and metrics JSON on the held-out split. `predict.py` serves cached single/batch inference with full probability distributions.

**`backend/app/services/`** — `firms_service.py` (FIRMS ingestion + 24 h cache), `osm_service.py` (Overpass + 26-state cache + tag taxonomy), `hotspot_service.py` (end-to-end orchestrator: ingest → enrich → hybrid classify → `HotspotAnalysis`).

## 📊 Model Performance & Evaluation Results

Fresh run on the current 968-row training set (774 train / 194 test), artifacts in `data/processed/hotspots/` + `backend/app/ml/models/training_metrics.json`:

```text
                     precision    recall  f1-score   support
  agricultural_burn       0.98      0.91      0.94        45
forest_natural_fire       1.00      1.00      1.00         1
          gas_flare       0.96      0.97      0.96        94
    industrial_fire       0.86      0.86      0.86         7
    mining_activity       0.90      0.94      0.92        47
       unclassified       0.00      0.00      0.00         0
           accuracy                           0.94       194
          macro avg       0.78      0.78      0.78       194
```

| Metric | Value |
|---|---|
| Test accuracy | **94.33%** (183/194) |
| Macro F1 (5 classes with test support) | **0.9359** |
| 5-fold CV accuracy | **95.15% ± 1.68%** |
| 5-fold CV macro F1 | **0.9124 ± 0.0538** |
| Top features (XGBoost gain) | `has_industrial_2km` 0.47, `bright_ti4` 0.18, `frp` 0.13, `dist_factory` 0.09, `bright_ti5` 0.02 |

> Note: `unclassified` (1 train sample) and `forest_natural_fire` (15 train) are thinly represented — see [Known Limitations](#-known-limitations--honest-baseline). Macro F1 0.9359 averages the 5 classes with test support; including the zero-support class it is 0.78.

## ⚡ Quickstart & Installation

**Prerequisites:** Python 3.10+ (3.11 recommended), a free NASA FIRMS Map Key ([register here](https://firms.modaps.eosdis.nasa.gov/api/area/)).

```powershell
# 1. Clone
git clone https://github.com/anshux1098/THERMOSCOPE-AI.git
cd THERMOSCOPE-AI

# 2. Virtual environment (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
# Linux/macOS: python3 -m venv venv; source venv/bin/activate

# 3. Dependencies (+ pytest for unit tests)
pip install -r requirements.txt
pip install pytest   # optional, not in requirements

# 4. Configure secrets (never commit .env)
Copy-Item .env.example .env   # cp .env.example .env on Linux/macOS
# Edit .env: FIRMS_MAP_KEY=your_key_here
```

## 🚀 Execution & Verification Commands

Run from the repo root with the venv active. Canonical module paths (`backend.app.ml.*`, `app.intelligence.*`):

```powershell
# Step 1: Pre-training integrity guard (must pass before training)
venv\Scripts\python scripts/check_data_integrity.py

# Step 2: Unit tests (requires pytest, see above)
venv\Scripts\python -m pytest tests/test_build_real_dataset.py -v

# Step 3: Rebuild the v2 feature dataset (642-hotspot snapshot × 24k OSM, ~2 min)
# NOTE: invoke by file path or in-process import — `python -m scripts.build_synth_v2`
# hangs in this environment (package-mode quirk, no output, near-zero CPU).
venv\Scripts\python scripts/build_synth_v2.py

# Step 4: Build weak-supervision training set
venv\Scripts\python -m backend.app.ml.dataset_builder

# Step 5: Train XGBoost (saves model + metrics to backend/app/ml/models/)
venv\Scripts\python -m backend.app.ml.train

# Step 6: Regenerate evaluation plots + report
venv\Scripts\python -m backend.app.ml.evaluate

# Step 7: Inference smoke test (predict_proba / batch_predict)
venv\Scripts\python -u -c "from app.ml.predict import predict_proba; print(predict_proba({'frp':65.0,'bright_ti4':345.0,'bright_ti5':305.0,'dist_refinery':12.0,'dist_factory':0.35,'dist_industrial_zone':0.35,'dist_oil_gas':8.5,'dist_mining':999.0,'dist_forest':5.0,'dist_agriculture':2.5,'dist_powerplant':3.2,'has_refinery_5km':0,'has_powerplant_5km':0,'has_factory_5km':1,'has_industrial_2km':1,'count_ind_5km':4,'count_ref_5km':0}))"
# → {'label': 'industrial_fire', 'probability': ~0.73, 'all_probabilities': {...}}

# Step 8: Labeling-function self-test (9/9 scenarios must PASS)
venv\Scripts\python -m app.intelligence.labeling_functions
```

## 🔀 Hybrid Intelligence Fusion (Phase C)

`backend/app/intelligence/hybrid_engine.py` combines both intelligence paths per hotspot:

- **Rule path**: 13 LF votes → majority consensus + vote breakdown.
- **ML path**: XGBoost `predict_proba` over the 17-feature vector.
- **Fusion**: agreement → `hybrid_agreement` with boosted confidence; disagreement → lower confidence + `requires_human_review=True`, both explanations preserved in `explanation_bullets`.
- Outputs land in `classified_hotspots_v2_enriched.csv` (`final_label`, `hybrid_confidence`, `decision_source`, `agreement`, `conflict`, `risk_score`, …) and validate against `gate2_check.py`.

## 🛡️ Data Integrity & Reproducibility Guarantees

- **No synthetic training fallbacks** — `check_data_integrity.py` fails the run (exit 1) if any `_is_synthetic_demo` marker or `allow_demo_fallback=True` exists in the training path. `build_demo_dataset.py` output can never leak into training.
- **Deterministic** — fixed `random_state=42` for splits, CV folds, and XGBoost; reruns reproduce metrics bit-for-bit (see `REPRODUCIBILITY.md`).
- **Strict unit discipline** — `dist_*` columns are kilometers, `distance_to_*_m` / `dist_*_m` are meters; 999 is the single missing-value sentinel recognized by `is_missing()`.
- **Dual naming convention** — v2 CSVs carry both `dist_*` (km) and `distance_to_*_m` (m); `get_distance_meters()` resolves either, so rule and ML paths can never desync on units.

## ⚠️ Known Limitations (Honest Baseline)

- Train accuracy is 1.0 vs 94.3% test — mild overfit; the model memorizes the small (968-row) weak-label set. More FIRMS history is the fix, not hyperparameter tuning.
- `unclassified` has 1 training sample and `forest_natural_fire` only 15 — expect ~0% recall there until OSM coverage / label yield improves.
- `mining_activity` recall depends on the 20-site mining proxy cache (`add_synthetic_mining.py`) — approximate coordinates of known mining belts, documented as such.
- `python -m scripts.build_synth_v2` hangs in this environment; use `python scripts/build_synth_v2.py` instead (identical code path, completes in ~2 min).

## 👥 Collaboration Workflow

```powershell
git checkout -b feat/<your-feature>
# ...edit, then verify: integrity guard → dataset_builder → train → evaluate...
git add <files>; git commit -m "feat: <what + why>"
git push -u origin feat/<your-feature>
# Open a Pull Request → a teammate reviews → merge to main
```

`.env` (FIRMS key) and `venv/` are gitignored — new clones start from `.env.example`.

## 📜 Problem Statement & Team

- **Smart India Hackathon (SIH 2026)** — Problem Statement ID **SIH26162**
- **Organization**: National Technical Research Organisation (NTRO) / Disaster Management
- **Repository**: https://github.com/anshux1098/THERMOSCOPE-AI
- **Team**: 6 members (collaborative)
- **Stack**: Python · XGBoost · FastAPI · OSM/Overpass · Pydantic · pandas/scikit-learn
