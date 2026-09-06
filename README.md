# THERMOSCOPE-AI

**AI-Based Detection and Classification of Industrial Fires and Persistent Thermal Sources**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-ready-green)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-multi--class-orange)](https://xgboost.readthedocs.io/)
[![SIH 2026](https://img.shields.io/badge/SIH-2026-red)](https://www.sih.gov.in/)
[![Test Accuracy](https://img.shields.io/badge/Test_Accuracy-98.81%25-brightgreen)](backend/app/ml/models/training_metrics.json)

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
- [FastAPI & Frontend Integration](#-fastapi--frontend-integration)
- [Hybrid Intelligence (Phase C)](#-hybrid-intelligence-fusion-phase-c)
- [Data Integrity](#%EF%B8%8F-data-integrity--reproducibility-guarantees)
- [Known Limitations](#-known-limitations--honest-baseline)
- [Team & Problem Statement](#-problem-statement--team)

## 📌 Executive Summary

Satellite thermal sensors (NASA VIIRS and MODIS) detect high-temperature infrared anomalies across the Indian subcontinent daily. But raw telemetry is just hot pixels — latitude, longitude, brightness temperature, Fire Radiative Power — with no distinction between an active industrial plant fire, an oil/gas flare, a surface coal mine, a forest wildfire, or agricultural stubble burning.

**THERMOSCOPE-AI** closes that gap with a multi-stage geospatial intelligence + machine learning pipeline:

1. **Near-real-time ingestion** — active thermal detections across India via the NASA FIRMS API (`VIIRS_SNPP_NRT`, `VIIRS_NOAA20_NRT`, `MODIS_NRT`), cached locally with 24 h freshness.
2. **Granular spatial enrichment** — **24,022** OpenStreetMap sites indexed (20,231 industrial: factories, industrial zones, power plants, refineries, oil/gas, mines + 3,791 forest/agriculture) with geodesic Haversine distance vectors per hotspot.
3. **Weak supervision engine** — **14** domain-expert Labeling Functions (LFs) with majority-vote consensus generate high-confidence labels with zero manual annotation.
4. **ML classifier** — multi-class XGBoost on 17 engineered features with stratified 5-fold cross-validation: **98.81% test accuracy** on the current 1,268-hotspot snapshot (see [Model Performance](#-model-performance--evaluation-results) for honest caveats).
5. **Hybrid Rules × ML intensity** — agreement boosts confidence; disagreement lowers confidence, flags `requires_human_review`, and preserves both explanations. Risk score derived from hybrid confidence.
6. **API-ready** — FastAPI backend (`backend/app/main.py`) exposing the end-to-end hotspot analysis as a thin HTTP layer for the frontend dashboard.

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
│   14 domain-expert LFs → majority-vote consensus       │
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
│  Evaluation (training)  │     │   Production Inference  │
│  ml/evaluate.py →       │     │   predict.py +          │
│  training_metrics.json  │     │   hybrid_engine.py      │
└─────────────────────────┘     └────────────┬────────────┘
                                             ▼
                               ┌────────────────────────────┐
                               │  FastAPI (backend/app)     │
                               │  POST /api/v1/hotspots/    │
                               │       analyze  →  Frontend │
                               └────────────────────────────┘
```

## 📁 Complete Project Structure & File Index

```text
THERMOSCOPE-AI/
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── main.py                  # FastAPI entry: CORS, /health, mounts /api/v1
│       ├── api/
│       │   ├── __init__.py
│       │   └── hotspots.py          # POST /api/v1/hotspots/analyze (thin adapter)
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py            # Pydantic settings: FIRMS key, DB, CORS, India bbox, cache
│       │   ├── constants.py         # 7-class taxonomy, display names, GIS colors, thresholds
│       │   ├── lineage.py           # Data lineage + integrity guards
│       │   └── paths.py             # Canonical path contracts
│       ├── geo/
│       │   ├── __init__.py
│       │   ├── distance.py          # Haversine engine (meters + km), batch helpers
│       │   ├── spatial_context.py   # Multi-category distance aggregation → schemas
│       │   └── spatial_features.py  # 17-feature spatial schema + sentinel semantics
│       ├── intelligence/
│       │   ├── __init__.py
│       │   ├── labeling_functions.py  # 14 explainable LFs + self-test harness (9/9)
│       │   ├── label_aggregator.py    # Majority-vote consensus, vote summaries
│       │   └── hybrid_engine.py       # Rules × ML fusion + human-review flag (Phase C)
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── dataset_builder.py     # 17-feature matrix + weak labels → training_dataset.csv
│       │   ├── train.py               # XGBoost + 5-fold CV, coverage-gap guards (training)
│       │   ├── evaluate.py            # Confusion matrix, feature importance, metrics (training)
│       │   ├── predict.py             # Production singleton: predict_proba() / batch_predict()
│       │   ├── imbalance.py           # [research] E4 class-weighting / oversample preprocessors
│       │   ├── experiment_runner.py   # [research] E4 controlled-experiment harness (regenerates experiment outputs on demand)
│       │   └── models/                # ★ Production artifacts (E3)
│       │       ├── hotspot_classifier.joblib
│       │       ├── feature_columns.joblib
│       │       ├── label_classes.joblib
│       │       ├── label_encoder.joblib
│       │       └── training_metrics.json   # canonical E3 metrics
│       ├── schemas/
│       │   ├── __init__.py
│       │   ├── hotspot.py            # FIRMS detection input records
│       │   ├── spatial_context.py    # Multi-distance geospatial context
│       │   └── analysis.py           # HotspotAnalysis: label, confidence, explanation
│       └── services/
│           ├── __init__.py
│           ├── firms_service.py      # FIRMS API ingestion, 24 h cache
│           ├── osm_service.py        # Overpass client + local cache, tag taxonomy
│           └── hotspot_service.py    # Orchestrator: ingest → enrich → hybrid classify → analyze
├── data/
│   ├── processed/hotspots/
│   │   ├── classified_hotspots_v2.csv          # ★ Canonical 50-col feature dataset (1,268 rows)
│   │   ├── classified_hotspots_v2_enriched.csv # ★ Canonical hybrid outputs (risk, bullets, review)
│   │   └── training_dataset.csv                # 17-feature matrix + consensus labels (1,268 × 20)
│   └── raw/
│       ├── firms_recent.csv                    # ★ Canonical FIRMS snapshot (1,268 rows)
│       └── osm/
│           ├── osm_industrial_sites.json       # 20,231 industrial/refinery/power/mining sites
│           └── osm_forest_agriculture.json     # 3,791 forest + agriculture sites
├── docs/
│   └── classification_logic.md                 # Methodology: threshold logic, LFs, evidence
├── research/
│   └── demo/
│       └── build_demo_dataset.py               # SYNTHETIC demo generator (never for training); writes only to research/demo/output/
├── scripts/                                    # production data tooling + batch pipeline
│   ├── __init__.py
│   ├── build_real_dataset.py   # ★ Canonical CSV producer: FIRMS × OSM → classified v2
│   ├── add_synthetic_mining.py # documented mining-cluster proxies for the OSM cache
│   ├── check_data_integrity.py # pre-training guard: synthetic markers / demo fallbacks
│   ├── fetch_firms_api.py      # live FIRMS API fetcher
│   ├── fetch_osm_daily.py      # daily Overpass fetcher (forest/agriculture landuse)
│   ├── fetch_mining.py         # Overpass queries for coal/iron-ore/bauxite mines
│   ├── fetch_state.py          # per-state bounding-box query utility
│   └── run_pipeline.py         # ★ batch hybrid enrichment CLI (resume/force/limit/dry-run)
├── tests/                       # 116-test suite (pytest)
│   ├── __init__.py
│   ├── test_api.py                  # FastAPI adapter: health + analyze contract
│   ├── test_build_real_dataset.py   # data-build guards (confidence, day/night, integrity)
│   ├── test_data_lineage.py         # canonical-path + lineage guards
│   ├── test_geospatial_audit.py     # spatial audit locks (units, radii, sentinels)
│   ├── test_imbalance.py            # E4 imbalance preprocessors + transitions
│   ├── test_phase_b_spatial_fixes.py
│   ├── test_phase_e2_improvements.py
│   └── test_pipeline_cli.py         # run_pipeline resume/force/limit/dry-run
├── reports/                        # phase evidence: E1–E4, e4/, audit/, structure guide
├── REPOSITORY_CLEANUP_PLAN.md      # cleanup audit + action plan (this exercise)
├── .env.example                    # config template (copy to .env, add FIRMS key)
├── .gitignore
├── REPRODUCIBILITY.md              # reproducibility + baseline integrity notes
├── requirements.txt
└── README.md
```

## 🔬 What Every Module & File Does

**`backend/app/core/`** — `config.py` reads `.env` via Pydantic `BaseSettings` (FIRMS key, DB URL, CORS, India bbox, cache hours; comma-separated bbox strings handled). `constants.py` is the single source of truth: 7-class taxonomy, display names, GIS colors, sensor types, proximity thresholds. `paths.py` defines every canonical data/model path. `lineage.py` validates training datasets and warns against any legacy/non-canonical data copy.

**`backend/app/geo/`** — `distance.py` implements the spherical Haversine formula (R = 6,371 km) with point-to-point, batch, and nearest-candidate helpers in meters or km. `spatial_context.py` aggregates distances across all 7 infrastructure categories into structured dicts/models. `spatial_features.py` defines the 17-feature spatial schema with sentinel distance semantics.

**`backend/app/intelligence/`** — `labeling_functions.py` holds **14** explainable detectors (industrial fire ×3, gas flare ×2, mining ×2, agriculture ×2, forest ×2, process heat ×2) plus safe helpers (`get_distance_meters` accepts meter fields and km aliases, `is_missing` treats 999/NaN/Inf as missing). Zero eager guessing — insufficient evidence returns abstain (`None`). Includes a 9-scenario self-test harness (run: `python -m app.intelligence.labeling_functions` → 9/9 PASS). `label_aggregator.py` applies majority voting; total abstention or ties fall back to `unclassified`. `hybrid_engine.py` fuses LF consensus with XGBoost probabilities into `final_label` + `hybrid_confidence` + `decision_source`, flagging disagreements for human review.

**`backend/app/ml/`** — `dataset_builder.py` runs the LFs over the v2 CSV, extracts 17 numeric features, writes `training_dataset.csv`. `train.py` fits XGBoost (200 trees, depth 6) on a stratified 80/20 split + 5-fold CV, with explicit coverage-gap warnings. `evaluate.py` regenerates confusion matrix, feature-importance chart, text report and metrics JSON on the held-out split. `predict.py` serves cached single/batch inference with full probability distributions (production). `imbalance.py` + `experiment_runner.py` are the Phase E4 controlled-experiment harness (class-weighting/oversampling matrix); it regenerates per-candidate outputs under `experiments/e4/` on demand — the previously stored E4 output artifacts were removed as redundant historical data.

**`backend/app/services/`** — `firms_service.py` (FIRMS ingestion + 24 h cache), `osm_service.py` (Overpass + state cache + tag taxonomy), `hotspot_service.py` (end-to-end orchestrator: ingest → enrich → hybrid classify → `HotspotAnalysis`).

**`backend/app/api/` + `main.py`** — thin FastAPI layer. `main.py` wires CORS from config and mounts the router; `api/hotspots.py` exposes `POST /api/v1/hotspots/analyze`, delegating entirely to `hotspot_service.analyze_single_hotspot` (no logic lives here).

**`scripts/`** — production data tooling. `fetch_*` pull FIRMS/OSM data into caches. `build_real_dataset.py` produces the canonical 1,268-row feature CSV. `run_pipeline.py` runs the frozen hybrid engine over the whole CSV batch (resume/force/limit/dry-run) and writes `classified_hotspots_v2_enriched.csv`. `check_data_integrity.py` is the pre-training guard.

**`research/`** — non-production engineering: `demo/` holds the synthetic demo generator (never for training; writes only to its own git-ignored output directory).

## 📊 Model Performance & Evaluation Results

Current **Phase E3** model. Training set: **1,015 rows** (stratified 80/20 of the 1,268-row snapshot, random_state 42) · held-out test: **253 rows**. Canonical metrics live in `backend/app/ml/models/training_metrics.json`.

| Metric | Value |
|---|---|
| Train accuracy | **100.0%** (1,015/1,015) |
| Test accuracy | **98.81%** (250/253) |
| Test macro F1 (all 7 classes) | **0.5838** |
| 5-fold CV accuracy | **98.13% ± 0.59%** |
| 5-fold CV macro F1 | **0.6642 ± 0.1320** |
| Top features (XGBoost gain) | `dist_factory` 0.31, `dist_forest` 0.19, `frp` 0.15, `dist_refinery` 0.10, `bright_ti4` 0.10, `dist_mining` 0.06 |

> **Honest caveats (do not over-read the 98.81%):** the weak-supervision label surface is small. Four classes have tiny training support (`agricultural_burn` 1, `gas_flare` 7, `industrial_process_heat` 2, `mining_activity` 3) and appear only ~13× in the held-out set, so **macro F1 (0.5838)** is the honest primary metric. 98.81% accuracy is dominated by the majority class. Phase E4 tested five class-imbalance mitigation strategies; **none was adopted** — the E3 model remains the production model.

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
venv\Scripts\python -m pytest tests -q

# Step 3: Rebuild the v2 feature dataset (real FIRMS × OSM, 1,268-hotspot snapshot, ~2 min)
venv\Scripts\python scripts/build_real_dataset.py

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

# Step 9: Batch hybrid enrichment (writes classified_hotspots_v2_enriched.csv)
venv\Scripts\python scripts/run_pipeline.py
```

## ⚡ FastAPI & Frontend Integration

```powershell
# Start the API (from repo root)
venv\Scripts\python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

- Live docs: `http://localhost:8000/docs`
- `GET /health` → `{"status": "ok", "engine": "thermoscope-ai"}`
- `POST /api/v1/hotspots/analyze` — accepts a `Hotspot` (lat/lon, FRP, brightness, confidence, date); returns `HotspotAnalysis` with spatial context + full hybrid classification (`final_label`, `hybrid_confidence`, `decision_source`, `agreement`, `conflict`, `requires_human_review`, `review_reason`, `explanation`).
- CORS is pre-configured for `http://localhost:5173` (Vite) and `http://localhost:8501` (Streamlit) via `core/config.py`.

## 🔀 Hybrid Intelligence Fusion (Phase C)

`backend/app/intelligence/hybrid_engine.py` combines both intelligence paths per hotspot:

- **Rule path**: 14 LF votes → majority consensus + vote breakdown.
- **ML path**: XGBoost `predict_proba` over the 17-feature vector.
- **Fusion**: agreement → `hybrid_agreement` with boosted confidence; disagreement → lower confidence + `requires_human_review=True`, both explanations preserved in `explanation`.
- Batch outputs land in `classified_hotspots_v2_enriched.csv` (`final_label`, `hybrid_confidence`, `decision_source`, `agreement`, `conflict`, `risk_score`, `explanation_bullets`, …).

## 🛡️ Data Integrity & Reproducibility Guarantees

- **No synthetic training fallbacks** — `check_data_integrity.py` fails the run (exit 1) if any `_is_synthetic_demo` marker or `allow_demo_fallback=True` exists in the training path. The synthetic demo generator lives in `research/demo/` and writes only to its own demo output directory — it can never overwrite the canonical dataset.
- **Deterministic** — fixed `random_state=42` for splits, CV folds, and XGBoost; reruns reproduce metrics bit-for-bit (see `REPRODUCIBILITY.md`).
- **Single canonical paths** — `core/paths.py` + `core/lineage.py` enforce one canonical location per dataset; legacy/archived copies have been removed from the repo, so production can never read a duplicate snapshot.
- **Strict unit discipline** — `dist_*` columns are kilometers, `distance_to_*_m` / `dist_*_m` are meters; 999 is the single missing-value sentinel recognized by `is_missing()`.
- **Dual naming convention** — v2 CSVs carry both `dist_*` (km) and `distance_to_*_m` (m); `get_distance_meters()` resolves either, so rule and ML paths can never desync on units.

## ⚠️ Known Limitations (Honest Baseline)

- The weak-label surface is small and class-imbalanced: `agricultural_burn` (1), `gas_flare` (7), `industrial_process_heat` (2), `mining_activity` (3) training rows. Macro F1 (0.5838) is the honest headline over accuracy (98.81%), which is majority-class-dominated.
- `mining_activity` recall depends on the 20-site mining proxy cache (`scripts/add_synthetic_mining.py`) — approximate coordinates of known mining belts, documented as such.
- More FIRMS history + OSM coverage is the fix; Phase E4 showed adding rows/gains without new evidence does not generalize.
- The OSM industrial cache is git-ignored and generated locally; clones must regenerate it (`fetch_mining.py`, `scripts/add_synthetic_mining.py`, stitching in `build_real_dataset.py`) before rebuilding data.

## 👥 Collaboration Workflow

```powershell
git checkout -b feat/<your-feature>
# ...edit, then verify: integrity guard → dataset_builder → train → evaluate → pytest...
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