# THERMOSCOPE-AI — Project Structure & Approach

> Smart India Hackathon 2026 · Problem Statement **SIH26162** (NTRO / Disaster Management)
> *"AI-Based Detection and Classification of Industrial Fires and Persistent Thermal Sources"*
> Python 3.11 · FastAPI · XGBoost · NASA FIRMS · OpenStreetMap Overpass

This document describes the current file/folder structure, the responsibility of every file, and the end-to-end approach used to solve the problem. Last updated against working tree `HEAD` (post Phase F).

---

## 1. Current file/folder structure

```
THERMOSCOPE-AI/
├── .env.example                     # Template for environment configuration (copy to .env)
├── .gitignore                       # Excludes .env, venv/, __pycache__/, models, caches
├── README.md                        # Project overview, structure index, module guide
├── REPRODUCIBILITY.md               # Training baseline, commit pin, dataset statistics
├── requirements.txt                 # Pinned dependencies (incl. fastapi, xgboost, pandas, streamlit, folium)
│
├── backend/                         # FastAPI application
│   └── app/
│       ├── main.py                  # Server entry point, router mounting
│       ├── api/hotspots.py          # HTTP adapter (REST contract for frontend)
│       ├── core/
│       │   ├── config.py            # Pydantic-settings env config (.env)
│       │   ├── constants.py         # 7-class taxonomy, thresholds, OSM site types
│       │   ├── lineage.py           # Dataset lineage / integrity protection
│       │   └── paths.py             # Canonical absolute paths (single source of truth)
│       ├── geo/
│       │   ├── distance.py          # Haversine distance helpers
│       │   ├── spatial_context.py   # OSM proximity context computation
│       │   └── spatial_features.py  # Unified spatial feature engineering
│       ├── intelligence/
│       │   ├── labeling_functions.py  # 14 weak-supervision labeling functions
│       │   ├── label_aggregator.py    # Vote aggregation from labeling functions
│       │   └── hybrid_engine.py       # Phase C Rules × ML fusion decision matrix
│       ├── ml/
│       │   ├── dataset_builder.py   # Training surface construction
│       │   ├── feature_schema.py    # Canonical feature column definitions
│       │   ├── imbalance.py         # Class-imbalance handling (SMOTE, weighting)
│       │   ├── train.py             # XGBoost training
│       │   ├── evaluate.py          # Metric computation, plots, report
│       │   ├── predict.py           # In-memory model inference
│       │   ├── experiment_runner.py # Experiment suite execution
│       │   └── models/              # Artifacts
│       │       ├── hotspot_classifier.joblib   # Trained XGBoost model
│       │       ├── feature_columns.joblib      # Feature order used at fit time
│       │       ├── label_classes.joblib        # Target class list
│       │       ├── label_encoder.joblib        # Class encoding
│       │       └── training_metrics.json       # Stored eval metrics
│       ├── schemas/
│       │   ├── hotspot.py           # Hotspot representation
│       │   ├── spatial_context.py   # Spatial-context response model
│       │   ├── analysis.py          # Analysis request model
│       │   └── analysis_response.py # Analysis response + ClassificationSummary
│       └── services/
│           ├── firms_service.py        # NASA FIRMS ingestion
│           ├── osm_service.py          # OSM Overpass fetch + cache
│           ├── hotspot_service.py      # End-to-end orchestration (phase D)
│           └── recommendation_service.py # Phase F response recommendations
│
├── data/
│   ├── raw/
│   │   ├── firms_recent.csv              # FIRMS VIIRS NRT snapshot (India bbox, 30 days)
│   │   └── osm/
│   │       ├── osm_industrial_sites.json     # OSM industrial/refinery/mining cache (~5.9 MB)
│   │       └── osm_forest_agriculture.json   # OSM forest/agriculture cache (~0.6 MB)
│   └── processed/hotspots/
│       ├── classified_hotspots_v2.csv         # Real classified dataset (642↔1,268 rows)
│       ├── training_dataset.csv               # Feature matrix consumed by train.py
│       └── classified_hotspots_v2_enriched.csv # Hybrid-engine output snapshot (1,268 rows)
│
├── scripts/                         # CLI orchestration
│   ├── fetch_firms_api.py           # Pull FIRMS API → raw FIRMS CSV
│   ├── fetch_osm_daily.py           # Daily OSM Overpass refresh of site caches
│   ├── fetch_mining.py              # Fetch mining-themed OSM sites
│   ├── fetch_state.py               # Per-state regional fetch helper
│   ├── build_real_dataset.py        # RAW FIRMS + RAW OSM → classified dataset (stage A/B)
│   ├── check_data_integrity.py      # Pre-training guard: block demo/synthetic data
│   ├── run_pipeline.py              # Batch end-to-end classification → enriched CSV
│   └── add_synthetic_mining.py      # Synthetic mining injector (research only)
│
├── research/
│   └── demo/
│       └── build_demo_dataset.py    # Synthetic demo dataset (UI only, NEVER for training)
│
├── docs/
│   └── classification_logic.md      # Methodology: taxonomy, LFs, features, decision matrix
│
├── reports/
│   ├── HYBRID_DECISION_AUDIT.md         # Phase A read-only audit of stored engine output
│   └── MVP_HYBRID_DECISION_POLICY_REPORT.md # Phase C decision-policy implementation report
│
└── tests/                           # pytest suite (130 tests passing)
    ├── test_api.py                  # API contract tests
    ├── test_api_response.py         # Response-schema/field contract tests
    ├── test_pipeline_cli.py         # run_pipeline.py CLI tests
    ├── test_hybrid_policy.py        # 11 deterministic decision-matrix cases
    ├── test_imbalance.py            # Imbalance-handling tests
    ├── test_build_real_dataset.py   # Dataset builder tests
    ├── test_data_lineage.py         # Lineage/integrity guard tests
    ├── test_geospatial_audit.py     # Spatial-feature audit
    ├── test_phase_b_spatial_fixes.py# Spatial-context regression tests
    ├── test_phase_e2_improvements.py# Batch-pipeline regression tests
    └── test_crit_fixes.py           # Critical-fix regression tests
```

Not shown: `.env` (gitignored local secrets), `venv/`, `__pycache__/`, git internals.

---

## 2. What every file does

| Path | Responsibility |
| --- | --- |
| `.env.example` | Documented template: `FIRMS_MAP_KEY`, `FIRMS_DAYS=30`, `FIRMS_DATASETS=VIIRS_SNPP_NRT,VIIRS_NOAA20_NRT`, `INDIA_BBOX=68,6,96,36`, `ITERATIONS`, `USE_ML=true`, `TRAIN_NEW_MODEL=true`, `VERBOSE`. Copy to `.env` locally. |
| `README.md` | Entry point — goal, SIH context, structure index, module-by-module guide, run commands. |
| `REPRODUCIBILITY.md` | Fixed baseline: commit `4f0ebc0` *"Completed Till Phase B"* (2026-09-04); 642 FIRMS rows, 423,232 OSM cache sites, 80/20 stratified split, mini-batch BFS public coordinates; no stochastic retrain without regeneration. |
| `requirements.txt` | Pinned dependencies; core: `fastapi`, `uvicorn`, `xgboost`, `scikit-learn`, `pandas`, `geopy`, `joblib`, `python-dotenv`, `pydantic-settings`, `requests`; frontend stack: `streamlit`, `folium`, `streamlit-folium`, `plotly`, `altair`. |
| `backend/app/main.py` | FastAPI app factory; mounts API router, docs, CORS setup. |
| `backend/app/api/hotspots.py` | HTTP adapter exposing the analysis endpoint. Thin controller — all logic stays in services/intelligence. |
| `backend/app/core/config.py` | Typed settings loaded from `.env` via pydantic-settings (`FIRMS_MAP_KEY`, bbox, flags like `USE_ML`, `TRAIN_NEW_MODEL`). |
| `backend/app/core/constants.py` | Canonical 7-class taxonomy, OSM tagging→class maps, proximity thresholds, `SENTINEL_DISTANCE_M` (~999 km), model thresholds. |
| `backend/app/core/lineage.py` | Guards against forbidden sources (e.g., demo data reaching training) and stale/legacy copies; enforces the canonical flow. |
| `backend/app/core/paths.py` | Single source of truth for every file path; expresses canonical flow: RAW FIRMS+OSM → `build_real_dataset` → `dataset_builder` → `train` → hybrid/predict → `run_pipeline`. Absolute, CWD-independent. |
| `backend/app/geo/distance.py` | Haversine great-circle distance between map points. |
| `backend/app/geo/spatial_context.py` | For a hotspot, find nearest OSM sites per class and compute proximity evidence. |
| `backend/app/geo/spatial_features.py` | Unified features (nearest-distance features, counts within gates, per-class min distances) feeding both rules and ML. |
| `backend/app/intelligence/labeling_functions.py` | 14 weak-supervision labeling functions emitting class votes from OSM proximity gates. |
| `backend/app/intelligence/label_aggregator.py` | Aggregates LF votes (`rule_active_votes`, `rule_prediction`, abstention). |
| `backend/app/intelligence/hybrid_engine.py` | Phase C fusion: deterministic decision matrix (Cases A–E) combining rule votes and ML probabilities into `classification_status` (confirmed/probable/uncertain) with honest confidence semantics. |
| `backend/app/ml/dataset_builder.py` | Builds `training_dataset.csv` (features + weak labels) from the classified dataset. |
| `backend/app/ml/feature_schema.py` | Canonical ordered feature columns; used at build, fit, and predict time to prevent drift. |
| `backend/app/ml/imbalance.py` | Class-imbalance mitigation (resampling/weights) for the minority industrial classes. |
| `backend/app/ml/train.py` | Trains the XGBoost classifier; `TRAIN_NEW_MODEL` toggles retraining; persists artifacts. |
| `backend/app/ml/evaluate.py` | Computes metrics (accuracy, macro F1, per-class report), confusion matrix, feature-importance plot. |
| `backend/app/ml/predict.py` | In-memory classifier inference with probability output. |
| `backend/app/ml/experiment_runner.py` | Runs experiment suites across data/model variants (investigative use). |
| `backend/app/ml/models/*` | Trained artifacts: `hotspot_classifier.joblib`, `feature_columns.joblib`, `label_classes.joblib`, `label_encoder.joblib`, `training_metrics.json`. |
| `backend/app/schemas/*` | Pydantic request/response models; `analysis_response.py` defines `ClassificationSummary` and the status/confidence field separation. |
| `backend/app/services/firms_service.py` | FIRMS ingestion: API fetch, caching, fallback handling. |
| `backend/app/services/osm_service.py` | Overpass queries for industrial/refinery/forest/agriculture/mining sites; JSON cache on disk. |
| `backend/app/services/hotspot_service.py` | Orchestrator: spatial context → rules → hybrid engine → prediction → presentation projection. |
| `backend/app/services/recommendation_service.py` | Phase F recommendations keyed on `classification_status` (escalation/verification guidance). |
| `scripts/fetch_firms_api.py` | Refresh `data/raw/firms_recent.csv` from the FIRMS MapKey API. |
| `scripts/fetch_osm_daily.py` | Refresh OSM caches via Overpass (bounded queries). |
| `scripts/fetch_mining.py` / `fetch_state.py` | Regional OSM harvesting helpers for extra site coverage. |
| `scripts/build_real_dataset.py` | Ingestion + weak-supervision labeling → `classified_hotspots_v2.csv` (real-data producer; never touches training directly). |
| `scripts/check_data_integrity.py` | Hard pre-training gate: rejects any demo/synthetic/`_is_synthetic_demo` rows so the model never trains on fake data. |
| `scripts/run_pipeline.py` | Batch CLI: reads classified hotspots → spatial features → hybrid engine → exports `classified_hotspots_v2_enriched.csv`; `--force` regenerates. |
| `scripts/add_synthetic_mining.py`/`research/demo/build_demo_dataset.py` | Research-only synthetic generators for UI/demo; explicitly excluded from the training path. |
| `docs/classification_logic.md` | Authoritative methodology reference (taxonomy, all 14 LFs, features, decision matrix, current-state metrics). |
| `reports/HYBRID_DECISION_AUDIT.md` | Phase A read-only audit: why ~92% abstain, ML evidence is not discarded, confidence-semantics contradiction root cause. |
| `reports/MVP_HYBRID_DECISION_POLICY_REPORT.md` | Phase C report: decision matrix tables, before/after numbers (1,268-row audit), changed files, validation. |
| `tests/*` | 130 passing pytest contracts covering API, pipeline CLI, hybrid policy, imbalance, lineage, geospatial audit, and regressions. |

---

## 3. Approach used to solve the problem

**Problem essence:** SIH26162 asks to automatically detect and *differentiate* real-time thermal anomalies — distinguishing genuine industrial fires and persistent industrial thermal sources (process heat, flares, mining heat) from false/benign signals such as agricultural burns, forest/grass fires, and non-event detections, using satellite thermal data.

### The canonical pipeline

```
 NASA FIRMS (VIIRS NRT, 30d, India bbox)          OpenStreetMap Overpass
                 │                                        │
                 ▼                                        ▼
         firms_recent.csv                   osm_industrial_sites.json
                                            osm_forest_agriculture.json
                 │                                        │
                 └──────────────┬─────────────────────────┘
                                ▼
              build_real_dataset.py   (Stage A/B: ingest + weak labels)
                                ▼
              classified_hotspots_v2.csv      (real dataset, 1,268 rows)
                                │
                                ▼
                 dataset_builder.py  ──►  training_dataset.csv
                                │
                                ▼
               train.py → XGBoost            (feature_schema/imbalance guards)
                                │
               ┌────────────────┴───────────────┐
               ▼                                 ▼
     hybrid_engine.py (Rules × ML)      predict.py (in-memory inference)
               │
               ▼
     run_pipeline.py ──► classified_hotspots_v2_enriched.csv
               │
               ▼
     hotspot_service → FastAPI /analyze → frontend (classification_status, confidence, recommendations)
```

### Stage-by-stage approach

1. **Data ingestion (real-world, reproducible)** — FIRMS VIIRS NRT fire hotspot CSV plus OSM industrial/refinery/forest/agriculture/mining site polygons via Overpass. Everything is cached raw; no derived data is regenerated silently. `paths.py` is the single source of truth so no stage reads a stale copy.

2. **Weak-supervision labeling (no manual annotation)** — 14 labeling functions in `labeling_functions.py` vote on a 7-class taxonomy
   (`industrial_fire`, `gas_flare`, `mining_activity`, `agricultural_burn`, `forest_natural_fire`, `industrial_process_heat`, `unclassified`)
   by measuring OSM-proximity evidence of the right class inside tight distance gates. `label_aggregator.py` merges votes; sparse evidence → honest abstention rather than a forced label.

3. **Spatial feature engineering** — unified features computed in `geo/spatial_features.py` (nearest-site distances per class, counts within gates) are the same inputs for rules and for the ML features.

4. **XGBoost supervised model** — trained on the weakly-labeled dataset with `imbalance.py` mitigating minority classes; `evaluate.py` records metrics and plots. Stored E3 production model: test accuracy **0.9881**, macro F1 **0.5838**.

5. **Rules × ML hybrid fusion (Phase C/D)** — `hybrid_engine.py` implements a deterministic decision matrix (Cases A–E): rules+ML agreement → `confirmed`; strong single-source evidence → `probable`; abstention/weak signals → `uncertain` (with human-review routing). Confidence semantics were audited and corrected so `unclassified` is reported as low/insufficient evidence — not inflated by raw model probability.

6. **Batch + live inference (Phase E)** — `run_pipeline.py` reproduces the identical per-hotspot analysis in batch (1,268/1,268, ~246 rows/s) and `hotspot_service.py` drives the same engine live in the API, guaranteeing train/deploy parity.

7. **Recommendations (Phase F)** — `recommendation_service.py` keys guidance on `classification_status`: verify-before-escalation for `probable`, verification sets for abstention.

### Data-integrity guarantees

- Synthetic/demo data is **never** allowed into the training path (`check_data_integrity.py`, `lineage.py`); `research/demo` exists only for UI demos.
- Reproducible baseline pinned in `REPRODUCIBILITY.md` (commit `4f0ebc0`, 642-row baseline) independent of the later 1,268-row snapshot.
- 130 pytest tests guard contracts, the decision matrix, geospatial features, lineage, and regressions.

### Current-state audit results (2026-09-06)

1,268 canonical hotspots: 101 `confirmed`, 0 `probable`, 1,167 `uncertain`. Abstention (~92%) is driven by spatial-evidence sparsity and an ML model that learned majority-abstention on no-evidence rows — the audit confirmed **no useful ML evidence is discarded** (0 rules-abstain rows carry a confident real-class ML prediction), so abstention is reported honestly and routed to human review rather than inflated into a decision.