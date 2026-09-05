# THERMOSCOPE-AI — Phase B / C / D Engineering Report

**Date:** 2026-09-06
**Scope covered:** Phase B (targeted defect fixes) → Phase C (canonical rebuild + retrain) → Phase D (before/after measurement) → Phase E decision.

---

## 1. Executive summary

Phase B executed exactly its approved scope: P0.1 (merge existing forest/agriculture OSM
cache), P0.2 (`firms_type` tri-state), P1.3 (single shared batch==live feature contract),
P1.4 (one canonical 999 sentinel, never `None`, never confused with the LFs' 45 km influence
threshold), and P1.5 (real neighbourhood counts with backward-compatible legacy aliases).

Phase C rebuilt the canonical dataset, feature matrix, and model through the production
pipeline. Phase D confirms the previously-constant `dist_forest`/`dist_agriculture` features
are now real (nunique 1 → 278 / 85; 177 forest-near and 34 agri-near hotspots), producing
`+3 forest_natural_fire` and `+1 agricultural_burn` weak labels, exactly as the Phase A
simulation predicted. All 77 tests pass, including the two Phase A ``pin-the-bug`` guards
which now assert the fixed invariants.

**Phase E decision: PARTIALLY** — see §6.

---

## 2. Phase B — Defect fixes implemented

| ID | Defect | Root cause | Fix | Proof |
|----|--------|-----------|-----|-------|
| P0.1 | `dist_forest`/`dist_agriculture` constant 999 in canonical CSV despite a 3,791-site cache | producer never loaded `osm_forest_agriculture.json` | `build_real_dataset` now loads + merges it (`scripts/build_real_dataset.py:156`); `osm_service` cache fallback merges it too (`backend/app/services/osm_service.py:552`) | `pytest tests/test_phase_b_spatial_fixes.py::TestBatchProducesRealForestAgri`; coverage audit |
| P0.2 | `firms_type is None` used as *positive* veg evidence; `-1` silently *negative* | missing data treated as evidence | new `get_firms_type_state` (Known/Unknown); unknown is never evidence; known non-veg (2/3/4) abstains (`backend/app/intelligence/labeling_functions.py`) | `TestFirmsTypeTriState` |
| P1.3 | batch and live computed features independently (counts hard-coded 0 live) | no shared contract | new `app/geo/spatial_features.py` = single source of truth for all spatial features; producer, `hotspot_service._build_feature_record`, and `run_pipeline._build_feature_row` all funnel through `compute_spatial_features` | `TestBatchLiveParity` |
| P1.4 | batch sentinel 999000 vs live 45000 | two modules each defined their own | one canonical `SENTINEL_DISTANCE_M == 999000.0`/`SENTINEL_DISTANCE_KM == 999.0`; `SPATIAL_EVIDENCE_INFLUENCE_M == 45000.0` named as the LF rule threshold; `None` forbidden in feature layer | `TestSentinelContract`; updated `TestConfigurationConsistency` |
| P1.5 | `count_ind_5km`/`count_ref_5km` are bucket codes misused as counts; live path hard-coded 0 | no real-count columns | real columns added: `industrial_sites_within_2km/5km`, `refinery_sites_within_3km/5km`, `forest_sites_within_5km`, `agriculture_sites_within_5km`, `count_forest_5km`, `count_agriculture_5km`; legacy aliases kept byte-compatible for the 17-col model; LF density gate uses the real count with bucket fallback | `TestRealCounts`; `TestIndustrialDensityGate` |

Supporting changes:
- `backend/app/geo/spatial_context.py` now re-exports the canonical sentinel (display schema
  may still carry `None`; the feature layer converts to 999000.0).
- `scripts/build_real_dataset.py` `_compute_distances_from_cache` retained as a compat probe
  delegating to the shared contract (keeps `test_geospatial_audit.py` imports working).
- `scripts/run_pipeline.py` km→m conversion maps any ≥ 999 km value to the canonical sentinel
  (never `None`, never false proximity); reads the new count columns; `dist_industry` falls
  back to `dist_factory`.
- `app/ml/train.py` gained a rare-class coverage guard (rows of a class that lands in the
  test split are moved into train) and skips degenerate CV folds where a fold would fit
  without the low label index. This is a **fit-validity guard only** — no hyperparameter,
  algorithm, or synthetic-data change.

## 3. Phase C — Canonical rebuild (all via the production pipeline)

1. `python scripts/build_real_dataset.py`
   - 1,268 rows written; 0 removed. Canonical path `data/processed/hotspots/classified_hotspots_v2.csv`.
   - Coverage (unchanged vs Phase A for every previously-merged category):
     industry 175 (13.8 %), refinery 30 (2.4 %), oil_gas 33 (2.6 %), mining 165 (13.0 %),
     power_plant 169 (13.3 %). **New real:** forest 282 (22.2 %), agriculture 84 (6.6 %).
2. `python -m backend.app.ml.dataset_builder` → LF consensus labels (see §4 table).
3. `python -m backend.app.ml.train` → same XGB config (n_estimators=200, depth 6, lr 0.1,
   multi:softprob); artifacts rewritten (`hotspot_classifier.joblib`, `feature_columns.joblib`,
   `label_classes.joblib`, `training_metrics.json`). 17-column feature schema untouched.
4. `python scripts/run_pipeline.py --force` → enriched CSV, 1,268/1,268 rows, 241 rows/sec.

## 4. Phase D — Before / after

### Feature health

| Metric | Before (Phase A) | After (Phase C) |
|--------|------------------|-----------------|
| `dist_forest` unique values | 1 (all 999.0) | **278** |
| `dist_agriculture` unique values | 1 (all 999.0) | **85** |
| hotspots within 15 km of forest | 0 (producer saw none) | **177** (exactly the audit-sim projection) |
| hotspots within 15 km of agri | 0 (producer saw none) | **34** (matches the audit-sim projection) |
| forest real min distance | — | 1,070.1 m |
| agri real min distance | — | 3,013.2 m |
| rows with ≥1 forest site ≤ 5 km | — | 64 |
| rows with ≥1 agri site ≤ 5 km | — | 27 |
| `industrial_sites_within_5km` real counts | not present (bucket codes only) | 0…165 per hotspot |
| `None`/NaN in any `distance_to_*_m` | — | 0 |

### Labels

| Label | LF consensus before | LF consensus after | Hybrid final after |
|-------|--------------------:|-------------------:|-------------------:|
| unclassified | 1,208 | 1,204 | 1,205 |
| industrial_fire | 49 | 49 | 49 |
| gas_flare | 8 | 8 | 7 |
| forest_natural_fire | 0 | **3** | **3** |
| mining_activity | 3 | 3 | 3 |
| agricultural_burn | 0 | **1** | **1** |
| **Total** | 1,268 | 1,268 | 1,268 |

- The `+3 forest_natural_fire` / `+1 agricultural_burn` deltas match the Phase A
  `geospatial_audit_sim` prediction exactly.
- `gas_flare` hybrid 8 → 7: one row that LF-consensus still marks `gas_flare` is scored
  `unclassified` by the hybrid fusion after retrain (new `dist_forest`/`dist_agriculture`
  features now carry ~4.2 % / ~2.4 % model importance). This is an expected fusion delta,
  transparent in the enriched CSV.

### Model (same config, retrained on fixed data)

- Train acc 0.9990, test acc 0.9960, test macro-F1 0.6660.
- CV accuracy 0.9872 ± 0.0032; CV macro-F1 0.5143 ± 0.0869 (4/5 usable folds).
- Coverage gaps remain honest: agricultural_burn 1, forest_natural_fire 3, gas_flare 7,
  mining_activity 3 train samples → these rare classes are not learnable yet.

### Verification gates (all green)

- `python scripts/check_data_integrity.py` — no synthetic marker; no `allow_demo_fallback=True`
  in training paths.
- `pytest tests` — **77 passed** (was 53 in Phase A; +24 new Phase B regression tests).
- Enriched CSV: 1,268 rows, 1,268 unique `input_index`, 0 duplicates, 0 synthetic columns.
- All 17 ML column names and the new real-count columns present.

## 5. Files changed

- `backend/app/geo/spatial_features.py` — **new**: canonical feature contract.
- `backend/app/geo/spatial_context.py` — canonical sentinel re-export.
- `backend/app/intelligence/labeling_functions.py` — `get_firms_type_state`, tri-state veg LFs,
  `SPATIAL_EVIDENCE_INFLUENCE_M`, real-count density gate.
- `backend/app/services/hotspot_service.py` — feature record via shared contract.
- `backend/app/services/osm_service.py` — cache fallback merges forest/agri cache.
- `scripts/build_real_dataset.py` — forest/agri merge, shared contract, real counts, compat probe.
- `scripts/run_pipeline.py` — sentinel-correct `_km`, new count columns, `dist_industry` fallback.
- `backend/app/ml/train.py` — rare-class fit-validity guard (no model-config change).
- `tests/test_phase_b_spatial_fixes.py` — **new** (24 tests).
- `tests/test_geospatial_audit.py`, `tests/test_build_real_dataset.py` — updated guards/fixtures.

## 6. Phase E decision

**PARTIALLY — data integrity and contract work is complete and verified; the classifier is
not yet production-ready.**

Reason:
- **What is now solid:** every P0/P1 defect is fixed and regression-tested; batch and live
  produce identical features; the dataset is clean, real, non-constant, and rebuildable; the
  model retrains deterministically with the same config.
- **What still blocks production use:** 95 % of rows remain `unclassified`, and 4 rare classes
  have < 15 training samples (feature/model-level sparsity). Because Phase B deliberately did
  NOT touch radii, thresholds, model algorithm, or synthetic augmentation, raw classification
  quality was never the target — and it has not yet moved.
- **Recommendation for the next phase (if approved — model/feature work only, no data
  fabrication):** density/dominance-derived features (e.g. forest-fraction in a sliding
  radius), class-weight/handling for rare classes, and evaluating the 177 forest-near rows as
  a labeled eval subset. Re-run Phase C/D exactly as done here to verify.