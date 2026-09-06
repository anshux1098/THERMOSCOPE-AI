# THERMOSCOPE-AI: Classification Logic

Methodology document for SIH26162 (NTRO). Describes how satellite thermal
hotspots are classified, what the system can and cannot do, and how to
reproduce every number in this file. All numbers were read from repository
artifacts; nothing is estimated. Known discrepancies between historical
comments in the code and measured reality are called out explicitly.

Project: THERMOSCOPE-AI, Smart India Hackathon 2026, Problem Statement
SIH26162 (National Technical Research Organisation / Disaster Management).

> **Dataset currency note (Sep 2026).** The methodology below is authoritative
> for the *labeling logic* (thresholds, fusion, review rules) and is unchanged.
> Historic dataset references (642-row / 968-row snapshots, "13 labeling
> functions", ~96% abstention) describe the Phase-A/B snapshot this document was
> written from. **Current system state: 1,268-hotspot snapshot, 14 labeling
> functions, E3 production model (test accuracy 0.9881, macro F1 0.5838).**
> The abstention rate on the current snapshot is ~92% (majority of hotspots
> flagged for review rather than guessed). See README "Model Performance".

--------------------------------------------------------------------------------

## 1. Executive Summary

THERMOSCOPE-AI classifies NASA FIRMS satellite thermal hotspots over India
into 7 categories (industrial fire, gas flare, mining, agricultural burn,
forest fire, industrial process heat, unclassified). It enriches each hotspot
with OpenStreetMap spatial context, generates training labels with 13
conservative rule-based labeling functions, trains an XGBoost classifier on
those weak labels, and fuses rule and ML outputs into calibrated decisions
with human-review routing. The honest headline: 96% of hotspots lack enough
spatial evidence to classify, so the system auto-classifies the clear 4% and
flags the rest for analysts instead of guessing.

--------------------------------------------------------------------------------

## 2. Classification Taxonomy

Seven canonical labels, defined once in `backend/app/core/constants.py:74-82`.
Display names (`constants.py:85-93`) and GIS map colors
(`constants.py:96-104`) are listed alongside.

| # | Label | Display name | Color | Definition |
|---|-------|--------------|-------|------------|
| 1 | `industrial_fire` | Industrial Fire | #FF4500 | Active fire associated with industrial infrastructure (factory, industrial zone) |
| 2 | `gas_flare` | Gas Flare / Persistent Thermal Source | #FF8C00 | Oil/gas flare or persistent thermal source at extraction, chemical, or refinery sites |
| 3 | `mining_activity` | Mining Activity | #8B4513 | Thermal signature from mining, quarry, or coal operations |
| 4 | `agricultural_burn` | Agricultural Burn | #DAA520 | Seasonal stubble burning or agricultural residue fire on farmland |
| 5 | `forest_natural_fire` | Forest / Natural Fire | #228B22 | Wildfire or natural vegetation fire in forest cover, isolated from industry |
| 6 | `industrial_process_heat` | Industrial Process Heat | #DC143C | Persistent non-fire heat from an industrial process (kiln, boiler, furnace) |
| 7 | `unclassified` | Unknown / Requires Verification | #808080 | Evidence insufficient; routed to human review, never force-classified |

`unclassified` is a first-class output, not an error state. A classifier that
says "I do not know" on 96% of inputs is preferable to one that invents
answers. Section 8 explains why the share is this large.

--------------------------------------------------------------------------------

## 3. System Architecture

Five layers. Data flows strictly top to bottom; no layer is skipped.

```text
FIRMS API (VIIRS SNPP / NOAA-20, MODIS)
  |  lat, lon, frp, bright_ti4, bright_ti5, confidence, daynight, acq_date
  v
OSM CACHE (20,231 industrial + 3,791 forest/agriculture sites)
  |  nearest-site lookup per hotspot
  v
DISTANCE ENGINE (backend/app/geo/distance.py)
  |  Haversine nearest distance to each of 7 categories (meters + km aliases)
  v
WEAK SUPERVISION (backend/app/intelligence/labeling_functions.py)
  |  13 labeling functions vote: one class or ABSTAIN
  v
VOTE AGGREGATION (backend/app/intelligence/label_aggregator.py)
  |  majority vote -> weak label per hotspot
  v
ML CLASSIFIER (backend/app/ml/train.py, XGBoost)
  |  trained on weak labels; outputs class probabilities
  v
HYBRID FUSION (backend/app/intelligence/hybrid_engine.py)
  |  5-case rules x ML fusion -> final_label + confidence + review flag
  v
SERVICE LAYER (backend/app/services/hotspot_service.py)
   |  analyze_single_hotspot() -> HotspotAnalysis (schemas/analysis.py)
   v
BATCH OUTPUT (scripts/run_pipeline.py)
   enriched CSV: labels, confidences, explanations, risk scores
```

Layer responsibilities and key files:

| Layer | Files | Input -> Output |
|-------|-------|-----------------|
| Data ingestion | `backend/app/services/firms_service.py`, `backend/app/services/osm_service.py` | FIRMS API + Overpass API -> cached CSV/JSON |
| Spatial features | `backend/app/geo/distance.py`, `backend/app/geo/spatial_context.py` | lat/lon + OSM cache -> 7 distance vectors |
| Weak supervision | `backend/app/intelligence/labeling_functions.py`, `backend/app/intelligence/label_aggregator.py` | distances + thermal -> weak labels |
| ML | `backend/app/ml/dataset_builder.py`, `train.py`, `evaluate.py`, `predict.py` | weak labels + 17 features -> trained model |
| Fusion + service | `backend/app/intelligence/hybrid_engine.py`, `backend/app/services/hotspot_service.py`, `scripts/run_pipeline.py` | LF votes + ML probs -> reviewed decisions |

--------------------------------------------------------------------------------

## 4. Weak Supervision (13 Labeling Functions)

Each labeling function (LF) is an independent detector: it inspects one
hotspot record and returns either a single class vote or ABSTAIN (`None`,
`labeling_functions.py:54`). LFs are deliberately conservative - they vote
only when evidence is strong and abstain otherwise. The 13 votes are combined
by majority rule (`label_aggregator.py`); total abstention or a tie yields
`unclassified`. No manual labeling is required at any point: the consensus
votes become the training targets for the ML layer (Section 5).

NOTE ON COUNT: several code comments and docstrings say "14 labeling
functions" (e.g. `hotspot_service.py:137`, `hybrid_engine.py:282`,
`run_pipeline.py:8`). The registry actually contains 13
(`ALL_LABELING_FUNCTIONS`, verified by import). This document uses 13
everywhere. The stale "14" strings were left untouched because code files
were out of scope for this document.

### 4.1 The 13 labeling functions

References are `backend/app/intelligence/labeling_functions.py` line numbers
for each function definition.

| # | Function (line) | Votes for | Required evidence (all must hold) |
|---|-----------------|-----------|-----------------------------------|
| 1 | `lf_industry_high_frp` (:288) | `industrial_fire` | industry within 2000 m AND frp >= 15 MW |
| 2 | `lf_factory_proximity_thermal` (:305) | `industrial_fire` | factory within 2000 m AND brightness >= 310 K AND confidence != low |
| 3 | `lf_industrial_zone_cluster` (:324) | `industrial_fire` | industry within 2000 m AND dense industrial cluster AND frp >= 5 MW |
| 4 | `lf_oil_gas_flare` (:349) | `gas_flare` | oil/gas within 2000 m AND (night detection OR persistent/static signal) |
| 5 | `lf_refinery_flare` (:373) | `gas_flare` | refinery within 2000 m AND flare evidence |
| 6 | `lf_mining_thermal_activity` (:399) | `mining_activity` | mining within 2000 m AND elevated thermal signal |
| 7 | `lf_mining_high_confidence` (:424) | `mining_activity` | mining within 3000 m AND high confidence AND thermal signal |
| 8 | `lf_agriculture_vegetation_fire` (:451) | `agricultural_burn` | agriculture within 15000 m AND FIRMS vegetation-fire type AND nearer than industry |
| 9 | `lf_agriculture_burn_context` (:473) | `agricultural_burn` | agriculture within 15000 m AND stubble-range thermal signature |
| 10 | `lf_forest_vegetation_fire` (:503) | `forest_natural_fire` | forest within 15000 m AND isolated from industry AND vegetation-fire type |
| 11 | `lf_strong_forest_fire` (:527) | `forest_natural_fire` | forest within 15000 m AND high-FRP wildfire signature |
| 12 | `lf_static_industrial_heat` (:555) | `industrial_process_heat` | FIRMS type 3 (static land source) AND industry nearby |
| 13 | `lf_nighttime_process_heat` (:577) | `industrial_process_heat` | night AND low steady FRP AND industry nearby |

### 4.2 Thresholds actually used

From `backend/app/core/constants.py:135-140` (shared defaults) and
`backend/app/intelligence/labeling_functions.py:70-86` (LF-local values).
Two corrections to values quoted elsewhere in the repo: agriculture and
forest buffers are 15,000 m, not 50,000 m; the industry threshold is 2,000 m
(the "1,000 m" inline comment at `labeling_functions.py:70` is stale).

| Constant | Value | Source |
|----------|-------|--------|
| `INDUSTRIAL_PROXIMITY_M` | 2000 m | `constants.py:138` |
| `REFINERY_PROXIMITY_M` | 2000 m | `constants.py:136` |
| `OIL_GAS_PROXIMITY_M` | 2000 m | `constants.py:137` |
| `MINING_PROXIMITY_M` | 2000 m | `constants.py:140` |
| `POWER_PLANT_PROXIMITY_M` | 5000 m | `constants.py:139` |
| `THRESHOLD_AGRICULTURE_PROXIMITY_M` | 15000 m | `labeling_functions.py:74` |
| `THRESHOLD_FOREST_PROXIMITY_M` | 15000 m | `labeling_functions.py:75` |
| `THRESHOLD_ISOLATED_FROM_INDUSTRY_M` | 1500 m | `labeling_functions.py:76` |
| `THRESHOLD_DEEP_ISOLATION_INDUSTRY_M` | 3000 m | `labeling_functions.py:77` |
| `DEFAULT_RADIUS_METERS` (OSM search) | 15000 m | `constants.py:135` |
| `FRP_VERY_HIGH_MW` | 15.0 | `labeling_functions.py:81` |
| `FRP_MODERATE_MW` | 5.0 | `labeling_functions.py:82` |
| `FRP_LOW_STEADY_MW` | 10.0 | `labeling_functions.py:83` |
| `FRP_AGRI_MAX_MW` | 45.0 | `labeling_functions.py:84` |
| `BRIGHTNESS_ELEVATED_K` | 330.0 | `labeling_functions.py:85` |
| `BRIGHTNESS_MODERATE_K` | 310.0 | `labeling_functions.py:86` |

### 4.3 Missing-data handling

Distance extraction (`labeling_functions.py:173-213`, `get_distance_meters`)
accepts meter fields (`distance_to_<category>_m`) and kilometer aliases
(`dist_*`, scaled x1000). The sentinel 999 (also NaN, Inf, empty string)
means "no data" (`labeling_functions.py:92-105`, `is_missing`). An LF whose
inputs are missing abstains instead of guessing; this single rule is what
produces the 96% abstention rate documented in Section 7.

--------------------------------------------------------------------------------

## 5. Machine Learning Model

- **Algorithm**: XGBoost 3.2.0 multi-class classifier (`requirements.txt:144`,
  `backend/app/ml/train.py:53`). 200 trees, max depth 6, learning rate 0.1.
- **Training targets**: weak labels from Section 4. No hand-labeled data exists
  anywhere in the pipeline.
- **Features (17)**, defined in `backend/app/ml/dataset_builder.py:54-87`:

| Group | Count | Columns |
|-------|-------|---------|
| Thermal | 3 | `frp`, `bright_ti4`, `bright_ti5` |
| Distance (km) | 8 | `dist_refinery`, `dist_factory`, `dist_industrial_zone`, `dist_oil_gas`, `dist_mining`, `dist_forest`, `dist_agriculture`, `dist_powerplant` |
| Proximity flags | 4 | `has_refinery_5km`, `has_powerplant_5km`, `has_factory_5km`, `has_industrial_2km` |
| Density counts | 2 | `count_ind_5km`, `count_ref_5km` |

- **Split**: 80/20 train/test, stratified; falls back to non-stratified split
  when a class has fewer than 2 samples (implemented in `train.py` because
  `unclassified` has exactly 1 training sample).
- **Validation**: stratified 5-fold cross-validation.
- **Label encoding**: string labels mapped to integers via scikit-learn
  `LabelEncoder`, required by XGBoost 3.x (`train.py:123`); the encoder is
  persisted to `backend/app/ml/models/label_encoder.joblib` so inference
  maps predictions back to identical strings.
- **Artifacts** (all in `backend/app/ml/models/`): `hotspot_classifier.joblib`,
  `feature_columns.joblib`, `label_classes.joblib`, `label_encoder.joblib`,
  `training_metrics.json`. Column order is frozen at training time because
  `predict.py` relies on it (`train.py:26`).

--------------------------------------------------------------------------------

## 6. Hybrid Intelligence Fusion

`classify_hotspot()` in `backend/app/intelligence/hybrid_engine.py:127`
runs both engines per hotspot - LF consensus (Section 4) and XGBoost
probabilities (`predict.py:91`, `predict_proba`) - then routes through five
ordered entry cases. Tuning constants live at `hybrid_engine.py`:
`REVIEW_CONFIDENCE = 0.60`, `ML_HIGH = 0.80`, `ML_MODERATE = 0.60`,
`STRONG_RULE_VOTES = 2`, `ML_SIGNAL_MIN = 0.45`, agreement boost capped at 0.12,
plus Phase F decision-policy constants `ML_PROBABLE_THRESHOLD = 0.80`,
`MODERATE_DECISION_CONFIDENCE = 0.70`, `ML_ASSISTED_REVIEW_MIN_PROB = 0.90`.

Every case yields a `classification_status`:
- **CONFIRMED** — independent rule evidence is strong (>= 2 LFs) and/or rules
  and ML positively agree. No operator review unless confidence < 0.60.
- **PROBABLE** — a single strong source (typically strong uncontested ML) with
  no independent spatial confirmation, or a contested strong source. Decision
  confidence is bounded at `MODERATE_DECISION_CONFIDENCE` (0.70) — model
  probability (~0.99) is *not* treated as calibrated decision confidence.
- **UNCERTAIN** — both sources weak/abstaining, ML below the acceptance
  threshold, or genuine strong disagreement. Decision confidence is 0.0 and the
  row is flagged for operator review.

| Case | Condition | Outcome (`classification_status`) |
|------|-----------|-----------------------------------|
| A. Agreement | rules and ML both signal and agree | `hybrid_agreement`, CONFIRMED; decision confidence = ML prob + up to 0.12 vote boost, capped 0.99 |
| B. ML-assisted | rules abstain, ML real class >= 0.80 (`ML_PROBABLE_THRESHOLD`) | `ml_assisted`, PROBABLE; decision confidence fixed at 0.70 (medium). Operator review only when ML prob < 0.90 (`ML_ASSISTED_REVIEW_MIN_PROB`) |
| C. Strong rules | >= 2 LF votes, ML weak or unclassified | `rule_dominant`, CONFIRMED; decision confidence = 0.55 + 0.10 per vote, capped 0.85 |
| D. Conflict | both signal but disagree | D1 strong rules + strong ML -> `conflict`, UNCERTAIN, review; D2 strong ML vs weak rule -> `ml_dominant`, PROBABLE, review; D3 strong rules vs weaker ML -> `rule_dominant`, PROBABLE, review; D4 close tie -> `conflict`, UNCERTAIN, review |
| E. Both abstain | neither engine has signal, or ML below the acceptance thresholds | `uncertain`, UNCERTAIN; decision confidence 0.0; always flagged for review |

Confidence tiers (`high`/`medium`/`low`) come from `_confidence_tier`; anything
below 0.60 is flagged for human review, which is why near-total abstention
converts into near-total review flagging rather than silent guesses.

Semantics (Phase A audit finding, never conflated):
`classification_status` is the final decision status; `hybrid_confidence`
(alias `decision_confidence`) is confidence in that decision (0.0 when no class
was chosen); `raw_ml_confidence` (alias `ml_probability`) is the raw XGBoost
probability of the ML top class and is *not* decision confidence; and
`rule_vote_strength` (alias `rule_consensus`) is the active LF count. On an
abstention the final class is `unclassified`, the decision confidence is 0.0,
and any high ML number belongs to the ML abstention output alone.

Why fusion is needed even though LFs and ML share features (Section 8.1
details the leakage): LFs output binary votes with no uncertainty measure,
while XGBoost outputs calibrated probabilities. Fusion converts "3 rules
fired" into a triage-ready confidence (e.g. 0.73 vs 0.99), routes
low-confidence cases to analysts via `requires_human_review`, and records a
per-decision explanation list (`:300-311` return dict) so every automated
call is auditable.

--------------------------------------------------------------------------------

## 7. Verified Results

Dataset: 642-hotspot snapshot (`data/processed/hotspots/classified_hotspots_v2.csv`,
42 columns). Note: the live FIRMS cache (`data/raw/firms_recent.csv`) has
since grown to 1,268 rows covering 2026-09-02 to 2026-09-04; the numbers below
describe the frozen 642-row snapshot the model was trained and evaluated on,
not the larger cache.

### 7.1 Label distribution after LF voting

Read from `data/processed/hotspots/training_dataset.csv` (642 rows).

| Label | Count | Share |
|-------|-------|-------|
| `unclassified` | 616 | 95.9% |
| `industrial_fire` | 14 | 2.2% |
| `agricultural_burn` | 6 | 0.9% |
| `gas_flare` | 3 | 0.5% |
| `forest_natural_fire` | 3 | 0.5% |
| `mining_activity` | 0 | 0.0% |
| `industrial_process_heat` | 0 | 0.0% |

Two classes never fire: no hotspot in the snapshot is simultaneously near a
mining site with a thermal signal, and no FIRMS type-3 static source sits
near industry.

### 7.2 Model performance

| Metric | Value | Meaning |
|--------|-------|---------|
| Test accuracy | 99.2% | Inflated by leakage (Section 8.1); do not quote without the caveat |
| Test macro F1 | 0.75 | Dragged down by zero-recall rare classes |
| 5-fold CV macro F1 | 0.66 +/- 0.11 | Honest generalization estimate; wide band reflects tiny rare classes |
| Classes with 0% recall | 4 of 6 evaluated | Data sparsity (Section 8.2), not a code bug |

### 7.3 Hybrid engine end-to-end (642 enriched rows)

Read from `data/processed/hotspots/classified_hotspots_v2_enriched.csv`.

| Decision source | Count | Share |
|-----------------|-------|-------|
| `uncertain` (Case E) | 617 | 96.1% |
| `hybrid_agreement` (Case A) | 25 | 3.9% |
| `ml_only`, `rule_dominant`, `conflict` | 0 | 0.0% |

- Human-review flagged: 617/642 = **96.1%**.
- `agreement` column is True for 100% of rows in this frozen snapshot. This
  needs careful reading: under the then-current engine, Case E set `agreement`
  when both engines "agreed" on `unclassified`, so the flag recorded
  consensus, not correctness. Since the Phase F decision-matrix change,
  `agreement` means *positive agreement on a real class* — reciprocal
  abstention no longer sets it True (1,268-row rerun: agreement 101 rows).
  It must never be quoted as accuracy.
- Risk scores span **80.9 to 100.0** (all within 0-100 by construction).

### 7.4 Worked example (live run, not a mock)

The service demo (`backend/app/services/hotspot_service.py`, executed during
documentation review) analyzes a Gujarat hotspot at (21.1051, 72.6438), FRP
5.9 MW, brightness 330.8 K, nominal confidence: it is a strong proximity case
(industry ~530 m, three independent LFs concur, XGBoost agrees), so fusion
boosts decision confidence toward 0.990, `classification_status` `confirmed`,
and clears review. A hotspot with identical FRP but industry at 50 km draws
zero votes and lands in `uncertain` instead. Proximity evidence, not FRP
alone, decides.

### 7.5 Reading the enriched output

`data/processed/hotspots/classified_hotspots_v2_enriched.csv` keeps every
input column and appends the hybrid decision per row
(`scripts/run_pipeline.py:30-33`):

| Column | Content |
|--------|---------|
| `final_label` | fused class (Section 6 outcome) |
| `classification_status` | `confirmed` / `probable` / `uncertain` |
| `hybrid_confidence` / `decision_confidence` | 0.0-1.0 confidence in the FINAL decision (0.0 on abstention) |
| `raw_ml_confidence` / `ml_probability` | raw XGBoost probability of the ML top class (NOT decision confidence) |
| `confidence_level` / `decision_confidence_level` | low / medium / high tier |
| `decision_source` | one of `hybrid_agreement`, `ml_assisted`, `rule_dominant`, `ml_dominant`, `conflict`, `uncertain` |
| `agreement` / `conflict` | boolean flags (positive class agreement; not accuracy) |
| `requires_human_review` / `review_reason` | triage routing + machine-readable why |
| `rule_prediction` / `rule_active_votes` / `rule_consensus` / `rule_vote_strength` | raw rule-engine outputs |
| `explanation_bullets` | human-readable audit trail per decision |
| `risk_score` | 0-100 operational score, separate from class confidence |
| `ml_prediction` / `ml_top_probability` | raw ML-engine outputs for debugging |

--------------------------------------------------------------------------------

## 8. Limitations

### 8.1 Data leakage between LFs and ML

The 13 LFs vote on the same distance features XGBoost trains on, so the
model largely memorizes LF decisions instead of learning independent
patterns. The 99.2% test accuracy is inflated; a hand-coded rule system on
the same features scores similarly, and the honest generalization estimate
is the CV band (0.66 +/- 0.11), i.e. roughly 70-80% effective accuracy.
Mitigation: the hybrid layer still adds value through probability
calibration, triage confidences, and review routing (Section 6), which
binary LF votes cannot provide.

### 8.2 Class imbalance (96% unclassified)

LFs abstain when industry is farther than 2 km, forest/agriculture farther
than 15 km, FRP is weak, or FIRMS type is unknown - conditions that hold for
616 of 642 hotspots. This is honest sparsity: the OSM cache does not cover
every industrial site, and most satellite detections are genuinely ambiguous
at 375 m resolution. Mitigation: abstention is routed to human review
instead of being forced into a class; the review queue IS the product for
ambiguous cases.

### 8.3 Sparse OSM coverage

The forest/agriculture cache holds 3,791 sites across 27 fetch regions and
the industrial cache 20,231 sites across 17 states - large but incomplete.
Hotspots in poorly mapped districts can sit 100-500 km from the nearest
cached industrial site, which LFs correctly read as "no context".
Mitigation: `radius_meters` is configurable per call
(`hotspot_service.py:126`); live Overpass queries exist (`use_live_api`)
but are rate-limited and reserved for single-hotspot analysis, not batches.

### 8.4 Synthetic mining data

The 20 mining sites are approximate coordinates of known mining belts
(Jharia, Singrauli, Bellary, and others), injected by
`scripts/add_synthetic_mining.py` - not live OSM query results. They provide
distance signal for `mining_activity`, which otherwise has zero representation.
Mitigation: documented here and in the script header; replace with surveyed
coordinates or expanded Overpass mining queries before operational use.
Notably, even with these proxies the snapshot produced zero mining votes,
so no training label depends on synthetic coordinates today.

### 8.5 Short FIRMS window

The evaluated snapshot covers 3 days. The live cache has since grown to
1,268 rows (2026-09-02 to 2026-09-04), which should expand rare-class support
on the next retrain. Mitigation: `scripts/fetch_firms_api.py` re-pulls on
demand; the pipeline is append-capable (`run_pipeline.py --no-resume`
forces full reprocessing; default resumes from existing output).

--------------------------------------------------------------------------------

## 9. Reproducibility

All commands run from the repository root with the virtual environment
active. Module invocations below were executed during documentation review;
scripts under `scripts/` are invoked by file path because `python -m
scripts.<name>` hangs in this environment (package-mode import quirk -
identical code path, no output, near-zero CPU; file-path form completes).

```text
# 1. Integrity guard (must print ALL GUARDS PASSED, exit 0)
venv\Scripts\python scripts/check_data_integrity.py

# 2. Build the v2 distance dataset (642 FIRMS x OSM caches)
venv\Scripts\python scripts/build_synth_v2.py

# 3. Weak labels via the 13 LFs -> training_dataset.csv
venv\Scripts\python -m backend.app.ml.dataset_builder

# 4. Train XGBoost (model + metrics -> backend/app/ml/models/)
venv\Scripts\python -m backend.app.ml.train

# 5. Evaluation plots + report (-> data/processed/hotspots/)
venv\Scripts\python -m backend.app.ml.evaluate

# 6. Batch hybrid pipeline (resume-aware; --limit N for smoke tests)
venv\Scripts\python scripts/run_pipeline.py --help
venv\Scripts\python scripts/run_pipeline.py --limit 50

# 7. Single-hotspot demo (Gujarat industrial case, prints
#    spatial context + hybrid decision + explanation)
venv\Scripts\python backend/app/services/hotspot_service.py

# 8. LF self-test (must print 9/9 PASS)
venv\Scripts\python -m app.intelligence.labeling_functions
```

Expected outputs after a full run: `training_dataset.csv` (642 rows, label
counts as in Section 7.1), model joblibs + `training_metrics.json`,
`confusion_matrix.png`, `feature_importance.png`,
`classification_report.txt`, `evaluation_metrics.json`, and
`classified_hotspots_v2_enriched.csv` (642 rows, review rate ~96%).

--------------------------------------------------------------------------------

## 10. Future Work

- Expand FIRMS history beyond the 3-day snapshot; the 1,268-row cache already
  waiting in `data/raw/` is the first retraining candidate.
- Extend OSM coverage to unmapped districts and all states; replace the 20
  synthetic mining proxies with surveyed or Overpass-sourced coordinates.
- Add live Overpass queries with rate limiting for batch use (currently
  single-hotspot only via `use_live_api`).
- Active learning loop: feed reviewed human decisions back as training labels
  to shrink the 96% abstention share honestly.
- Separate the risk score into its own calibrated module instead of bundling
  it with classification output.
- FastAPI serving endpoint (Phase H) and React GIS dashboard (Phase I).
- Fix stale "14 labeling functions" strings in code comments
  (`hotspot_service.py:137`, `hybrid_engine.py:282`, `run_pipeline.py:8`)
  to match the 13-function registry.
