# MVP Hybrid Classification Audit & Decision Policy — Final Report

Date: 2026-09-06 · Range: full read-only audit, decision-policy implementation, pipeline regeneration, test + API validation. Companion doc: `reports/HYBRID_DECISION_AUDIT.md` (Phase A detail).

## 1. Root cause

The MVP flagged two problems: (a) ~92% of hotspots are `unclassified`, and (b) an API semantic contradiction — a *final decision* of `unclassified` that nevertheless printed ~99.8% "confidence", "high confidence level", and a `risk_score` of ~99.8.

**Audit finding (Phase A):** the unclassified rate is NOT caused by the hybrid matrix discarding ML evidence. On all 1,164 rules-abstain rows the ML model's top class is itself `unclassified` (median probability 0.9997) — **0 rows carry a confident real-class ML prediction the matrix could have used** (0 rows at any real-class probability ≥ 0.45). The engine's ML-only branch was structurally unreachable on this snapshot. Root causes:

1. **Spatial-evidence sparsity** — 1,164/1,268 hotspots (92%) have no labeling-function trigger (no industry/forest/agriculture/refinery/mining entity inside the proximity gates). Rules correctly abstain.
2. **ML majority-abstention behavior** — the weak-supervision training surface is dominated by `unclassified`; XGBoost learned to emit that class on no-evidence rows.
3. **Semantic conflation (the bug)** — the stored artifact computed `hybrid_confidence`, its confidence tier, and `risk_score` from `raw_ml_confidence` (probability of the abstention class). Result: 1,166 unclassified rows at `hybrid_confidence` ≈ 0.9997 min 0.65, `confidence_level=high` (1,164) / `medium` (2), `risk_score` ≈ 99.8.

No retraining was required or performed; the canonical input dataset was not modified. Improvement is delivered purely through **honest semantics + deterministic decision policy**.

## 2. Decision matrix (implemented in `hybrid_engine.py`, ordered evaluation)

Constants (all centralized in `hybrid_engine.py`): `HIGH_CONFIDENCE = 0.80`, `MODERATE_CONFIDENCE = 0.60`, `REVIEW_CONFIDENCE = 0.60`, `ML_HIGH = 0.80`, `ML_MODERATE = 0.60`, `ML_SIGNAL_MIN = 0.45`, `STRONG_RULE_VOTES = 2`, `ML_PROBABLE_THRESHOLD = 0.80`, `MODERATE_DECISION_CONFIDENCE = 0.70`, `ML_ASSISTED_REVIEW_MIN_PROB = 0.90`.

| Case | Condition | classification_status | decision_source | decision confidence | human review |
|------|-----------|----------------------|-----------------|---------------------|--------------|
| A | rules + ML both signal, agree | confirmed | hybrid_agreement | min(0.99, ml_prob + 0.04·votes) | no |
| B | rules abstain, ML real class ≥ 0.80 | probable | ml_assisted | 0.70 (moderate) | only if ml_prob < 0.90 |
| C | ≥ 2 LF votes, ML weak/absent | confirmed | rule_dominant | min(0.85, 0.55 + 0.10·votes) | no |
| D1 | ≥ 2 votes AND ML ≥ 0.80, disagree | uncertain | conflict | 0.0 | yes |
| D2 | ML ≥ 0.80 vs weak rule (1 vote) | probable | ml_dominant | 0.70 | yes |
| D3 | ≥ 2 votes vs ML < 0.80, disagree | probable | rule_dominant | 0.70 | yes |
| D4 | weak conflict / close tie | uncertain | conflict | 0.0 | yes |
| E | both abstain / below thresholds | uncertain | uncertain | 0.0 | yes |

Justification for `ML_PROBABLE_THRESHOLD = 0.80`: every real-class ML prediction on the snapshot sits ≥ 0.60, and 0.80 is the existing high-confidence tier; the Phase A count of eligible Case B rows on current data is **0**, so this constant is a forward-compatibility policy, not an inflation lever. For `probable` outcomes the decision confidence is bounded at 0.70 because model probability (~0.99) is not calibrated validated accuracy.

Semantic field separation (internal `/analyze` + presentation): `classification_status` (confirmed/probable/uncertain) ≠ `ml_probability`/`raw_ml_confidence` (raw XGBoost top-class probability) ≠ `decision_confidence`/`hybrid_confidence` (confidence in the final decision; 0.0 on abstention) ≠ `rule_vote_strength`/`rule_active_votes` (active LF count). `agreement` now means *positive agreement on a real class* — reciprocal abstention no longer counts.

## 3. Audit numbers — before vs after (1,268 canonical rows)

| Metric | Before (pre-fix engine) | After (new policy, `--force`) |
| --- | --- | --- |
| Decided | 102 | 101 |
| — industrial_fire | 47 | 47 |
| — forest_natural_fire | 41 | 41 |
| — gas_flare | 8 | **7** (close-tie now honestly uncertain) |
| — mining_activity | 3 | 3 |
| — industrial_process_heat | 2 | 2 |
| — agricultural_burn | 1 | 1 |
| Unclassified | 1,166 | 1,167 |
| **classification_status — confirmed** | n/a (not tracked) | **101** |
| **classification_status — probable** | n/a | **0** (audit-predicted: Case B unreachable on this model) |
| **classification_status — uncertain** | n/a | **1,167** |
| decision_source: hybrid_agreement | 100 | 100 |
| decision_source: rule_dominant | 1 | 1 |
| decision_source: ml_dominant / ml_only | 1 / 0 | 0 / 0 (replaced by `conflict`, `ml_assisted`) |
| decision_source: conflict | (flag only) | 1 |
| decision_source: uncertain | 1,166 | 1,166 |
| agreement flag = True | 1,267 (incl. reciprocal abstention) | **101** (meaningful) |
| conflict flag = True | 1 | 1 |
| requires_human_review | 1,167 | 1,167 |
| Unclassified `hybrid_confidence` | ≈ 0.9997 (min 0.6516) | **0.0 × 1,167** |
| Unclassified `confidence_level` | high 1,164 / medium 2 | low 1,167 |
| Unclassified `risk_score` | ≈ 99.8 (max 100) | 0.0 |
| Contradictory explanation text | 0 in stored enriched CSV (live API already fixed); legacy wording existed in the now-removed `backend/app/ml/experiments/e4*` historical outputs | **0** in runtime output |

## 4. Changed files

- `backend/app/intelligence/hybrid_engine.py` — decision matrix, `classification_status`, policy constants, honest per-case explanations, semantic aliases (`decision_confidence`, `ml_probability`, `rule_consensus`, `rule_vote_strength`, `decision_confidence_level`), `agreement` redefined, agent `ml_only` → `ml_assisted`.
- `backend/app/schemas/analysis_response.py` — `ClassificationSummary` + `classification_status`, `rule_prediction`, `rule_vote_strength`.
- `backend/app/services/hotspot_service.py` — presentation projection maps new fields, status-coherent fallback.
- `backend/app/services/recommendation_service.py` — Phase F policy keyed on `classification_status` (`probable` → verify-before-escalation; review reminder only when engine requests it; abstention carries its own verification set).
- `scripts/run_pipeline.py` — persists the new semantics columns.
- `docs/classification_logic.md` — decision matrix + semantics documentation updated.
- `tests/test_hybrid_policy.py` (new) — 11 deterministic cases incl. all matrix branches + threshold canary.
- `tests/test_api.py`, `tests/test_api_response.py`, `tests/test_pipeline_cli.py` — contracts extended for the new fields/source.
- `data/processed/hotspots/classified_hotspots_v2_enriched.csv` — regenerated (`--force`, sanctioned derived-output rewrite; canonical input untouched).
- `reports/HYBRID_DECISION_AUDIT.md` + this report (new).

## 5. Validation

- **pytest**: 130 passed (119 prior + 11 new matrix tests), 0 regressions.
- **Pipeline**: `python scripts/run_pipeline.py --force` → 1,268/1,268 processed, 0 errors, 5.2 s (≈246 rows/s), keep-last dedupe by stable `input_index`, no duplicates.
- **API** (FastAPI TestClient, both endpoints, unclassified payload): `/health` 200; `/api/v1/hotspots/analyze` 200 (~385 ms) exposing `final_label`, `classification_status=uncertain`, `hybrid_confidence=0.0`, `decision_confidence=0.0`, `raw_ml_confidence=0.9983` kept separate, `rule_vote_strength=0`; `/api/v1/hotspots/analyze/presentation` 200 (~377 ms) with `confidence_score=null`, `confidence_level=insufficient_evidence`, `decision_status=uncertain`, `classification_status=uncertain`, `ml_confidence=0.9983` labeled as ML-only, non-empty evidence bullets, deduplicated recommendations, no generic review reminder on abstention, no contradictory wording.
- **Determinism**: engine is pure over features + frozen model; thresholds centralized.

## 6. Honest limitations

- `probable` (ML-assisted) currently classifies **0** of the 1,268 rows because the production XGBoost emits the `unclassified` abstention class on every no-evidence row. The `ml_assisted` branch is live and tested but unreachable on this snapshot — gaining real PROBABLE rows requires new/retrained ML signal, not threshold tuning (deliberately out of scope).
- Case D1/D4 replaces the old "pick a winner at 50%" tie-break with UNCERTAIN: decided count drops 102 → 101 (one close-tie `gas_flare`). This is the intended honesty change.
- Risk tier for `unclassified` remains `critical` (`risk_level_for` label-based policy: an unknown thermal event is routed to review before escalation). `risk_score` now correctly reflects decision confidence (0.0 on abstention) and is not a probability claim.
- Model accuracy claims are unchanged and remain caveated (macro F1 0.5838 is the honest headline; 98.81% accuracy is majority-class-dominated). No new accuracy claims were introduced; `ml_probability` is labeled as model output, not validated accuracy.
- Historical experiment outputs under `backend/app/ml/experiments/e4*` (which retained the old contradictory phrasing) were removed as redundant generated data; the `imbalance.py` + `experiment_runner.py` harness regenerates them on demand. Nothing in the runtime path is affected.