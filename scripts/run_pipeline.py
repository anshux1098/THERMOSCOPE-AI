"""
run_pipeline.py
Phase E — Batch pipeline for THERMOSCOPE-AI (SIH26162).

End-to-end CLI that:
  1. Loads hotspots from the canonical classified_hotspots_v2.csv
     (data/processed/hotspots/classified_hotspots_v2.csv)
  2. For each hotspot, runs the Hybrid Intelligence Engine
     (13 labeling functions + XGBoost ML)
  3. Writes an enriched CSV with classification results

Modes:
    Default (resume):
        python scripts/run_pipeline.py
        Skip rows already present in the output (keyed by stable `input_index`).

    Force (reprocess everything, but replace by stable row id — never duplicates):
        python scripts/run_pipeline.py --force

    Limited test runs:
        python scripts/run_pipeline.py --limit 10
        python scripts/run_pipeline.py --limit 10 --force

    Dry run (no processing, no writes):
        python scripts/run_pipeline.py --dry-run

    Resume key:
        `input_index` = the row index in the input classified CSV. Deterministic
        and stable across runs. Force mode REPLACES the existing output row for
        the same input_index (keep-last), so forced reruns never duplicate rows.

Output:
    Classification per hotspot appended to the input columns:
    final_label, hybrid_confidence, decision_source, agreement, conflict,
    requires_human_review, review_reason, explanation_bullets, risk_score,
    rule_prediction, rule_active_votes, ml_prediction, ml_top_probability.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Make backend package importable
repo_root = Path(__file__).resolve().parents[1]
backend_dir = str(repo_root / "backend")
root_dir = str(repo_root)
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd

from app.intelligence.hybrid_engine import classify_hotspot
from app.intelligence.label_aggregator import apply_labeling_functions, compute_vote_summary
from app.core.paths import CLASSIFIED_DATASET_PATH, ENRICHED_DATASET_PATH
from app.core.lineage import validate_classified_dataset
from app.geo.spatial_features import SENTINEL_DISTANCE_M

SENTINEL_KM_THRESHOLD = 999.0

DEFAULT_INPUT_CSV = str(CLASSIFIED_DATASET_PATH)
DEFAULT_OUTPUT_CSV = str(ENRICHED_DATASET_PATH)

# Column that identifies a row in the output. Equal to the input CSV row index.
RESUME_KEY = "input_index"


def _km(v):
    """Convert km value to meters, preserving the canonical 999 sentinel.

    Phase B P1.4 fix: any km value >= 999.0 is the "no nearby entity" sentinel
    (999.0 km == 999000.0 m == SENTINEL_DISTANCE_M), so it maps to the sentinel
    meters — not None and never a fabricated sub-45km proximity.
    """
    if v is None or pd.isna(v):
        return None
    v = float(v)
    if v >= SENTINEL_KM_THRESHOLD:
        return SENTINEL_DISTANCE_M
    return v * 1000.0


def _build_feature_row(row: pd.Series) -> Dict[str, Any]:
    """Convert a v2 CSV row into the feature dict the hybrid engine expects."""
    return {
        # FIRMS thermal / metadata
        "frp": float(row["frp"]) if not pd.isna(row.get("frp")) else None,
        "bright_ti4": float(row.get("bright_ti4", 0)) if not pd.isna(row.get("bright_ti4")) else None,
        "bright_ti5": float(row.get("bright_ti5", 0)) if not pd.isna(row.get("bright_ti5")) else None,
        "confidence": str(row.get("confidence", "n")),
        "daynight": str(row.get("daynight", "D")),
        "acq_date": str(row.get("acq_date", "")),
        "acq_time": str(row.get("acq_time", "")),
        "satellite": str(row.get("satellite", "")),
        # Canonical meter fields
        "distance_to_industry_m": _km(row.get("dist_industry") if row.get("dist_industry") is not None else row.get("dist_factory")),
        "distance_to_refinery_m": _km(row.get("dist_refinery")),
        "distance_to_oil_gas_m": _km(row.get("dist_oil_gas")),
        "distance_to_mining_m": _km(row.get("dist_mining")),
        "distance_to_agriculture_m": _km(row.get("dist_agriculture")),
        "distance_to_forest_m": _km(row.get("dist_forest")),
        "distance_to_power_plant_m": _km(row.get("dist_powerplant")),
        # km aliases
        "dist_industry": row.get("dist_industry") if row.get("dist_industry") is not None else row.get("dist_factory"),
        "dist_factory": row.get("dist_factory"),
        "dist_industrial_zone": row.get("dist_industrial_zone"),
        "dist_refinery": row.get("dist_refinery"),
        "dist_oil_gas": row.get("dist_oil_gas"),
        "dist_mining": row.get("dist_mining"),
        "dist_agriculture": row.get("dist_agriculture"),
        "dist_forest": row.get("dist_forest"),
        "dist_powerplant": row.get("dist_powerplant"),
        "dist_power_plant": row.get("dist_power_plant"),
        # Flag/count columns
        "has_industrial_2km": int(row.get("has_industrial_2km", 0)),
        "has_factory_5km": int(row.get("has_factory_5km", 0)),
        "has_refinery_5km": int(row.get("has_refinery_5km", 0)),
        "has_powerplant_5km": int(row.get("has_powerplant_5km", 0)),
        "has_forest_5km": int(row.get("has_forest_5km", 0)),
        "has_agriculture_5km": int(row.get("has_agriculture_5km", 0)),
        # REAL neighbourhood counts (Phase B P1.5)
        "industrial_sites_within_2km": int(row.get("industrial_sites_within_2km", 0)),
        "industrial_sites_within_5km": int(row.get("industrial_sites_within_5km", 0)),
        "refinery_sites_within_3km": int(row.get("refinery_sites_within_3km", 0)),
        "refinery_sites_within_5km": int(row.get("refinery_sites_within_5km", 0)),
        "forest_sites_within_5km": int(row.get("forest_sites_within_5km", 0)),
        "agriculture_sites_within_5km": int(row.get("agriculture_sites_within_5km", 0)),
        "count_forest_5km": int(row.get("count_forest_5km", 0)),
        "count_agriculture_5km": int(row.get("count_agriculture_5km", 0)),
        "firms_type": row.get("firms_type"),
        # Legacy ML-schema count aliases (DEPRECATED, kept byte-compatible)
        "count_ind_5km": int(row.get("count_ind_5km", 0)),
        "count_ref_5km": int(row.get("count_ref_5km", 0)),
    }


def _load_existing_output(output_csv: str):
    """
    Load the full existing output (ALL columns — never read only the resume key)
    and return (existing_df, processed_indices).

    Reading only the resume key was a bug that silently destroyed every other
    column when merging, producing a one-column enriched CSV.
    """
    if not Path(output_csv).exists():
        return None, set()

    try:
        existing_df = pd.read_csv(output_csv)
    except Exception:
        return None, set()

    if RESUME_KEY not in existing_df.columns:
        return None, set()

    processed = set()
    for v in existing_df[RESUME_KEY].tolist():
        try:
            processed.add(int(v))
        except (TypeError, ValueError):
            continue
    return existing_df, processed


def process_hotspots(
    input_csv: str = DEFAULT_INPUT_CSV,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    start: int = 0,
    limit: Optional[int] = None,
    resume: bool = True,
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Process hotspots from the canonical classified CSV and write enriched results.

    Args:
        input_csv: Canonical classified_hotspots_v2.csv path.
        output_csv: Enriched output path.
        start: Start index (for resuming).
        limit: Max number of hotspots to process (None = all selected).
        resume: If True, skip input rows whose RESUME_KEY already exists in output.
        force: If True, reprocess selected rows even if already in output. Existing
               records are REPLACED by stable row id (no duplicates).
        dry_run: If True, only compute and print what would be processed. No
                 classification, no writes, no file modification.
        verbose: Print detailed progress.

    Returns:
        Stats dict: total, processed, skipped, errors, label/decision distributions.
    """
    print(f"[pipeline] Loading {input_csv}...")
    # DATA CONTRACT: validates canonical path, schema, non-empty, not synthetic.
    df = validate_classified_dataset(Path(input_csv))
    total = len(df)
    print(f"  -> {total} rows, {len(df.columns)} columns")

    if limit is not None:
        end = min(start + limit, total)
    else:
        end = total
    selected = list(range(start, end))

    existing_df, processed_indices = _load_existing_output(output_csv)

    # Determine the set of rows that would be processed.
    if force:
        to_process = list(selected)
        would_skip = 0
    else:
        to_process = [i for i in selected if i not in processed_indices]
        would_skip = len(selected) - len(to_process)

    mode = "FORCE" if force else ("RESUME" if resume else "FRESH")
    if dry_run:
        print(f"\n  Mode          : {mode} (dry run — no data modified)")
        print(f"  Input rows    : {total}")
        print(f"  Already done  : {len(processed_indices)}")
        print(f"  Selected      : {len(selected)}")
        print(f"  Would process : {len(to_process)}")
        print(f"  Would skip    : {would_skip}")
        print(f"  Output (untouched): {output_csv}")
        return {
            "total": total,
            "selected": len(selected),
            "processed": 0,
            "skipped": would_skip,
            "errors": 0,
            "dry_run": True,
        }

    print(f"\n  Mode          : {mode}")
    print(f"  Input rows    : {total}")
    print(f"  Processing    : {len(to_process)} row(s) (limit={limit if limit is not None else 'all'})")

    results = []
    errors = []
    label_counts = {}
    decision_source_counts = {}
    n_agreement = 0
    n_review = 0
    t0 = time.time()

    for n, i in enumerate(to_process, start=1):
        try:
            row = df.iloc[i]
            features = _build_feature_row(row)
            result = classify_hotspot(features)

            rec = {
                RESUME_KEY: i,
                "latitude": row.get("latitude"),
                "longitude": row.get("longitude"),
                "frp": row.get("frp"),
                "bright_ti4": row.get("bright_ti4"),
                "acq_date": row.get("acq_date"),
                "dist_industry_km": row.get("dist_industry"),
                "dist_forest_km": row.get("dist_forest"),
                "dist_agriculture_km": row.get("dist_agriculture"),
                "dist_oil_gas_km": row.get("dist_oil_gas"),
                "dist_refinery_km": row.get("dist_refinery"),
                "dist_mining_km": row.get("dist_mining"),
                "dist_powerplant_km": row.get("dist_powerplant"),
                "final_label": result["final_label"],
                "hybrid_confidence": result["hybrid_confidence"],
                "raw_ml_confidence": result["raw_ml_confidence"],
                "confidence_level": result["confidence_level"],
                "decision_source": result["decision_source"],
                "agreement": result["agreement"],
                "conflict": result["conflict"],
                "requires_human_review": result["requires_human_review"],
                "review_reason": result.get("review_reason"),
                "rule_prediction": result["rule_engine"]["prediction"],
                "rule_active_votes": result["rule_engine"]["active_votes"],
                "ml_prediction": result["ml_engine"]["prediction"],
                "ml_top_probability": result["ml_engine"]["confidence"],
                "risk_score": round(result["hybrid_confidence"] * 100, 1),
                "explanation_bullets": json.dumps(result.get("explanation", []), ensure_ascii=False),
                "n_explanation_bullets": len(result.get("explanation", [])),
            }
            results.append(rec)

            label = result["final_label"]
            label_counts[label] = label_counts.get(label, 0) + 1
            src = result["decision_source"]
            decision_source_counts[src] = decision_source_counts.get(src, 0) + 1
            if result["agreement"]:
                n_agreement += 1
            if result["requires_human_review"]:
                n_review += 1

            if verbose and n % 50 == 0:
                elapsed = time.time() - t0
                rate = n / elapsed if elapsed > 0 else 0
                print(f"  [{n}/{len(to_process)}] {rate:.1f} rows/sec")

        except Exception as e:
            errors.append({"input_index": i, "error": str(e)[:200]})
            if verbose and len(errors) <= 5:
                print(f"  [error] row {i}: {str(e)[:100]}")

    # Write results. Force / resume replace records by stable RESUME_KEY:
    # concatenate existing + new, keep-last per input_index, never duplicates.
    new_df = pd.DataFrame(results)
    if existing_df is not None and len(existing_df) > 0:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=[RESUME_KEY], keep="last")
        combined = combined.sort_values(RESUME_KEY, kind="mergesort").reset_index(drop=True)
        combined.to_csv(output_csv, index=False)
    else:
        new_df.to_csv(output_csv, index=False)

    elapsed = time.time() - t0
    stats = {
        "total": total,
        "selected": len(selected),
        "processed": len(results),
        "skipped": would_skip,
        "errors": len(errors),
        "elapsed_sec": round(elapsed, 1),
        "rate_per_sec": round(len(results) / elapsed, 2) if elapsed > 0 else 0,
        "label_counts": label_counts,
        "decision_source_counts": decision_source_counts,
        "n_agreement": n_agreement,
        "n_review": n_review,
        "dry_run": False,
    }

    print("\n" + "=" * 66)
    print("PIPELINE SUMMARY")
    print("=" * 66)
    print(f"Input rows:              {total}")
    print(f"Selected for processing: {len(selected)}")
    print(f"Successfully processed:  {len(results)}")
    print(f"Skipped:                 {would_skip}")
    print(f"Errors:                  {len(errors)}")
    print()
    if len(results) > 0:
        print("Label distribution:")
        for label, n in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"  {label}: {n} ({100*n/len(results):.1f}%)")
        print("\nDecision sources:")
        for src, n in sorted(decision_source_counts.items(), key=lambda x: -x[1]):
            print(f"  {src}: {n} ({100*n/len(results):.1f}%)")
        print(f"\nFusion agreement:        {n_agreement}/{len(results)} "
              f"({100*n_agreement/len(results):.1f}%)")
        print(f"Review flagged:          {n_review}/{len(results)} "
              f"({100*n_review/len(results):.1f}%)")
    print(f"\nOutput:                  {output_csv}")
    print(f"Duration:                {stats['elapsed_sec']} seconds")
    if elapsed > 0:
        print(f"Throughput:              {stats['rate_per_sec']} rows/sec")
    print("=" * 66)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="THERMOSCOPE-AI batch pipeline: classify hotspots end-to-end"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV, help="Input classified CSV path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV, help="Enriched output CSV path")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--limit", type=int, default=None, help="Max hotspots to process")
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not skip already-processed rows (fresh run)")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess selected rows even if already in output "
                             "(existing records replaced by stable row id, never duplicated)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report what would be processed; write nothing")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    args = parser.parse_args()

    print("=" * 66)
    print("THERMOSCOPE-AI — Phase E Batch Pipeline")
    print("=" * 66)
    print()

    if args.force and args.no_resume:
        print("[pipeline] Note: --force already reprocesses everything; "
              "--no-resume is redundant and ignored.")

    stats = process_hotspots(
        input_csv=args.input,
        output_csv=args.output,
        start=args.start,
        limit=args.limit,
        resume=not args.no_resume,
        force=args.force,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

    print()
    print("=" * 66)
    print("Phase E pipeline complete.")
    print("=" * 66)


if __name__ == "__main__":
    main()