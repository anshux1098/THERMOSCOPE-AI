"""
run_pipeline.py
Phase E — Batch pipeline for THERMOSCOPE-AI (SIH26162).

End-to-end CLI that:
  1. Loads hotspots from classified_hotspots_v2.csv
  2. For each hotspot, runs the Hybrid Intelligence Engine
     (14 labeling functions + XGBoost ML)
  3. Writes an enriched CSV with classification results

Optimized for performance:
  - Model loaded ONCE at start, not per hotspot
  - Vectorized feature extraction
  - Resume support: skip already-processed rows

Usage:
    # Process all hotspots (default)
    venv/Scripts/python -m scripts.run_pipeline

    # Process first N hotspots
    venv/Scripts/python -m scripts.run_pipeline --limit 50

    # Process from a specific index
    venv/Scripts/python -m scripts.run_pipeline --start 100 --limit 50

    # Use live OSM queries (slow, use only for single hotspots)
    venv/Scripts/python -m scripts.run_pipeline --limit 1 --live-osm

Output:
    data/processed/hotspots/classified_hotspots_v2_enriched.csv
    Columns: original + final_label, hybrid_confidence, decision_source,
             agreement, conflict, requires_human_review, review_reason,
             explanation_bullets, risk_score, rule_votes, ml_predictions
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make backend package importable
backend_dir = str(Path(__file__).resolve().parents[1])
root_dir = str(Path(__file__).resolve().parents[0])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd

from app.intelligence.hybrid_engine import classify_hotspot
from app.intelligence.label_aggregator import apply_labeling_functions, compute_vote_summary


DEFAULT_INPUT_CSV = "data/processed/hotspots/classified_hotspots_v2.csv"
DEFAULT_OUTPUT_CSV = "data/processed/hotspots/classified_hotspots_v2_enriched.csv"


def _build_feature_row(row: pd.Series) -> Dict[str, Any]:
    """Convert a v2 CSV row into the feature dict the hybrid engine expects."""
    def _km(v):
        """Convert km value to meters, preserving 999 sentinel."""
        if v is None or v == 999.0 or pd.isna(v):
            return None
        return float(v) * 1000.0

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
        "distance_to_industry_m": _km(row.get("dist_industry")),
        "distance_to_refinery_m": _km(row.get("dist_refinery")),
        "distance_to_oil_gas_m": _km(row.get("dist_oil_gas")),
        "distance_to_mining_m": _km(row.get("dist_mining")),
        "distance_to_agriculture_m": _km(row.get("dist_agriculture")),
        "distance_to_forest_m": _km(row.get("dist_forest")),
        "distance_to_power_plant_m": _km(row.get("dist_powerplant")),
        # km aliases
        "dist_industry": row.get("dist_industry"),
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
        "count_ind_5km": int(row.get("count_ind_5km", 0)),
        "count_ref_5km": int(row.get("count_ref_5km", 0)),
        "count_forest_5km": int(row.get("count_forest_5km", 0)),
        "count_agriculture_5km": int(row.get("count_agriculture_5km", 0)),
    }


def process_hotspots(
    input_csv: str,
    output_csv: str,
    start: int = 0,
    limit: Optional[int] = None,
    resume: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Process hotspots from input_csv and write enriched results to output_csv.

    Args:
        input_csv: Path to classified_hotspots_v2.csv
        output_csv: Path to write enriched CSV
        start: Start index (for resuming)
        limit: Max number of hotspots to process (None = all)
        resume: If True and output_csv exists, skip already-processed rows
        verbose: Print progress

    Returns:
        Stats dict: total, processed, errors, label distribution
    """
    print(f"[pipeline] Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    total = len(df)
    print(f"  -> {total} rows, {len(df.columns)} columns")

    if limit is not None:
        end = min(start + limit, total)
    else:
        end = total

    # Check for resume
    processed_indices = set()
    existing_df = None
    if resume and Path(output_csv).exists():
        try:
            existing_df = pd.read_csv(output_csv, usecols=["input_index"])
            processed_indices = set(existing_df["input_index"].tolist())
            print(f"  -> Resume: skipping {len(processed_indices)} already-processed rows")
        except Exception:
            pass

    results = []
    errors = []
    label_counts = {}
    decision_source_counts = {}
    n_agreement = 0
    n_review = 0
    t0 = time.time()

    for i in range(start, end):
        if i in processed_indices:
            continue

        try:
            row = df.iloc[i]
            features = _build_feature_row(row)
            result = classify_hotspot(features)

            # Collect result
            rec = {
                "input_index": i,
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

            # Stats
            label = result["final_label"]
            label_counts[label] = label_counts.get(label, 0) + 1
            src = result["decision_source"]
            decision_source_counts[src] = decision_source_counts.get(src, 0) + 1
            if result["agreement"]:
                n_agreement += 1
            if result["requires_human_review"]:
                n_review += 1

            if verbose and (i - start + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i - start + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{i - start + 1}/{end - start}] {rate:.1f} rows/sec")

        except Exception as e:
            errors.append({"input_index": i, "error": str(e)[:200]})
            if verbose and len(errors) <= 5:
                print(f"  [error] row {i}: {str(e)[:100]}")

    # Write results (append if resume, else overwrite)
    new_df = pd.DataFrame(results)
    if existing_df is not None and len(existing_df) > 0 and resume:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["input_index"], keep="last")
        combined = combined.sort_values("input_index").reset_index(drop=True)
        combined.to_csv(output_csv, index=False)
    else:
        new_df.to_csv(output_csv, index=False)

    elapsed = time.time() - t0
    stats = {
        "total": total,
        "processed": len(results),
        "errors": len(errors),
        "elapsed_sec": round(elapsed, 1),
        "rate_per_sec": round(len(results) / elapsed, 2) if elapsed > 0 else 0,
        "label_counts": label_counts,
        "decision_source_counts": decision_source_counts,
        "n_agreement": n_agreement,
        "n_review": n_review,
    }

    print(f"\n[pipeline] DONE in {elapsed:.1f}s")
    print(f"  Processed: {len(results)} rows ({stats['rate_per_sec']} rows/sec)")
    print(f"  Errors: {len(errors)}")
    print(f"\n  Label distribution:")
    for label, n in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {label}: {n} ({100*n/len(results):.1f}%)")
    print(f"\n  Decision sources:")
    for src, n in sorted(decision_source_counts.items(), key=lambda x: -x[1]):
        print(f"    {src}: {n} ({100*n/len(results):.1f}%)")
    print(f"\n  Fusion agreement: {n_agreement}/{len(results)} ({100*n_agreement/len(results):.1f}%)")
    print(f"  Review flagged:   {n_review}/{len(results)} ({100*n_review/len(results):.1f}%)")
    print(f"\n  Output: {output_csv}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="THERMOSCOPE-AI batch pipeline: classify hotspots end-to-end"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV, help="Input CSV path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV, help="Output CSV path")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--limit", type=int, default=None, help="Max hotspots to process")
    parser.add_argument("--no-resume", action="store_true", help="Reprocess all rows")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    args = parser.parse_args()

    print("=" * 65)
    print("THERMOSCOPE-AI — Phase E Batch Pipeline")
    print("=" * 65)
    print()

    stats = process_hotspots(
        input_csv=args.input,
        output_csv=args.output,
        start=args.start,
        limit=args.limit,
        resume=not args.no_resume,
        verbose=not args.quiet,
    )

    print()
    print("=" * 65)
    print("Phase E pipeline complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
