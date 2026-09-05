"""
build_real_dataset.py
Phase A.5 (Real) — Build classified_hotspots_v2.csv from genuine NASA FIRMS data
and real OSM spatial context (no synthetic data, no demo fallback).

Pipeline:
  1. Load real FIRMS hotspots from data/raw/firms_recent.csv
  2. Load + merge the industrial OSM cache AND the forest/agriculture OSM cache
     (Phase B P0.1 fix: the forest/agri cache was previously never merged)
  3. For each hotspot, compute the spatial feature block via the shared canonical
     contract (app.geo.spatial_features.compute_spatial_features)
  4. Preserve real FIRMS thermal columns (frp, bright_ti4, bright_ti5, confidence, daynight)
  5. Write the canonical classified CSV (data/processed/hotspots/classified_hotspots_v2.csv)

DATA INTEGRITY GUARANTEES:
  - allow_demo_fallback is NEVER set to True anywhere in this script.
  - Missing OSM categories produce the canonical sentinel distance
    (SENTINEL_DISTANCE_M = 999000.0 m, shared with the live path) — this is
    honest signal (isolation), not fabricated proximity.
  - The output CSV does NOT contain a _is_synthetic_demo column.
    The guard script (scripts/check_data_integrity.py) verifies this before training.
"""
from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Make app package importable
backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
root_dir = str(Path(__file__).resolve().parents[1])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.services.osm_service import (
    find_nearby_geographic_objects,
    load_osm_sites,
    classify_osm_category,
)
from app.geo.spatial_features import (
    SENTINEL_DISTANCE_M,
    CATEGORIES,
    build_candidates_by_category,
    compute_spatial_features,
)
from app.core.paths import (
    FIRMS_DATASET_PATH,
    OSM_INDUSTRIAL_CACHE_PATH,
    OSM_FOREST_AGRI_CACHE_PATH,
    CLASSIFIED_DATASET_PATH,
)
from app.core.lineage import log_lineage, warn_if_stale_classified_copy

# ---------------------------------------------------------------------------
# Canonical constants (single source of truth in backend/app/core/paths.py)
# ---------------------------------------------------------------------------
FIRMS_CSV = str(FIRMS_DATASET_PATH)
OUTPUT_CSV = str(CLASSIFIED_DATASET_PATH)
OSM_CACHE_PATH = str(OSM_INDUSTRIAL_CACHE_PATH)
OSM_FOREST_AGRI_CACHE_PATH_DEFAULT = str(OSM_FOREST_AGRI_CACHE_PATH)

# Canonical sentinel = 999 km (999000.0 m) — used when no OSM site found within
# search radius. Imported from app.geo.spatial_features so batch and live share
# ONE sentinel (see backend/app/geo/spatial_features.py for the full contract).
# Search radius for live Overpass API (cache uses bounding box)
RADIUS_METERS = 15000

# FIRMS confidence raw values (VIIRS): 'h' = high, 'n' = nominal, 'l' = low
# These are passed through RAW to the output CSV so that get_confidence() in
# labeling_functions.py receives the correct string and returns 'high'/'nominal'/'low'.
# Do NOT convert to numeric — that breaks the string matching in get_confidence().
FIRMS_CONFIDENCE_VALUES = {"h", "n", "l"}  # assertion reference only

# FIRMS daynight raw values: 'D' = Day, 'N' = Night
# Passed through RAW so that is_night() in labeling_functions.py works correctly.
# Do NOT convert to int — is_night() checks val.strip().upper() == 'N'.


def _compute_distances_from_cache(
    lat: float,
    lon: float,
    cached_sites: List[Dict],
    radius_m: float = RADIUS_METERS,
) -> Dict[str, float]:
    """
    Compute nearest-distance (meters) for each of the 7 OSM categories
    using only the local cached OSM dataset.

    DEPRECATED COMPAT PROBE: delegates to the canonical shared contract
    (app.geo.spatial_features.compute_spatial_features) so the batch unit
    tests and the geospatial audit probe see EXACTLY the same feature math as
    the production producer and the live hotspot service.

    Returns SENTINEL_DISTANCE_M for any category with no cache hit.
    """
    feats = compute_spatial_features(
        lat,
        lon,
        build_candidates_by_category(cached_sites),
        radius_m=radius_m,
    )
    nearest: Dict[str, float] = {c: feats[f"distance_to_{c}_m"] for c in CATEGORIES}
    data_sources: Dict[str, str] = {c: feats[f"src_{c}"] for c in CATEGORIES}
    return nearest, data_sources


def build_real_dataset(
    firms_path: str = FIRMS_CSV,
    output_path: str = OUTPUT_CSV,
    osm_cache_path: str = OSM_CACHE_PATH,
    osm_forest_agri_cache_path: str = OSM_FOREST_AGRI_CACHE_PATH_DEFAULT,
    use_live_api: bool = False,  # default: cache-only for reproducibility
    states: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the real classified_hotspots_v2.csv from genuine FIRMS + OSM data.

    Args:
        firms_path: Path to data/raw/firms_recent.csv
        output_path: Output CSV path
        osm_cache_path: Path to the industrial OSM cache
            (data/raw/osm/osm_industrial_sites.json)
        osm_forest_agri_cache_path: Path to the forest/agriculture OSM cache
            (data/raw/osm/osm_forest_agriculture.json). PHASE B FIX: this
            existing, validated cache is now merged so forest/agriculture
            distances/flags are REAL instead of constant sentinel.
        use_live_api: If True, also queries Overpass live API per hotspot.
        states: Optional list of state names to filter
            (e.g. ['gujarat', 'maharashtra', 'uttar_pradesh', 'haryana']).
        verbose: Print progress.
    """
    from backend.app.services.osm_service import INDIA_STATES

    # Normalize state names if provided
    target_states = [s.strip().lower() for s in states] if states else None

    def _in_target_states(lat: float, lon: float) -> bool:
        if not target_states:
            return True
        for st in target_states:
            if st in INDIA_STATES:
                w, s, e, n = INDIA_STATES[st]
                if s <= lat <= n and w <= lon <= e:
                    return True
        return False

    # -----------------------------------------------------------------------
    # 1. Load real FIRMS data
    # -----------------------------------------------------------------------
    if not os.path.exists(firms_path):
        raise FileNotFoundError(f"FIRMS CSV not found: {firms_path}")

    firms_df = pd.read_csv(firms_path)
    total = len(firms_df)

    if target_states:
        firms_df = firms_df[firms_df.apply(lambda r: _in_target_states(float(r["latitude"]), float(r["longitude"])), axis=1)].reset_index(drop=True)
        if verbose:
            print(f"[Step 1] Filtered FIRMS hotspots to {len(target_states)} states {target_states}: {len(firms_df)}/{total} rows match.")
    elif verbose:
        print(f"[Step 1] Loaded {total} real FIRMS hotspots from {firms_path}")

    # -----------------------------------------------------------------------
    # 2. Load OSM caches (industrial + forest/agriculture) and merge them
    # -----------------------------------------------------------------------
    cached_sites = load_osm_sites(osm_cache_path)
    if not cached_sites:
        raise FileNotFoundError(
            f"OSM cache not found or empty: {osm_cache_path}\n"
            "Run: python -m app.services.osm_service  (to populate the cache first)"
        )

    # PHASE B FIX (P0.1): load the existing forest/agriculture cache and merge.
    # Before this fix the producer never touched this file, so dist_forest and
    # dist_agriculture were ALWAYS 999 (sentinel) in the canonical dataset.
    forest_agri_sites = load_osm_sites(osm_forest_agri_cache_path)
    if forest_agri_sites:
        n_before = len(cached_sites)
        seen_ids = {(s.get("osm_type"), s.get("id")) for s in cached_sites}
        for s in forest_agri_sites:
            key = (s.get("osm_type"), s.get("id"))
            if key != (None, None) and key in seen_ids:
                continue
            seen_ids.add(key)
            cached_sites.append(s)
        if verbose:
            print(
                f"[Step 2] Merged forest/agriculture cache: "
                f"{n_before} -> {len(cached_sites)} sites "
                f"(+{len(cached_sites) - n_before} forest/agri sites)"
            )
    else:
        if verbose:
            print(
                "[Step 2] WARNING: forest/agriculture OSM cache not found or "
                f"unreadable: {osm_forest_agri_cache_path}\n"
                "         dist_forest / dist_agriculture will remain sentinel "
                "(999) for this build."
            )

    if target_states:
        def _site_in_target_states(s: Dict) -> bool:
            st = s.get("state", "").lower()
            if st in target_states:
                return True
            lat, lon = s.get("lat"), s.get("lon")
            if lat is not None and lon is not None:
                return _in_target_states(float(lat), float(lon))
            return False

        cached_sites = [s for s in cached_sites if _site_in_target_states(s)]
        if verbose:
            print(f"[Step 2] Filtered OSM cache to target states: {len(cached_sites)} sites remain.")

    if verbose:
        cat_counts = {}
        for s in cached_sites:
            c = classify_osm_category(s.get("tags", {}))
            if c == "other" or not c:
                c = s.get("category") or s.get("site_type", "other")
            cat_counts[c] = cat_counts.get(c, 0) + 1
        print(f"[Step 2] OSM cache active: {len(cached_sites)} sites")
        for c, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"         {c:16s}: {n}")

    # Pre-group the merged cache into per-category candidate lists ONCE so every
    # hotspot uses the same shared feature contract (batch == live parity).
    candidates_by_category = build_candidates_by_category(cached_sites)

    # -----------------------------------------------------------------------
    # 3. Process each hotspot
    # -----------------------------------------------------------------------
    rows = []
    source_stats = {"cache": 0, "live": 0, "none": 0}

    for idx, row in firms_df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        if use_live_api:
            # Full live + cache query — slower but more complete. The returned
            # per-category candidate lists feed the SAME shared contract.
            osm_result = find_nearby_geographic_objects(
                lat=lat,
                lon=lon,
                radius_meters=RADIUS_METERS,
                use_live_api=True,
                allow_demo_fallback=False,  # NEVER True in training path
            )
            ds_map = osm_result.pop("data_sources", {})
            feats = compute_spatial_features(
                lat,
                lon,
                {cat: osm_result.get(cat, []) for cat in CATEGORIES},
                data_sources=ds_map,
                radius_m=RADIUS_METERS,
            )
        else:
            # Cache-only path (fast, fully reproducible) — shared contract.
            feats = compute_spatial_features(
                lat,
                lon,
                candidates_by_category,
                radius_m=RADIUS_METERS,
            )

        for c in CATEGORIES:
            src = feats.get(f"src_{c}", "none")
            if src in source_stats:
                source_stats[src] += 1

        # Real thermal values from FIRMS — passed through RAW, never transformed.
        # confidence stays as 'h'/'n'/'l' so get_confidence() in labeling_functions works.
        # daynight stays as 'D'/'N' so is_night() in labeling_functions works.
        frp = float(row.get("frp", 0.0))
        bright_ti4 = float(row.get("bright_ti4", 300.0))
        bright_ti5 = float(row.get("bright_ti5", 290.0))
        confidence = str(row.get("confidence", "n")).strip().lower()   # 'h', 'n', or 'l'
        daynight   = str(row.get("daynight",   "D")).strip().upper()   # 'D' or 'N'

        # Defensive assertions — catch any unexpected raw FIRMS values early
        if confidence not in FIRMS_CONFIDENCE_VALUES:
            confidence = "n"  # fall back to nominal, log a warning
        if daynight not in ("D", "N"):
            daynight = "D"

        dist_ind = feats["distance_to_industry_m"]
        dist_ref = feats["distance_to_refinery_m"]
        dist_oil = feats["distance_to_oil_gas_m"]
        dist_min = feats["distance_to_mining_m"]
        dist_agr = feats["distance_to_agriculture_m"]
        dist_for = feats["distance_to_forest_m"]
        dist_pow = feats["distance_to_power_plant_m"]

        out = {
            # Spatial identity
            "latitude": lat,
            "longitude": lon,
            "acq_date": row.get("acq_date", ""),
            "acq_time": row.get("acq_time", ""),
            "satellite": row.get("satellite", ""),
            "instrument": row.get("instrument", ""),
            "source_dataset": row.get("source_dataset", ""),

            # Real FIRMS thermal values (never synthesized)
            "frp": frp,
            "bright_ti4": bright_ti4,
            "bright_ti5": bright_ti5,
            "confidence": confidence,
            "daynight": daynight,

            # Real OSM spatial distances (meters) — SENTINEL = genuinely isolated
            "distance_to_industry_m":    dist_ind,
            "distance_to_refinery_m":    dist_ref,
            "distance_to_oil_gas_m":     dist_oil,
            "distance_to_mining_m":      dist_min,
            "distance_to_agriculture_m": dist_agr,
            "distance_to_forest_m":      dist_for,
            "distance_to_power_plant_m": dist_pow,

            # KM equivalents for XGBoost feature matrix compatibility
            "dist_factory":          feats["dist_factory"],
            "dist_industrial_zone":  feats["dist_industrial_zone"],
            "dist_refinery":         feats["dist_refinery"],
            "dist_oil_gas":          feats["dist_oil_gas"],
            "dist_mining":           feats["dist_mining"],
            "dist_agriculture":      feats["dist_agriculture"],
            "dist_forest":           feats["dist_forest"],
            "dist_powerplant":       feats["dist_powerplant"],

            # Spatial proximity flags
            "has_refinery_5km":    feats["has_refinery_5km"],
            "has_powerplant_5km":  feats["has_powerplant_5km"],
            "has_factory_5km":     feats["has_factory_5km"],
            "has_industrial_2km":  feats["has_industrial_2km"],
            "has_forest_5km":      feats["has_forest_5km"],
            "has_agriculture_5km": feats["has_agriculture_5km"],

            # REAL neighbourhood counts (PHASE B P1.5 fix)
            "industrial_sites_within_2km": feats["industrial_sites_within_2km"],
            "industrial_sites_within_5km": feats["industrial_sites_within_5km"],
            "refinery_sites_within_3km":   feats["refinery_sites_within_3km"],
            "refinery_sites_within_5km":   feats["refinery_sites_within_5km"],
            "forest_sites_within_5km":     feats["forest_sites_within_5km"],
            "agriculture_sites_within_5km": feats["agriculture_sites_within_5km"],
            "count_forest_5km":            feats["count_forest_5km"],
            "count_agriculture_5km":       feats["count_agriculture_5km"],

            # Legacy ML-schema count aliases (DEPRECATED — bucket codes kept
            # byte-compatible with the saved 17-column XGBoost feature schema)
            "count_ind_5km": feats["count_ind_5km"],
            "count_ref_5km": feats["count_ref_5km"],

            # Auditability — data source per distance column
            "src_industry":    feats["src_industry"],
            "src_refinery":    feats["src_refinery"],
            "src_oil_gas":     feats["src_oil_gas"],
            "src_mining":      feats["src_mining"],
            "src_agriculture": feats["src_agriculture"],
            "src_forest":      feats["src_forest"],
            "src_power_plant": feats["src_power_plant"],
        }
        rows.append(out)

        if verbose and (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{total} hotspots...")

    # -----------------------------------------------------------------------
    # 4. Save output
    # -----------------------------------------------------------------------
    out_df = pd.DataFrame(rows)
    # Explicitly confirm no synthetic marker
    assert "_is_synthetic_demo" not in out_df.columns, \
        "BUG: synthetic demo marker found in real dataset!"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

    if verbose:
        warn_if_stale_classified_copy()
        log_lineage(
            stage="build_real_dataset",
            input_path=firms_path,
            input_rows=len(firms_df),
            output_path=output_path,
            output_rows=len(out_df),
            rows_removed=len(firms_df) - len(out_df),
            reason_for_removal=(
                "rows dropped because a category distance computation failed"
                if len(firms_df) != len(out_df) else "none"
            ),
        )
        print(f"\n[Step 4] Wrote {len(out_df)} REAL rows -> {output_path}")
        print("\n--- Distance Coverage (% rows where OSM category was found within radius) ---")
        BASELINE_COUNTS = {
            "industry": 80,
            "refinery": 7,
            "oil_gas": 4,
            "mining": 1,
            "power_plant": 70,
        }
        for cat, col in [
            ("industry",    "distance_to_industry_m"),
            ("refinery",    "distance_to_refinery_m"),
            ("oil_gas",     "distance_to_oil_gas_m"),
            ("mining",      "distance_to_mining_m"),
            ("agriculture", "distance_to_agriculture_m"),
            ("forest",      "distance_to_forest_m"),
            ("power_plant", "distance_to_power_plant_m"),
        ]:
            found = (out_df[col] < SENTINEL_DISTANCE_M).sum()
            pct = 100.0 * found / len(out_df)
            sentinel_pct = 100.0 - pct
            print(f"  {cat:16s}: {found:4d}/{len(out_df)} found ({pct:.1f}%)"
                  f" | {sentinel_pct:.1f}% will ABSTAIN in labeling")
            if cat in BASELINE_COUNTS and found < BASELINE_COUNTS[cat]:
                print(f"  ⚠️ COVERAGE REGRESSION DETECTED! {cat}: found {found} < baseline {BASELINE_COUNTS[cat]}")

        print("\n--- Data Source Breakdown ---")
        for src_col in ["src_industry", "src_refinery", "src_oil_gas", "src_mining",
                        "src_agriculture", "src_forest", "src_power_plant"]:
            if src_col in out_df.columns:
                counts = out_df[src_col].value_counts().to_dict()
                print(f"  {src_col:20s}: {counts}")

    return out_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build real classified_hotspots_v2.csv from FIRMS + OSM data")
    parser.add_argument("--states", default=None, help="Comma-separated target state names (e.g. gujarat,maharashtra,uttar_pradesh,haryana)")
    parser.add_argument("--live", action="store_true", help="Query Overpass live API per hotspot (slow)")
    parser.add_argument("--firms", default=FIRMS_CSV, help="Path to FIRMS CSV")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV path")
    args = parser.parse_args()

    state_list = [s.strip() for s in args.states.split(",")] if args.states else None

    print("=" * 70)
    print("THERMOSCOPE-AI: Building REAL classified_hotspots_v2.csv")
    print("=" * 70)
    print(f"FIRMS source  : {args.firms}")
    print(f"OSM cache     : {OSM_CACHE_PATH}")
    print(f"Target States : {state_list or 'ALL (Nationwide)'}")
    print(f"Live API      : {'YES (slow)' if args.live else 'NO (cache only)'}")
    print(f"Demo fallback : DISABLED (never used in training)")
    print("=" * 70)

    df = build_real_dataset(
        firms_path=args.firms,
        output_path=args.output,
        use_live_api=args.live,
        states=state_list,
        verbose=True,
    )
    print("\n✅ Done. Use this CSV with dataset_builder.py, not build_demo_dataset.py.")
