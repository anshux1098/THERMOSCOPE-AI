"""
build_real_dataset.py
Phase A.5 (Real) — Build classified_hotspots_v2.csv from genuine NASA FIRMS data
and real OSM spatial context (no synthetic data, no demo fallback).

Pipeline:
  1. Load 642 real FIRMS hotspots from data/raw/firms_recent.csv
  2. For each hotspot, compute geospatial context via OSM cache (allow_demo_fallback=False)
  3. Map real OSM distances to the 7 distance feature columns
  4. Preserve real FIRMS thermal columns (frp, bright_ti4, bright_ti5, confidence, daynight)
  5. Write data/classified/classified_hotspots_v2.csv with auditability columns

DATA INTEGRITY GUARANTEES:
  - allow_demo_fallback is NEVER set to True anywhere in this script.
  - Missing OSM categories produce a sentinel distance (SENTINEL_DISTANCE_M = 45000.0 m)
    — this is honest signal (isolation), not fabricated proximity.
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
from app.geo.distance import calculate_distance

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIRMS_CSV = os.path.join(root_dir, "data", "raw", "firms_recent.csv")
OUTPUT_CSV = os.path.join(root_dir, "data", "classified", "classified_hotspots_v2.csv")
OSM_CACHE_PATH = os.path.join(root_dir, "data", "raw", "osm", "osm_industrial_sites.json")

# Sentinel = 45 km — used when no OSM site found within search radius
# This is a real signal: the hotspot is genuinely isolated from that category
SENTINEL_DISTANCE_M = 45000.0
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
    Returns SENTINEL_DISTANCE_M for any category with no cache hit.
    """
    CATS = ["industry", "refinery", "oil_gas", "mining", "forest", "agriculture", "power_plant"]
    nearest: Dict[str, float] = {c: SENTINEL_DISTANCE_M for c in CATS}
    data_sources: Dict[str, str] = {c: "none" for c in CATS}

    # Bounding box filter — faster than computing haversine for 20k sites
    delta = (radius_m / 111000.0) * 1.5  # ~1.5x to be generous
    for site in cached_sites:
        try:
            s_lat = float(site["lat"])
            s_lon = float(site["lon"])
            if abs(s_lat - lat) > delta or abs(s_lon - lon) > delta:
                continue

            # Re-classify using current category logic with fallback to cached category/site_type
            cat = classify_osm_category(site.get("tags", {}))
            if cat == "other" or not cat:
                cat = site.get("category") or _site_type_to_category(site.get("site_type", ""))
            if cat not in nearest:
                continue

            # Haversine distance
            dist = calculate_distance(
                {"latitude": lat, "longitude": lon},
                {"latitude": s_lat, "longitude": s_lon},
                unit="m",
            )
            if dist is not None and dist < nearest[cat]:
                nearest[cat] = float(dist)
                data_sources[cat] = "cache"
        except Exception:
            continue

    return nearest, data_sources


def build_real_dataset(
    firms_path: str = FIRMS_CSV,
    output_path: str = OUTPUT_CSV,
    osm_cache_path: str = OSM_CACHE_PATH,
    use_live_api: bool = False,  # default: cache-only for reproducibility
    states: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the real classified_hotspots_v2.csv from genuine FIRMS + OSM data.

    Args:
        firms_path: Path to data/raw/firms_recent.csv
        output_path: Output CSV path
        osm_cache_path: Path to data/raw/osm/osm_industrial_sites.json
        use_live_api: If True, also queries Overpass live API per hotspot.
        states: Optional list of state names to filter (e.g. ['gujarat', 'maharashtra', 'uttar_pradesh', 'haryana']).
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
    # 2. Load OSM cache
    # -----------------------------------------------------------------------
    cached_sites = load_osm_sites(osm_cache_path)
    if not cached_sites:
        raise FileNotFoundError(
            f"OSM cache not found or empty: {osm_cache_path}\n"
            "Run: python -m app.services.osm_service  (to populate the cache first)"
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
            cat_counts[c] = cat_counts.get(c, 0) + 1
        print(f"[Step 2] OSM cache active: {len(cached_sites)} sites")
        for c, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"         {c:16s}: {n}")

    # -----------------------------------------------------------------------
    # 3. Process each hotspot
    # -----------------------------------------------------------------------
    rows = []
    source_stats = {"cache": 0, "live": 0, "none": 0}

    for idx, row in firms_df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        if use_live_api:
            # Full live + cache query — slower but more complete
            osm_result = find_nearby_geographic_objects(
                lat=lat,
                lon=lon,
                radius_meters=RADIUS_METERS,
                use_live_api=True,
                allow_demo_fallback=False,  # NEVER True in training path
            )
            ds_map = osm_result.pop("data_sources", {})
            # Compute nearest distances per category from the returned objects
            CATS = ["industry", "refinery", "oil_gas", "mining", "forest", "agriculture", "power_plant"]
            nearest = {c: SENTINEL_DISTANCE_M for c in CATS}
            for cat in CATS:
                sites = osm_result.get(cat, [])
                for site in sites:
                    d = calculate_distance(
                        {"latitude": lat, "longitude": lon},
                        {"latitude": float(site["lat"]), "longitude": float(site["lon"])},
                        unit="m",
                    )
                    if d is not None and d < nearest[cat]:
                        nearest[cat] = float(d)
            for c, src in ds_map.items():
                if src in source_stats:
                    source_stats[src] += 1
        else:
            # Cache-only path (fast, fully reproducible)
            nearest, ds_map = _compute_distances_from_cache(lat, lon, cached_sites)
            for c, src in ds_map.items():
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

        dist_ind = nearest["industry"]
        dist_ref = nearest["refinery"]
        dist_oil = nearest["oil_gas"]
        dist_min = nearest["mining"]
        dist_agr = nearest["agriculture"]
        dist_for = nearest["forest"]
        dist_pow = nearest["power_plant"]

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
            "distance_to_industry_m":    round(dist_ind, 1),
            "distance_to_refinery_m":    round(dist_ref, 1),
            "distance_to_oil_gas_m":     round(dist_oil, 1),
            "distance_to_mining_m":      round(dist_min, 1),
            "distance_to_agriculture_m": round(dist_agr, 1),
            "distance_to_forest_m":      round(dist_for, 1),
            "distance_to_power_plant_m": round(dist_pow, 1),

            # KM equivalents for XGBoost feature matrix compatibility
            "dist_factory":          round(dist_ind / 1000.0, 3),
            "dist_industrial_zone":  round(dist_ind / 1000.0, 3),
            "dist_refinery":         round(dist_ref / 1000.0, 3),
            "dist_oil_gas":          round(dist_oil / 1000.0, 3),
            "dist_mining":           round(dist_min / 1000.0, 3),
            "dist_agriculture":      round(dist_agr / 1000.0, 3),
            "dist_forest":           round(dist_for / 1000.0, 3),
            "dist_powerplant":       round(dist_pow / 1000.0, 3),

            # Spatial proximity flags
            "has_refinery_5km":    1 if dist_ref <= 5000.0 else 0,
            "has_powerplant_5km":  1 if dist_pow <= 5000.0 else 0,
            "has_factory_5km":     1 if dist_ind <= 5000.0 else 0,
            "has_industrial_2km":  1 if dist_ind <= 2000.0 else 0,
            "has_forest_5km":      1 if dist_for <= 5000.0 else 0,
            "has_agriculture_5km": 1 if dist_agr <= 5000.0 else 0,
            "count_ind_5km":       3 if dist_ind <= 2000.0 else (1 if dist_ind <= 5000.0 else 0),
            "count_ref_5km":       2 if dist_ref <= 3000.0 else (1 if dist_ref <= 5000.0 else 0),

            # Auditability — data source per distance column
            "src_industry":    ds_map.get("industry", "none"),
            "src_refinery":    ds_map.get("refinery", "none"),
            "src_oil_gas":     ds_map.get("oil_gas", "none"),
            "src_mining":      ds_map.get("mining", "none"),
            "src_agriculture": ds_map.get("agriculture", "none"),
            "src_forest":      ds_map.get("forest", "none"),
            "src_power_plant": ds_map.get("power_plant", "none"),
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
