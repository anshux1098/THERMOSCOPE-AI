"""
build_synth_v2.py
Phase A.5 — Build classified_hotspots_v2.csv with all 7 distance categories.

Data sources:
  - Industrial categories (industry, refinery, oil_gas, mining, power_plant):
    from data/raw/osm/osm_industrial_sites.json (existing 20k cache)
  - Forest + agriculture: from data/raw/osm/osm_forest_agriculture.json
    (built incrementally by scripts/fetch_osm_daily.py) — REAL OSM data
    if you've run the daily fetcher. Otherwise falls back to synthetic
    points from osm_service._get_demo_fallback_features().

Pipeline:
  1. Load 642 FIRMS hotspots from data/raw/firms_recent.csv
  2. Load industrial cache (or use synthetic fallback)
  3. Load real OSM forest/agri cache (or use synthetic fallback)
  4. For each hotspot, find NEAREST site per category (Haversine distance)
  5. Write classified_hotspots_v2.csv with all 8 distance columns

The script is fast (~5 seconds for 642 hotspots) because distances are
computed locally — no Overpass calls at this stage.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Make app package importable
backend_dir = str(Path(__file__).resolve().parents[1])
root_dir = str(Path(__file__).resolve().parents[0])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd

from app.geo.distance import haversine_distance
from app.services.osm_service import _get_demo_fallback_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIRMS_CSV = "data/raw/firms_recent_india.csv"
INDUSTRIAL_CACHE = "data/raw/osm/osm_industrial_sites.json"
FOREST_AGRI_CACHE = "data/raw/osm/osm_forest_agriculture.json"
OUTPUT_CSV = "data/processed/hotspots/classified_hotspots_v2.csv"
MISSING_KM = 999.0

# Map our category names to the industrial cache's site_type
INDUSTRIAL_SITE_TYPE_MAP = {
    "refinery": "refinery",
    "oil_gas": "oil_gas",
    "mining": "mining",
    "industry": ["factory", "industrial_zone"],
    "power_plant": "power_plant",
}


# ---------------------------------------------------------------------------
# Cache loaders
# ---------------------------------------------------------------------------
def _load_industrial_sites() -> Dict[str, List[Dict[str, Any]]]:
    """
    Load and classify industrial sites by category.
    Returns: {"industry": [...], "refinery": [...], "oil_gas": [...], "mining": [...], "power_plant": [...]}
    """
    if not os.path.exists(INDUSTRIAL_CACHE):
        print(f"  [WARN] Industrial cache not found: {INDUSTRIAL_CACHE}")
        return {}

    with open(INDUSTRIAL_CACHE) as f:
        raw = json.load(f)

    by_cat: Dict[str, List[Dict[str, Any]]] = {
        "industry": [],
        "refinery": [],
        "oil_gas": [],
        "mining": [],
        "power_plant": [],
    }
    for s in raw:
        site_type = s.get("site_type", "other_industrial")
        if site_type == "volcano":
            continue  # skip volcano
        try:
            lat = float(s.get("lat"))
            lon = float(s.get("lon"))
        except (TypeError, ValueError):
            continue
        entry = {
            "id": s.get("id"),
            "name": s.get("name", ""),
            "lat": lat,
            "lon": lon,
        }
        if site_type in ("refinery",):
            by_cat["refinery"].append(entry)
        elif site_type in ("oil_gas",):
            by_cat["oil_gas"].append(entry)
        elif site_type in ("mining",):
            by_cat["mining"].append(entry)
        elif site_type in ("factory", "industrial_zone", "other_industrial"):
            by_cat["industry"].append(entry)
        elif site_type in ("power_plant", "power_infrastructure"):
            by_cat["power_plant"].append(entry)

    return by_cat


def _load_real_forest_agri() -> Tuple[Dict[str, List[Dict[str, Any]]], bool]:
    """
    Load real OSM forest + agriculture sites from the daily cache.
    Returns: ({"forest": [...], "agriculture": [...]}, was_real_data_loaded)
    """
    if not os.path.exists(FOREST_AGRI_CACHE):
        return {"forest": [], "agriculture": []}, False

    with open(FOREST_AGRI_CACHE) as f:
        raw = json.load(f)

    by_cat: Dict[str, List[Dict[str, Any]]] = {"forest": [], "agriculture": []}
    for s in raw:
        cat = s.get("category")
        if cat not in by_cat:
            continue
        try:
            lat = float(s.get("lat"))
            lon = float(s.get("lon"))
        except (TypeError, ValueError):
            continue
        by_cat[cat].append(
            {
                "id": s.get("id"),
                "name": s.get("name", ""),
                "lat": lat,
                "lon": lon,
                "state": s.get("state", "?"),
            }
        )

    has_data = len(raw) > 0
    return by_cat, has_data


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------
def find_nearest_km(hot_lat: float, hot_lon: float, sites: List[Dict[str, Any]]) -> float:
    if not sites:
        return MISSING_KM
    best = MISSING_KM
    for s in sites:
        d = haversine_distance(hot_lat, hot_lon, s["lat"], s["lon"]) / 1000.0
        if d < best:
            best = d
    return best


def has_within_km(hot_lat: float, hot_lon: float, sites: List[Dict[str, Any]], radius_km: float) -> int:
    if not sites:
        return 0
    return int(any(
        haversine_distance(hot_lat, hot_lon, s["lat"], s["lon"]) / 1000.0 <= radius_km
        for s in sites
    ))


def count_within_km(hot_lat: float, hot_lon: float, sites: List[Dict[str, Any]], radius_km: float) -> int:
    if not sites:
        return 0
    return sum(
        1 for s in sites
        if haversine_distance(hot_lat, hot_lon, s["lat"], s["lon"]) / 1000.0 <= radius_km
    )


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------
def build():
    if not os.path.exists(FIRMS_CSV):
        raise FileNotFoundError(f"FIRMS CSV not found: {FIRMS_CSV}")

    print(f"[build_v2] Loading {FIRMS_CSV}...")
    firms = pd.read_csv(FIRMS_CSV)
    print(f"  -> {len(firms)} FIRMS hotspots (all India)")

    # Load industrial sites
    print(f"[build_v2] Loading industrial cache {INDUSTRIAL_CACHE}...")
    industrial_by_cat = _load_industrial_sites()
    industrial_counts = {k: len(v) for k, v in industrial_by_cat.items()}
    print(f"  -> {industrial_counts}")

    # Load real OSM forest/agri (or use synthetic)
    print(f"[build_v2] Loading real OSM forest/agri cache {FOREST_AGRI_CACHE}...")
    real_forest_agri, has_real = _load_real_forest_agri()
    if has_real:
        print(f"  -> REAL OSM data found: {len(real_forest_agri['forest'])} forest, {len(real_forest_agri['agriculture'])} agriculture")
    else:
        print(f"  -> No real OSM cache. Will use synthetic fallback (run scripts/fetch_osm_daily.py to get real data).")

    rows = []
    synthetic_used = 0
    real_used = 0

    for idx, row in firms.iterrows():
        try:
            hot_lat = float(row["latitude"])
            hot_lon = float(row["longitude"])
        except (KeyError, TypeError, ValueError):
            continue

        out = {
            "latitude": hot_lat,
            "longitude": hot_lon,
            "frp": row.get("frp"),
            "bright_ti4": row.get("bright_ti4"),
            "bright_ti5": row.get("bright_ti5"),
            "daynight": row.get("daynight"),
            "satellite": row.get("satellite"),
            "confidence": row.get("confidence"),
            "acq_date": row.get("acq_date"),
            "acq_time": row.get("acq_time"),
        }

        # INDUSTRIAL categories — always use real industrial cache
        for cat in ["refinery", "oil_gas", "mining", "industry", "power_plant"]:
            sites = industrial_by_cat.get(cat, [])
            d = find_nearest_km(hot_lat, hot_lon, sites)
            out[f"dist_{cat}"] = round(d, 3) if d < MISSING_KM else MISSING_KM

        # FOREST + AGRICULTURE — use real OSM if available, else synthetic
        forest_sites = real_forest_agri.get("forest", [])
        agri_sites = real_forest_agri.get("agriculture", [])

        if not forest_sites or not agri_sites:
            # No real data — use synthetic fallback for this hotspot
            synth = _get_demo_fallback_features(hot_lat, hot_lon)
            forest_sites = forest_sites or synth.get("forest", [])
            agri_sites = agri_sites or synth.get("agriculture", [])
            synthetic_used += 1
        else:
            real_used += 1

        d_forest = find_nearest_km(hot_lat, hot_lon, forest_sites)
        out["dist_forest"] = round(d_forest, 3) if d_forest < MISSING_KM else MISSING_KM

        d_agri = find_nearest_km(hot_lat, hot_lon, agri_sites)
        out["dist_agriculture"] = round(d_agri, 3) if d_agri < MISSING_KM else MISSING_KM

        # Column aliases that dataset_builder / LFs expect
        out["dist_refinery_m"] = out["dist_refinery"]
        out["dist_factory"] = out["dist_industry"]
        out["dist_industrial_zone"] = out["dist_industry"]
        out["dist_oil_gas_m"] = out["dist_oil_gas"]
        out["dist_mining_m"] = out["dist_mining"]
        out["dist_forest_m"] = out["dist_forest"]
        out["dist_agriculture_m"] = out["dist_agriculture"]
        out["dist_powerplant"] = out["dist_power_plant"]

        # Flag + count features
        out["has_refinery_5km"] = has_within_km(hot_lat, hot_lon, industrial_by_cat.get("refinery", []), 5.0)
        out["has_powerplant_5km"] = has_within_km(hot_lat, hot_lon, industrial_by_cat.get("power_plant", []), 5.0)
        out["has_factory_5km"] = has_within_km(hot_lat, hot_lon, industrial_by_cat.get("industry", []), 5.0)
        out["has_forest_5km"] = has_within_km(hot_lat, hot_lon, forest_sites, 5.0)
        out["has_agriculture_5km"] = has_within_km(hot_lat, hot_lon, agri_sites, 5.0)
        out["has_industrial_2km"] = has_within_km(hot_lat, hot_lon, industrial_by_cat.get("industry", []), 2.0)
        out["count_ind_5km"] = count_within_km(hot_lat, hot_lon, industrial_by_cat.get("industry", []), 5.0)
        out["count_ref_5km"] = count_within_km(hot_lat, hot_lon, industrial_by_cat.get("refinery", []), 5.0)
        out["count_forest_5km"] = count_within_km(hot_lat, hot_lon, forest_sites, 5.0)
        out["count_agriculture_5km"] = count_within_km(hot_lat, hot_lon, agri_sites, 5.0)

        rows.append(out)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[build_v2] Wrote {len(out_df)} rows -> {OUTPUT_CSV}")
    print(f"\n[build_v2] Forest/agriculture data source:")
    print(f"  hotspots with REAL OSM data: {real_used}")
    print(f"  hotspots with SYNTHETIC fallback: {synthetic_used}")
    print(f"\n[build_v2] Distance column coverage (non-999 counts):")
    for c in [c for c in out_df.columns if c.startswith("dist_")]:
        if c in out_df.columns:
            nu = (out_df[c] < MISSING_KM).sum()
            print(f"  {c}: {nu}/{len(out_df)} real values ({100 * nu / len(out_df):.1f}%)")
    return out_df


if __name__ == "__main__":
    build()
