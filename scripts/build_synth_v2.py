"""
build_synth_v2.py
Phase A.5 — Build classified_hotspots_v2.csv using the EXISTING demo fallback
in osm_service.py (no Overpass calls — Overpass is rate-limiting us).

Strategy:
  - Take ALL 642 real FIRMS hotspots (full India, not just North)
  - For each hotspot, use osm_service._get_demo_fallback_features() to
    generate synthetic but realistic nearby sites for all 7 categories
    (industry, refinery, oil_gas, mining, forest, agriculture, power_plant)
  - For each category, find the NEAREST site (Haversine distance in km)
  - Skip volcano (team decision: no data, not needed)
  - Write classified_hotspots_v2.csv with all distance columns

The fallback data is documented in osm_service._get_demo_fallback_features
and represents typical "near an industrial fire" / "near a forest" /
"near agriculture" arrangements. It's synthetic but spatially-realistic
for demo purposes.

For SIH demo this is acceptable. For production we'd replace with real
Overpass queries — those are rate-limited from shared IPs.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

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
FIRMS_CSV = "data/raw/firms_recent.csv"
OUTPUT_CSV = "data/processed/hotspots/classified_hotspots_v2.csv"
MISSING_KM = 999.0


def find_nearest_km(hot_lat: float, hot_lon: float, sites: List[Dict[str, Any]]) -> float:
    best = MISSING_KM
    for s in sites:
        d = haversine_distance(hot_lat, hot_lon, s["lat"], s["lon"]) / 1000.0
        if d < best:
            best = d
    return best


def has_within_km(hot_lat: float, hot_lon: float, sites: List[Dict[str, Any]], radius_km: float) -> int:
    return int(any(
        haversine_distance(hot_lat, hot_lon, s["lat"], s["lon"]) / 1000.0 <= radius_km
        for s in sites
    ))


def count_within_km(hot_lat: float, hot_lon: float, sites: List[Dict[str, Any]], radius_km: float) -> int:
    return sum(
        1 for s in sites
        if haversine_distance(hot_lat, hot_lon, s["lat"], s["lon"]) / 1000.0 <= radius_km
    )


def build():
    if not os.path.exists(FIRMS_CSV):
        raise FileNotFoundError(f"FIRMS CSV not found: {FIRMS_CSV}")

    print(f"[build_synth] Loading {FIRMS_CSV}...")
    firms = pd.read_csv(FIRMS_CSV)
    print(f"  -> {len(firms)} FIRMS hotspots total (all India)")

    # Use ALL hotspots, not filtered
    north = firms.copy()
    print(f"  -> Using all {len(north)} hotspots for v2 CSV")

    rows = []
    for idx, row in north.iterrows():
        try:
            hot_lat = float(row["latitude"])
            hot_lon = float(row["longitude"])
        except (KeyError, TypeError, ValueError):
            continue

        # Use the EXISTING demo fallback to get synthetic but realistic nearby sites
        # for all 7 categories (industry/refinery/oil_gas/mining/forest/agriculture/power_plant)
        features = _get_demo_fallback_features(hot_lat, hot_lon)

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

        # 6 categories (skip volcano)
        for cat in ["refinery", "oil_gas", "mining", "industry", "power_plant", "forest", "agriculture"]:
            sites = features.get(cat, [])
            d = find_nearest_km(hot_lat, hot_lon, sites)
            out[f"dist_{cat}"] = round(d, 3) if d < MISSING_KM else MISSING_KM

        # Specific column names the dataset_builder / LFs expect
        out["dist_refinery_m"] = out["dist_refinery"]
        out["dist_factory"] = out["dist_industry"]  # alias used by LFs
        out["dist_industrial_zone"] = out["dist_industry"]
        out["dist_oil_gas_m"] = out["dist_oil_gas"]
        out["dist_mining_m"] = out["dist_mining"]
        out["dist_forest_m"] = out["dist_forest"]
        out["dist_agriculture_m"] = out["dist_agriculture"]
        out["dist_powerplant"] = out["dist_power_plant"]

        # Flag + count features
        out["has_refinery_5km"] = has_within_km(hot_lat, hot_lon, features.get("refinery", []), 5.0)
        out["has_powerplant_5km"] = has_within_km(hot_lat, hot_lon, features.get("power_plant", []), 5.0)
        out["has_factory_5km"] = has_within_km(hot_lat, hot_lon, features.get("industry", []), 5.0)
        out["has_forest_5km"] = has_within_km(hot_lat, hot_lon, features.get("forest", []), 5.0)
        out["has_agriculture_5km"] = has_within_km(hot_lat, hot_lon, features.get("agriculture", []), 5.0)
        out["has_industrial_2km"] = has_within_km(hot_lat, hot_lon, features.get("industry", []), 2.0)
        out["count_ind_5km"] = count_within_km(hot_lat, hot_lon, features.get("industry", []), 5.0)
        out["count_ref_5km"] = count_within_km(hot_lat, hot_lon, features.get("refinery", []), 5.0)
        out["count_forest_5km"] = count_within_km(hot_lat, hot_lon, features.get("forest", []), 5.0)
        out["count_agriculture_5km"] = count_within_km(hot_lat, hot_lon, features.get("agriculture", []), 5.0)

        rows.append(out)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[build_synth] Wrote {len(out_df)} rows -> {OUTPUT_CSV}")
    print(f"\n[build_synth] Distance column coverage (non-999 counts):")
    for c in [c for c in out_df.columns if c.startswith("dist_") and c.endswith("_m") or c in ("dist_factory", "dist_industrial_zone", "dist_powerplant")]:
        if c in out_df.columns:
            nu = (out_df[c] < MISSING_KM).sum()
            print(f"  {c}: {nu}/{len(out_df)} real values ({100 * nu / len(out_df):.1f}%)")
    return out_df


if __name__ == "__main__":
    build()
