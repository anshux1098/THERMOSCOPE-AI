"""
spatial_features.py
Canonical spatial feature-generation contract shared by the BATCH producer
(scripts/build_real_dataset.py) and the LIVE analysis path
(backend/app/geo/spatial_context.py -> backend/app/services/hotspot_service.py).

WHY THIS MODULE EXISTS (Phase B audit finding P1.3 / P1.4 / P1.5):
- Before Phase B the batch and live paths each computed spatial features
  differently (batch: 999000 m sentinel + hard-coded count buckets; live:
  45000 m sentinel + hard-coded count_*_5km=0). That divergence was a proven
  pipeline defect.
- This module is the SINGLE source of truth for:
    1. Sentinel semantics: one canonical "no nearby entity" sentinel for BOTH
       paths. Always numeric (never None / NaN / -1 / 0).
    2. Nearest-distance computation (haversine over bbox-culled candidates).
    3. Real count semantics: industrial_sites_within_2km / ..._5km,
       refinery_sites_within_3km / ..._5km, etc. — actual neighbourhood counts,
       NOT distance bucket codes.
    4. Deprecated ML-schema aliases (count_ind_5km / count_ref_5km) kept
       byte-compatible with the saved 17-column XGBoost feature schema.
- This module NEVER changes search radii, LF thresholds, or FIRMS FRP
  semantics.

SENTINEL SEMANTICS (canonical, documented):
- SENTINEL_DISTANCE_M (999000.0 m)  <=> "no nearby entity of this category was
  found within the search radius". Same meaning in m and km columns.
- SPATIAL_EVIDENCE_INFLUENCE_M (45000.0 m) <=> "distances at or above this are
  treated as negligible spatial influence by the labeling functions". This is a
  rule threshold, DISTINCT from "no entity found". It is kept as a named
  constant so labels never use a bare literal again.
- None is NEVER a valid feature value. None may still appear inside the Pydantic
  DISPLAY schema (SpatialContext) for the UI; the feature layer converts it to
  SENTINEL_DISTANCE_M before LFs / ML run.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure app package is discoverable
backend_dir = str(Path(__file__).resolve().parents[2])
root_dir = str(Path(__file__).resolve().parents[3])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.geo.distance import calculate_distance

# ---------------------------------------------------------------------------
# Canonical sentinel constants (single source of truth)
# ---------------------------------------------------------------------------
SENTINEL_DISTANCE_M: float = 999000.0
SENTINEL_DISTANCE_KM: float = 999.0

# Rule-level "negligible spatial influence" threshold used by the LFs.
# Kept here (not inside labeling_functions) so every consumer shares it.
SPATIAL_EVIDENCE_INFLUENCE_M: float = 45_000.0

# ---------------------------------------------------------------------------
# Canonical category list (matches app.core.constants.OSM_CATEGORIES)
# ---------------------------------------------------------------------------
CATEGORIES: List[str] = [
    "industry",
    "refinery",
    "oil_gas",
    "mining",
    "forest",
    "agriculture",
    "power_plant",
]

SRC_COLUMN_MAP: Dict[str, str] = {c: f"src_{c}" for c in CATEGORIES}


def categorize_site(site: Dict[str, Any]) -> Optional[str]:
    """
    Return the canonical category (one of CATEGORIES) for a raw OSM site dict,
    or None if the site is not one of the 7 categories.

    Resolution order (mirrors the producer's historical behaviour):
      1. classify_osm_category(site['tags'])
      2. fallback to the cached 'category' field   (forest/agri cache has this)
      3. fallback to _site_type_to_category(site['site_type'])
    """
    from app.services.osm_service import (
        classify_osm_category,
        _site_type_to_category,
    )

    tags = site.get("tags") or {}
    cat = classify_osm_category(tags)
    if cat not in CATEGORIES:
        cat = site.get("category")
    if cat not in CATEGORIES:
        cat = _site_type_to_category(str(site.get("site_type", "")))
    return cat if cat in CATEGORIES else None


def build_candidates_by_category(
    sites: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group raw cached OSM sites into per-category candidate lists.

    Also de-duplicates by (osm_type, id), drops sites without valid lat/lon,
    and discards sites that are not one of the canonical 7 categories.
    """
    out: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CATEGORIES}
    seen = set()
    for s in sites or []:
        key = (s.get("osm_type"), s.get("id"))
        if key != (None, None):
            if key in seen:
                continue
            seen.add(key)
        cat = categorize_site(s)
        if cat is None:
            continue
        try:
            s_lat = float(s["lat"])
            s_lon = float(s["lon"])
        except (TypeError, ValueError, KeyError):
            continue
        out[cat].append(
            {
                "lat": s_lat,
                "lon": s_lon,
                "id": s.get("id"),
                "osm_type": s.get("osm_type"),
                "name": s.get("name", ""),
            }
        )
    return out


def compute_spatial_features(
    lat: float,
    lon: float,
    candidates_by_category: Dict[str, List[Dict[str, Any]]],
    data_sources: Optional[Dict[str, str]] = None,
    radius_m: float = 15_000.0,
) -> Dict[str, Any]:
    """
    Compute the full canonical spatial feature block for a hotspot.

    Args:
        lat, lon: Hotspot coordinates.
        candidates_by_category: {category: [site dicts with 'lat'/'lon']}.
            A site may also carry a pre-computed 'distance_meters' key (from
            app.geo.spatial_context) which is reused to avoid recomputation.
        data_sources: Optional {category: "cache"|"live"|"demo"|"none"} audit
            trail. When omitted, it is derived from nearest-hits ("cache").
        radius_m: Search radius (meters). Used ONLY for the coarse bbox cull,
            never to change classification thresholds.

    Returns:
        A flat feature dict with (always numeric, never None/NaN):
          - distance_to_{cat}_m  : nearest distance in meters (SENTINEL if none)
          - dist_* km aliases    : km equivalents (SENTINEL_DISTANCE_KM if none)
          - has_*_5km / has_industrial_2km flags (0/1)
          - REAL counts: industrial_sites_within_2km/5km,
            refinery_sites_within_3km/5km, forest_sites_within_5km,
            agriculture_sites_within_5km
          - count_forest_5km / count_agriculture_5km (same real counts)
          - Legacy ML-schema count aliases: count_ind_5km / count_ref_5km
            (bucket codes, byte-compatible with the saved 17-col model)
          - src_{cat} audit trail ("cache"/"live"/"demo"/"none")
    """
    feats: Dict[str, Any] = {}
    for c in CATEGORIES:
        feats[f"distance_to_{c}_m"] = SENTINEL_DISTANCE_M
        feats[SRC_COLUMN_MAP[c]] = "none"

    count_2km = {"industry": 0}                       # industry <= 2000 m
    count_3km = {"refinery": 0}                       # refinery <= 3000 m
    count_5km = {c: 0 for c in CATEGORIES}            # all categories <= 5000 m

    # Coarse bounding-box gate (~1.5x the search radius) before haversine.
    delta = (radius_m / 111000.0) * 1.5

    for cat in CATEGORIES:
        sites = candidates_by_category.get(cat) or []
        nearest_m = SENTINEL_DISTANCE_M
        for site in sites:
            try:
                s_lat = float(site.get("lat"))
                s_lon = float(site.get("lon"))
            except (TypeError, ValueError):
                continue
            if abs(s_lat - lat) > delta or abs(s_lon - lon) > delta:
                continue

            # Reuse a pre-computed distance if present, else haversine.
            dist_m = site.get("distance_meters")
            try:
                if dist_m is None:
                    raise ValueError
                dist_m = float(dist_m)
            except (TypeError, ValueError):
                dist_m = calculate_distance(
                    {"latitude": lat, "longitude": lon},
                    {"latitude": s_lat, "longitude": s_lon},
                    unit="m",
                )
            if dist_m is None:
                continue

            if dist_m < nearest_m:
                nearest_m = float(dist_m)

            if cat == "industry":
                if dist_m <= 2000.0:
                    count_2km[cat] += 1
                if dist_m <= 5000.0:
                    count_5km[cat] += 1
            elif cat == "refinery":
                if dist_m <= 3000.0:
                    count_3km[cat] += 1
                if dist_m <= 5000.0:
                    count_5km[cat] += 1
            else:
                if dist_m <= 5000.0:
                    count_5km[cat] += 1

        if nearest_m != SENTINEL_DISTANCE_M:
            feats[f"distance_to_{cat}_m"] = round(nearest_m, 1)
            feats[SRC_COLUMN_MAP[cat]] = "cache"

    # Explicit data-source audit trail (live path) overrides the derived value.
    if data_sources:
        for c in CATEGORIES:
            feats[SRC_COLUMN_MAP[c]] = data_sources.get(c, feats[SRC_COLUMN_MAP[c]])

    dist_m = {c: feats[f"distance_to_{c}_m"] for c in CATEGORIES}

    def _km(cat: str) -> float:
        d = dist_m[cat]
        if d == SENTINEL_DISTANCE_M:
            return SENTINEL_DISTANCE_KM
        return round(float(d) / 1000.0, 3)

    # KM aliases (industry has three historical aliases, power_plant two).
    feats.update(
        {
            "dist_industry": _km("industry"),
            "dist_factory": _km("industry"),
            "dist_industrial_zone": _km("industry"),
            "dist_refinery": _km("refinery"),
            "dist_oil_gas": _km("oil_gas"),
            "dist_mining": _km("mining"),
            "dist_forest": _km("forest"),
            "dist_agriculture": _km("agriculture"),
            "dist_powerplant": _km("power_plant"),
            "dist_power_plant": _km("power_plant"),
        }
    )

    # Proximity flags (matching the historic producer's boundary semantics).
    feats["has_refinery_5km"] = 1 if dist_m["refinery"] <= 5000.0 else 0
    feats["has_powerplant_5km"] = 1 if dist_m["power_plant"] <= 5000.0 else 0
    feats["has_factory_5km"] = 1 if dist_m["industry"] <= 5000.0 else 0
    feats["has_industrial_2km"] = 1 if dist_m["industry"] <= 2000.0 else 0
    feats["has_forest_5km"] = 1 if dist_m["forest"] <= 5000.0 else 0
    feats["has_agriculture_5km"] = 1 if dist_m["agriculture"] <= 5000.0 else 0

    # REAL neighbourhood counts (the P1.5 fix).
    feats["industrial_sites_within_2km"] = count_2km["industry"]
    feats["industrial_sites_within_5km"] = count_5km["industry"]
    feats["refinery_sites_within_3km"] = count_3km["refinery"]
    feats["refinery_sites_within_5km"] = count_5km["refinery"]
    feats["forest_sites_within_5km"] = count_5km["forest"]
    feats["agriculture_sites_within_5km"] = count_5km["agriculture"]
    feats["count_forest_5km"] = count_5km["forest"]
    feats["count_agriculture_5km"] = count_5km["agriculture"]

    # LEGACY ML-schema count aliases (DEPRECATED). Bucket codes only, kept so
    # the saved 17-column XGBoost feature schema keeps loading byte-compatible
    # values. New code MUST use the real *_sites_within_* columns above.
    feats["count_ind_5km"] = (
        3 if dist_m["industry"] <= 2000.0
        else (1 if dist_m["industry"] <= 5000.0 else 0)
    )
    feats["count_ref_5km"] = (
        2 if dist_m["refinery"] <= 3000.0
        else (1 if dist_m["refinery"] <= 5000.0 else 0)
    )

    return feats


__all__ = [
    "SENTINEL_DISTANCE_M",
    "SENTINEL_DISTANCE_KM",
    "SPATIAL_EVIDENCE_INFLUENCE_M",
    "CATEGORIES",
    "SRC_COLUMN_MAP",
    "categorize_site",
    "build_candidates_by_category",
    "compute_spatial_features",
]