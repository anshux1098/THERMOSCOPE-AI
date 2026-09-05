"""
geospatial_audit.py
================================================================================
AUDIT INSTRUMENTATION ONLY — forensics for the THERMOSCOPE-AI geospatial pipeline.

THIS SCRIPT:
  - NEVER modifies production data, thresholds, models, or labels.
  - Reads the canonical classified CSV, both OSM caches, FIRMS raw, and LF code.
  - Recomputes TRUE nearest distances from raw caches (independent of the
    production bounding-box gate) so we can measure what a wider discovery
    radius WOULD find.
  - Produces reproducible artifacts under reports/audit/:
      spatial_feature_inventory.csv
      spatial_feature_statistics.csv
      radius_sensitivity_report.csv
      labeling_function_spatial_activation.csv
      unclassified_root_cause_analysis.csv
      *_histogram.png, radius_sensitivity.png, lf_activation.png,
      feature_missingness.png, unclassified_rootcause.png

Usage:
    python scripts/geospatial_audit.py
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------------
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

repo_root = Path(__file__).resolve().parents[1]
backend_dir = str(repo_root / "backend")
root_dir = str(repo_root)
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.core.paths import (
    CLASSIFIED_DATASET_PATH,
    FIRMS_DATASET_PATH,
    OSM_INDUSTRIAL_CACHE_PATH,
    OSM_FOREST_AGRI_CACHE_PATH,
    ENRICHED_DATASET_PATH,
)
from app.services.osm_service import classify_osm_category
from app.intelligence.labeling_functions import (
    ALL_LABELING_FUNCTIONS,
    apply_labeling_functions,
    aggregate_votes,
    THRESHOLD_INDUSTRY_PROXIMITY_M,
    THRESHOLD_REFINERY_PROXIMITY_M,
    THRESHOLD_OIL_GAS_PROXIMITY_M,
    THRESHOLD_MINING_PROXIMITY_M,
    THRESHOLD_AGRICULTURE_PROXIMITY_M,
    THRESHOLD_FOREST_PROXIMITY_M,
)

OUT_DIR = repo_root / "reports" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SENTINEL_M = 999000.0
EARTH_RADIUS_M = 6371000.0

CATEGORIES = [
    "industry", "refinery", "oil_gas", "mining",
    "forest", "agriculture", "power_plant",
]

# Source of each category's raw sites
SOURCE_MAP = {
    "industry": "osm_industrial_sites.json",
    "refinery": "osm_industrial_sites.json",
    "oil_gas": "osm_industrial_sites.json",
    "mining": "osm_industrial_sites.json",
    "power_plant": "osm_industrial_sites.json",
    "forest": "osm_forest_agriculture.json",
    "agriculture": "osm_forest_agriculture.json",
}

# LF evidence thresholds (meters) — production, read-only
LF_THRESHOLDS_M = {
    "industry": THRESHOLD_INDUSTRY_PROXIMITY_M,      # 2000 (docstring says 1000)
    "refinery": THRESHOLD_REFINERY_PROXIMITY_M,      # 2000
    "oil_gas": THRESHOLD_OIL_GAS_PROXIMITY_M,        # 2000
    "mining": THRESHOLD_MINING_PROXIMITY_M,          # 2000
    "agriculture": THRESHOLD_AGRICULTURE_PROXIMITY_M,  # 15000
    "forest": THRESHOLD_FOREST_PROXIMITY_M,            # 15000
    "power_plant": None,  # no LF uses power_plant
}

# has_* / count_* build flags in build_real_dataset.py (meters) — read-only
BUILD_FLAGS_M = {
    "has_refinery_5km": ("refinery", 5000),
    "has_powerplant_5km": ("power_plant", 5000),
    "has_factory_5km": ("industry", 5000),
    "has_industrial_2km": ("industry", 2000),
    "has_forest_5km": ("forest", 5000),
    "has_agriculture_5km": ("agriculture", 5000),
}

# Diagnostic discovery/evidence candidate radii (meters) — never applied to prod
SENSITIVITY_RADII_M = [
    100, 250, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000,
]


def _haversine_matrix(lats: np.ndarray, lons: np.ndarray,
                      s_lats: np.ndarray, s_lons: np.ndarray) -> np.ndarray:
    """Pairwise haversine distance (meters) between each hotspot and each site."""
    lat1 = np.radians(lats)[:, None]
    lat2 = np.radians(s_lats)[None, :]
    dlat = np.radians(s_lats)[None, :] - lat1
    dlon = np.radians(s_lons)[None, :] - np.radians(lons)[:, None]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def load_sites(path: Path) -> List[Dict[str, Any]]:
    raw = json.load(open(path, encoding="utf-8"))
    if isinstance(raw, dict) and "sites" in raw:
        return raw["sites"]
    return raw


def _categorize(site: Dict[str, Any]) -> str:
    cat = classify_osm_category(site.get("tags") or {})
    if cat not in CATEGORIES:
        cat = site.get("category", "other")
    return cat if cat in CATEGORIES else "other"


def recompute_nearest_distances(lats, lons, sites_by_cat) -> pd.DataFrame:
    """Independent nearest-distance recomputation from raw caches (meters).
    Also computes the production bounding-box gate (delta = 15000/111000*1.5)."""
    delta = (15000.0 / 111000.0) * 1.5
    cols: Dict[str, Any] = {}
    cols_bbox: Dict[str, Any] = {}
    for cat in CATEGORIES:
        sites = sites_by_cat.get(cat, [])
        if not sites:
            cols[cat] = np.full(len(lats), SENTINEL_M)
            cols_bbox[cat] = np.full(len(lats), SENTINEL_M)
            continue
        s_lat = np.array([float(s["lat"]) for s in sites])
        s_lon = np.array([float(s["lon"]) for s in sites])
        d = _haversine_matrix(lats, lons, s_lat, s_lon)
        cols[cat] = d.min(axis=1)
        # production gate: only sites within the square bbox count as "found"
        in_bbox = (np.abs(s_lat[None, :] - lats[:, None]) <= delta) & (
            np.abs(s_lon[None, :] - lons[:, None]) <= delta
        )
        d_in = np.where(in_bbox, d, SENTINEL_M)
        cols_bbox[cat] = d_in.min(axis=1)
    return pd.DataFrame(cols), pd.DataFrame(cols_bbox)


def main() -> None:
    print("=" * 78)
    print("GEOSPATIAL AUDIT — THERMOSCOPE-AI (AUDIT INSTRUMENTATION ONLY)")
    print("=" * 78)

    # ------------------------------------------------------------------ raw data
    firms = pd.read_csv(FIRMS_DATASET_PATH)
    classified = pd.read_csv(CLASSIFIED_DATASET_PATH)
    ind_sites = load_sites(OSM_INDUSTRIAL_CACHE_PATH)
    fa_sites = load_sites(OSM_FOREST_AGRI_CACHE_PATH)

    print(f"FIRMS hotspots            : {len(firms)}")
    print(f"Classified rows           : {len(classified)}")
    print(f"Industrial OSM cache      : {len(ind_sites)} sites")
    print(f"Forest/agri OSM cache     : {len(fa_sites)} sites")

    sites_by_cat: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CATEGORIES}
    for site in ind_sites:
        c = _categorize(site)
        if c in sites_by_cat:
            sites_by_cat[c].append(site)
    for site in fa_sites:
        c = _categorize(site)
        if c in sites_by_cat:
            sites_by_cat[c].append(site)
    print("\nRaw-cache category entity counts (what actually exists):")
    for c in CATEGORIES:
        print(f"  {c:14s}: {len(sites_by_cat[c])}")

    lats = classified["latitude"].to_numpy().astype(float)
    lons = classified["longitude"].to_numpy().astype(float)
    true_d, bbox_d = recompute_nearest_distances(lats, lons, sites_by_cat)

    # ------------------------------------------------------------------ stats
    print("\nBuilding spatial feature statistics...")
    stats_rows: List[Dict[str, Any]] = []
    for col in classified.columns:
        s = classified[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        n_missing = int(s.isna().sum())
        vals = s.dropna()
        if vals.empty:
            stats_rows.append({
                "column": col, "dtype": str(s.dtype), "n": 0, "missing": n_missing,
                "zeros": 0, "nunique": 0, "min": None, "max": None,
                "mean": None, "median": None, "std": None,
                "p25": None, "p50": None, "p75": None, "p90": None,
                "p95": None, "p99": None, "constant_value": None,
            })
            continue
        q = vals.quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        stats_rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "n": int(len(vals)),
            "missing": n_missing,
            "zeros": int((vals == 0).sum()),
            "nunique": int(vals.nunique()),
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
            "mean": round(float(vals.mean()), 4),
            "median": round(float(vals.median()), 4),
            "std": round(float(vals.std()), 4),
            "p25": round(float(q[0.25]), 4),
            "p50": round(float(q[0.50]), 4),
            "p75": round(float(q[0.75]), 4),
            "p90": round(float(q[0.90]), 4),
            "p95": round(float(q[0.95]), 4),
            "p99": round(float(q[0.99]), 4),
            "constant_value": float(vals.unique()[0]) if vals.nunique() == 1 else None,
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUT_DIR / "spatial_feature_statistics.csv", index=False)
    print(f"  wrote spatial_feature_statistics.csv ({len(stats_df)} rows)")

    # ------------------------------------------------------------------ inventory
    inventory: List[Dict[str, Any]] = []
    for col, (cat, r) in BUILD_FLAGS_M.items():
        activation = float((classified[col] == 1).mean() * 100)
        found = int((classified[f"distance_to_{cat}_m"] < SENTINEL_M).sum())
        inventory.append({
            "feature": col,
            "entity": cat,
            "source": SOURCE_MAP[cat],
            "spatial_logic": "nearest distance to OSM entity",
            "radius_m": r,
            "unit": "m",
            "defined_in": "scripts/build_real_dataset.py",
            "hardcoded_or_configurable": "hardcoded inline",
            "used_by": "dataset_builder FEATURE_COLUMNS / ML",
            "activation_pct": round(activation, 3),
            "hotspots_found": found,
        })
    for cat in CATEGORIES:
        mcol = f"distance_to_{cat}_m"
        if mcol in classified.columns:
            found = int((classified[mcol] < SENTINEL_M).sum())
            inventory.append({
                "feature": mcol,
                "entity": cat,
                "source": SOURCE_MAP[cat],
                "spatial_logic": "haversine nearest-site distance (m)",
                "radius_m": None,
                "unit": "m",
                "defined_in": "app/geo/distance.py",
                "hardcoded_or_configurable": "N/A (continuous)",
                "used_by": "labeling_functions.get_distance_meters, spatial_context",
                "activation_pct": round(100.0 * found / len(classified), 3),
                "hotspots_found": found,
            })
    for cat in CATEGORIES:
        dcol = f"dist_{cat}"
        if dcol in classified.columns:
            found = int((classified[dcol] < 999.0).sum())
            inventory.append({
                "feature": dcol,
                "entity": cat,
                "source": SOURCE_MAP[cat],
                "spatial_logic": "km alias (m/1000)",
                "radius_m": None,
                "unit": "km",
                "defined_in": "scripts/build_real_dataset.py",
                "hardcoded_or_configurable": "N/A (continuous)",
                "used_by": "dataset_builder DISTANCE_COLUMNS / ML",
                "activation_pct": round(100.0 * found / len(classified), 3),
                "hotspots_found": found,
            })
    inv_df = pd.DataFrame(inventory)
    inv_df.to_csv(OUT_DIR / "spatial_feature_inventory.csv", index=False)
    print(f"  wrote spatial_feature_inventory.csv ({len(inv_df)} rows)")

    # ------------------------------------------------------------------ radius sensitivity (true distances from raw caches)
    print("\nRadius sensitivity analysis (TRUE distances recomputed from raw caches)...")
    sens_rows: List[Dict[str, Any]] = []
    for cat in CATEGORIES:
        d = true_d[cat].to_numpy()
        for r in SENSITIVITY_RADII_M:
            pct = float((d <= r).mean() * 100)
            n = int((d <= r).sum())
            sens_rows.append({
                "category": cat, "radius_m": r, "radius_km": r / 1000.0,
                "n_within": n, "pct_within": round(pct, 3),
            })
    # also report production bbox-gated counts for comparison
    for cat in CATEGORIES:
        db = bbox_d[cat].to_numpy()
        n_found = int((db < SENTINEL_M).sum())
        sens_rows.append({
            "category": cat, "radius_m": "prod_bbox", "radius_km": "prod_bbox",
            "n_within": n_found,
            "pct_within": round(100.0 * n_found / len(classified), 3),
        })
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(OUT_DIR / "radius_sensitivity_report.csv", index=False)
    print(f"  wrote radius_sensitivity_report.csv ({len(sens_df)} rows)")

    # ------------------------------------------------------------------ LF activation on the real classified rows
    print("\nRunning all labeling functions on classified rows (read-only)...")
    lf_fired: Dict[str, int] = collections.Counter()
    lf_abstain: Dict[str, int] = collections.Counter()
    lf_by_label: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    row_votes: List[Dict[str, Optional[str]]] = []
    labels = []
    for idx, row in classified.iterrows():
        votes = apply_labeling_functions(row)
        row_votes.append(votes)
        labels.append(aggregate_votes(votes))
        for lf_name, vote in votes.items():
            if vote is not None:
                lf_fired[lf_name] += 1
                lf_by_label[lf_name][vote] += 1
            else:
                lf_abstain[lf_name] += 1

    lf_rows: List[Dict[str, Any]] = []
    # per-LF spatial dependency map (from source analysis of labeling_functions.py)
    lf_dep: Dict[str, List[str]] = {
        "lf_industry_high_frp": ["industry", "frp"],
        "lf_factory_proximity_thermal": ["industry", "brightness", "confidence"],
        "lf_industrial_zone_cluster": ["industry", "density", "frp"],
        "lf_oil_gas_flare": ["oil_gas", "night", "firms_type", "frp"],
        "lf_refinery_flare": ["refinery", "night", "firms_type", "persistence"],
        "lf_mining_thermal_activity": ["mining", "frp/brightness", "confidence"],
        "lf_mining_high_confidence": ["mining", "confidence", "frp", "industry"],
        "lf_agriculture_vegetation_fire": ["agriculture", "firms_type", "industry"],
        "lf_agriculture_burn_context": ["agriculture", "frp", "brightness", "industry"],
        "lf_forest_vegetation_fire": ["forest", "firms_type", "industry"],
        "lf_strong_forest_fire": ["forest", "frp", "industry", "agriculture"],
        "lf_static_industrial_heat": ["firms_type", "industry", "frp"],
        "lf_nighttime_process_heat": ["night", "frp", "brightness", "industry"],
    }
    for lf in ALL_LABELING_FUNCTIONS:
        name = lf.__name__
        fired = lf_fired.get(name, 0)
        total = len(classified)
        lf_rows.append({
            "labeling_function": name,
            "fired": fired,
            "abstained": total - fired,
            "fire_rate_pct": round(100.0 * fired / total, 3),
            "primary_spatial_dependency": ";".join(lf_dep.get(name, [])),
            "votes_by_label": dict(lf_by_label[name]),
        })
    lf_df = pd.DataFrame(lf_rows)
    lf_df.to_csv(OUT_DIR / "labeling_function_spatial_activation.csv", index=False)
    print(f"  wrote labeling_function_spatial_activation.csv ({len(lf_df)} rows)")

    # ------------------------------------------------------------------ unclassified root cause
    print("\nUnclassified root-cause decomposition...")
    uncl_mask = [lbl == "unclassified" for lbl in labels]
    uncl_idx = [i for i, m in enumerate(uncl_mask) if m]
    print(f"  consensus 'unclassified': {len(uncl_idx)}/{len(labels)}")

    # Stage analysis for unclassified rows
    # Stage1 = distance gates (which LFs had an entity within threshold)
    # We recompute distances used by LFs directly from the classified row columns
    # (the actual features the LFs consumed).
    def _dist_m(row, cat):
        v = row.get(f"distance_to_{cat}_m")
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return SENTINEL_M
        return float(v)

    dist_cols = {c: f"distance_to_{c}_m" for c in CATEGORIES}
    n_spatial_blocked = 0          # no LF had an entity within its LF threshold
    n_evidence_blocked = 0         # entity present but secondary evidence failed

    reachable_by_row: List[int] = []
    firms_type_missing_blocks = 0
    night_required_blocks = 0
    thermal_insufficient = 0
    for i in uncl_idx:
        row = classified.iloc[i]
        reach = 0
        for cat, thr in LF_THRESHOLDS_M.items():
            if thr is None:
                continue
            d = _dist_m(row, cat)
            if d < 45000.0 and d <= thr:
                reach += 1
        reachable_by_row.append(reach)
        if reach == 0:
            n_spatial_blocked += 1
        else:
            n_evidence_blocked += 1
            # sub-cause instrumentation
            night_avail = str(row.get("daynight", "")).strip().upper() == "N"
            firms_type_avail = "firms_type" in row.index and not pd.isna(row.get("firms_type"))
            if not firms_type_avail:
                firms_type_missing_blocks += 1
            if not night_avail:
                night_required_blocks += 1
            frp = row.get("frp")
            frame = 0.0
            try:
                frame = float(frp) if not pd.isna(frp) else 0.0
            except (TypeError, ValueError):
                pass
            if frame < 5.0:
                thermal_insufficient += 1

    # Attribution buckets
    n_all = len(uncl_idx)
    root_rows = [
        {"root_cause": "All LFs blocked at SPATIAL gate (no entity within LF evidence threshold)",
         "count": n_spatial_blocked,
         "pct_of_unclassified": round(100.0 * n_spatial_blocked / max(n_all, 1), 2)},
        {"root_cause": "Entity within LF threshold but SECONDARY evidence failed",
         "count": n_evidence_blocked,
         "pct_of_unclassified": round(100.0 * n_evidence_blocked / max(n_all, 1), 2)},
        {"root_cause": "  - FIRMS `type` (firms_type) column missing from dataset",
         "count": firms_type_missing_blocks,
         "pct_of_unclassified": round(100.0 * firms_type_missing_blocks / max(n_all, 1), 2)},
        {"root_cause": "  - Night evidence unavailable (daynight != 'N')",
         "count": night_required_blocks,
         "pct_of_unclassified": round(100.0 * night_required_blocks / max(n_all, 1), 2)},
        {"root_cause": "  - FRP below 5 MW thermal threshold",
         "count": thermal_insufficient,
         "pct_of_unclassified": round(100.0 * thermal_insufficient / max(n_all, 1), 2)},
    ]
    root_df = pd.DataFrame(root_rows)
    root_df.to_csv(OUT_DIR / "unclassified_root_cause_analysis.csv", index=False)
    print(f"  wrote unclassified_root_cause_analysis.csv ({len(root_df)} rows)")

    # ------------------------------------------------------------------ validation: stored vs recomputed distance
    print("\nValidating stored distances vs recomputed haversine (units / math check)...")
    diffs = []
    for cat in CATEGORIES:
        mcol = f"distance_to_{cat}_m"
        if mcol not in classified.columns:
            continue
        stored = classified[mcol].to_numpy().astype(float)
        recomputed = true_d[cat].to_numpy()
        mask = stored < SENTINEL_M - 1e-6  # stored found rows
        if mask.sum() == 0:
            print(f"  {cat:14s}: NO stored found rows (all sentinel)")
            continue
        diff = np.abs(stored[mask] - recomputed[mask])
        # recompute may be slightly smaller (bbox tolerance) — report max abs diff
        print(f"  {cat:14s}: {'%d' % mask.sum()} found | max|stored-recomputed| = {diff.max():.1f} m | mean = {diff.mean():.1f} m")
    # Also confirm stored *found* matches the production bbox gate
    print("\nConsistency: stored-found vs production-bbox-recomputed found:")
    for cat in CATEGORIES:
        mcol = f"distance_to_{cat}_m"
        if mcol not in classified.columns:
            continue
        stored_found = int((classified[mcol] < SENTINEL_M - 1e-6).sum())
        bbox_found = int((bbox_d[cat] < SENTINEL_M - 1e-6).sum())
        print(f"  {cat:14s}: stored={stored_found}  bbox_recomputed={bbox_found}")

    # ------------------------------------------------------------------ plots
    print("\nGenerating diagnostic plots...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Radius sensitivity curves
    plt.figure(figsize=(10, 6))
    for cat in CATEGORIES:
        sub = sens_df[(sens_df["category"] == cat) & (sens_df["radius_m"] != "prod_bbox")]
        x = sub["radius_m"].astype(float).to_numpy() / 1000.0
        y = sub["pct_within"].astype(float).to_numpy()
        plt.plot(x, y, marker="o", label=cat)
    plt.xlabel("Discovery radius (km)")
    plt.ylabel("% hotspots with entity within radius")
    plt.title("Radius sensitivity — % activation vs discovery radius (true recomputed distances)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "radius_sensitivity.png", dpi=120)
    plt.close()

    # 2. LF activation bar chart
    plt.figure(figsize=(10, 6))
    lfs_sorted = lf_df.sort_values("fire_rate_pct")["labeling_function"].tolist()
    rates = lf_df.set_index("labeling_function").loc[lfs_sorted, "fire_rate_pct"]
    plt.barh(rates.index, rates.values)
    plt.xlabel("Fire rate (%)")
    plt.title("Labeling function activation on 1268 real classified rows")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "lf_activation.png", dpi=120)
    plt.close()

    # 3. Feature missingness (sentinel/constant for distance columns)
    plt.figure(figsize=(10, 5))
    sent_perc = []
    cols_d = [f"distance_to_{c}_m" for c in CATEGORIES if f"distance_to_{c}_m" in classified.columns]
    for c in cols_d:
        sent_perc.append(100.0 * (classified[c] >= SENTINEL_M).mean())
    plt.bar([c.replace("distance_to_", "").replace("_m", "") for c in cols_d], sent_perc)
    plt.ylabel("% rows with NO entity (sentinel 999 km)")
    plt.title("Spatial feature missingness — % of 1268 hotspots with no entity found")
    plt.ylim(0, 105)
    for i, v in enumerate(sent_perc):
        plt.text(i, v + 2, f"{v:.0f}%", ha="center")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_missingness.png", dpi=120)
    plt.close()

    # 4. Unclassified root cause bar
    plt.figure(figsize=(10, 5))
    rc = root_rows[:2]
    plt.bar([r["root_cause"].split("(")[0][:60] for r in rc],
            [r["count"] for r in rc])
    plt.ylabel("unclassified rows")
    plt.title(f"Unclassified root cause (n={n_all})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "unclassified_rootcause.png", dpi=120)
    plt.close()

    # 5. Distance distribution histograms for found categories
    for cat in ["industry", "refinery", "oil_gas", "mining", "power_plant"]:
        mcol = f"distance_to_{cat}_m"
        if mcol not in classified.columns:
            continue
        vals = classified[mcol]
        found = vals[vals < SENTINEL_M]
        if len(found) == 0:
            continue
        plt.figure(figsize=(7, 4))
        plt.hist(found / 1000.0, bins=30, color="steelblue")
        plt.xlabel("Nearest distance (km)")
        plt.ylabel("hotspots")
        plt.title(f"{cat} — nearest-distance distribution ({len(found)} found)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{cat}_distance_histogram.png", dpi=120)
        plt.close()
    print(f"  plots -> {OUT_DIR}")

    print("\n" + "=" * 78)
    print("AUDIT COMPLETE — artifacts in reports/audit/")
    print("=" * 78)


if __name__ == "__main__":
    main()