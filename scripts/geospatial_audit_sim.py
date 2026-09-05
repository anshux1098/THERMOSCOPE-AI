"""
geospatial_audit_sim.py
================================================================================
AUDIT SIMULATION (READ-ONLY) — "what would change if the forest/agriculture OSM
cache had been merged into feature generation, exactly like the industrial cache?"

No production file, threshold, label, or model is modified. This runs entirely
in memory on copies of the classified rows and answers the audit question:

  - How many forest/agriculture LFs would fire?
  - How would the weak-supervision label distribution change?
  - How much of the 95.78% spatial-gate unclassified block is attributable to
    the un-merged forest/agriculture cache?

Usage:
    python scripts/geospatial_audit_sim.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

from app.core.paths import CLASSIFIED_DATASET_PATH, OSM_FOREST_AGRI_CACHE_PATH
from app.services.osm_service import classify_osm_category
from app.intelligence.labeling_functions import (
    ALL_LABELING_FUNCTIONS,
    apply_labeling_functions,
    aggregate_votes,
)


def _haversine_matrix(lats, lons, s_lats, s_lons):
    lat1 = np.radians(lats)[:, None]
    lat2 = np.radians(s_lats)[None, :]
    dlat = np.radians(s_lats)[None, :] - lat1
    dlon = np.radians(s_lons)[None, :] - np.radians(lons)[:, None]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * 6371000.0 * np.arcsin(np.sqrt(a))


def main() -> None:
    print("=" * 78)
    print("AUDIT SIMULATION — forest/agriculture cache merge impact (READ-ONLY)")
    print("=" * 78)

    fa_sites = json.load(open(OSM_FOREST_AGRI_CACHE_PATH, encoding="utf-8"))

    def _categorize(site):
        c = classify_osm_category(site.get("tags") or {})
        if c not in ("forest", "agriculture"):
            c = site.get("category")
        return c

    forest = [s for s in fa_sites if _categorize(s) == "forest"]
    agri = [s for s in fa_sites if _categorize(s) == "agriculture"]
    print(f"forest sites in raw cache: {len(forest)} | agriculture: {len(agri)}")

    classified = pd.read_csv(CLASSIFIED_DATASET_PATH)
    lats = classified["latitude"].to_numpy().astype(float)
    lons = classified["longitude"].to_numpy().astype(float)

    def nearest(cat):
        if not cat:
            return np.full(len(lats), 999000.0)
        d = _haversine_matrix(
            lats, lons,
            np.array([float(s["lat"]) for s in cat]),
            np.array([float(s["lon"]) for s in cat]),
        )
        return d.min(axis=1)

    d_forest = nearest(forest)
    d_agri = nearest(agri)

    # Baseline (current production behavior) label distribution from LFs
    print("\n--- BASELINE (current) ---")
    base_labels = [aggregate_votes(apply_labeling_functions(r))
                   for _, r in classified.iterrows()]
    base_counts = collections.Counter(base_labels)
    for k, v in sorted(base_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:24s}: {v}")

    # Simulation: overwrite forest/agri distances in-memory (copies only)
    # Behavior matches _compute_distances_from_cache: nearest distance, sentinel
    # otherwise. Keep flags consistent (has_forest_5km / has_agriculture_5km).
    print("\n--- SIMULATION (forest/agri cache MERGED into features) ---")
    sim_rows = classified.copy()
    sim_rows["distance_to_forest_m"] = np.round(d_forest, 1)
    sim_rows["distance_to_agriculture_m"] = np.round(d_agri, 1)
    sim_rows["dist_forest"] = np.round(d_forest / 1000.0, 3)
    sim_rows["dist_agriculture"] = np.round(d_agri / 1000.0, 3)
    sim_rows["has_forest_5km"] = (d_forest <= 5000.0).astype(int)
    sim_rows["has_agriculture_5km"] = (d_agri <= 5000.0).astype(int)

    lf_counts = collections.Counter()
    labels = []
    for _, row in sim_rows.iterrows():
        votes = apply_labeling_functions(row)
        for lf, v in votes.items():
            if v is not None:
                lf_counts[lf] += 1
        labels.append(aggregate_votes(votes))
    sim_counts = collections.Counter(labels)

    for k, v in sorted(sim_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:24s}: {v}")

    print("\nPer-LF fire counts (baseline -> simulated):")
    for lf in ALL_LABELING_FUNCTIONS:
        base = sum(1 for lbl, v in
                   zip(base_labels + [None] * (len(sim_rows) - len(base_labels)),
                       [apply_labeling_functions(r) for r in
                        classified.iloc[:0]]) if False)  # placeholder
    # Cleaner: recompute baseline LF counts
    base_lf = collections.Counter()
    for _, r in classified.iterrows():
        for lf, v in apply_labeling_functions(r).items():
            if v is not None:
                base_lf[lf] += 1
    for lf in ALL_LABELING_FUNCTIONS:
        n = lf.__name__
        print(f"  {n:32s}: baseline={base_lf.get(n, 0):4d}  simulated={lf_counts.get(n, 0):4d}")

    # How many currently-spatial-blocked rows become reachable?
    print("\nImpact on the 95.8% spatial-gate block:")
    forest_within_15 = int((d_forest <= 15000.0).sum())
    agri_within_15 = int((d_agri <= 15000.0).sum())
    uncl_before = base_counts.get("unclassified", 0)
    uncl_after = sim_counts.get("unclassified", 0)
    print(f"  hotspots within 15 km (LF forest threshold) of a forest site   : {forest_within_15}")
    print(f"  hotspots within 15 km (LF agriculture threshold) of farmland   : {agri_within_15}")
    print(f"  unclassified labels: {uncl_before} -> {uncl_after} "
          f"(net change {uncl_after - uncl_before:+d})")
    print("=" * 78)


if __name__ == "__main__":
    main()