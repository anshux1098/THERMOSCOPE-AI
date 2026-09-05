"""
fetch_osm_daily.py
Daily incremental OSM fetcher. Fetches forest + agriculture sites for
Indian states that have FIRMS hotspots, appends to a JSON cache.

Usage:
    # Run each day — fetches states that aren't already in the cache
    venv/Scripts/python -u -m scripts.fetch_osm_daily

Design:
    - CACHE_FILE: data/raw/osm/osm_forest_agriculture.json
    - First run: fetches all 9 hotspot states, saves to cache
    - Subsequent runs: only fetches states missing from cache
    - Per-state Overpass query (urllib — fast)
    - Per-state timeout: 60s
    - Failed states are NOT removed — they stay in the "missing" list
      so the next day's run tries them again
    - Sleeps 2s between queries (polite to Overpass)
"""
from __future__ import annotations

import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Make app package importable
repo_root = Path(__file__).resolve().parents[1]
backend_dir = str(repo_root / "backend")
root_dir = str(repo_root)
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from collections import Counter
from app.services.osm_service import classify_osm_category

# ---------------------------------------------------------------------------
# Paths + state list
# ---------------------------------------------------------------------------
CACHE_FILE = Path("data/raw/osm/osm_forest_agriculture.json")
FIRMS_CSV = "data/raw/firms_recent.csv"

# 9 Indian states where FIRMS hotspots actually exist (from earlier state analysis)
HOTSPOT_STATES = {
    "tamil_nadu":     (76.20, 8.05, 80.40, 13.60),    # 150 hotspots
    "karnataka":      (74.05, 11.55, 78.60, 18.45),   # 33
    "andhra_pradesh": (76.75, 12.65, 84.75, 19.15),   # 27
    "maharashtra":    (72.60, 15.60, 80.90, 22.05),   # 17
    "rajasthan":      (69.50, 23.10, 78.25, 30.25),   # 16
    "punjab":         (73.85, 29.55, 76.95, 32.55),   # 14
    "haryana":        (74.45, 27.65, 77.60, 30.95),   # 10
    "gujarat":        (68.15, 20.10, 73.95, 24.70),   # 10
    "madhya_pradesh": (21.10, 26.90, 74.05, 82.80),   # 1
}

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

HEADERS = {
    "User-Agent": "THERMOSCOPE-AI/1.0 (NTRO SIH26162 demo; contact: thermoscope@example.com)",
    "Accept": "application/json",
}


# ---------------------------------------------------------------------------
# Overpass fetch (urllib — fast, ~1-5s per query)
# ---------------------------------------------------------------------------
def _fetch_overpass_urllib(query: str, timeout: int = 60) -> Optional[Dict[str, Any]]:
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    last_err: Optional[str] = None
    for url in OVERPASS_URLS:
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={**HEADERS, "Content-Type": "application/x-www-form-urlencoded"},
            )
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                raw = resp.read()
            return json.loads(raw)
        except Exception as e:
            last_err = f"{url} -> {type(e).__name__}: {str(e)[:120]}"
            time.sleep(1)
    raise RuntimeError(f"All Overpass mirrors failed: {last_err}")


def _state_forest_agri_query(bbox: Tuple[float, float, float, float]) -> str:
    """Overpass query for forest + agriculture within a state bbox."""
    w, s, e, n = bbox
    return (
        f"[out:json][timeout:50];\n"
        f"(\n"
        f'  nwr["landuse"="forest"]({s},{w},{n},{e});\n'
        f'  nwr["natural"="wood"]({s},{w},{n},{e});\n'
        f'  nwr["natural"="tree_row"]({s},{w},{n},{e});\n'
        f'  nwr["natural"="scrub"]({s},{w},{n},{e});\n'
        f'  nwr["landuse"="farmland"]({s},{w},{n},{e});\n'
        f'  nwr["landuse"="farm"]({s},{w},{n},{e});\n'
        f'  nwr["landuse"="orchard"]({s},{w},{n},{e});\n'
        f'  nwr["landuse"="meadow"]({s},{w},{n},{e});\n'
        f");\n"
        f"out center 200;\n"
    )


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------
def _load_cache() -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Load existing cache. Returns (sites_list, set_of_states_already_cached)."""
    if not CACHE_FILE.exists():
        return [], set()
    with open(CACHE_FILE) as f:
        sites = json.load(f)
    states_seen = {s.get("state") for s in sites if s.get("state")}
    return sites, states_seen


def _save_cache(sites: List[Dict[str, Any]]) -> None:
    os.makedirs(CACHE_FILE.parent, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(sites, f, indent=2)


# ---------------------------------------------------------------------------
# Main daily fetch
# ---------------------------------------------------------------------------
def fetch_daily(verbose: bool = True) -> Dict[str, Any]:
    """
    Fetch forest + agriculture sites for any states missing from the cache.

    Returns a summary dict:
        {
            "states_fetched": [list of states fetched this run],
            "states_failed":  [list of states that failed],
            "sites_added":    N (int — number of new sites added to cache),
            "total_sites":    M (int — total sites in cache after this run)
        }
    """
    existing_sites, states_done = _load_cache()

    if verbose:
        print(f"[fetch_daily] Cache: {CACHE_FILE}")
        print(f"  existing sites: {len(existing_sites)}")
        print(f"  states already cached: {sorted(states_done) or '(none)'}")

    # States we need but don't have yet
    todo = {s: bb for s, bb in HOTSPOT_STATES.items() if s not in states_done}

    if not todo:
        if verbose:
            print("[fetch_daily] All states already in cache. Nothing to do.")
            print(f"  ({len(existing_sites)} sites from {len(states_done)} states)")
        return {
            "states_fetched": [],
            "states_failed": [],
            "sites_added": 0,
            "total_sites": len(existing_sites),
        }

    if verbose:
        print(f"[fetch_daily] Need to fetch: {len(todo)} state(s) -> {sorted(todo.keys())}")

    # Dedupe against existing (same OSM id + category)
    existing_keys: Set[Tuple[str, Any]] = {(s["category"], s.get("id")) for s in existing_sites}
    new_sites: List[Dict[str, Any]] = []
    states_fetched: List[str] = []
    states_failed: List[str] = []

    t_start = time.time()
    for i, (state_name, bbox) in enumerate(todo.items(), 1):
        query = _state_forest_agri_query(bbox)
        if verbose:
            print(f"  [{i}/{len(todo)}] {state_name:20s}  fetching...  ", end="", flush=True)
        try:
            data = _fetch_overpass_urllib(query, timeout=60)
            elements = data.get("elements", [])
            added = 0
            for elem in elements:
                tags = elem.get("tags", {})
                if elem.get("type") == "node":
                    el_lat, el_lon = elem.get("lat"), elem.get("lon")
                elif "center" in elem:
                    el_lat = elem["center"].get("lat")
                    el_lon = elem["center"].get("lon")
                else:
                    continue
                if el_lat is None or el_lon is None:
                    continue
                cat = classify_osm_category(tags)
                if cat not in ("forest", "agriculture"):
                    continue
                key = (cat, elem.get("id"))
                if key in existing_keys:
                    continue
                existing_keys.add(key)
                new_sites.append(
                    {
                        "id": elem.get("id"),
                        "name": tags.get("name") or f"{cat.title()} area",
                        "category": cat,
                        "lat": float(el_lat),
                        "lon": float(el_lon),
                        "state": state_name,
                    }
                )
                added += 1
            elapsed = time.time() - t_start
            if verbose:
                print(f"OK ({len(elements):3d} elements, {added:3d} new forest/agri) [{elapsed:.0f}s]")
            states_fetched.append(state_name)
        except Exception as e:
            elapsed = time.time() - t_start
            if verbose:
                print(f"FAILED ({type(e).__name__}: {str(e)[:80]}) [{elapsed:.0f}s]")
            states_failed.append(state_name)
        time.sleep(2)  # be polite

    # Persist (always — so next run sees partial progress)
    all_sites = existing_sites + new_sites
    _save_cache(all_sites)

    if verbose:
        print()
        print(f"[fetch_daily] Saved cache -> {CACHE_FILE}")
        print(f"  total sites now: {len(all_sites)}")
        if all_sites:
            cats = Counter(s["category"] for s in all_sites)
            states = Counter(s.get("state", "?") for s in all_sites)
            print(f"  by category: {dict(cats)}")
            print(f"  by state:    {dict(states)}")
        print()
        print(f"[fetch_daily] Run summary:")
        print(f"  fetched this run:  {states_fetched}")
        print(f"  failed this run:   {states_failed}")
        print(f"  new sites added:   {len(new_sites)}")
        print(f"  total in cache:    {len(all_sites)}")
        if states_failed:
            print()
            print(f"  [next day] Re-run this script to retry failed states.")

    return {
        "states_fetched": states_fetched,
        "states_failed": states_failed,
        "sites_added": len(new_sites),
        "total_sites": len(all_sites),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fetch_daily()
