"""
fetch_state.py
Fetch forest + agriculture for a SINGLE Indian state from Overpass.

Usage:
    venv/Scripts/python -u -m scripts.fetch_state --state gujarat
    venv/Scripts/python -u -m scripts.fetch_state --state tamil_nadu
    venv/Scripts/python -u -m scripts.fetch_state --state puis-haryana

States with FIRMS hotspots (9 states):
    tamil_nadu, karnataka, andhra_pradesh, maharashtra,
    rajasthan, punjab, haryana, gujarat, madhya_pradesh

Output:
    - Prints summary to stdout
    - Appends to data/raw/osm/osm_forest_agriculture.json (accumulative)
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Make app importable
backend_dir = str(Path(__file__).resolve().parents[1])
root_dir = str(Path(__file__).resolve().parents[0])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.services.osm_service import classify_osm_category

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_FILE = Path("data/raw/osm/osm_forest_agriculture.json")

STATES: Dict[str, Any] = {
    # Main states
    "tamil_nadu":     (76.20, 8.05, 80.40, 13.60),
    "karnataka":      (74.05, 11.55, 78.60, 18.45),
    "andhra_pradesh": (76.75, 12.65, 84.75, 19.15),
    "maharashtra":    (72.60, 15.60, 80.90, 22.05),
    "rajasthan":      (69.50, 23.10, 78.25, 30.25),
    "punjab":         (73.85, 29.55, 76.95, 32.55),
    "haryana":        (74.45, 27.65, 77.60, 30.95),
    "gujarat":        (68.15, 20.10, 73.95, 24.70),
    "madhya_pradesh": (74.05, 21.10, 82.80, 26.90),
    "jammu_kashmir":  (73.75, 32.30, 80.30, 35.50),
    "himachal_pradesh":(75.55, 30.40, 79.00, 33.20),
    "uttarakhand":    (77.55, 28.70, 81.05, 31.45),
    "delhi":          (76.85, 28.40, 77.35, 28.90),
    "uttar_pradesh":  (77.10, 23.85, 84.70, 30.45),
    "kerala":         (74.85, 8.20, 77.40, 12.80),
    "telangana":      (77.25, 15.85, 81.85, 19.95),
    "odisha":         (81.50, 17.50, 87.50, 22.55),
    "jharkhand":      (83.30, 21.95, 87.95, 25.35),
    "chhattisgarh":   (80.25, 17.85, 84.40, 24.10),
    "west_bengal":    (85.85, 21.50, 89.95, 27.30),
    "assam":          (89.70, 24.10, 96.05, 27.95),
    "bihar":          (83.30, 24.30, 88.30, 27.85),
    # Sub-regions for bigger states (split because full bbox times out)
    "mp_west":        (74.05, 21.10, 78.50, 26.90),
    "mp_east":        (78.50, 21.10, 82.80, 26.90),
    "up_north":       (77.10, 23.85, 80.30, 27.00),
    "up_central":     (77.10, 23.85, 84.70, 23.85),
    "up_east":        (77.10, 27.00, 84.70, 30.45),
    "wb_west":        (85.85, 21.50, 87.50, 27.30),
    "wb_central":     (87.50, 21.50, 89.95, 27.30),
    "wb_east":        (85.85, 21.50, 89.95, 24.30),
    "bihar_north":    (83.30, 27.85, 85.80, 27.85),
    "bihar_south":    (83.30, 24.30, 85.80, 24.30),
    "bihar_central":  (85.80, 24.30, 88.30, 27.85),
}

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

HEADERS = {
    "User-Agent": "THERMOSCOPE-AI/1.0 (NTRO SIH26162 demo)",
    "Accept": "application/json",
}

TIMEOUT = 90  # seconds per mirror per query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fetch_overpass(query: str, timeout: int = TIMEOUT) -> Optional[Dict[str, Any]]:
    """
    Try all Overpass mirrors, return first successful response.
    Returns None if all mirrors fail.
    """
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    errors: List[str] = []
    for i, url in enumerate(OVERPASS_URLS, 1):
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
            err = f"{type(e).__name__}: {str(e)[:120]}"
            errors.append(f"  [{i}] {url.split('//')[1]}: {err}")
    # All failed — print what we know
    print("  All mirrors failed:")
    for e in errors:
        print(e)
    return None


def query_state(state: str, bbox: Tuple[float, float, float, float]) -> Optional[Dict[str, Any]]:
    """Build and run the Overpass query for one state."""
    w, s, e, n = bbox
    query = (
        f"[out:json][timeout:60];\n"
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
    return fetch_overpass(query)


def load_cache() -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Load existing cache. Returns (sites, set_of_cached_states)."""
    if not CACHE_FILE.exists():
        return [], set()
    with open(CACHE_FILE) as f:
        sites = json.load(f)
    states = {s.get("state") for s in sites if s.get("state")}
    return sites, states


def save_cache(sites: List[Dict[str, Any]]) -> None:
    os.makedirs(CACHE_FILE.parent, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(sites, f, indent=2)


def process_elements(elements: List[Dict], state: str, existing_keys: Set) -> Tuple[int, List[Dict]]:
    """
    Filter elements to forest/agriculture, skip duplicates.
    Returns: (new_count, new_sites_list)
    """
    new_sites: List[Dict[str, Any]] = []
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
        new_sites.append({
            "id": elem.get("id"),
            "name": tags.get("name") or f"{cat}",
            "category": cat,
            "lat": float(el_lat),
            "lon": float(el_lon),
            "state": state,
        })
        added += 1
    return added, new_sites


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def fetch_single_state(state: str) -> bool:
    """
    Fetch one state. Returns True if successful, False if failed.
    """
    bbox = STATES.get(state)
    if bbox is None:
        print(f"ERROR: Unknown state '{state}'.")
        print(f"\nAvailable states ({len(STATES)}):")
        for s in sorted(STATES.keys()):
            print(f"  {s}")
        return False

    print(f"\n{'='*60}")
    print(f"Fetching: {state}")
    print(f"Bounding box: west={bbox[0]}, south={bbox[1]}, east={bbox[2]}, north={bbox[3]}")
    print(f"{'='*60}")

    # Load existing cache to avoid duplicates
    existing_sites, existing_states = load_cache()
    existing_keys: Set = {(s["category"], s.get("id")) for s in existing_sites}

    print(f"Existing cache: {len(existing_sites)} sites, {len(existing_states)} states")
    if state in existing_states:
        print(f"[SKIP] {state} already in cache. Nothing to fetch.")
        print(f"{'='*60}\n")
        return True

    # Fetch
    print(f"Querying Overpass...")
    t0 = time.time()
    data = query_state(state, bbox)
    elapsed = time.time() - t0

    if data is None:
        print(f"FAILED after {elapsed:.0f}s")
        print(f"{'='*60}")
        return False

    elements = data.get("elements", [])
    added, new_sites = process_elements(elements, state, existing_keys)

    # Save
    all_sites = existing_sites + new_sites
    save_cache(all_sites)

    cats = Counter(s["category"] for s in new_sites)
    print(f"SUCCESS after {elapsed:.0f}s")
    print(f"  elements returned: {len(elements)}")
    print(f"  new forest/agri sites: {added}")
    if cats:
        print(f"  by category: {dict(cats)}")
    print(f"  total cache now: {len(all_sites)} sites from {len(existing_states) + (1 if state not in existing_states else 0)} states")
    print(f"{'='*60}\n")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch forest+agriculture OSM data for ONE Indian state"
    )
    parser.add_argument(
        "--state",
        help="State name (see available states below)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available state names and exit",
    )
    args = parser.parse_args()

    if args.list:
        print(f"Available states ({len(STATES)}):")
        for s in sorted(STATES.keys()):
            print(f"  {s}")
        sys.exit(0)

    success = fetch_single_state(args.state)
    sys.exit(0 if success else 1)
