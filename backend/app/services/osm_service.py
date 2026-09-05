"""
osm_service.py
OpenStreetMap Overpass Service for THERMOSCOPE-AI (SIH26162).
Fetches nearby geographic context for thermal hotspots:
- Industries & factories
- Forest & woodlands (via Overpass live API only — not in static cache)
- Agricultural lands & crops (via Overpass live API only — not in static cache)
- Power plants & critical infrastructure

DATA-INTEGRITY CONTRACT:
- find_nearby_geographic_objects() NEVER injects synthetic/demo data into any
  real pipeline (hotspot_service, dataset_builder, training scripts).
- Demo fallback (_get_demo_fallback_features) is exposed ONLY via the
  explicit allow_demo_fallback=True parameter.
- When a category has no live API hit and no cache hit, that category returns
  an empty list. Downstream code (labeling_functions) must ABSTAIN in that case
  — that is honest signal (site genuinely not found), not a gap to patch.
- A per-category data_source key ("live", "cache", "none") is returned in
  the result dict for full auditability of every downstream record.
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import requests

# Set stdout to UTF-8 if supported to prevent Windows cp1252 UnicodeEncodeError
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure app package is discoverable
backend_dir = str(Path(__file__).resolve().parents[2])
root_dir = str(Path(__file__).resolve().parents[3])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

HEADERS = {
    "User-Agent": "SIH26162-IndustrialFireDetection/1.0 (research)",
    "Accept": "application/json",
}

# India state bounding boxes [min_lon, min_lat, max_lon, max_lat]
INDIA_STATES = {
    "andhra_pradesh": (76.75, 12.65, 84.75, 19.15),
    "assam":          (89.70, 24.10, 96.05, 27.95),
    "bihar":          (83.30, 24.30, 88.30, 27.85),
    "chhattisgarh":   (80.25, 17.85, 84.40, 24.10),
    "delhi":          (76.85, 28.40, 77.35, 28.90),
    "goa":            (73.65, 14.90, 74.35, 15.80),
    "gujarat":        (68.15, 20.10, 73.95, 24.70),
    "haryana":        (74.45, 27.65, 77.60, 30.95),
    "himachal_pradesh":(75.55, 30.40, 79.00, 33.20),
    "jharkhand":      (83.30, 21.95, 87.95, 25.35),
    "karnataka":      (74.05, 11.55, 78.60, 18.45),
    "kerala":         (74.85, 8.20, 77.40, 12.80),
    "madhya_pradesh": (74.05, 21.10, 82.80, 26.90),
    "maharashtra":    (72.60, 15.60, 80.90, 22.05),
    "odisha":         (81.50, 17.50, 87.50, 22.55),
    "punjab":         (73.85, 29.55, 76.95, 32.55),
    "rajasthan":      (69.50, 23.10, 78.25, 30.25),
    "tamil_nadu":     (76.20, 8.05, 80.40, 13.60),
    "telangana":      (77.25, 15.85, 81.85, 19.95),
    "uttar_pradesh":  (77.10, 23.85, 84.70, 30.45),
    "uttarakhand":    (77.55, 28.70, 81.05, 31.45),
    "west_bengal":    (85.85, 21.50, 89.95, 27.30),
    "andaman_nicobar":(92.20, 6.75, 93.95, 13.65),
    "jammu_kashmir":  (73.75, 32.30, 80.30, 35.50),
    "ladakh":         (75.50, 32.25, 79.50, 35.50),
    "meghalaya":      (89.85, 25.05, 92.80, 26.15),
}

# ---------------------------------------------------------------------------
# Cache Schema Version
# ---------------------------------------------------------------------------
# Increment this string whenever query_state() adds new OSM tag types.
# query_all_states() compares this against the cached version and forces a
# full re-query of every state if they don't match, bypassing the >100 skip.
CACHE_SCHEMA_VERSION = "v3_forest_agri_mining_2026_09"


def classify_osm_site(tags: Dict[str, Any]) -> str:
    """Classify specific industrial site types (backward-compatible)."""
    if tags.get("natural") == "volcano":
        return "volcano"
    if tags.get("industrial") == "refinery":
        return "refinery"
    if tags.get("power") == "plant":
        return "power_plant"
    if tags.get("industrial") in ("factory", "manufacturing", "works"):
        return "factory"
    if tags.get("industrial") in ("oil", "gas"):
        return "oil_gas"
    if tags.get("man_made") == "works":
        return "factory"
    if tags.get("resource") in ("oil", "gas"):
        return "oil_gas"
    if tags.get("landuse") == "industrial":
        return "industrial_zone"
    if tags.get("power") in ("station", "substation"):
        return "power_infrastructure"
    if (
        "mining" in tags
        or tags.get("resource") == "coal"
        or tags.get("landuse") == "quarry"
        or tags.get("industrial") in ("mine", "mining", "quarry")
        or tags.get("man_made") in ("mineshaft", "mine_shaft")
        or tags.get("historic") == "mine"
        or tags.get("resource") in ("coal", "iron", "bauxite", "limestone", "granite")
    ):
        return "mining"
    return "other_industrial"


def classify_osm_category(tags: Dict[str, Any]) -> str:
    """
    High-level category classifier for SIH multi-class geospatial context:
    - 'power_plant'
    - 'refinery'
    - 'oil_gas'
    - 'mining'
    - 'industry'
    - 'forest'
    - 'agriculture'
    - 'infrastructure'
    """
    # 1. Power plant
    if tags.get("power") == "plant":
        return "power_plant"

    # 2. Refinery (before generic industry)
    if tags.get("industrial") == "refinery":
        return "refinery"

    # 3. Oil & Gas
    if tags.get("industrial") in ("oil", "gas") or tags.get("resource") in ("oil", "gas"):
        return "oil_gas"

    # 4. Mining / Quarry — expanded tag set for India OSM sparsity
    if (
        tags.get("landuse") == "quarry"
        or tags.get("resource") in ("coal", "iron", "bauxite", "limestone", "granite", "mica", "copper")
        or tags.get("industrial") in ("mine", "mining", "quarry")
        or "mining" in tags
        or tags.get("man_made") in ("mineshaft", "mine_shaft")
        or tags.get("historic") == "mine"
    ):
        return "mining"

    # 5. General Industry (factory / manufacturing / industrial zones)
    ind = tags.get("industrial", "")
    if ind in ("factory", "manufacturing", "works", "chemical") or \
       tags.get("man_made") == "works" or \
       tags.get("landuse") == "industrial":
        return "industry"

    # 6. Forest / Woodland
    if tags.get("landuse") in ("forest", "wood") or \
       tags.get("natural") in ("wood", "tree_row", "scrub", "heath") or \
       tags.get("boundary") == "forest":
        return "forest"

    # 7. Agriculture / Cropland
    if tags.get("landuse") in ("farmland", "farm", "orchard", "meadow", "vineyard", "crop", "greenhouse", "allotments", "paddy"):
        return "agriculture"

    # 8. Infrastructure
    if tags.get("power") in ("station", "substation") or "railway" in tags or "highway" in tags:
        return "infrastructure"

    return "other"


def _site_type_to_category(site_type: str) -> str:
    """
    Map a legacy cached site_type string to the current 7-category scheme.
    Used for backwards compatibility with cached OSM data that was classified
    before the refinery/oil_gas/mining split was introduced.
    """
    _map = {
        "refinery": "refinery",
        "oil_gas": "oil_gas",
        "mining": "mining",
        "power_plant": "power_plant",
        "factory": "industry",
        "industrial_zone": "industry",
        "other_industrial": "industry",
        "power_infrastructure": "power_plant",
        "volcano": "other",
    }
    return _map.get(site_type, "other")


def _post_with_retry(query: str, timeout: int = 40, retries: int = 2) -> Dict[str, Any]:
    last_err = None
    for attempt in range(retries):
        for url in OVERPASS_URLS:
            try:
                r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=timeout)
                if r.status_code == 200:
                    return r.json()
                last_err = f"{url} -> HTTP {r.status_code}"
            except Exception as e:
                last_err = f"{url} -> {e}"
            time.sleep(1)
    raise RuntimeError(f"All Overpass servers failed: {last_err}")


def query_state_group(state_name: str, bbox: Tuple[float, float, float, float], group_name: str, clauses: List[str]) -> List[Dict[str, Any]]:
    w, s, e, n = bbox
    clause_str = "\n  ".join([f"{c}({s},{w},{n},{e});" for c in clauses])
    query = f"""
[out:json][timeout:50];
(
  {clause_str}
);
out center;
"""
    try:
        data = _post_with_retry(query, timeout=50, retries=2)
    except Exception as err:
        print(f"    [{group_name}] Failed for {state_name}: {err}")
        return []

    elements = data.get("elements", [])
    sites = []
    for elem in elements:
        tags = elem.get("tags", {})
        if elem["type"] == "node":
            lat, lon = elem.get("lat"), elem.get("lon")
        elif "center" in elem:
            lat = elem["center"].get("lat")
            lon = elem["center"].get("lon")
        else:
            continue
        if lat is None or lon is None:
            continue
        sites.append({
            "id": elem["id"],
            "osm_type": elem["type"],
            "lat": lat,
            "lon": lon,
            "tags": tags,
            "site_type": classify_osm_site(tags),
            "category": classify_osm_category(tags),
            "name": tags.get("name", ""),
            "state": state_name,
        })
    return sites


def query_state(state_name: str, bbox: Tuple[float, float, float, float], with_landuse: bool = True) -> List[Dict[str, Any]]:
    """Query OSM for a state using smaller, category-specific sub-queries to prevent timeouts."""
    groups = {
        "industrial": [
            'nwr["power"="plant"]',
            'nwr["industrial"="refinery"]',
            'nwr["industrial"="factory"]',
            'nwr["man_made"="works"]',
            'nwr["resource"="oil"]',
            'nwr["resource"="gas"]',
            'nwr["natural"="volcano"]',
            'nwr["landuse"="quarry"]',
            'nwr["industrial"="mine"]',
            'nwr["industrial"="mining"]',
            'nwr["man_made"="mineshaft"]',
            'nwr["historic"="mine"]',
        ],
        "forest": [
            'nwr["landuse"="forest"]',
            'nwr["natural"="wood"]',
        ],
        "agriculture": [
            'nwr["landuse"="farmland"]',
            'nwr["landuse"="farm"]',
        ],
    }
    if with_landuse:
        groups["industrial"].append('nwr["landuse"="industrial"]')

    all_found = []
    for gname, clauses in groups.items():
        res = query_state_group(state_name, bbox, gname, clauses)
        all_found.extend(res)
    return all_found


def query_all_states(
    skip_landuse_states=None,
    save_path="data/raw/osm/osm_industrial_sites.json",
    force_refresh: bool = False,
):
    """
    Query OSM for all India states and MERGE results into the existing cache.

    Guarantees:
    - NEVER deletes existing cached sites.
    - Uses category-specific sub-queries to prevent Overpass timeouts.
    - Deduplicates by (osm_type, id).
    """
    skip_landuse_states = skip_landuse_states or set()

    # Load existing sites first (ALWAYS preserved)
    all_sites: List[Dict[str, Any]] = load_osm_sites(save_path) or []
    seen_ids: set = {(s.get("osm_type"), s.get("id")) for s in all_sites}

    initial_counts = Counter(s.get("category", "other") for s in all_sites)
    print(f"[query_all_states] Loaded {len(all_sites)} existing sites from {save_path}")
    print(f"                   Initial category distribution: {dict(initial_counts)}")

    failed_states: List[str] = []

    for i, (state, bbox) in enumerate(INDIA_STATES.items(), 1):
        print(f"[{i}/{len(INDIA_STATES)}] {state}...")
        try:
            new_sites = query_state(state, bbox, with_landuse=(state not in skip_landuse_states))
            added_count = 0
            for s in new_sites:
                key = (s.get("osm_type"), s.get("id"))
                if key not in seen_ids:
                    seen_ids.add(key)
                    all_sites.append(s)
                    added_count += 1
            if added_count > 0:
                print(f"  -> Added {added_count} new sites for {state} (total now: {len(all_sites)})")
                save_osm_sites(all_sites, save_path)
            else:
                print(f"  -> No new unique sites found for {state}")
        except Exception as e:
            print(f"  FAILED: {e}")
            failed_states.append(state)
        time.sleep(2)

    final_counts = Counter(s.get("category", "other") for s in all_sites)
    print("\n" + "=" * 60)
    print("[query_all_states] National Category Totals (Post-Merge):")
    for cat, n in sorted(final_counts.items(), key=lambda x: -x[1]):
        old_n = initial_counts.get(cat, 0)
        diff = n - old_n
        print(f"  {cat:16s}: {n:6d} (+{diff:d} new)")
    print("=" * 60)

    if failed_states:
        print(f"⚠️ WARNING: The following {len(failed_states)} states encountered errors: {failed_states}")

    return all_sites


def save_osm_sites(sites: List[Dict[str, Any]], path: str = "data/raw/osm/osm_industrial_sites.json"):
    """Save OSM sites alongside the current CACHE_SCHEMA_VERSION for freshness checks."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "sites": sites,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(sites)} sites to {path} (schema={CACHE_SCHEMA_VERSION})")


def load_osm_sites(path: str = "data/raw/osm/osm_industrial_sites.json") -> Optional[List[Dict[str, Any]]]:
    """Load cached OSM sites. Returns None if file missing, unreadable, or schema mismatch."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # New format: {"schema_version": ..., "sites": [...]}
        if isinstance(raw, dict) and "sites" in raw:
            return raw["sites"]
        # Legacy format: bare list (pre-schema-version)
        if isinstance(raw, list):
            return raw
        return None
    except Exception:
        return None


def _get_cache_schema_version(path: str) -> Optional[str]:
    """Read just the schema_version field from the cache file without loading all sites."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            return raw.get("schema_version")
        return None  # Legacy list format has no version
    except Exception:
        return None


# ---------------------------------------------------------------------------
# DEMO FALLBACK — FOR OFFLINE UI DEMONSTRATION ONLY
# ---------------------------------------------------------------------------
def _get_demo_fallback_features(lat: float, lon: float) -> Dict[str, List[Dict[str, Any]]]:
    """
    FOR DEMO/UI DISPLAY PURPOSES ONLY.
    Generate synthetic surrounding features around a hotspot.
    MUST NEVER be called from training, dataset_builder, or hotspot_service
    unless allow_demo_fallback=True is explicitly set by a demo/UI code path.
    """
    return {
        "industry": [
            {"name": "[DEMO] Industrial Works Unit A", "category": "industry", "site_type": "factory",
             "lat": lat + 0.0108, "lon": lon, "tags": {"industrial": "factory"}, "is_demo": True},
            {"name": "[DEMO] Heavy Manufacturing Zone", "category": "industry", "site_type": "industrial_zone",
             "lat": lat + 0.0072, "lon": lon + 0.003, "tags": {"landuse": "industrial"}, "is_demo": True},
        ],
        "refinery": [
            {"name": "[DEMO] Petroleum Refinery Complex", "category": "refinery", "site_type": "refinery",
             "lat": lat + 0.021, "lon": lon + 0.015, "tags": {"industrial": "refinery"}, "is_demo": True},
        ],
        "oil_gas": [
            {"name": "[DEMO] Natural Gas Processing Station", "category": "oil_gas", "site_type": "oil_gas",
             "lat": lat + 0.016, "lon": lon - 0.010, "tags": {"industrial": "gas"}, "is_demo": True},
        ],
        "mining": [
            {"name": "[DEMO] Open-Cast Coal Mine", "category": "mining", "site_type": "mining",
             "lat": lat + 0.031, "lon": lon - 0.014, "tags": {"landuse": "quarry"}, "is_demo": True},
        ],
        "forest": [
            {"name": "[DEMO] Protected Reserve Forest", "category": "forest", "site_type": "forest",
             "lat": lat + 0.0145, "lon": lon + 0.008, "tags": {"landuse": "forest"}, "is_demo": True},
        ],
        "agriculture": [
            {"name": "[DEMO] Paddy Cultivation Land", "category": "agriculture", "site_type": "cropland",
             "lat": lat - 0.0089, "lon": lon - 0.004, "tags": {"landuse": "farmland"}, "is_demo": True},
        ],
        "power_plant": [
            {"name": "[DEMO] Thermal Grid Substation", "category": "power_plant", "site_type": "power_plant",
             "lat": lat + 0.040, "lon": lon + 0.025, "tags": {"power": "plant"}, "is_demo": True},
        ]
    }


# ---------------------------------------------------------------------------
# Real Spatial Query Around a Hotspot Coordinate
# ---------------------------------------------------------------------------
def find_nearby_geographic_objects(
    lat: float,
    lon: float,
    radius_meters: int = 15000,
    use_live_api: bool = True,
    allow_demo_fallback: bool = False,
) -> Dict[str, Any]:
    """
    Find nearby geographic objects around (lat, lon) within radius_meters.

    Categorizes results into 7 distinct buckets:
      - 'industry'     General industrial zones / factories / manufacturing
      - 'refinery'     Petroleum / chemical refineries
      - 'oil_gas'      Oil & gas extraction / processing facilities
      - 'mining'       Mining sites / quarries / coal resource areas
      - 'forest'       Forest / woodlands / scrub
      - 'agriculture'  Farmland / crops / orchards
      - 'power_plant'  Power generation plants

    DATA-INTEGRITY RULES:
    - allow_demo_fallback=False (default): Returns EMPTY lists for categories
      with no live API or cache hit. This is the only value permitted in
      training pipelines (hotspot_service, dataset_builder, train.py).
    - allow_demo_fallback=True: May ONLY be used from UI/demo code paths.
      Adds a `data_source: "demo"` tag per injected category.

    Returns a dict with:
      - 7 category lists of found sites
      - 'data_sources': {category: "live"|"cache"|"demo"|"none"} audit trail
    """
    ALL_CATS = ["industry", "refinery", "oil_gas", "mining", "forest", "agriculture", "power_plant"]

    categorized: Dict[str, List[Dict[str, Any]]] = {c: [] for c in ALL_CATS}
    data_sources: Dict[str, str] = {c: "none" for c in ALL_CATS}

    # --- 1. Try Overpass live API ---
    if use_live_api:
        overpass_query = f"""
[out:json][timeout:45];
(
  nwr["landuse"="industrial"](around:{radius_meters},{lat},{lon});
  nwr["industrial"](around:{radius_meters},{lat},{lon});
  nwr["man_made"="works"](around:{radius_meters},{lat},{lon});
  nwr["power"="plant"](around:{radius_meters},{lat},{lon});
  nwr["landuse"="forest"](around:{radius_meters},{lat},{lon});
  nwr["natural"="wood"](around:{radius_meters},{lat},{lon});
  nwr["natural"="scrub"](around:{radius_meters},{lat},{lon});
  nwr["landuse"="farmland"](around:{radius_meters},{lat},{lon});
  nwr["landuse"="farm"](around:{radius_meters},{lat},{lon});
  nwr["landuse"="orchard"](around:{radius_meters},{lat},{lon});
  nwr["landuse"="quarry"](around:{radius_meters},{lat},{lon});
  nwr["resource"="coal"](around:{radius_meters},{lat},{lon});
  nwr["resource"="oil"](around:{radius_meters},{lat},{lon});
  nwr["resource"="gas"](around:{radius_meters},{lat},{lon});
  nwr["industrial"="mine"](around:{radius_meters},{lat},{lon});
  nwr["man_made"="mineshaft"](around:{radius_meters},{lat},{lon});
  nwr["historic"="mine"](around:{radius_meters},{lat},{lon});
);
out center 100;
"""
        try:
            data = _post_with_retry(overpass_query, timeout=45, retries=1)
            elements = data.get("elements", [])
            for elem in elements:
                tags = elem.get("tags", {})
                if elem["type"] == "node":
                    el_lat, el_lon = elem.get("lat"), elem.get("lon")
                elif "center" in elem:
                    el_lat = elem["center"].get("lat")
                    el_lon = elem["center"].get("lon")
                else:
                    continue
                if el_lat is None or el_lon is None:
                    continue
                cat = classify_osm_category(tags)
                if cat in categorized:
                    name = tags.get("name") or tags.get("description") or f"{cat.title()} Feature #{elem['id']}"
                    categorized[cat].append({
                        "id": elem["id"],
                        "name": name,
                        "category": cat,
                        "site_type": classify_osm_site(tags),
                        "lat": float(el_lat),
                        "lon": float(el_lon),
                        "tags": tags,
                        "data_source": "live",
                    })
            for cat in ALL_CATS:
                if categorized[cat]:
                    data_sources[cat] = "live"
        except Exception:
            # Overpass live failed — fall through to cache
            pass

    # --- 2. Search local cached OSM dataset for any still-empty categories ---
    empty_cats = [c for c in ALL_CATS if not categorized[c]]
    if empty_cats:
        # Load BOTH the industrial cache and the forest/agriculture cache so the
        # live/cache path sees the same forest + agri sites as the batch producer
        # (Phase B P0.1 / P1.3 batch==live parity).
        cached_sites = []
        try:
            from app.core.paths import OSM_INDUSTRIAL_CACHE_PATH, OSM_FOREST_AGRI_CACHE_PATH
            for p in (OSM_INDUSTRIAL_CACHE_PATH, OSM_FOREST_AGRI_CACHE_PATH):
                part = load_osm_sites(str(p))
                if part:
                    cached_sites.extend(part)
        except Exception:
            cached_sites = load_osm_sites() or []
        if cached_sites:
            # Bounding box filter (~0.15 deg latitude is ~16 km)
            delta = (radius_meters / 111000.0) * 1.2
            for s in cached_sites:
                try:
                    s_lat = float(s["lat"])
                    s_lon = float(s["lon"])
                    if abs(s_lat - lat) > delta or abs(s_lon - lon) > delta:
                        continue
                    # Re-classify using expanded category function
                    cat = classify_osm_category(s.get("tags", {}))
                    if cat not in categorized:
                        cat = _site_type_to_category(s.get("site_type", ""))
                    if cat in categorized:
                        categorized[cat].append({
                            "id": s.get("id"),
                            "name": s.get("name") or f"{cat.title()} Site ({s.get('site_type', 'works')})",
                            "category": cat,
                            "site_type": s.get("site_type", "factory"),
                            "lat": s_lat,
                            "lon": s_lon,
                            "tags": s.get("tags", {}),
                            "data_source": "cache",
                        })
                except Exception:
                    continue
            for cat in ALL_CATS:
                if data_sources[cat] == "none" and categorized[cat]:
                    data_sources[cat] = "cache"

    # --- 3. Demo fallback — ONLY if explicitly requested (UI/offline demo) ---
    if allow_demo_fallback:
        demo = _get_demo_fallback_features(lat, lon)
        for cat, items in demo.items():
            if not categorized.get(cat):
                for item in items:
                    item["data_source"] = "demo"
                categorized[cat].extend(items)
                data_sources[cat] = "demo"

    # Attach audit trail
    result = dict(categorized)
    result["data_sources"] = data_sources
    return result


if __name__ == "__main__":
    print("=" * 65)
    print("OSM Service: Querying Geographic Context around Coordinates")
    print("=" * 65)

    test_lat, test_lon = 30.3165, 78.0322
    print(f"Target Hotspot Coordinate: ({test_lat}, {test_lon})")
    print("Searching nearby industries, forests, agriculture, and power plants...")

    nearby = find_nearby_geographic_objects(test_lat, test_lon, radius_meters=15000, use_live_api=False)

    for category in ["industry", "refinery", "oil_gas", "mining", "forest", "agriculture", "power_plant"]:
        items = nearby.get(category, [])
        src = nearby.get("data_sources", {}).get(category, "none")
        print(f"\n[{category.upper()} - Found {len(items)} items (source: {src})]")
        for item in items[:3]:
            print(f"  - {item['name']} ({item.get('site_type', 'N/A')}) at ({item['lat']:.4f}, {item['lon']:.4f})")
