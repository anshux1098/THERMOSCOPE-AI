"""
osm_service.py
OpenStreetMap Overpass Service for THERMOSCOPE-AI.
Fetches nearby geographic context for thermal hotspots:
- Industries & factories
- Forest & woodlands
- Agricultural lands & crops
- Power plants & critical infrastructure
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
    if "mining" in tags or tags.get("resource") == "coal":
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

    # 4. Mining / Quarry
    if (
        tags.get("landuse") == "quarry"
        or tags.get("resource") == "coal"
        or tags.get("industrial") in ("mine", "mining", "quarry")
        or "mining" in tags
    ):
        return "mining"

    # 5. General Industry (factory / manufacturing / industrial zones)
    ind = tags.get("industrial", "")
    if ind in ("factory", "manufacturing", "works", "chemical") or \
       tags.get("man_made") == "works" or \
       tags.get("landuse") == "industrial":
        return "industry"

    # 6. Forest / Woodland
    if tags.get("landuse") == "forest" or tags.get("natural") in ("wood", "tree_row", "scrub"):
        return "forest"

    # 7. Agriculture / Cropland
    if tags.get("landuse") in ("farmland", "farm", "orchard", "meadow", "vineyard", "crop", "greenhouse"):
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


def query_state(state_name: str, bbox: Tuple[float, float, float, float], with_landuse: bool = True) -> List[Dict[str, Any]]:
    w, s, e, n = bbox
    query = f"""
[out:json][timeout:30];
(
  nwr["power"="plant"]({s},{w},{n},{e});
  nwr["industrial"="refinery"]({s},{w},{n},{e});
  nwr["industrial"="factory"]({s},{w},{n},{e});
  nwr["man_made"="works"]({s},{w},{n},{e});
  nwr["resource"="oil"]({s},{w},{n},{e});
  nwr["resource"="gas"]({s},{w},{n},{e});
  nwr["natural"="volcano"]({s},{w},{n},{e});
  {"nwr['landuse'='industrial'](" + str(s) + "," + str(w) + "," + str(n) + "," + str(e) + ");" if with_landuse else ""}
);
out center;
"""
    data = _post_with_retry(query, timeout=45, retries=2)
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


def query_all_states(skip_landuse_states=None, save_path="data/raw/osm_industrial_sites.json"):
    skip_landuse_states = skip_landuse_states or set()
    all_sites = []
    seen_ids = set()

    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            all_sites = json.load(f)
        for s in all_sites:
            seen_ids.add((s.get("osm_type"), s.get("id")))
        print(f"Resuming with {len(all_sites)} existing sites")

    for i, (state, bbox) in enumerate(INDIA_STATES.items(), 1):
        existing_for_state = [s for s in all_sites if s.get("state") == state]
        if existing_for_state and len(existing_for_state) > 100:
            print(f"[{i}/{len(INDIA_STATES)}] {state}: SKIP (already have {len(existing_for_state)} sites)")
            continue

        print(f"[{i}/{len(INDIA_STATES)}] {state}...")
        try:
            sites = query_state(state, bbox, with_landuse=(state not in skip_landuse_states))
            for s in sites:
                key = (s.get("osm_type"), s.get("id"))
                if key not in seen_ids:
                    seen_ids.add(key)
                    all_sites.append(s)
            save_osm_sites(all_sites, save_path)
        except Exception as e:
            print(f"  FAILED: {e}")
        time.sleep(2)

    return all_sites


def save_osm_sites(sites: List[Dict[str, Any]], path: str = "data/raw/osm_industrial_sites.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sites, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(sites)} sites to {path}")


def load_osm_sites(path: str = "data/raw/osm_industrial_sites.json") -> Optional[List[Dict[str, Any]]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dynamic Spatial Query Around a Hotspot Coordinate
# ---------------------------------------------------------------------------

def _get_demo_fallback_features(lat: float, lon: float) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate realistic surrounding features around a hotspot for offline demonstration.
    Provides candidate sites for all 7 distance categories:
    industry, refinery, oil_gas, mining, forest, agriculture, power_plant.
    """
    return {
        "industry": [
            {"name": "Industrial Works Unit A", "category": "industry", "site_type": "factory", "lat": lat + 0.0108, "lon": lon, "tags": {"industrial": "factory"}},
            {"name": "Precision Engineering Works (Unit B)", "category": "industry", "site_type": "factory", "lat": lat + 0.00405, "lon": lon, "tags": {"industrial": "factory"}},
            {"name": "Heavy Manufacturing Zone C", "category": "industry", "site_type": "industrial_zone", "lat": lat + 0.0072, "lon": lon + 0.003, "tags": {"landuse": "industrial"}},
        ],
        "refinery": [
            {"name": "Petroleum Refinery Complex", "category": "refinery", "site_type": "refinery", "lat": lat + 0.021, "lon": lon + 0.015, "tags": {"industrial": "refinery"}},
        ],
        "oil_gas": [
            {"name": "Natural Gas Processing Station", "category": "oil_gas", "site_type": "oil_gas", "lat": lat + 0.016, "lon": lon - 0.010, "tags": {"industrial": "gas"}},
            {"name": "Crude Oil Extraction Field", "category": "oil_gas", "site_type": "oil_gas", "lat": lat - 0.012, "lon": lon + 0.018, "tags": {"resource": "oil"}},
        ],
        "mining": [
            {"name": "Open-Cast Coal Mine", "category": "mining", "site_type": "mining", "lat": lat + 0.031, "lon": lon - 0.014, "tags": {"landuse": "quarry", "resource": "coal"}},
        ],
        "forest": [
            {"name": "Protected Reserve Forest", "category": "forest", "site_type": "forest", "lat": lat + 0.0145, "lon": lon + 0.008, "tags": {"landuse": "forest"}},
            {"name": "Secondary Wooded Ridge", "category": "forest", "site_type": "forest", "lat": lat - 0.019, "lon": lon + 0.012, "tags": {"natural": "wood"}},
        ],
        "agriculture": [
            {"name": "Paddy Cultivation Land", "category": "agriculture", "site_type": "cropland", "lat": lat - 0.0089, "lon": lon - 0.004, "tags": {"landuse": "farmland"}},
            {"name": "Agro Mixed Farmland", "category": "agriculture", "site_type": "cropland", "lat": lat + 0.012, "lon": lon - 0.009, "tags": {"landuse": "farmland"}},
        ],
        "power_plant": [
            {"name": "Thermal Grid Substation", "category": "power_plant", "site_type": "power_plant", "lat": lat + 0.040, "lon": lon + 0.025, "tags": {"power": "plant"}},
        ]
    }


def find_nearby_geographic_objects(
    lat: float,
    lon: float,
    radius_meters: int = 15000,
    use_live_api: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
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

    Tries Overpass live API first. If unreachable or offline, searches cached OSM dataset
    and applies fallback demo features to ensure fail-safe operation.
    """
    categorized: Dict[str, List[Dict[str, Any]]] = {
        "industry": [],
        "refinery": [],
        "oil_gas": [],
        "mining": [],
        "forest": [],
        "agriculture": [],
        "power_plant": [],
    }

    if use_live_api:
        overpass_query = f"""
[out:json][timeout:30];
(
  nwr["landuse"="industrial"](around:{radius_meters},{lat},{lon});
  nwr["industrial"](around:{radius_meters},{lat},{lon});
  nwr["man_made"="works"](around:{radius_meters},{lat},{lon});
  nwr["power"="plant"](around:{radius_meters},{lat},{lon});
  nwr["landuse"="forest"](around:{radius_meters},{lat},{lon});
  nwr["natural"="wood"](around:{radius_meters},{lat},{lon});
  nwr["landuse"="farmland"](around:{radius_meters},{lat},{lon});
  nwr["landuse"="farm"](around:{radius_meters},{lat},{lon});
  nwr["landuse"="orchard"](around:{radius_meters},{lat},{lon});
  nwr["landuse"="quarry"](around:{radius_meters},{lat},{lon});
  nwr["resource"="coal"](around:{radius_meters},{lat},{lon});
  nwr["resource"="oil"](around:{radius_meters},{lat},{lon});
  nwr["resource"="gas"](around:{radius_meters},{lat},{lon});
);
out center 80;
"""
        try:
            data = _post_with_retry(overpass_query, timeout=30, retries=1)
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
                        "tags": tags
                    })
        except Exception:
            # Overpass live failed, proceed to local cache / demo fallback
            pass

    # Check local cached dataset if industry is empty (covers refinery, oil_gas, mining too)
    if not any([categorized["industry"], categorized["refinery"], categorized["oil_gas"], categorized["mining"]]):
        cached_sites = load_osm_sites()
        if cached_sites:
            # Bounding box filter (~0.15 deg latitude is ~16 km)
            delta = (radius_meters / 111000.0) * 1.2
            for s in cached_sites:
                try:
                    s_lat = float(s["lat"])
                    s_lon = float(s["lon"])
                    if abs(s_lat - lat) <= delta and abs(s_lon - lon) <= delta:
                        # Re-classify with the expanded category function
                        cat = classify_osm_category(s.get("tags", {}))
                        # Also fall back to site_type-based mapping for legacy cached data
                        if cat not in categorized:
                            site_type = s.get("site_type", "")
                            cat = _site_type_to_category(site_type)
                        if cat in categorized:
                            categorized[cat].append({
                                "id": s.get("id"),
                                "name": s.get("name") or f"{cat.title()} Site ({s.get('site_type', 'works')})",
                                "category": cat,
                                "site_type": s.get("site_type", "factory"),
                                "lat": s_lat,
                                "lon": s_lon,
                                "tags": s.get("tags", {})
                            })
                except Exception:
                    continue

    # Fallback to demo items if any category is empty (for live demos/offline)
    demo_fallback = _get_demo_fallback_features(lat, lon)
    for cat, items in demo_fallback.items():
        if not categorized.get(cat):
            categorized[cat].extend(items)

    return categorized


if __name__ == "__main__":
    print("=" * 65)
    print("OSM Service: Querying Geographic Context around Coordinates")
    print("=" * 65)

    test_lat, test_lon = 30.3165, 78.0322
    print(f"Target Hotspot Coordinate: ({test_lat}, {test_lon})")
    print("Searching nearby industries, forests, agriculture, and power plants...")

    nearby = find_nearby_geographic_objects(test_lat, test_lon, radius_meters=15000, use_live_api=False)

    for category, items in nearby.items():
        print(f"\n[{category.upper()} - Found {len(items)} items]")
        for item in items[:3]:
            print(f"  - {item['name']} ({item.get('site_type', 'N/A')}) at ({item['lat']:.4f}, {item['lon']:.4f})")
