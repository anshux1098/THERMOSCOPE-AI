"""
osm_query_states.py
Query OpenStreetMap Overpass API state-by-state for Indian industrial sites.
This avoids the 91-second timeout by splitting into smaller queries.
Owner: P1 - Data Engineer
"""
import os
import json
import time
import requests
from collections import Counter

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

HEADERS = {
    "User-Agent": "SIH26162-IndustrialFireDetection/1.0 (research)",
    "Accept": "application/json",
}


def classify_osm_site(tags):
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


def _post_with_retry(query, timeout=60, retries=2):
    last_err = None
    for attempt in range(retries):
        for url in OVERPASS_URLS:
            try:
                r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=timeout)
                if r.status_code == 200:
                    return r.json()
                last_err = f"{url} -> {r.status_code}"
            except Exception as e:
                last_err = f"{url} -> {e}"
            time.sleep(1)
    raise RuntimeError(f"All Overpass servers failed: {last_err}")


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


def query_state(state_name, bbox, with_landuse=True):
    w, s, e, n = bbox
    if with_landuse:
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
  nwr["landuse"="industrial"]({s},{w},{n},{e});
);
out center;
"""
    else:
        # Smaller query omitting landuse to avoid timeout
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
);
out center;
"""
    data = _post_with_retry(query, timeout=50, retries=2)
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
            "name": tags.get("name", ""),
            "state": state_name,
        })
    return sites


def query_all_states(skip_landuse_states=None, save_path="data/raw/osm_industrial_sites.json"):
    """Query each state. skip_landuse_states is a set of state names
    where we skip the landuse=industrial tag (slower queries).
    """
    skip_landuse_states = skip_landuse_states or set()
    all_sites = []
    seen_ids = set()
    failed = []

    # Load existing data to resume
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            all_sites = json.load(f)
        for s in all_sites:
            seen_ids.add((s["osm_type"], s["id"]))
        print(f"Resuming with {len(all_sites)} existing sites")

    # Two passes: first pass without landuse for all (faster)
    # then second pass for only industrial zones in big states
    print("=" * 60)
    print(f"Querying {len(INDIA_STATES)} Indian states (state-by-state)")
    print("=" * 60)

    for i, (state, bbox) in enumerate(INDIA_STATES.items(), 1):
        # Skip if already have data for this state
        existing_for_state = [s for s in all_sites if s.get("state") == state]
        if existing_for_state and len(existing_for_state) > 100:
            print(f"\n[{i}/{len(INDIA_STATES)}] {state}: SKIP (already have {len(existing_for_state)} sites)")
            continue

        print(f"\n[{i}/{len(INDIA_STATES)}] {state}...")
        try:
            sites = query_state(state, bbox, with_landuse=(state not in skip_landuse_states))
            added = 0
            for s in sites:
                key = (s["osm_type"], s["id"])
                if key not in seen_ids:
                    seen_ids.add(key)
                    all_sites.append(s)
                    added += 1
            print(f"  Got {len(sites)} sites, {added} new (total: {len(all_sites)})")
            # SAVE AFTER EVERY STATE (so we don't lose progress)
            save_osm_sites(all_sites, save_path)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(state)
            save_osm_sites(all_sites, save_path)  # save what we have
        time.sleep(2)  # be nice to the server

    # Second pass removed (saves time, and we already have 10k+ sites)

    # Skip second pass for large states (gujarat, maharashtra etc) - they time out
    # We already have enough data from first pass

    if failed:
        print(f"\nFailed states: {failed}")
    print(f"\nTotal unique sites: {len(all_sites)}")
    return all_sites


def save_osm_sites(sites, path="data/raw/osm_industrial_sites.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sites, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(sites)} sites to {path}")


def load_osm_sites(path="data/raw/osm_industrial_sites.json"):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    sites = query_all_states()
    if sites:
        save_osm_sites(sites)
        counts = Counter(s["site_type"] for s in sites)
        print("\nSite type distribution:")
        for st, n in counts.most_common():
            print(f"  {st}: {n}")
