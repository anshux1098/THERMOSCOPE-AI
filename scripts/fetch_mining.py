"""
fetch_mining.py
Fetch mining sites from Overpass — add to industrial cache.

Mining tags: landuse=quarry, resource=coal, industrial=mine
"""
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

backend_dir = str(Path(__file__).resolve().parents[1])
root_dir = str(Path(__file__).resolve().parents[0])
for p in (backend_dir, root_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

HEADERS = {
    "User-Agent": "THERMOSCOPE-AI/1.0",
    "Accept": "application/json",
}

CACHE = "data/raw/osm/osm_industrial_sites.json"


def fetch(query: str, timeout: int = 90) -> dict:
    data = urllib.parse.urlencode({"data": query}).encode()
    for url in OVERPASS_URLS:
        try:
            req = urllib.request.Request(url, data=data, headers={**HEADERS, "Content-Type": "application/x-www-form-urlencoded"})
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                return json.loads(r.read())
        except Exception as e:
            print(f"  {url.split('//')[1]}: {type(e).__name__} ({str(e)[:80]})")
    raise RuntimeError("All mirrors failed")


def main():
    if not os.path.exists(CACHE):
        print(f"Cache not found: {CACHE}")
        return

    with open(CACHE) as f:
        sites = json.load(f)
    existing_ids = {(s.get("osm_type"), s.get("id")) for s in sites}

    print(f"Existing: {len(sites)} sites ({Counter(s.get('site_type','?') for s in sites)})")

    # India-wide mining query
    print("\nFetching mining sites (India-wide)...")
    t0 = time.time()
    query = (
        "[out:json][timeout:90];\n"
        "(nwr[landuse=quarry](20,-10,40,40);nwr[resource=coal](20,-10,40,40);"
        "nwr[industrial=mine](20,-10,40,40);nwr[industrial=mining](20,-10,40,40);"
        "nwr[mining](20,-10,40,40);nwr[place=village][ mining ](20,-10,40,40););\n"
        "out center 200;\n"
    )
    data = fetch(query)
    elapsed = time.time() - t0
    elements = data.get("elements", [])
    print(f"  Got {len(elements)} elements in {elapsed:.1f}s")

    mining_added = 0
    for elem in elements:
        tags = elem.get("tags", {})
        if elem.get("type") == "node":
            lat, lon = elem.get("lat"), elem.get("lon")
        elif "center" in elem:
            lat = elem["center"].get("lat")
            lon = elem["center"].get("lon")
        else:
            continue
        if lat is None or lon is None:
            continue
        key = (elem.get("type"), elem.get("id"))
        if key in existing_ids:
            continue
        existing_ids.add(key)
        sites.append({
            "id": elem.get("id"),
            "osm_type": elem.get("type"),
            "lat": float(lat),
            "lon": float(lon),
            "tags": tags,
            "site_type": "mining",
            "category": "mining",
            "name": tags.get("name") or tags.get("operator") or f"mining_{elem.get('id')}",
            "state": "unknown",
        })
        mining_added += 1

    with open(CACHE, "w") as f:
        json.dump(sites, f, indent=2)

    print(f"\nAdded {mining_added} mining sites")
    print(f"Total cache: {len(sites)} sites")
    print(f"Mining now: {sum(1 for s in sites if s.get('site_type')=='mining')}")


if __name__ == "__main__":
    main()
