"""
add_synthetic_mining.py
Add known Indian mining locations to the industrial cache.
Sites from known mining regions — Jharkhand (coal belt), Odisha (iron ore),
Chhattisgarh (coal), Karnataka (iron/ manganese), Rajasthan (zinc).

These are approximate coordinates of major mine clusters — used as proxy
locations when real Overpass mining data isn't available.
"""
import json
import os
from collections import Counter
from pathlib import Path

CACHE = "data/raw/osm/osm_industrial_sites.json"


# Known mining regions in India (approximate cluster centers)
MINING_REGIONS = [
    # Jharkhand coal belt — largest coal mining region
    ("Jharia coalfield", 23.70, 86.45, "Jharkhand"),
    ("Bokaro coalfield", 23.65, 86.05, "Jharkhand"),
    ("Hutar coalfield", 23.75, 86.10, "Jharkhand"),
    # Odisha iron ore belt
    ("Keonjhar iron ore belt", 21.70, 85.50, "Odisha"),
    ("Mayurbhanj iron ore", 21.95, 85.75, "Odisha"),
    ("Sundergarh iron ore", 21.75, 84.20, "Odisha"),
    # Chhattisgarh coal
    ("Korba coalfield", 22.35, 82.90, "Chhattisgarh"),
    ("Bhilai coalfield", 21.55, 80.85, "Chhattisgarh"),
    ("Mahan coal block", 21.55, 81.05, "Chhattisgarh"),
    # Karnataka
    ("Bellary iron ore belt", 15.20, 76.50, "Karnataka"),
    ("Hospet mining region", 15.30, 76.35, "Karnataka"),
    # Rajasthan
    ("Zawar zinc mines", 24.75, 75.30, "Rajasthan"),
    ("Kayatha zinc deposits", 24.50, 75.10, "Rajasthan"),
    ("Udaipur mining region", 24.55, 73.85, "Rajasthan"),
    # Odisha manganese
    ("Gangpur manganese", 21.65, 85.25, "Odisha"),
    # MP
    ("Singrauli coal belt", 24.25, 82.25, "Madhya Pradesh"),
    ("Satna coalfield", 24.55, 81.05, "Madhya Pradesh"),
    # Andhra
    ("Krishna-Godavari onshore", 16.50, 81.75, "Andhra Pradesh"),
    # Gujarat
    ("Cambay basin oil", 22.50, 72.50, "Gujarat"),
    # Assam
    ("Assam oil fields", 26.75, 94.05, "Assam"),
]


def main():
    if not os.path.exists(CACHE):
        print(f"Cache not found: {CACHE}")
        return

    with open(CACHE) as f:
        sites = json.load(f)

    existing_ids = {(s.get("osm_type"), s.get("id")) for s in sites}
    print(f"Existing: {len(sites)} sites")
    print(f"Current mining: {sum(1 for s in sites if s.get('site_type')=='mining')}")

    added = 0
    for name, lat, lon, state in MINING_REGIONS:
        if ("mining", f"synth_{name}") in existing_ids:
            continue
        sites.append({
            "id": f"synth_{name}",
            "osm_type": "node",
            "lat": lat,
            "lon": lon,
            "tags": {
                "landuse": "quarry",
                "name": name,
                "description": "Known mining region (synthetic proxy)",
            },
            "site_type": "mining",
            "category": "mining",
            "name": name,
            "state": state,
        })
        existing_ids.add(("node", f"synth_{name}"))
        added += 1

    with open(CACHE, "w") as f:
        json.dump(sites, f, indent=2)

    print(f"\nAdded {added} mining sites")
    print(f"Total cache: {len(sites)}")
    print(f"Mining now: {sum(1 for s in sites if s.get('site_type')=='mining')}")
    print(f"\nTypes: {dict(Counter(s.get('site_type','?') for s in sites))}")


if __name__ == "__main__":
    main()
